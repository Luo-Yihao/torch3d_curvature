import torch
import nvdiffrast.torch as dr

import ops.mesh_geometry as mg
from pytorch3d.structures import Meshes

from PIL import Image

LIGHT_DIR = [0., 0., -1.]    #3

'''
=======================================================
Implementations from [Continous Remeshing For Inverse Rendering](https://github.com/Profactor/continuous-remeshing).
=======================================================
'''
def _translation(x, y, z, device):
    return torch.tensor([[1., 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]],device=device) #4,4

def _projection(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    p = torch.zeros([4,4],device=device)
    p[0,0] = 2*n/(r-l)
    p[0,2] = (r+l)/(r-l)
    p[1,1] = 2*n/(t-b) * (-1 if flip_y else 1)
    p[1,2] = (t+b)/(t-b)
    p[2,2] = -(f+n)/(f-n)
    p[2,3] = -(2*f*n)/(f-n)
    p[3,2] = -1
    return p #4,4

def make_star_cameras(az_count,pol_count,distance:float=10.,r=None, n=None, f=None, image_size=[512,512],device='cuda'):
    if r is None:
        r = 1/distance
    if n is None:
        n = 1
    if f is None:
        f = 50
    A = az_count
    P = pol_count
    C = A * P

    phi = torch.arange(0,A) * (2*torch.pi/A)
    phi_rot = torch.eye(3,device=device)[None,None].expand(A,1,3,3).clone()
    phi_rot[:,0,2,2] = phi.cos()
    phi_rot[:,0,2,0] = -phi.sin()
    phi_rot[:,0,0,2] = phi.sin()
    phi_rot[:,0,0,0] = phi.cos()
    
    theta = torch.arange(1,P+1) * (torch.pi/(P+1)) - torch.pi/2
    theta_rot = torch.eye(3,device=device)[None,None].expand(1,P,3,3).clone()
    theta_rot[0,:,1,1] = theta.cos()
    theta_rot[0,:,1,2] = -theta.sin()
    theta_rot[0,:,2,1] = theta.sin()
    theta_rot[0,:,2,2] = theta.cos()

    mv = torch.empty((C,4,4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:,:3,:3] = (theta_rot @ phi_rot).reshape(C,3,3)
    mv = _translation(0, 0, -distance, device) @ mv

    return mv, _projection(r,device, n=n, f=f)

def _warmup(glctx):
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device='cuda', **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

class NormalsRenderer:
    
    _glctx:dr.RasterizeGLContext = None
    
    def __init__(
            self,
            mv: torch.Tensor, #C,4,4
            proj: torch.Tensor, #C,4,4
            image_size: "tuple[int,int]",
            ):
        self._mvp = proj @ mv #C,4,4
        self._image_size = image_size
        self._glctx = dr.RasterizeCudaContext()
        _warmup(self._glctx)

    def render(self,
            vertices: torch.Tensor, #V,3 float
            normals: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        vert_col = (normals+1)/2 #V,3
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        return col #C,H,W,4

def calc_face_normals(
        vertices:torch.Tensor, #V,3 first vertex may be unreferenced
        faces:torch.Tensor, #F,3 long, first face may be all zero
        normalize:bool=False,
        )->torch.Tensor: #F,3
    """
         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    """
    full_vertices = vertices[faces] #F,C=3,3
    v0,v1,v2 = full_vertices.unbind(dim=1) #F,3
    face_normals = torch.cross(v1-v0,v2-v0, dim=1) #F,3
    if normalize:
        face_normals = torch.nn.functional.normalize(face_normals, eps=1e-6, dim=1) #TODO inplace?
    return face_normals #F,3

def calc_vertex_normals(
        vertices:torch.Tensor, #V,3 first vertex may be unreferenced
        faces:torch.Tensor, #F,3 long, first face may be all zero
        face_normals:torch.Tensor=None, #F,3, not normalized
        )->torch.Tensor: #F,3

    F = faces.shape[0]

    if face_normals is None:
        face_normals = calc_face_normals(vertices,faces)
    
    vertex_normals = torch.zeros((vertices.shape[0],3,3),dtype=vertices.dtype,device=vertices.device) #V,C=3,3
    vertex_normals.scatter_add_(dim=0,index=faces[:,:,None].expand(F,3,3),src=face_normals[:,None,:].expand(F,3,3))
    vertex_normals = vertex_normals.sum(dim=1) #V,3
    return torch.nn.functional.normalize(vertex_normals, eps=1e-6, dim=1)

'''
Our adaptations
'''

class AlphaRenderer(NormalsRenderer):
    '''
    Renderer that renders
    * normal
    * depth
    * shillouette
    '''
    
    def __init__(
            self,
            mv: torch.Tensor, #C,4,4
            proj: torch.Tensor, #C,4,4
            image_size: "tuple[int,int]",
            ):
        super().__init__(mv,proj,image_size)
        self._mv = mv
        self._proj = proj
        self.eps = 1e-4

    def forward(self,
                verts: torch.Tensor,
                normals: torch.Tensor,
                faces: torch.Tensor,
                curv_rescale: float = 10.0) -> torch.Tensor:
        '''
        Single pass without transparency.
        '''
        V = verts.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((verts, torch.ones(V, 1, device=verts.device)), axis=-1)  # V,3 -> V,4
        verts_clip = vert_hom @ self._mvp.transpose(-2, -1)  # C,V,4
        rast_out, _ = dr.rasterize(self._glctx,
                                verts_clip,
                                faces,
                                resolution=self._image_size,
                                grad_db=False)  # C,H,W,4

        # View-space normal
        vert_normals_hom = torch.cat((normals, torch.zeros(V, 1, device=verts.device)), axis=-1)  # V,3 -> V,4
        vert_normals_view = vert_normals_hom @ self._mv.transpose(-2, -1)  # C,V,4
        vert_normals_view = vert_normals_view[..., :3]  # C,V,3
        vert_normals_view[vert_normals_view[..., 2] > 0.] = \
            -vert_normals_view[vert_normals_view[..., 2] > 0.]
        vert_normals_view = vert_normals_view.contiguous()

        # View-space light direction
        lightdir = torch.tensor(LIGHT_DIR, dtype=torch.float32, device=verts.device)  # 3
        lightdir = lightdir.view((1, 1, 1, 3))  # 1,1,1,3

        # Pixel normals in view space
        pixel_normals_hom, _ = dr.interpolate(normals, rast_out, faces)  # C,H,W,3
        pixel_normals_hom = pixel_normals_hom / torch.clamp(
            torch.norm(pixel_normals_hom, p=2, dim=-1, keepdim=True), min=1e-5)

        pixel_normals_view, _ = dr.interpolate(vert_normals_view, rast_out, faces)  # C,H,W,3
        pixel_normals_view = pixel_normals_view / torch.clamp(
            torch.norm(pixel_normals_view, p=2, dim=-1, keepdim=True), min=1e-5)

        # Diffuse shading
        diffuse = torch.sum(lightdir * pixel_normals_view, -1, keepdim=True)  # C,H,W,1
        diffuse = torch.clamp(diffuse, min=0.0, max=1.0)
        diffuse = diffuse[..., [0, 0, 0]]  # C,H,W,3

        # Depth
        verts_clip_w = verts_clip[..., [3]]
        verts_clip_w[torch.logical_and(verts_clip_w >= 0.0, verts_clip_w < self.eps)] = self.eps
        verts_clip_w[torch.logical_and(verts_clip_w < 0.0, verts_clip_w > -self.eps)] = -self.eps

        verts_depth = (verts_clip[..., [2]] / verts_clip_w)  # C,V,1
        depth, _ = dr.interpolate(verts_depth, rast_out, faces)  # C,H,W,1
        depth = (depth + 1.) * 0.5  # Normalize depth to [0, 1]

        depth[rast_out[..., -1] == 0] = 1.0  # Exclude background
        depth = 1 - depth  # Invert depth for visualization
        max_depth = depth.max()
        min_depth = depth[depth > 0.0].min()  # Exclude background
        depth_info = {'raw': depth, 'max': max_depth, 'min': min_depth}

        # Silhouette (alpha)
        alpha = torch.clamp(rast_out[..., [-1]], max=1)  # C,H,W,1

        # Convert normals to RGB
        normals_rgb_hom = (pixel_normals_hom + 1.0) * 0.5  # Shift and scale normals from [-1, 1] to [0, 1]


        # Curvature 
        with torch.no_grad():
            meshes = Meshes(verts=[verts], faces=[faces])
            gs_curvatures_vert = mg.get_gaussian_curvature_vertices_packed(meshes).view(-1, 1) # V,1
            mean_curvature_vert = mg.get_mean_curvature_vertices_packed(meshes).view(-1, 1) # V,1
            total_curvature_vert = mean_curvature_vert**2 - 2*gs_curvatures_vert
            
            gs_curvatures_vert = torch.tanh(gs_curvatures_vert/curv_rescale**2)  # C,H,W,1
            gs_curvatures_vert = (gs_curvatures_vert + 1.0) * 0.5
            gs_curvatures, _ = dr.interpolate(gs_curvatures_vert, rast_out, faces)  # C,H,W,1
            gs_curvatures = torch.clamp(gs_curvatures, min=0.0, max=1.0)

            mean_curvature_vert = torch.tanh(mean_curvature_vert/curv_rescale)  # C,H,W,1
            mean_curvature_vert = torch.exp(mean_curvature_vert)/np.exp(1.0)
            mean_curvature, _ = dr.interpolate(mean_curvature_vert, rast_out, faces)  # C,H,W,1
            mean_curvature = torch.clamp(mean_curvature, min=0.0, max=1.0)

            total_curvature_vert = torch.tanh(total_curvature_vert/curv_rescale**2)  # C,H,W,1
            total_curvature, _ = dr.interpolate(total_curvature_vert, rast_out, faces)  # C,H,W,1
            total_curvature = torch.clamp(total_curvature, min=0.0, max=1.0)

        # Combine output: diffuse, depth, alpha, and normals (in RGB)
        col = torch.concat((diffuse, depth, alpha, normals_rgb_hom, 
                         gs_curvatures, mean_curvature, total_curvature), dim=-1)
        col = dr.antialias(col, rast_out, verts_clip, faces)  # C,H,W,8

        return col, depth_info


class MultiChannelRenderer:

    def __init__(self, verts: torch.Tensor, faces: torch.Tensor, device: str):

        # geometry info;
        self.gt_vertices = verts
        self.gt_faces = faces
        self.gt_vertex_normals = calc_vertex_normals(verts, faces)

        # rendered images;
        self.gt_images = None
        self.gt_depth_info = None

    def render(self, renderer: AlphaRenderer):

        target_images, target_depth_info = \
            renderer.forward(self.gt_vertices, self.gt_vertex_normals, self.gt_faces)

        self.gt_images = target_images
        self.gt_depth_info = target_depth_info

        return self.gt_images


    def diffuse_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,:3]

    def depth_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,[3,3,3]]

    def shillouette_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,[4,4,4]]
    
    def normal_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,5:8]

    def gs_curvatures_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,[8,8,8]]
    
    def mean_curvature_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,[9,9,9]]
    
    def total_curvature_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,[10,10,10]]
    
    def curvatures_rgb_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,[10,9,8]]

def compute_verts_depth(verts: torch.Tensor, mv: torch.Tensor, proj: torch.Tensor):
    '''
    Compute depth for each vertices using [mv, proj].

    @ verts: [# point, 3]
    @ mv: [# batch, 4, 4]
    @ proj: [# batch, 4, 4]
    '''

    verts_hom = torch.cat((verts, torch.ones_like(verts[:, [0]])), dim=-1)            # [V, 4]
    verts_hom = verts_hom.unsqueeze(0).unsqueeze(-1)                            # [1, V, 4, 1]
    e_mv = mv.unsqueeze(1)                                                      # [B, 1, 4, 4]
    e_proj = proj.unsqueeze(1)                                                  # [B, 1, 4, 4]
    verts_view = e_mv @ verts_hom                                               # [B, V, 4, 1]
    verts_proj = e_proj @ verts_view                                            # [B, V, 4, 1]
    verts_proj_w = verts_proj[..., [3], 0]                                      # [B, V, 1]

    # clamp w;
    verts_proj_w[torch.logical_and(verts_proj_w >= 0.0, verts_proj_w < 1e-4)] = 1e-4
    verts_proj_w[torch.logical_and(verts_proj_w < 0.0, verts_proj_w > -1e-4)] = -1e-4

    verts_ndc = verts_proj[..., :3, 0] / verts_proj_w                           # [B, V, 3]
    verts_depth = verts_ndc[..., 2]                                             # [B, V]

    return verts_depth

def compute_faces_view_normal(verts: torch.Tensor, faces: torch.Tensor, mv: torch.Tensor):
    '''
    Compute face normals in the view space using [mv].

    @ verts: [# point, 3]
    @ faces: [# face, 3]
    @ mv: [# batch, 4, 4]
    '''
    faces_normals = calc_face_normals(verts, faces, True)           # [F, 3]
    faces_normals_hom = torch.cat((faces_normals, torch.zeros_like(faces_normals[:, [1]])), dim=-1)   # [F, 4]
    faces_normals_hom = faces_normals_hom.unsqueeze(0).unsqueeze(-1)                    # [1, F, 4, 1]
    e_mv = mv.unsqueeze(1)                                                              # [B, 1, 4, 4]
    faces_normals_view = e_mv @ faces_normals_hom                                       # [B, F, 4, 1]
    faces_normals_view = faces_normals_view[:, :, :3, 0]                                # [B, F, 3]
    faces_normals_view[faces_normals_view[..., 2] > 0] = \
        -faces_normals_view[faces_normals_view[..., 2] > 0]                               # [B, F, 3]

    return faces_normals_view

def compute_faces_intense(verts: torch.Tensor, faces: torch.Tensor, mv: torch.Tensor, lightdir: torch.Tensor):
    '''
    Compute face intense using [mv] and [lightdir].

    @ verts: [# point, 3]
    @ faces: [# face, 3]
    @ mv: [# batch, 4, 4]
    @ lightdir: [# batch, 3]
    '''
    faces_normals_view = compute_faces_view_normal(verts, faces, mv)                        # [B, F, 3]
    faces_attr = torch.sum(lightdir.unsqueeze(1) * faces_normals_view, -1, keepdim=True)       # [B, F, 1]
    faces_attr = torch.clamp(faces_attr, min=0.0, max=1.0)                                     # [B, F, 1]
    faces_intense = faces_attr[..., 0]                                                    # [B, F]

    return faces_intense

def save_image(img, path):
    img = img.cpu().numpy()
    img = img * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)

if __name__ == "__main__":

    import argparse
    import os
    import numpy as np
    from pytorch3d.io import load_objs_as_meshes
    import trimesh
    import warnings
    warnings.filterwarnings("ignore")
    
    ## python render.py --mesh_path data_example/Kar.obj --result_dir results/ --num_viewpoints 6 --image_size 256 --device cuda:0
    '''
    Ground truth mesh
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, default="data_example/Kar.obj")
    # parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--result_dir", type=str, default="results/")

    parser.add_argument("--num_viewpoints", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    

    args = parser.parse_args()

    DEVICE = torch.device(args.device)

    mesh_name = os.path.basename(args.mesh_path).split('/')[-1]
    mesh_name = mesh_name.split('.')[0]

    image_save_path = os.path.join(args.result_dir, mesh_name)
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path, exist_ok=True)
    

    mesh_path = args.mesh_path
    mesh_tem = load_objs_as_meshes([mesh_path], device=DEVICE)
    mesh_tem = mg.normalize_mesh(mesh_tem, 0.88)

    verts, faces = mesh_tem.verts_list()[0], mesh_tem.faces_list()[0]
    
    print("===== Ground truth mesh =====")
    print("Id: ", mesh_name)
    print("Number of vertices: ", verts.shape[0])
    print("Number of faces: ", faces.shape[0])
    print("=============================")

    # save gt mesh;
    mesh = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy())
    mesh.export(os.path.join(image_save_path, "gt_mesh.obj"))

    '''
    Multi-channel renderer
    '''
    num_viewpoints = int(float(args.num_viewpoints))
    image_size = int(float(args.image_size))

    mv, proj = make_star_cameras(num_viewpoints, num_viewpoints, distance=2.0, r=0.6, n=1.0, f=3.0)
    proj = proj.unsqueeze(0).expand(mv.shape[0], -1, -1)
    renderer = AlphaRenderer(mv, proj, [image_size, image_size])

    gt_manager = MultiChannelRenderer(verts, faces, DEVICE)
    gt_manager.render(renderer)
    
    gt_diffuse_map = gt_manager.diffuse_images()
    gt_depth_map = gt_manager.depth_images()
    gt_shil_map = gt_manager.shillouette_images()
    gt_normals_map = gt_manager.normal_images()
    gt_gs_curv_map = gt_manager.gs_curvatures_images()
    gt_mean_curv_map = gt_manager.mean_curvature_images()
    gt_curv_rgb_map = gt_manager.curvatures_rgb_images()


    for i in range(len(gt_diffuse_map)):
        save_image(gt_diffuse_map[i], os.path.join(image_save_path, "diffuse_{}.png".format(i)))
        save_image(gt_depth_map[i], os.path.join(image_save_path, "depth_{}.png".format(i)))
        save_image(gt_shil_map[i], os.path.join(image_save_path, "shil_{}.png".format(i)))
        save_image(gt_normals_map[i], os.path.join(image_save_path, "normals_{}.png".format(i)))
        save_image(gt_gs_curv_map[i], os.path.join(image_save_path, "gs_curv_{}.png".format(i)))
        save_image(gt_mean_curv_map[i], os.path.join(image_save_path, "mean_curv_{}.png".format(i)))
        save_image(gt_curv_rgb_map[i], os.path.join(image_save_path, "curv_rgb_{}.png".format(i)))

    print("Saved images to: ", image_save_path)
    print("======== Done! ========")