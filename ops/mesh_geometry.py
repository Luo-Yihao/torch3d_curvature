import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh 
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian
from pytorch3d.ops import packed_to_padded
from torch_scatter import scatter
from einops import rearrange, repeat



def normalize_mesh(mesh, rescalar=0.99):
    """
    Normalize the mesh to fit in the unit sphere
    Args:
        mesh: Meshes object or trimesh object
        rescalar: float, the scale factor to rescale the mesh
    """
    if isinstance(mesh, Meshes):
       
        bbox = mesh.get_bounding_boxes()
        B = bbox.shape[0]
        center = (bbox[:, :, 0] + bbox[:, :, 1]) / 2
        center = center.view(B, 1, 3)
        size = (bbox[:, :, 1] - bbox[:, :, 0]) 

        scale = 2.0 / (torch.max(size, dim=1)[0]+1e-8).view(B, 1)*rescalar
        scale = scale.view(B, 1, 1)
        mesh = mesh.update_padded((mesh.verts_padded()-center)*scale)
        return mesh

    elif isinstance(mesh, trimesh.Trimesh):
        bbox_min, bbox_max = mesh.bounds
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min

        # Scale factor to normalize to [-1, 1]
        scale_factor = 2.0 / np.max(bbox_size)  # Ensures the longest side fits in [-1, 1]

        # Apply translation and scaling
        mesh.apply_translation(-bbox_center)  # Move the mesh center to the origin
        mesh.apply_scale(scale_factor)

    return mesh


def vert_feature_packed_padded(Surfaces: Meshes, feature):
    """
    Compute the feature of each vertices in a mesh
    Args:
        Surfaces: Meshes object
        feature: Tensor of shape (V, d) where V is the number of vertices and d is the feature dimension
    Returns:
        vert_feature: Tensor of shape (N,1) where N is the number of vertices
        the feature of a vertices is defined as the sum of the feature of the triangles that contains this vertices divided by the dual area of this vertices
    """

    vert_first_idx = Surfaces.mesh_to_verts_packed_first_idx()
    vert_feature_padded = packed_to_padded(feature, vert_first_idx, max_size=Surfaces.verts_padded().shape[1])
    return vert_feature_padded


def face_feature_packed_padded(Surfaces: Meshes, feature):
    """
    Compute the feature of each faces in a mesh
    Args:
        Surfaces: Meshes object
        feature: Tensor of shape (F, d) where F is the number of faces and d is the feature dimension
    Returns:
        face_feature: Tensor of shape (N,1) where N is the number of vertices
        the feature of a vertices is defined as the sum of the feature of the faces that contains this vertices divided by the dual area of this vertices
    """

    face_first_idx = Surfaces.mesh_to_faces_packed_first_idx()
    face_feature_padded = packed_to_padded(feature, face_first_idx, max_size=Surfaces.faces_padded().shape[1])
    return face_feature_padded


def get_faces_coordinates_padded(meshes: Meshes):
    """
    Get the faces coordinates of the meshes in padded format.
    return:
        face_coord_padded: [B, F, 3, 3]
    """
    face_mesh_first_idx = meshes.mesh_to_faces_packed_first_idx()
    face_coord_packed = get_faces_coordinates_packed(meshes.verts_packed(), meshes.faces_packed())

    face_coord_padded = packed_to_padded(face_coord_packed, face_mesh_first_idx, max_size=meshes.faces_padded().shape[1])

    return face_coord_padded



def get_faces_coordinates_packed(*args):
    """
    Get the faces coordinates of the meshes in padded format.
    return:
        face_coord_padded: [F, 3, 3]
    """
    if len(args) == 1:
        if isinstance(args[0], Meshes):
            vertices_packed = args[0].verts_packed()
            faces_packed = args[0].faces_packed()
        elif isinstance(args[0], trimesh.Trimesh):
            vertices_packed = args[0].vertices.astype(np.float32)
            faces_packed = args[0].faces.astype(np.int64)
    
    elif len(args) == 2:
        vertices_packed = args[0]
        faces_packed = args[1]

    face_coord_packed = vertices_packed[faces_packed,:]

    return face_coord_packed


def get_faces_angle_packed(*args):
    """
    Compute the angle of each face.
    Returns:
        angles: Tensor of shape (N,3) where N is the number of faces
    """
    

    if len(args) == 1:
        if isinstance(args[0], trimesh.Trimesh):
            return args[0].face_angles
        
    Face_coord = get_faces_coordinates_packed(*args)

    if not isinstance(Face_coord, torch.Tensor):
        Face_coord = torch.tensor(Face_coord, dtype=torch.float32)

    A = Face_coord[:,1,:] - Face_coord[:,0,:]
    B = Face_coord[:,2,:] - Face_coord[:,1,:]
    C = Face_coord[:,0,:] - Face_coord[:,2,:]
    
    angle_0 = torch.arccos(-torch.sum(A*C,dim=1)/(1e-10+(torch.norm(A,dim=1)*torch.norm(C,dim=1))))
    angle_1 = torch.arccos(-torch.sum(A*B,dim=1)/(1e-10+(torch.norm(A,dim=1)*torch.norm(B,dim=1))))
    angle_2 = torch.arccos(-torch.sum(B*C,dim=1)/(1e-10+(torch.norm(B,dim=1)*torch.norm(C,dim=1))))
    angles = torch.stack([angle_0,angle_1,angle_2],dim=1)

    if len(args) == 2 and not isinstance(args[1], torch.Tensor):
        angles = angles.detach().cpu().numpy()
        
    return angles


def get_dual_area_weights_packed(Surfaces: Meshes)->torch.Tensor:
    """
    Compute the dual area weights of 3 vertices of each triangles in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_weight: Tensor of shape (N,3) where N is the number of triangles
        the dual area of a vertices in a triangles is defined as the area of the sub-quadrilateral divided by three perpendicular bisectors
    """
    angles = get_faces_angle_packed(Surfaces)
    if not isinstance(angles, torch.Tensor):
        angles = torch.from_numpy(angles)
    
    angles_roll = torch.stack([angles, angles.roll(-1,dims=1), angles.roll(1,dims=1)],dim=1)

    sinanlge = torch.sin(angles)

    cosdiffangle = torch.cos(angles_roll[...,-2]-angles_roll[...,-1])

    dual_area_weight = sinanlge*cosdiffangle

    dual_area_weight = dual_area_weight/(torch.sum(dual_area_weight,dim=-1,keepdim=True)+1e-8)

    dual_area_weight = torch.clamp(dual_area_weight,0,1)

    dual_area_weight = dual_area_weight/dual_area_weight.sum(dim=-1,keepdim=True)

    return dual_area_weight


def get_dual_area_vertex_packed(Surfaces: Meshes, return_type = 'packed')->torch.Tensor:
    """
    Compute the dual area of each vertices in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_per_vertex: Tensor of shape (N,1) where N is the number of vertices
        the dual area of a vertices is defined as the sum of the dual area of the triangles that contains this vertices
    """
        
    dual_weights = get_dual_area_weights_packed(Surfaces)

    if isinstance(Surfaces, trimesh.Trimesh):
        face_areas = torch.from_numpy(Surfaces.area_faces).unsqueeze(-1).float()
        face2vertex_index = Surfaces.faces.reshape(-1)
        face2vertex_index = torch.from_numpy(face2vertex_index).long()
    elif isinstance(Surfaces, Meshes):
        face_areas = (Surfaces.faces_areas_packed().unsqueeze(-1))
        face2vertex_index = Surfaces.faces_packed().view(-1)
    else:
        raise ValueError("Surfaces must be a Meshes or a Trimesh object")

    dual_area_per_vertex = scatter((dual_weights*face_areas).view(-1), face2vertex_index, reduce='sum')

    return dual_area_per_vertex

def dual_gather_from_face_features_to_vertices_packed(mesh: Meshes, features_faces: torch.Tensor, mode='dual') -> torch.Tensor:
    """
    Gather face features to vertices with dual area weighting.
    Args:
        mesh: Meshes object representing a batch of meshes.
        features_faces: Tensor of shape (F, D) where F is the number of faces and D is the number of features.
        mode: str, either 'mean' or 'dual'.
    Returns:
        Tensor of shape (V, D) where V is the number of vertices.
    """
    if isinstance(mesh, Meshes):
        F = mesh.faces_packed().shape[0]
        face_2_vert_index = mesh.faces_packed()
        face_area = mesh.faces_areas_packed()
    elif isinstance(mesh, trimesh.Trimesh):
        F = mesh.faces.shape[0]
        face_2_vert_index = torch.tensor(mesh.faces, dtype=torch.long)
        face_area = torch.tensor(mesh.area_faces, dtype=torch.float32)

    D = features_faces.shape[-1]

    assert features_faces.shape[0] == F, "features_faces must have the same number of faces as the mesh."

    if isinstance(mesh, Meshes):
        F = mesh.faces_packed().shape[0]
        V = mesh.verts_packed().shape[0]
    elif isinstance(mesh, trimesh.Trimesh):
        F = mesh.faces.shape[0]
        V = mesh.vertices.shape[0]

    if mode == 'mean':
        return scatter(repeat(features_faces, 'F D -> (F V) D', V=3), face_2_vert_index, dim=0, reduce='mean', dim_size=V)

    dual_area_face_splitted = get_dual_area_weights_packed(mesh)*(face_area.view(F,1))
    face_features_weighted = dual_area_face_splitted*features_faces
    
    face_2_vert_index = rearrange(face_2_vert_index, 'F V -> (F V) 1', F=F, V=3)
    interal_features = scatter(face_features_weighted.view(-1,D), face_2_vert_index, dim=0, reduce='sum', dim_size=V)
    dual_area_vert = get_dual_area_vertex_packed(mesh)
    return interal_features*(dual_area_vert.reciprocal().view(-1,1))


def dual_interpolation_from_verts_to_faces_packed(mesh: Meshes, features_verts: torch.Tensor, mode='mean') -> torch.Tensor:
    """
    Interpolate vertex features to faces with dual area weighting.
    Args:
        mesh: Meshes object representing a batch of meshes.
        features_verts: Tensor of shape (V, D) where V is the number of vertices and D is the number of features.
        mode: str, either 'mean' or 'dual'.
    Returns:
        Tensor of shape (F, D) where F is the number of faces.
    """
    D = features_verts.shape[1]
    if isinstance(mesh, Meshes):
        faces = mesh.faces_packed()
        V = faces.shape[0]
        F = mesh.faces_packed().shape[0]
    elif isinstance(mesh, trimesh.Trimesh):
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        V = mesh.vertices.shape[0]
        F = faces.shape[0]
    assert features_verts.shape[0] == V, "features_verts must have the same number of vertices as the mesh."
    dual_weights = get_dual_area_weights_packed(mesh).view(F,3,1)

    features_verts_faces = features_verts[faces,:]
    if mode == 'mean':
        return (features_verts_faces).mean(dim=1)
    elif mode == 'dual':
        return (dual_weights*features_verts_faces).sum(dim=1)
    
def get_soft_volume(mesh: Meshes):
    """
    Compute the (soft) volume of the mesh using the Gauss-Integral theorem.
    """
    if isinstance(mesh, Meshes):
        face_areas = mesh.faces_areas_packed()
        face_normals = mesh.faces_normals_packed()
        face_barycenters = mesh.verts_packed()[mesh.faces_packed()].mean(dim=1)
        volume_ele = ((face_barycenters*face_normals).sum(dim=-1)*face_areas)
        volume_ele_padded = face_feature_packed_padded(mesh, volume_ele.view(-1,1))
        vol = volume_ele_padded.sum(dim=1)/3
        vol = vol.view(-1)
    elif isinstance(mesh, trimesh.Trimesh):
        face_areas = torch.from_numpy(mesh.area_faces).float()
        face_normals = torch.from_numpy(mesh.face_normals).float()
        face_barycenters = torch.from_numpy(mesh.triangles_center).float()
        vol = ((face_barycenters*face_normals).sum(dim=-1)*face_areas).sum()/3
    return vol


#### ----------------------------------Curvature --------------------------------------------

def get_gaussian_curvature_vertices_packed(Surfaces: Meshes, return_density=False)->torch.Tensor:
    """
    Compute the gaussian curvature of each vertices in a mesh by local Gauss-Bonnet theorem
    Args:
        Surfaces: Meshes object
        return_topology: bool, if True, return the Euler characteristic and genus of the mesh
    Returns:
        gaussian_curvature: Tensor of shape (N,1) where N is the number of vertices
        the gaussian curvature of a vertices is defined as the sum of the angles of the triangles that contains this vertices minus 2*pi and divided by the dual area of this vertices
    """
    if isinstance(Surfaces, trimesh.Trimesh):
        face2vertex_index = Surfaces.faces.reshape(-1)
        face2vertex_index = torch.from_numpy(face2vertex_index).long()
    elif isinstance(Surfaces, Meshes):

        face2vertex_index = Surfaces.faces_packed().view(-1)

    angle_face = get_faces_angle_packed(Surfaces)

    if not isinstance(angle_face, torch.Tensor):
        angle_face = torch.from_numpy(angle_face).float()


    dual_area_per_vertex = get_dual_area_vertex_packed(Surfaces)

    angle_sum_per_vertex = scatter(angle_face.view(-1), face2vertex_index, reduce='sum')
    if return_density:
        curvature = (2*np.pi - angle_sum_per_vertex)
    else:
        curvature = (2*np.pi - angle_sum_per_vertex)/(dual_area_per_vertex+1e-8)

    return curvature



class SolidAngle(nn.Module):
    def __init__(self):
        """
        Compute the solid angle of a batch of triangles
        Input: batch_of_three_vectors: [B, 3, 3]
        Output: solid_angle: [B,]
        """
        super(SolidAngle, self).__init__()
    def forward(self, batch_of_three_vectors):
        assert batch_of_three_vectors.shape[-1] == 3
        assert batch_of_three_vectors.shape[-2] == 3

        a_vert = batch_of_three_vectors[...,0,:] # [B, 3]    
        b_vert = batch_of_three_vectors[...,1,:] # [B, 3]
        c_vert = batch_of_three_vectors[...,2,:] # [B, 3]
        
        face_det = (a_vert * b_vert.cross(c_vert)).sum(dim=-1) # [B,]

        abc = batch_of_three_vectors.norm(dim=-1).prod(dim=-1) # [B,3]-->[B,]

        ab = (a_vert*b_vert).sum(-1) # [B,]
        bc = (b_vert*c_vert).sum(-1) # [B,]
        ac = (a_vert*c_vert).sum(-1) # [B,]

        solid_angle = 2*torch.arctan2(face_det, (abc + bc*a_vert.norm(dim=-1) + ac*b_vert.norm(dim=-1) + ab*c_vert.norm(dim=-1))) # []

        return solid_angle


### Curvature from faces

def get_gaussian_curvature_faces_packed(meshes: Meshes, return_density=False)->torch.Tensor:
    """
    Compute the gaussian curvature of each faces in a mesh
    Args:
        meshes: Meshes object
        return_density: bool, if True, return the gaussian curvature density
    Returns:
        gaussian_curvature: Tensor of shape (N,1) where N is the number of faces
        the gaussian curvature of a face is defined as the solid angle of the face divided by the area of the face
    """

    if isinstance(meshes, trimesh.Trimesh):
        face_coord_packed = get_faces_coordinates_packed(meshes)
        face_coord_packed = torch.from_numpy(face_coord_packed).float()
        trg_mesh_face_normal = meshes.vertex_normals
        trg_mesh_face_normal = trg_mesh_face_normal[meshes.faces,:]
        trg_mesh_face_normal = torch.from_numpy(trg_mesh_face_normal).float()
        face_area = torch.from_numpy(meshes.area_faces).float()

    elif isinstance(meshes, Meshes):
        face_coord_packed = get_faces_coordinates_packed(meshes)
        trg_mesh_normal = meshes.verts_normals_packed()
        trg_mesh_face_normal = trg_mesh_normal[meshes.faces_packed(),:]
        face_area = meshes.faces_areas_packed()
    

    normal_area = SolidAngle()(trg_mesh_face_normal)

    if return_density:
        return normal_area
    else:
        return normal_area/(face_area+1e-10)
    

    
def get_gaussian_curvature_vertices_from_face_packed(mesh: Meshes, mode='dual'):
    """
    Rather than computing the gaussian curvature at the vertices, we can compute it at the faces and then gather it to the vertices.
    Args:
        mesh: Meshes object
        mode: str, either 'mean' or 'dual'. If 'dual', use dual area weighting.
    """
    gc_face_packed = get_gaussian_curvature_faces_packed(mesh).view(-1,1)
    gc_vertex_packed = dual_gather_from_face_features_to_vertices_packed(mesh, gc_face_packed, mode=mode)
    return gc_vertex_packed


def get_mean_curvature_vertices_packed(mesh: Meshes):
    """
    Compute the mean curvature at the vertices of a mesh.
    """
    if isinstance(mesh, trimesh.Trimesh):
        meshes = Meshes(verts=[torch.tensor(mesh.vertices).float()], faces=[torch.tensor(mesh.faces)])
    elif isinstance(mesh, Meshes):
        meshes = mesh
        
    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    L, inv_areas = cot_laplacian(verts_packed, faces_packed)
    L = L/2

    L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
    mean_curvature_vector = (L.mm(verts_packed) - L_sum * verts_packed) * 0.5 * inv_areas*3
    mean_curvature = -(mean_curvature_vector*meshes.verts_normals_packed()).sum(dim=-1)
    return mean_curvature

### ------------------------------------------------------------------------------