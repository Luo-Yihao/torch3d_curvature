import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter

from pytorch3d.structures import Meshes
from einops import rearrange, einsum, repeat

import numpy as np

def faces_angle(meshs: Meshes)->torch.Tensor:
    """
    Compute the angle of each face in a mesh
    Args:
        meshs: Meshes object
    Returns:
        angles: Tensor of shape (N,3) where N is the number of faces
    """
    Face_coord = meshs.verts_packed()[meshs.faces_packed()]
    A = Face_coord[:,1,:] - Face_coord[:,0,:]
    B = Face_coord[:,2,:] - Face_coord[:,1,:]
    C = Face_coord[:,0,:] - Face_coord[:,2,:]
    
    angle_0 = torch.arccos(-torch.sum(A*C,dim=1)/(1e-10+(torch.norm(A,dim=1)*torch.norm(C,dim=1))))
    

    angle_1 = torch.arccos(-torch.sum(A*B,dim=1)/(1e-10+(torch.norm(A,dim=1)*torch.norm(B,dim=1))))
    angle_2 = torch.arccos(-torch.sum(B*C,dim=1)/(1e-10+(torch.norm(B,dim=1)*torch.norm(C,dim=1))))
    angles = torch.stack([angle_0,angle_1,angle_2],dim=1)
    return angles

def dual_area_weights_faces(Surfaces: Meshes)->torch.Tensor:
    """
    Compute the dual area weights of 3 vertices of each triangles in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_weight: Tensor of shape (N,3) where N is the number of triangles
        the dual area of a vertices in a triangles is defined as the area of the sub-quadrilateral divided by three perpendicular bisectors
    """
    angles = faces_angle(Surfaces)
    sin2angle = torch.sin(2*angles)
    dual_area_weight = sin2angle/(torch.sum(sin2angle,dim=-1,keepdim=True)+1e-8)
    dual_area_weight = (dual_area_weight[:,[2,0,1]]+dual_area_weight[:,[1,2,0]])/2
    
    
    return dual_area_weight


def dual_area_vertex(Surfaces: Meshes)->torch.Tensor:
    """
    Compute the dual area of each vertices in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_per_vertex: Tensor of shape (N,1) where N is the number of vertices
        the dual area of a vertices is defined as the sum of the dual area of the triangles that contains this vertices
    """
    dual_weights = dual_area_weights_faces(Surfaces)
    dual_areas = dual_weights*Surfaces.faces_areas_packed().view(-1,1)

    face2vertex_index = Surfaces.faces_packed().view(-1)

    dual_area_per_vertex = scatter(dual_areas.view(-1), face2vertex_index, reduce='sum')
    
    return dual_area_per_vertex.view(-1,1)


def gaussian_curvature(Surfaces: Meshes,return_topology=False)->torch.Tensor:
    """
    Compute the gaussian curvature of each vertices in a mesh by local Gauss-Bonnet theorem
    Args:
        Surfaces: Meshes object
        return_topology: bool, if True, return the Euler characteristic and genus of the mesh
    Returns:
        gaussian_curvature: Tensor of shape (N,1) where N is the number of vertices
        the gaussian curvature of a vertices is defined as the sum of the angles of the triangles that contains this vertices minus 2*pi and divided by the dual area of this vertices
    """

    face2vertex_index = Surfaces.faces_packed().view(-1)

    angle_face = faces_angle(Surfaces)

    dual_weights = dual_area_weights_faces(Surfaces)

    dual_areas = dual_weights*Surfaces.faces_areas_packed().view(-1,1)

    dual_area_per_vertex = scatter(dual_areas.view(-1), face2vertex_index, reduce='sum')

    angle_sum_per_vertex = scatter(angle_face.view(-1), face2vertex_index, reduce='sum')

    curvature = (2*torch.pi - angle_sum_per_vertex)/(dual_area_per_vertex+1e-8)

    # Euler_chara = torch.sparse.mm(vertices_to_meshid.float().T,(2*torch.pi - sum_angle_for_vertices).T).T/torch.pi/2
    # Euler_chara = torch.round(Euler_chara)
    # print('Euler_characteristic:',Euler_chara)
    # Genus = (2-Euler_chara)/2
    #print('Genus:',Genus)
    if return_topology:
        Euler_chara = (curvature*dual_area_per_vertex).sum()/2/torch.pi
        return curvature, Euler_chara
    return curvature

def Average_from_verts_to_face(Surfaces: Meshes, feature_verts: torch.Tensor)->torch.Tensor:
    """
    Compute the average of feature vectors defined on vertices to faces by dual area weights
    Args:
        Surfaces: Meshes object
        feature_verts: Tensor of shape (N,C) where N is the number of vertices, C is the number of feature channels
    Returns:
        vect_faces: Tensor of shape (F,C) where F is the number of faces
    """
    assert feature_verts.shape[0] == Surfaces.verts_packed().shape[0]

    dual_weight = dual_area_weights_faces(Surfaces).view(-1,3,1)

    feature_faces = feature_verts[Surfaces.faces_packed(),:]
    
    wg = dual_weight*feature_faces
    return wg.sum(dim=-2)

### winding number

def Electric_strength(q, p):
    """
    q: (M, 3) - charge position
    p: (N, 3) - field position
    """
    assert q.shape[-1] == 3 and len(q.shape) == 2, "q should be (M, 3)"
    assert p.shape[-1] == 3 and len(p.shape) == 2, "p should be (N, 3)"
    q = q.unsqueeze(1).repeat(1, p.shape[0], 1)
    p = p.unsqueeze(0)
    return (p-q)/(torch.norm(p-q, dim=-1, keepdim=True)**3+1e-6)



def Winding_Occupancy(mesh_tem: Meshes, points: torch.Tensor):
    """
    Involving the winding number to evaluate the occupancy of the points relative to the mesh
    mesh_tem: the reference mesh
    points: the points to be evaluated Nx3
    """
    dual_areas = dual_area_vertex(mesh_tem)

    normals_areaic = mesh_tem.verts_normals_packed() * dual_areas.view(-1,1)

    face_elefields_temp = Electric_strength(points, mesh_tem.verts_packed())
    winding_field = einsum(face_elefields_temp, normals_areaic, 'm n c, n c -> m')/4/np.pi

    return winding_field


class Differentiable_Voxelizer(nn.Module):
    def __init__(self, bbox_density=128):
        super(Differentiable_Voxelizer, self).__init__()
        self.bbox_density = bbox_density

    def forward(self, mesh_src: Meshes, output_resolution=256):
        """
        mesh_src: the source mesh to be voxelized (should be rescaled into the normalized coordinates [-1,1])
        return_type: the type of the return
        """


        # random sampling in bounding box
        resolution = self.bbox_density
        bbox = mesh_src.get_bounding_boxes()[0]

        # grid sampling in bounding box
        bbox_length = (bbox[:, 1] - bbox[:, 0])
        step_lengths = bbox_length.max() / resolution
        step = (bbox_length / step_lengths).int() + 1

        x = torch.linspace(bbox[0, 0], bbox[0, 1], steps=step[0], device=mesh_src.device)
        y = torch.linspace(bbox[1, 0], bbox[1, 1], steps=step[1], device=mesh_src.device)
        z = torch.linspace(bbox[2, 0], bbox[2, 1], steps=step[2], device=mesh_src.device)

        

        x_index, y_index, z_index = torch.meshgrid(x, y, z)

        slice_length_ranking, slice_direction_ranking = torch.sort(step, descending=False)

        # change the order of the coordinates for the acceleration 
        slice_direction_ranking_reverse = torch.argsort(slice_direction_ranking,descending=False)


        coordinates = torch.stack([x_index, y_index, z_index], dim=-1)


        coordinates = coordinates.permute(slice_direction_ranking.tolist() + [3])


        coordinates = rearrange(coordinates, 'x y z c -> x (y z) c', c = 3, x = slice_length_ranking[0], y = slice_length_ranking[1], z = slice_length_ranking[2])
        occupency_fields = []
        for i in range(0, coordinates.shape[0]):
            tem_charge = coordinates[i]

            occupency_temp = torch.sigmoid((Winding_Occupancy(mesh_src, tem_charge)-0.5)*100)

            occupency_fields.append(occupency_temp)

        occupency_fields = torch.stack(occupency_fields, dim=0)

        # embedding the bounding box into the whole space
        resolution_whole = output_resolution
        bbox_index = (bbox +1)*resolution_whole//2
        X_b, Y_b, Z_b = bbox_index.int().tolist()
        whole_image = torch.zeros(resolution_whole, resolution_whole, resolution_whole, device=mesh_src.device)

        bbox_transformed = rearrange(occupency_fields, 'x (y z) -> x y z', x = slice_length_ranking[0], y = slice_length_ranking[1], z = slice_length_ranking[2])

        bbox_transformed = bbox_transformed.permute(slice_direction_ranking_reverse.tolist()).unsqueeze(0).unsqueeze(0)

        bbox_transformed = F.interpolate(bbox_transformed, size=(X_b[1]-X_b[0]+1, Y_b[1]-Y_b[0]+1, Z_b[1]-Z_b[0]+1), mode='trilinear')
        bbox_transformed = bbox_transformed.squeeze(0).squeeze(0)

        whole_image[X_b[0]:X_b[1]+1, Y_b[0]:Y_b[1]+1, Z_b[0]:Z_b[1]+1] = bbox_transformed

        whole_image = (whole_image.permute(2, 1, 0)).unsqueeze(0)

        return whole_image
    





def normalize_mesh(mesh, rescalar=1.1):
    bbox = mesh.get_bounding_boxes()
    center = (bbox[:, :, 0] + bbox[:, :, 1]) / 2
    size = (bbox[:, :, 1] - bbox[:, :, 0]) 
    scale = 2.0 / (torch.max(size)*rescalar+1e-8)

    mesh = mesh.update_padded((mesh.verts_padded()-center)*scale+center)
    return mesh

