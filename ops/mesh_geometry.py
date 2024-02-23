import torch

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

def Winding_Occupancy(mesh_tem, points):
    """
    Involving the winding number to evaluate the occupancy of the points relative to the mesh
    mesh_tem
    points: the points to be evaluated
    charge: the charge position
    """
    face_coordinates= mesh_tem.verts_packed()[mesh_tem.faces_packed()]
    face_centers = face_coordinates.mean(-2)
    normals_areaic = mesh_tem.faces_normals_packed() * mesh_tem.faces_areas_packed().unsqueeze(-1)


    face_elefields_temp = Electric_strength(points, face_centers)
    winding_field = einsum(face_elefields_temp, normals_areaic, 'm n c, n c -> m')/4/np.pi

    winding_field = torch.sigmoid((winding_field-0.2)*20)

    return winding_field



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