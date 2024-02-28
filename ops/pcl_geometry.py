import torch
import torch.nn as nn
import pytorch3d 
from pytorch3d.ops import knn_points,knn_gather
from torch_scatter import scatter

import numpy as np


class Differentiable_Global_Geometry_PointCloud(nn.Module):

    def __init__(self, pointscloud, k = 50):
        super().__init__()
        self.k = k

        self.pointscloud = pointscloud
        self.knn_info, self.frames_field = self.frames_field_from_local_statistics(pointscloud, k)


    def frames_field_from_local_statistics(self, pointscloud, k):

        # undo global mean for stability
        knn_info = knn_points(pointscloud,pointscloud,K=k,return_nn=True)

        # compute knn & covariance matrix 
        knn_points_centered = knn_info.knn - knn_info.knn.mean(-2,keepdim=True)
        covs_field = torch.matmul(knn_points_centered.transpose(-1,-2),knn_points_centered) /(knn_points_centered.shape[-1]-1)
        frames_field = torch.linalg.eigh(covs_field).eigenvectors.transpose(-1,-2) # B x N x 3 x 3

        frames_field[:,:,1,:] = frames_field[:,:,1,:]*frames_field.det().unsqueeze(-1)

        return knn_info, frames_field
    

    def local_voronoi_area(self, frames_field: torch.Tensor=None, local_W: int=128, n_temp: int=2000):
        """
        Compute the local exclusive voronoi area for each point in the point cloud using the tangent plane projection

        Args:
            frames_field (torch.Tensor): the local frames field, if None, use the one stored in the class
            local_W (int): the resolution of the local patch
            n_temp (int): the number of points to process at a time to save memory for large point clouds
        """
        
        k = self.k
        knn_info = self.knn_info

        B, N = self.pointscloud.shape[:2]

        device = self.pointscloud.device

        X, Y = torch.meshgrid(torch.linspace(-1, 1, local_W), torch.linspace(-1, 1, local_W))

        points_meshgrid = torch.stack((X, Y), dim=-1).view(1,-1,2).to(device)
        # local normals difference
        if frames_field is None:
            frames_field = self.frames_field
        

        # ----------------- tangent plane projection ---------------

        local_pt_difference = knn_info.knn - self.pointscloud[:,:,None,:] # B x N x K x 3


        tangent1_field = frames_field[:,:,1,:] # B x N x  3
        tangent2_field = frames_field[:,:,2,:]  # B x N x  3


        # project the difference onto the tangent plane, getting the differential of the gaussian map
        local_dpt_tangent1 = (local_pt_difference * tangent1_field[:,:,None,:]).sum(-1,keepdim=True) # B x N x K x 1
        local_dpt_tangent2 = (local_pt_difference * tangent2_field[:,:,None,:]).sum(-1,keepdim=True) # B x N x K x 1
        local_dpt_tangent = torch.cat((local_dpt_tangent1,local_dpt_tangent2),dim=-1) # B x N x K x 2
        
        # local normals difference
        if frames_field is None:
            frames_field = self.frames_field
        



                # ----------------- local exclusive voronoi area ----------
        local_rescalar = 1.1

        local_knn_bbox_tangent = torch.stack((local_dpt_tangent.min(dim=-2).values,local_dpt_tangent.max(dim=-2).values),dim=-1)*local_rescalar

        local_knn_bbox_tangent_length = local_knn_bbox_tangent[...,1:]-local_knn_bbox_tangent[...,0:1]

        local_knn_coord_tangent = (local_dpt_tangent - local_knn_bbox_tangent[...,0:1].transpose(-1,-2))/local_knn_bbox_tangent_length.max(-2,keepdim=True).values.repeat(1,1,1,2)*2-1 # B x N x K x 2

        try:
            local_closest_idx = knn_points(points_meshgrid.repeat(N,1,1), local_knn_coord_tangent.view(-1,k,2),K=1).idx

            local_center_voronoi = torch.where(local_closest_idx == 0)

            local_center_voronoi = scatter(torch.ones_like(local_center_voronoi[0]), 
                                            local_center_voronoi[0],
                                            dim_size=N)
        except:  # memory saving strategy for large point clouds
            local_center_voronoi = torch.zeros(B, N, 1).to(device)
            n_temp = n_temp # split the point cloud into smaller chunks
            for n in range(N//n_temp+1):

                local_knn_coord_tangent_temp = local_knn_coord_tangent[:, n*n_temp:(n+1)*n_temp].view(-1,k,2)

                local_closest_idx = knn_points(points_meshgrid.repeat(local_knn_coord_tangent_temp.shape[0],1,1), local_knn_coord_tangent_temp,K=1).idx

                local_center_voronoi_temp = torch.where(local_closest_idx == 0)

                local_center_voronoi_temp = scatter(torch.ones_like(local_center_voronoi_temp[0]), 
                                                local_center_voronoi_temp[0],
                                                dim_size=local_knn_coord_tangent_temp.shape[0])


                local_center_voronoi[:, n*n_temp:(n+1)*n_temp] = local_center_voronoi_temp.view(B, -1, 1)
                



        voronoi_area_list = local_center_voronoi.view(local_knn_bbox_tangent_length[:,:,0:1,0].shape)

        voronoi_area_list = voronoi_area_list*(local_knn_bbox_tangent_length.max(-2).values**2)/(local_W-1)**2


        return voronoi_area_list
    

    
        
    def renew_frames_field(self,frames_field: torch.Tensor):
        self.frames_field.data = (frames_field.data).to(self.pointscloud.device)



    def weingarten_fields(self, frames_field: torch.Tensor=None):
        """
        Compute the weingarten map for the point cloud using the tangent plane projection
        
        """

        B, N = self.pointscloud.shape[:2]

        k = self.k
        device = self.pointscloud.device
        knn_info = self.knn_info
        
        # local normals difference
        if frames_field is None:
            frames_field = self.frames_field
        

        # ----------------- tangent plane projection ---------------

        local_pt_difference = knn_info.knn - self.pointscloud[:,:,None,:] # B x N x K x 3

        normals_field  = frames_field[:,:,0,:] # B x N x  3
        tangent1_field = frames_field[:,:,1,:] # B x N x  3
        tangent2_field = frames_field[:,:,2,:]  # B x N x  3
    

        # project the difference onto the tangent plane, getting the differential of the gaussian map

        # local normals difference
        local_normals_difference = knn_gather(normals_field,knn_info.idx) - normals_field.unsqueeze(-2) # B x N x K x 3

        # project the difference onto the tangent plane, getting the differential of the gaussian map
        local_dpt_tangent1 = (local_pt_difference * tangent1_field[:,:,None,:]).sum(-1,keepdim=True) # B x N x K x 1
        local_dpt_tangent2 = (local_pt_difference * tangent2_field[:,:,None,:]).sum(-1,keepdim=True) # B x N x K x 1
        local_dpt_tangent = torch.cat((local_dpt_tangent1,local_dpt_tangent2),dim=-1) # B x N x K x 2

        local_dnormals_tangent1 = (local_normals_difference * tangent1_field[:,:,None,:]).sum(-1,keepdim=True)
        local_dnormals_tangent2 = (local_normals_difference * tangent2_field[:,:,None,:]).sum(-1,keepdim=True)
        local_dnormals_tangent = torch.cat((local_dnormals_tangent1,local_dnormals_tangent2),dim=-1) # B x N x K x 2
        

        # ----------------- weingarten map ---------------------

        # estimate the weingarten map by solving a least squares problem: W = Dn^T Dp (Dp^T Dp)^-1
        XXT = torch.matmul(local_dpt_tangent.transpose(-1,-2),local_dpt_tangent)
        YXT = torch.matmul(local_dnormals_tangent.transpose(-1,-2),local_dpt_tangent)
        XYT = torch.matmul(local_dpt_tangent.transpose(-1,-2),local_dnormals_tangent)
        #Weingarten_fields_0 = torch.matmul(YXT,torch.inverse(XXT+1e-8*torch.eye(2).type_as(XXT))) ## the unsymetric version


        # solve the sylvester equation to get the shape operator (symmetric version)
        S = YXT + XYT

        XXT_eig = torch.linalg.eigh(XXT)
        Q = XXT_eig.eigenvectors
        #D = torch.diag_embed(XXT_eig.eigenvalues)
        # XX^T = Q^T D Q
        Q_TSQ = torch.matmul(Q.transpose(-1,-2),torch.matmul(S,Q))

        a = XXT_eig.eigenvalues[:,:,0]
        b = XXT_eig.eigenvalues[:,:,1]
        a_b = a+b
        a2_a_b = torch.stack((2*a,a_b),dim=-1).view(B,-1,1,2)
        a_b_b2 = torch.stack((a_b,2*b),dim=-1).view(B,-1,1,2)
        c = torch.stack((a2_a_b,a_b_b2),dim=-2).view(B,-1,2,2)

        E =  (1/(c+1e-8)) * Q_TSQ
        Weingarten_fields = torch.matmul(Q,torch.matmul(E,Q.transpose(-1,-2)))


        return Weingarten_fields

        

    def gaussian_curvature(self, frames_field: torch.Tensor=None):
        Weingarten_fields = self.weingarten_fields(frames_field)

        gaussian_curvature_pcl = torch.det(Weingarten_fields)

        return gaussian_curvature_pcl


    def differentiable_euler_number(self, frames_field: torch.Tensor=None, local_W: int=128, n_temp: int=2000):

        voronoi_area_list = self.local_voronoi_area(frames_field, local_W, n_temp)
        gaussian_curvature_pcl = self.gaussian_curvature(frames_field)
        euler_number = (gaussian_curvature_pcl.view(-1)*voronoi_area_list.view(-1)).sum()/2/np.pi

        return euler_number
    


if __name__ == '__main__':
 
    from pytorch3d.structures import Meshes, Pointclouds
    from pytorch3d.ops import sample_points_from_meshes  
    from pytorch3d.utils import ico_sphere, torus
    mesh_temp = ico_sphere(4, device='cuda:0')
    # mesh_temp = torus(r=1, R=2, sides=40, rings=20, device='cuda:0')
    from mesh_geometry import normalize_mesh
    mesh_temp = normalize_mesh(mesh_temp)
    sample_points = sample_points_from_meshes(mesh_temp, 100000, return_normals=False)
    sample_points = sample_points.to('cuda:0')

    pcl_geometry = Differentiable_Global_Geometry_PointCloud(sample_points, k=50)
    print(pcl_geometry.differentiable_euler_number(local_W=128, n_temp=2000))

    

