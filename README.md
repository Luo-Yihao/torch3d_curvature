# Torch3D Curvature
`Torch3D Curvature` is the official implementation of the paper [*Differentiable Topology Estimating for 3D Shapes*](https://arxiv.org/abs/) (Coming soon), developed by Yihao Luo. This project introduces a differentiable method for estimating the curvature and topological characteristics of 3D shapes across multiple representations. Entirely built in PyTorch, it seamlessly integrates with deep learning frameworks, making it the first project to provide global topological estimation of 3D shapes in an explicit and differentiable manner.

![Gaussian Curvatures of the Mesh of Kitty](images/KittyCurvature.jpg "Gaussian Curvatures of the mesh of Kitty by Torch3D_Curvature")

### Features

- [x] **Gaussian Curvature of Meshes**: Compute Gaussian curvature of triangular meshes via the discrete Gauss-Bonnet theorem with GPU acceleration.
- [x] **Curvature of Pointclouds**: Estimate Gaussian and mean curvature for point clouds using a novel, stable method based on the Gaussian map and Weingarten map.
- [x] **Differentiable Voronoi Diagrams**: Compute the local area element for point clouds with differentiable Voronoi diagrams.
- [x] **Topological Characteristics**: Estimate topological characteristics such as the Euler characteristic and genus of surfaces.
- [x] **Support for Multiple Representations**: Supports meshes, point clouds (oriented or non-oriented), voxels, implicit functions, and Gaussian splatting.

**Note**: This project corrects several conceptual issues found in `PyTorch3D`. For instance, `pytorch3d.ops.estimate_pointcloud_local_coord_frames` incorrectly derives frames using the covariance matrix of each point in a point cloud. In contrast, `Torch3D_Curvature.ops.pcl_curvature.Curvature_pcl` computes the correct eigenvector of the Weingarten map, derived from `Torch3D_Curvature.ops.pcl_curvature.Weingarten_maps`.

### Future Work
- [ ] Explore a zero-shot deep learning model for curvature and topological characteristic estimation.

### Requirements

The core functionality depends on the following libraries:
- ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-blue.svg)
- ![PyTorch3D](https://img.shields.io/badge/PyTorch3D-0.7.5-blue.svg)
- ![PyTorch_Scatter](https://img.shields.io/badge/PyTorch_Scatter-2.1-blue.svg)

For visualization and testing:
- ![Trimesh](https://img.shields.io/badge/Trimesh-3.9-blue.svg)
- ![Pyvista](https://img.shields.io/badge/Pyvista-0.43-blue.svg)

Please install the required packages according to their official guidelines.

### Getting Started
To get started, clone the repository:
```bash
git clone git@github.com:Luo-Yihao/Torch3d_Curvature.git
cd Torch3d_Curvature
```
Then run the `TUTORIAL.ipynb` notebook for a quick start.


### Usage Example 

#### For Mesh

```Python
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from ops.mesh_curvature import Gaussian_curvature, 
Dual_area_for_vertices, Average_from_verts_to_face

# Load the kitty mesh.
trg_obj = 'data_example/kitty.obj'

# We read the target 3D model using load_obj
verts, faces, aux = load_obj(trg_obj)
faces_idx = faces.verts_idx
verts = verts

# We construct a Meshes structure for the target mesh
trg_mesh = Meshes(verts=[verts], faces=[faces_idx])

# Dual areas of he vertices
dual_area_vertex = Dual_area_for_vertices(trg_mesh)

# gaussian curvature of the vertices and topological characteristics
gaussian_curvature, Euler_chara, Genus = Gaussian_curvature(trg_mesh,return_topology=True)

# gaussian curvature of the faces with weights of Dual areas
gaussian_curvature_faces = Average_from_verts_to_face(trg_mesh,gaussian_curvature.T).cpu().numpy()

# discrete Gauss-Bonnet theorem
print("Gauss-Bonnet theorem: integral of gaussian_curvature - 2*pi*X = ",(gaussian_curvature*dual_area_vertex).sum().cpu().numpy() - 2*np.pi*mesh_np.euler_number)
```
The result print as follows:
```Python
"Gauss-Bonnet theorem: integral of gaussian_curvature - 2*pi*X =  0.000017"
```

#### For PointCloud

```Python
num_samples = 10000 # 100000 
pointscloud, normals_gt = sample_points_from_meshes(trg_mesh, num_samples, return_normals=True)

k = 50 # number of nearest neighbors
Weingarten_fields, normals_field, tangent1_field, tangent2_field = Weingarten_maps(pointscloud,k)

# construct the principal frame by principal curved directions and normals
principal_curvatures, principal_direction, normals_field =  Curvature_pcl(pointscloud,k,return_princpals=True)
principal_frame = torch.cat([principal_direction,normals_field[:,:,None,:]],dim=-2)

## Estimate the Topological characteristics with various k neighborhoods (for stableness test)
for i in range(5,10):
    pcl_with_frames = Differentiable_Global_Geometry_PointCloud(pointscloud, k = 5*i, normals = None) # = normals_gt
    print('Estimated Genus',  1-torch.round(pcl_with_frames.differentiable_euler_number()[0]/2).item())

    print('Estimated euler',  pcl_with_frames.differentiable_euler_number()[0].item())
```
The results come as follows and visualization is shown in the top image:
```Python
Estimated Genus 1.0
Estimated euler -0.27381613850593567
Estimated Genus 1.0
Estimated euler -0.17997273802757263
Estimated Genus 1.0
Estimated euler -0.10879017412662506
Estimated Genus 1.0
Estimated euler -0.08370624482631683
Estimated Genus 1.0
Estimated euler -0.029316892847418785
```


### Mathematical Foundation

The Gaussian map is a mapping that transforms each point on a surface to a point on the unit sphere. The Weingarten map, the differential of the Gaussian map, transforms each tangent vector on a surface to the tangent vector of the unit sphere linearly. Due to the two tangent spaces coinciding, the Weingarten map can be represented as a $2\times2$ matrix field.  The curvature at a point on the surface can be computed using the Weingarten map as follows:

Let $N:M\to S^2$ be the unit normal vector at the point.
Let $dN: T_p(M)\to T_{N(p)}S^2$ be Weingarten map, written as $W(p)$. Here the differential and stable estimation of the Weingarten map is non-trivial, where we involved an algorithm to solve the Sylvester equation to achieve the Hermitian solution of the Weingarten map.

The Gaussian curvature can then be computed as $K_p = \det(W(p))$ and the Mean curvature equals to $H_p = {\frac{1}{2}}{\rm Tr}(W(p))$.

The Gauss-Bonnet theorem unveils the deep connection between the local geometry and global topology, which says the integral of Gaussian curvature giving the Euler characteristics of a closed surface: 
$$\int_M KdA = 2\pi* \chi,$$
where $dA$, the local area element, is estimated by differentiable Voronoi diagram conducted on local tangent spaces of each K-nearest neighborhood.

### Contact
For questions or feedback, feel free to contact Yihao Luo at email y.luo23@imperial.ac.uk.

### Citation
If you find this project useful, please cite the following paper:
```
@article{luo2021differentiable,
  title={Differentiable Topology Estimating for 3D Shapes},
  author={Luo, Yihao},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```