import open3d as o3d
import numpy as np
import torch
import os
from math import floor, ceil
from scipy.spatial import Delaunay

# main reference: https://www.open3d.org/docs/latest/tutorial/Advanced/voxelization.html

# The pcd voxelization consists of several parts
# 1. Explanation of the script called "pcd_voxelization.py". The pcd input is segmented point cloud of one specific object (e.g. a beer bottle). Fisrt step is to create voxel grid of the object surface 
# with voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud,voxel_size=voxel_size)
# 2. Explanation for script "pcd_voxelization_voxel_carving.py" and "pcd_voxelization_mesh.py". Besides the object surface, we also need the dense voxel grid of the object. The dense voxel grid is a 3D grid of voxels that covers the entire object. 
# This process is called "voxel carving". The voxel carving is done by the function o3d.geometry.VoxelGrid.create_dense(width=cubic_size,height=cubic_size,depth=cubic_size,voxel_size=voxel_resolution)
# The whole voxel carving process is in "pcd_voxelization_voxel_carving.py", however, the mesh of the object pcd is needed and that's why there is another script called "pcd_voxelization_mesh.py".



# tableware_ply_path = "/home/hamilton/Master_thesis/data_cup1/exp_default/pcd_exp_default.ply" 
# tableware_ply_path = "/home/hamilton/Master_thesis/data_cup1/exp_default/tableware_4_6_bowl.ply" # a cup
# tableware_ply_path = "/home/hamilton/Master_thesis/data_cup1/exp_default/tableware_4_6_bowl_denoised.ply" # a cup
# tableware_ply_path = "/home/hamilton/Master_thesis/data_cup1/exp_default_2/tableware_4_5_bowl_denoised.ply" # a plate
# tableware_ply_path = "./data_CG/tableware_4_16_bowl_denoised.ply" # a cup
scene_id = "tableware_5_12"
tableware_ply_path = f"/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/{scene_id}/exps/exp_default/{scene_id}_bowl_denoised.ply"
# tableware_ply_path = "/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/tableware_5_12/exps/exp_default/tableware_5_12_bowl_denoised.ply"


point_cloud = o3d.io.read_point_cloud(tableware_ply_path)
print("Original point cloud:")
print(point_cloud)
o3d.visualization.draw_geometries([point_cloud], window_name="Original Point Cloud")


# mesh = pcd2mesh(point_cloud) 

output_filename = os.path.join("T2SQNet_private/data/voxelization_data", f"{scene_id}_carved_voxel.ply") # save to T2SQNet_private/data/voxelization_data/{scene_id}_carved_voxel.ply
# camera_path = os.path.abspath("../../test_data/sphere.ply")





# voxel_size = 0.004843219465611634  # use the predefined voxel size for specific tableware from voxelize_config.yml # Bowl: 0.004843219465611634 # Mug: 0.001832966443807953
# voxelized_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
# print("Voxelized point cloud:")
# print(voxelized_point_cloud)
# output_ply_path = "./data_CG/tableware_4_16_bowl_denoised_voxelized.ply"
# o3d.io.write_point_cloud(output_ply_path, voxelized_point_cloud)
# print(f"Voxelized point cloud saved to: {output_ply_path}")
# o3d.visualization.draw_geometries([voxelized_point_cloud], window_name="Voxelized Point Cloud")
# voxel_points = np.asarray(voxelized_point_cloud.points) #e.g. voxel_points.shape = (1717, 3)
# print(f"Number of points after voxelization: {len(voxel_points)}")

# w = floor(2 *  0.11456040273799706 / 0.001832966443807953 + 0.5) # w = 125
# h = floor(2 * 0.11133156195208502 / 0.001832966443807953 + 0.5) # h = 121
# d = floor(2 * 0.06027486266068399 / 0.001832966443807953 + 0.5) # d = 66

# target_points = 125 * 121 * 66  # Total elements required

# def interpolate_points(points, num_new_points):
#     tri = Delaunay(points)
#     simplices = tri.simplices
#     new_points = []
#     for _ in range(num_new_points):
#         simplex = simplices[np.random.randint(len(simplices))]
#         vertices = points[simplex]
#         weights = np.random.dirichlet([1, 1, 1])
#         new_point = np.dot(weights, vertices)
#         new_points.append(new_point)

#     return np.array(new_points)

# num_new_points = target_points - len(voxel_points)
# interpolated_points = interpolate_points(voxel_points, num_new_points)
# resampled_points = np.vstack([voxel_points, interpolated_points])
# print("Resampled points shape:", resampled_points.shape)

# reshaped_points = resampled_points[:target_points].reshape(125, 121, 66, 3)  
# print("Reshaped points shape:", reshaped_points.shape)