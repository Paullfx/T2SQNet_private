import open3d as o3d
import numpy as np
import torch
from math import floor, ceil
from scipy.spatial import Delaunay

#tableware_ply_path = "/home/hamilton/Master_thesis/data_cup1/exp_default/pcd_exp_default.ply" 
# tableware_ply_path = "/home/hamilton/Master_thesis/data_cup1/exp_default/tableware_4_6_bowl.ply" # a cup
tableware_ply_path = "/home/hamilton/Master_thesis/data_cup1/exp_default/tableware_4_6_bowl_denoised.ply" # a cup

#tableware_ply_path = "/home/hamilton/Master_thesis/data_cup1/exp_default_2/tableware_4_5_bowl_denoised.ply" # a plate


point_cloud = o3d.io.read_point_cloud(tableware_ply_path)


print("Original point cloud:")
print(point_cloud)


o3d.visualization.draw_geometries([point_cloud], window_name="Original Point Cloud")

voxel_size = 0.0018 # from voxelize_config.yml 


voxelized_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)


print("Voxelized point cloud:")
print(voxelized_point_cloud)

output_ply_path = "/home/hamilton/Master_thesis/data_cup1/exp_default/tableware_voxelized.ply"
o3d.io.write_point_cloud(output_ply_path, voxelized_point_cloud)

print(f"Voxelized point cloud saved to: {output_ply_path}")

o3d.visualization.draw_geometries([voxelized_point_cloud], window_name="Voxelized Point Cloud")


voxel_points = np.asarray(voxelized_point_cloud.points) #e.g. voxel_points.shape = (1717, 3)
print(f"Number of points after voxelization: {len(voxel_points)}")

# w = floor(2 *  0.11456040273799706 / 0.001832966443807953 + 0.5) # w = 125
# h = floor(2 * 0.11133156195208502 / 0.001832966443807953 + 0.5) # h = 121
# d = floor(2 * 0.06027486266068399 / 0.001832966443807953 + 0.5) # d = 66

target_points = 125 * 121 * 66  # Total elements required

def interpolate_points(points, num_new_points):
    tri = Delaunay(points)
    simplices = tri.simplices
    new_points = []
    for _ in range(num_new_points):
        simplex = simplices[np.random.randint(len(simplices))]
        vertices = points[simplex]
        weights = np.random.dirichlet([1, 1, 1])
        new_point = np.dot(weights, vertices)
        new_points.append(new_point)

    return np.array(new_points)

num_new_points = target_points - len(voxel_points)
interpolated_points = interpolate_points(voxel_points, num_new_points)
resampled_points = np.vstack([voxel_points, interpolated_points])
print("Resampled points shape:", resampled_points.shape)

reshaped_points = resampled_points[:target_points].reshape(125, 121, 66, 3)  
print("Reshaped points shape:", reshaped_points.shape)