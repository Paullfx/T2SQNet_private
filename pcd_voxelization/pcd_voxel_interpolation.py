import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay

# Point cloud file path
tableware_ply_path = "/home/hamilton/Master_thesis/data_cup1/exp_default/tableware_4_6_bowl_denoised.ply"

# Load point cloud
point_cloud = o3d.io.read_point_cloud(tableware_ply_path)
print("Original point cloud:")
print(point_cloud)

# Visualize original point cloud
o3d.visualization.draw_geometries([point_cloud], window_name="Original Point Cloud")

# Voxelization
voxel_size = 0.0018  # Voxel size from configuration
voxelized_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
voxel_points = np.asarray(voxelized_point_cloud.points)
print("Voxelized point cloud:")
print(f"Number of points after voxelization: {len(voxel_points)}")

# Save voxelized point cloud
output_ply_path = "/home/hamilton/Master_thesis/data_cup1/exp_default/tableware_voxelized.ply"
o3d.io.write_point_cloud(output_ply_path, voxelized_point_cloud)
print(f"Voxelized point cloud saved to: {output_ply_path}")

# Visualize voxelized point cloud
o3d.visualization.draw_geometries([voxelized_point_cloud], window_name="Voxelized Point Cloud")

# Target dimensions for interpolation
w = 125  # Width
h = 121  # Height
d = 66   # Depth
target_points = w * h * d  # Total number of points

# Interpolation function with 4 vertices (fix for 3D Delaunay)
def interpolate_points(points, num_new_points):
    tri = Delaunay(points)
    simplices = tri.simplices
    new_points = []
    for _ in range(num_new_points):
        simplex = simplices[np.random.randint(len(simplices))]
        vertices = points[simplex]
        weights = np.random.dirichlet([1, 1, 1, 1])  # Adjust to 4 weights for 3D Delaunay
        new_point = np.dot(weights, vertices)
        new_points.append(new_point)
    return np.array(new_points)

# Calculate the number of new points needed
num_new_points = target_points - len(voxel_points)
print(f"Number of new points to interpolate: {num_new_points}")

# Generate interpolated points
interpolated_points = interpolate_points(voxel_points, num_new_points)

# Combine original and interpolated points
resampled_points = np.vstack([voxel_points, interpolated_points])
print("Resampled points shape:", resampled_points.shape)

# Reshape to target dimensions
reshaped_points = resampled_points[:target_points].reshape(w, h, d, 3)
print("Reshaped points shape:", reshaped_points.shape)

# Generate colors for the point cloud (example: random colors)
colors = np.random.rand(resampled_points.shape[0], 3)

# Create a new point cloud with colors
colored_point_cloud = o3d.geometry.PointCloud()
colored_point_cloud.points = o3d.utility.Vector3dVector(resampled_points)
colored_point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Visualize the colored point cloud
o3d.visualization.draw_geometries([colored_point_cloud], window_name="Colored Point Cloud")
