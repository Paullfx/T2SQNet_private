import numpy as np
import open3d as o3d

# https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Measurement+of+areas+on+a+sphere+using+Fibonacci+and+latitude%E2%80%93longitude+lattices&btnG=

import numpy as np
import open3d as o3d

def generate_fibonacci_sphere(num_points):
    """
    Generate uniformly distributed points on a sphere using the Fibonacci lattice.
    """
    N = (num_points - 1) // 2  # Ensure odd P = 2N + 1
    phi = (1 + 5**0.5) / 2  # Golden ratio

    indices = np.arange(-N, N + 1, dtype=float)
    lat = np.arcsin(2 * indices / (2 * N + 1))
    lon = 2 * np.pi * indices / phi

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    points = np.vstack((x, y, z)).T

    # Create point cloud
    sphere_pcd = o3d.geometry.PointCloud()
    sphere_pcd.points = o3d.utility.Vector3dVector(points)

    return sphere_pcd

# Generate and visualize the Fibonacci sphere
num_views = 641  # Odd number required
sphere_pcd = generate_fibonacci_sphere(num_views)

# Visualize without setting colors
o3d.visualization.draw_geometries([sphere_pcd], window_name="Fibonacci Sphere")


# Parameters
num_views = 642  # Total number of views
output_file = "./utils_SQfitting_pcd_voxelization/sphere3.ply"  # Output file name

# Generate the sphere and save it as a PLY file
sphere_mesh = generate_fibonacci_sphere(num_views)
o3d.io.write_point_cloud(output_file, sphere_mesh)
o3d.visualization.draw_geometries([sphere_mesh], window_name="Fibonacci Sphere")

print(f"Sphere mesh with {num_views} vertices saved to {output_file}")
