import numpy as np
import open3d as o3d

def generate_latitude_longitude_lattice(delta):
    """
    Generate a latitude-longitude grid (equal-angle grid) on a sphere.
    
    Parameters:
    delta (float): Angular spacing between grid lines in degrees.

    Returns:
    o3d.geometry.PointCloud: A point cloud representing the grid on the sphere.
    """
    k = int(180 / delta)  # Number of latitude steps
    P = 2 * k * (k - 1) + 2  # Total number of points (Equation from the paper)

    latitudes = np.linspace(-90, 90, k)  # Latitude from -90° to 90°
    longitudes = np.linspace(-180, 180, 2 * k)  # Longitude from -180° to 180°

    points = []
    
    for lat in latitudes:
        for lon in longitudes:
            # Convert latitude and longitude to radians
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)

            # Convert spherical coordinates to Cartesian coordinates
            x = np.cos(lat_rad) * np.cos(lon_rad)
            y = np.cos(lat_rad) * np.sin(lon_rad)
            z = np.sin(lat_rad)

            points.append([x, y, z])

    points = np.array(points)

    # Create point cloud
    sphere_pcd = o3d.geometry.PointCloud()
    sphere_pcd.points = o3d.utility.Vector3dVector(points)

    return sphere_pcd

def save_ply_file(point_cloud, filename):
    """
    Save the generated sphere lattice as a PLY file.

    Parameters:
    point_cloud (o3d.geometry.PointCloud): The point cloud to save.
    filename (str): The output file name.
    """
    o3d.io.write_point_cloud(filename, point_cloud)

def visualize_point_cloud(point_cloud):
    """
    Visualize the point cloud using Open3D.

    Parameters:
    point_cloud (o3d.geometry.PointCloud): The point cloud to visualize.
    """
    print("Displaying the Latitude-Longitude Lattice...")
    o3d.visualization.draw_geometries([point_cloud], window_name="Latitude-Longitude Lattice")

# Parameters
delta = 10  # Angular spacing in degrees (smaller values increase resolution)
output_file = "./utils_SQfitting_pcd_voxelization/latitude_longitude_lattice.ply"  

# Generate the Latitude-Longitude Lattice
sphere_pcd = generate_latitude_longitude_lattice(delta)
save_ply_file(sphere_pcd, output_file)

# Visualize the lattice
visualize_point_cloud(sphere_pcd)

print(f"Latitude-Longitude lattice saved to {output_file}")
