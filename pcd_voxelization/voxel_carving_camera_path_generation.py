import numpy as np
import open3d as o3d

# https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Measurement+of+areas+on+a+sphere+using+Fibonacci+and+latitude%E2%80%93longitude+lattices&btnG=

def generate_sphere_mesh(num_points):
    # Generate uniformly distributed points on a sphere using Fibonacci sampling
    indices = np.arange(0, num_points, dtype=float) + 0.5 # arange(start, stop): Values are generated within the half-open interval [start, stop).
    phi = np.arccos(1 - 2 * indices / num_points)  # Polar angle
    theta = np.pi * (1 + 5**0.5) * indices         # Azimuthal angle

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Combine coordinates into a (num_points, 3) array
    points = np.vstack((x, y, z)).T

    # Create a triangle mesh with vertices (no faces for now)
    sphere_mesh = o3d.geometry.TriangleMesh()
    sphere_mesh.vertices = o3d.utility.Vector3dVector(points)
    sphere_mesh.triangles = o3d.utility.Vector3iVector([])  # No triangles
    return sphere_mesh

def save_ply_file(mesh, filename):
    # Save the generated sphere mesh as a PLY file
    o3d.io.write_triangle_mesh(filename, mesh)

# Parameters
num_views = 642  # Total number of views
output_file = "sphere.ply"  # Output file name

# Generate the sphere and save it as a PLY file
sphere_mesh = generate_sphere_mesh(num_views)
save_ply_file(sphere_mesh, output_file)

print(f"Sphere mesh with {num_views} vertices saved to {output_file}")
