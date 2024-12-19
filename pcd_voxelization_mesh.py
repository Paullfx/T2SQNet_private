# convert the segmented point cloud into mesh, such that it can be used for the dense voxel grid generation
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


# reference: https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html, focus on "Poisson surface reconstruction"
# important intuition of parameter "depth", higher depth, more detailed mesh



tableware_ply_path = "./data_CG/tableware_4_16_bowl_denoised.ply" # a cup
point_cloud = o3d.io.read_point_cloud(tableware_ply_path)


print(point_cloud)
o3d.visualization.draw_geometries([point_cloud],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[1.8900, 3.2596, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])


print('run Poisson surface reconstruction')
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=9) # depth can be adjustable here
print(mesh)
o3d.visualization.draw_geometries([mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[1.8900, 3.2596, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])


# use pseudo-color to visualize the density
print('visualize densities')
densities = np.asarray(densities)
density_colors = plt.get_cmap('plasma')(
    (densities - densities.min()) / (densities.max() - densities.min()))
density_colors = density_colors[:, :3]
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = mesh.vertices
density_mesh.triangles = mesh.triangles
density_mesh.triangle_normals = mesh.triangle_normals
density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
o3d.visualization.draw_geometries([density_mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[1.8900, 3.2596, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])


# remove low density vertices
print('remove low density vertices')
vertices_to_remove = densities < np.quantile(densities, 0.01) #output bool
mesh.remove_vertices_by_mask(vertices_to_remove) # o3d.geometry.TriangleMesh.remove_vertices_by_mask requires boolean values with the number of vertices
print(mesh)
o3d.visualization.draw_geometries([mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[1.8900, 3.2596, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])