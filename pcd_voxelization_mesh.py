# convert the segmented point cloud into mesh, such that it can be used for the dense voxel grid generation
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


# reference: https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html, focus on "Poisson surface reconstruction"
# important intuition of parameter "depth", higher depth, more detailed mesh
# need to firsty generate normals for the mesh


# load and visualize the segmented pcd from CG pipleine

#tableware_ply_path = "./data_CG/tableware_4_16_bowl_denoised.ply" # a cup
tableware_ply_path = "/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/tableware_5_12/exps/exp_default/tableware_5_12_bowl_denoised.ply"

point_cloud = o3d.io.read_point_cloud(tableware_ply_path)
print(point_cloud)
#o3d.visualization.draw_geometries([point_cloud])



# normal estimation
point_cloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
point_cloud.estimate_normals()
#o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)
point_cloud.orient_normals_consistent_tangent_plane(100) # It is observed that the normal reorientation leads to a worse (even more incomplete) mesh representation
#o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)




# print('run Poisson surface reconstruction')
# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         point_cloud, depth=9) # depth can be adjustable here
# print(mesh)

print('run Poisson surface reconstruction')
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    point_cloud, depth=9)  # depth can be adjustable here
print(mesh)

# o3d.visualization.draw_geometries([mesh])


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
o3d.visualization.draw_geometries([density_mesh])


# remove low density vertices
print('remove low density vertices')
vertices_to_remove = densities < np.quantile(densities, 0.09) #set the density threshold #0.15 or even higher to crop the low density mesh of beer bottle
mesh.remove_vertices_by_mask(vertices_to_remove) # o3d.geometry.TriangleMesh.remove_vertices_by_mask requires boolean values with the number of vertices
print(mesh)

# Recalculate the densities for the remaining vertices
# Note: Open3D does not maintain per-vertex densities after filtering; you'll need to map the original densities back.
filtered_densities = densities[~vertices_to_remove]  # Only keep densities for remaining vertices
filtered_density_colors = plt.get_cmap('plasma')(
    (filtered_densities - filtered_densities.min()) / (filtered_densities.max() - filtered_densities.min())
)[:, :3]  # Map densities to colors

# Apply pseudo-color based on the new density values
filtered_density_mesh = o3d.geometry.TriangleMesh()
filtered_density_mesh.vertices = mesh.vertices
filtered_density_mesh.triangles = mesh.triangles
filtered_density_mesh.triangle_normals = mesh.triangle_normals
filtered_density_mesh.vertex_colors = o3d.utility.Vector3dVector(filtered_density_colors)

# Visualize the filtered mesh with pseudo-color
o3d.visualization.draw_geometries([filtered_density_mesh])

#o3d.visualization.draw_geometries([mesh])