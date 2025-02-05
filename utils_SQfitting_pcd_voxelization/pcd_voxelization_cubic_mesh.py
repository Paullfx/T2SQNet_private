import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

# reference: https://towardsdatascience.com/how-to-automate-voxel-modelling-of-3d-point-cloud-with-python-459f4d43a227

# 

####### load point cloud #######

scene_id = "tableware_6_1"
tableware = "laptop"
tableware_ply_path = f"/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/{scene_id}/exps/exp_default/{scene_id}_{tableware}_denoised.ply"
point_cloud = o3d.io.read_point_cloud(tableware_ply_path)
print(point_cloud)

# pointcloud boundings
point_cloud_bbox = point_cloud.get_axis_aligned_bounding_box()
point_cloud_extent = point_cloud_bbox.get_extent()
print(f"Point Cloud Size: {len(point_cloud.points)} points")
print(f"Point Cloud Bounding Box: Length = {point_cloud_extent[0]}, Width = {point_cloud_extent[1]}, Height = {point_cloud_extent[2]}")

# vis window 1, raw pcd
print('The first window visualize the original segmented pcd from CG')
o3d.visualization.draw_geometries([point_cloud]) 

############ surface voxel grid ############
print('voxelization in process')
voxel_size = 0.01  # use the predefined voxel size for specific tableware from voxelize_config.yml 
# Bowl: 0.004843219465611634 # Mug: 0.001832966443807953 #HandlessCup: 0.002201045924570001 # Laptop: 0.01
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud,
                                                            voxel_size=voxel_size)

# voxel grid boundings
voxel_grid_bbox = voxel_grid.get_axis_aligned_bounding_box()
voxel_grid_extent = voxel_grid_bbox.get_extent()
print(f"Voxel Grid Size: {len(voxel_grid.get_voxels())} voxels")
print(f"Voxel Grid Bounding Box: Length = {voxel_grid_extent[0]}, Width = {voxel_grid_extent[1]}, Height = {voxel_grid_extent[2]}")

### vis window 2, surface voxel grid ###
print('The second window visualize the voxel_grid ')
o3d.visualization.draw_geometries([voxel_grid])

### add color on z axis ###, comment this part if you want to add color on x axis
# voxels = voxel_grid.get_voxels() # extract the filled voxels
# z_indices = np.array([v.grid_index[2] for v in voxels])
# z_min, z_max = z_indices.min(), z_indices.max()
# z_normalized = (z_indices - z_min) / (z_max - z_min)
# colormap = plt.cm.jet
# colors = colormap(z_normalized)[:, :3]
# for v, color in zip(voxels, colors):
#     v.color = color

### add color on x axis ###, comment this part if you want to add color on z axis
voxels = voxel_grid.get_voxels() # extract the filled voxels
x_indices = np.array([v.grid_index[0] for v in voxels])
x_min, x_max = x_indices.min(), x_indices.max()
x_normalized = (x_indices - x_min) / (x_max - x_min)
colormap = plt.cm.jet
colors = colormap(x_normalized)[:, :3]
for v, color in zip(voxels, colors):
    v.color = color


############ cubic mesh ############

min_bound = voxel_grid.get_min_bound()
max_bound = voxel_grid.get_max_bound()
grid_size = max_bound - min_bound
voxel_size = voxel_grid.voxel_size
voxel_count_estimate = np.prod(np.ceil(grid_size / voxel_size).astype(int))
print(f"Estimated total number of voxels (including empty ones): {voxel_count_estimate}")

occupied_voxel_count = len(voxels)
print(f"Number of occupied voxels: {occupied_voxel_count}")



vox_mesh=o3d.geometry.TriangleMesh()
for v in voxels:
   cube=o3d.geometry.TriangleMesh.create_box(width=1, height=1,
   depth=1)
   cube.paint_uniform_color(v.color)
   cube.translate(v.grid_index, relative=False) # if false, the center is moved to the translation vector (v.grid_index)
   vox_mesh+=cube

vox_mesh.translate([0.5,0.5,0.5], relative=True) #if true, the translation vector is directly added to the geometry coordinates
# This is explained by the fact that when we created our initial voxel grid, 
# the reference was the lowest left point of the voxel instead of the barycenter (which is positioned at [0.5,0.5,0.5] relatively in the unit cube).

vox_mesh.scale(voxel_size, [0,0,0]) # Then, we scale our model by the voxel size, to transform each cube unit into its real size. 
# This makes use of the scale method that takes two arguments. The first is the scaling factor, and the second is the center used when scaling.

vox_mesh.translate(voxel_grid.origin, relative=True) # Finally, we need to translate our voxel assembly to its true original position by translating using the voxel grid origin relatively.
vox_mesh.merge_close_vertices(0.0000001)

# vox_mesh bounding
vox_mesh_bbox = vox_mesh.get_axis_aligned_bounding_box()
vox_mesh_extent = vox_mesh_bbox.get_extent()
print(f"Voxel Mesh Bounding Box: Length = {vox_mesh_extent[0]}, Width = {vox_mesh_extent[1]}, Height = {vox_mesh_extent[2]}")
print(f"Voxel Mesh Vertex Count: {len(vox_mesh.vertices)}")
print(f"Voxel Mesh Triangle Count: {len(vox_mesh.triangles)}")


### vis window 3, cubic mesh ###
print('The third window visualize the generated cubic mesh')
o3d.visualization.draw_geometries([vox_mesh])



############ save the ply ############
output_folder = "./data_cubic_mesh"
os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, f"{scene_id}_cubic_mesh.ply")
o3d.io.write_triangle_mesh(output_file_path, vox_mesh)
print(f"Voxel mesh saved to: {output_file_path}")


