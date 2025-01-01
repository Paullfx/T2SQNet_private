import open3d as o3d
import numpy as np

####### load point cloud #######
tableware_ply_path = "/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/tableware_5_12/exps/exp_default/tableware_5_12_bowl_denoised.ply"
point_cloud = o3d.io.read_point_cloud(tableware_ply_path)
print(point_cloud)
o3d.visualization.draw_geometries([point_cloud])

# surface voxel grid
print('voxelization')
voxel_size = 0.004843219465611634  # use the predefined voxel size for specific tableware from voxelize_config.yml 
# Bowl: 0.004843219465611634 # Mug: 0.001832966443807953
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud,
                                                            voxel_size=voxel_size)
o3d.visualization.draw_geometries([voxel_grid])

####### print results #######
min_bound = voxel_grid.get_min_bound()
max_bound = voxel_grid.get_max_bound()
grid_size = max_bound - min_bound
# print(f"Voxel grid minimum bound: {min_bound}")
# print(f"Voxel grid maximum bound: {max_bound}")
# print(f"Voxel grid size (physical dimensions): {grid_size}")

voxel_size = voxel_grid.voxel_size
voxel_count_estimate = np.prod(np.ceil(grid_size / voxel_size).astype(int))
print(f"Estimated total number of voxels (including empty ones): {voxel_count_estimate}")
voxels = voxel_grid.get_voxels() # extract the filled voxels
occupied_voxel_count = len(voxels)
print(f"Number of occupied voxels: {occupied_voxel_count}")


####### dense voxel grid #######

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
o3d.visualization.draw_geometries([vox_mesh])
# o3d.io.write_triangle_mesh(input_path+”voxel_mesh_h.ply”, vox_mesh)