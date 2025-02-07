import open3d as o3d


cubic_size = 2
voxel_resolution = 128


voxel_carving = o3d.geometry.VoxelGrid.create_dense(
    width=cubic_size,
    height=cubic_size,
    depth=cubic_size,
    voxel_size=cubic_size / voxel_resolution,
    origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],
    #origin=[0, 0, 0],
    color=[1.0, 0.7, 0.0])

origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)
print(" The x, y, z axis are rendered as red, green, and blue arrows.")
o3d.visualization.draw_geometries([voxel_carving, origin_frame], window_name="Voxel Carving")