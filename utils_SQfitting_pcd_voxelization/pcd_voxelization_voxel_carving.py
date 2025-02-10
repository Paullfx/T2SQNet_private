import open3d as o3d
import numpy as np
import os

# Reference: https://www.open3d.org/docs/latest/tutorial/Advanced/voxelization.html

# convert cartesian coordinates into spherical coordinates
def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]

# get rotation matrix from multiplying two rotation matrices
def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                        [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
                        [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x) # fisrtly rotate around x-axis, then rotate around y-axis

# construct transformation matrix with rotation matrix and translation vector
def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0, 0, 2]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans


def preprocess(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0 # 0.5* space diagonal
    vertices = np.asarray(model.vertices) #verstices of the mesh
    vertices -= center
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model, center, scale


def restore_voxel_grid(voxel_grid, center, scale): #newest
    # Create a new VoxelGrid for the restored voxels
    restored_voxel_grid = o3d.geometry.VoxelGrid()
    restored_voxel_grid.voxel_size = voxel_grid.voxel_size * scale  # Restore voxel size

    # Calculate the new origin for the restored voxel grid
    restored_voxel_grid.origin = center - (voxel_grid.origin * scale)

    # Iterate over the original voxels to compute their new positions
    for voxel in voxel_grid.get_voxels():
        # Compute the original voxel's center position in the scaled space
        voxel_center = voxel_grid.origin + np.array(voxel.grid_index, dtype=float) * voxel_grid.voxel_size

        # Scale and translate the voxel center to the original space
        restored_center = voxel_center * scale + center

        # Compute the new grid index for the restored voxel
        restored_grid_index = np.floor((restored_center - restored_voxel_grid.origin) / restored_voxel_grid.voxel_size).astype(int)

        # Create a new voxel with the restored grid index and original color
        restored_voxel = o3d.geometry.Voxel(restored_grid_index, voxel.color)

        # Add the restored voxel to the new voxel grid
        restored_voxel_grid.add_voxel(restored_voxel)

    return restored_voxel_grid



# voxel carving method
def voxel_carving(mesh,
                  camera_path,
                  cubic_size,
                  voxel_resolution,
                  w=300,
                  h=300,
                  use_depth=True,
                  surface_method='pointcloud'):
    mesh.compute_vertex_normals()
    camera_sphere = o3d.io.read_triangle_mesh(camera_path)

    # setup dense voxel grid
    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size,
        height=cubic_size,
        depth=cubic_size,
        voxel_size=cubic_size / voxel_resolution,
        origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],
        #origin=[0, 0, 0],
        color=[1.0, 0.7, 0.0])

    # rescale geometry and align the object (e.g. handleless cup) with the camera sphere
    camera_sphere,camera_center, camera_scale = preprocess(camera_sphere)
    print("camera_center :", camera_center)
    print("camera_scale :", camera_scale)
    print("number of camera sphere vertices:", len(camera_sphere.vertices))
    print("Original mesh size:", mesh.get_max_bound() - mesh.get_min_bound())
    mesh, mesh_center, mesh_scale = preprocess(mesh)
    print("mesh_center :", mesh_center)
    print("mesh_scale :", mesh_scale)
    camera_centers = np.zeros((len(camera_sphere.vertices), 3))

    # # Visualize the camera poses
    # camera_pcd = o3d.geometry.PointCloud()
    # camera_pcd.points = o3d.utility.Vector3dVector(camera_centers)


    # setup visualizer to render depthmaps
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=True) # set visible to False if you don't want to see the rendering
    vis.add_geometry(mesh)
    #vis.add_geometry(camera_pcd) # only for visualization purpose
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)  # Adjust size for visibility
    # vis.add_geometry(origin_frame)  # The global coordinate axes

    pcd_agg = o3d.geometry.PointCloud()
    
    for cid, xyz in enumerate(camera_sphere.vertices): # enumerate through all the vertices of the camera sphere to get camera poses
        # get new camera pose
        trans = get_extrinsic(xyz)
        param.extrinsic = trans
        c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())
        camera_centers[cid, :] = c[:3] # cid index from 0 to . raw cid, and all columns. c[:3] include the first three elements of 4-dim c
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True) # 

        #  # Create a small coordinate frame at the camera position
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # Adjust size as needed
        # frame.transform(trans)  # Apply the camera transformation
        # vis.add_geometry(frame)  # Add the frame to the visualizer

        # capture depth image and make a point cloud
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)
        pcd_agg += o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth),
            param.intrinsic,
            param.extrinsic,
            depth_scale=1)

        # depth map carving method
        if use_depth:
            voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)
        else:
            voxel_carving.carve_silhouette(o3d.geometry.Image(depth), param)
        #print("Carve view %03d/%03d" % (cid + 1, len(camera_sphere.vertices)))
    vis.destroy_window()
    print("Reshaped mesh size:", mesh.get_max_bound() - mesh.get_min_bound())
    print("Dense voxel grid size:", voxel_carving.get_max_bound() - voxel_carving.get_min_bound())
    # print(f"Saving voxel_carving to {output_filename}")
    # o3d.io.write_voxel_grid(output_filename, voxel_carving)


    # add voxel grid survace
    print('Surface voxel grid from %s' % surface_method)
    if surface_method == 'pointcloud':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd_agg,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
    elif surface_method == 'mesh':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
    else:
        raise Exception('invalid surface method')
    voxel_carving_surface = voxel_surface + voxel_carving
    print("Original dense voxel hull size:", voxel_carving.get_max_bound() - voxel_carving.get_min_bound())
    voxel_carving_surface = restore_voxel_grid(voxel_carving_surface, mesh_center, mesh_scale)
    voxel_carving = restore_voxel_grid(voxel_carving, mesh_center, mesh_scale)
    voxel_surface = restore_voxel_grid(voxel_surface, mesh_center, mesh_scale)
    
    print("Restored dense voxel carving size:", voxel_carving.get_max_bound() - voxel_carving.get_min_bound())
    print("Restored dense voxel carving + surface size:", voxel_carving_surface.get_max_bound() - voxel_carving_surface.get_min_bound())
    xyz_min = voxel_carving_surface.get_min_bound()
    xyz_max = voxel_carving_surface.get_max_bound()
    bbox =[0.5*(xyz_min[0]+xyz_max[0]),0.5*(xyz_min[1]+xyz_max[1]),0.5*(xyz_min[2]+xyz_max[2]),0.5 *(xyz_max[0]-xyz_min[0]),0.5*(xyz_max[1]-xyz_min[1]),0.5*(xyz_max[2]-xyz_min[2])]
    print(bbox)



    return voxel_carving_surface, voxel_carving, voxel_surface, bbox

### load the data 

scene_id = "tableware_5_12"

mesh_folder = "./data_cubic_mesh"
mesh_file_path = os.path.join(mesh_folder, f"{scene_id}_cubic_mesh.ply")
mesh = o3d.io.read_triangle_mesh(mesh_file_path)
#add a print statement to check the mesh
print(mesh)

origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
print(" The x, y, z axis are rendered as red, green, and blue arrows.")
o3d.visualization.draw_geometries([mesh,origin_frame])

#o3d.visualization.draw_geometries([mesh])

output_folder = "./results_voxel_carving"
scene_subfolder = os.path.join(output_folder, scene_id)
os.makedirs(scene_subfolder, exist_ok=True)
voxel_grid_filename = os.path.join(scene_subfolder, f"{scene_id}_voxel_grid.ply")
voxel_carving_filename = os.path.join(scene_subfolder, f"{scene_id}_voxel_carving.ply")
voxel_surface_filename = os.path.join(scene_subfolder, f"{scene_id}_voxel_surface.ply")

camera_path = os.path.join(".", "old_sphere.ply") #sphere.ply
#camera_path = os.path.join(".", "utils_SQfitting_pcd_voxelization", "sphere2.ply")

visualization = True
cubic_size = 2.0
voxel_resolution = 128.0


### run the voxel carving method ###
voxel_grid, voxel_carving, voxel_surface,bbox = voxel_carving(
    mesh, camera_path, cubic_size, voxel_resolution)

# store the voxel grid 
o3d.io.write_voxel_grid(voxel_grid_filename, voxel_grid)
o3d.io.write_voxel_grid(voxel_carving_filename, voxel_carving)
o3d.io.write_voxel_grid(voxel_surface_filename, voxel_surface)

print(f"Voxel grids saved to {scene_subfolder}")

### visualize the results ###
print("surface voxels")
print(voxel_surface)
#o3d.visualization.draw_geometries([voxel_surface])
o3d.visualization.draw_geometries([voxel_surface,origin_frame])

print("carved voxels")
print(voxel_carving)
#o3d.visualization.draw_geometries([voxel_carving])
o3d.visualization.draw_geometries([voxel_carving,origin_frame])

print("combined voxels (carved + surface)")
print(voxel_grid)
#o3d.visualization.draw_geometries([voxel_grid])
o3d.visualization.draw_geometries([voxel_grid,origin_frame])

# def restore_voxel_grid(voxel_grid, center, scale):
    
#     restored_voxel_grid = o3d.geometry.VoxelGrid()
#     print("voxel_grid.voxel_size:", voxel_grid.voxel_size)
#     restored_voxel_grid.voxel_size = voxel_grid.voxel_size * scale  # restore voxel size
#     print("restored_voxel_grid.voxel_size:", restored_voxel_grid.voxel_size)
#     restored_voxels = []
#     print("total voxel number of voxel hull:", len(voxel_grid.get_voxels()))
#     for voxel in voxel_grid.get_voxels():
#         #print("voxel.grid_index:", voxel.grid_index)
#         voxel_index = np.array(voxel.grid_index, dtype=float)
#         #print("voxel_index:", voxel_index)

#         # compute the voxel center of every voxel in restored grid
#         voxel_center = voxel_index * voxel_grid.voxel_size 
#         voxel_center = voxel_center * scale  
#         voxel_center += center 
#         print("voxel_center:", voxel_center)
#         restored_voxels.append(o3d.geometry.Voxel(voxel_center, voxel.color))

    
#     print("length of restored voxel list:", len(restored_voxels))
#     restored_voxel_grid.origin = center 
#     for voxel in restored_voxels:
#         #print("voxel:", voxel)
#         restored_voxel_grid.add_voxel(voxel)

#     return restored_voxel_grid





# def restore_voxel_grid(voxel_grid, center, scale):
#     transformed_voxels = []

#     for voxel in voxel_grid.get_voxels():
#         # Convert voxel center to original scale and position
#         voxel.grid_index = voxel.grid_index * scale + center
#         transformed_voxels.append(voxel)

#     return voxel_grid

# def restore_voxel_grid(voxel_grid, center, scale):
#     # Extract voxel centers and restore them to original coordinates
#     restored_voxels = []
    
#     for voxel in voxel_grid.get_voxels():
#         # Convert voxel center from normalized space back to original
#         voxel_center = np.array(voxel.grid_index, dtype=float) * (2.0 / voxel_grid.voxel_size)  # Convert indices to coordinates
#         voxel_center = voxel_center * scale + center  # Scale and translate back

#         # Create a new voxel with restored coordinates
#         restored_voxels.append(o3d.geometry.Voxel(voxel_center, voxel.color))

#     # Create a new voxel grid with restored positions
#     restored_voxel_grid = o3d.geometry.VoxelGrid()
#     restored_voxel_grid.voxel_size = voxel_grid.voxel_size * scale  # Restore original voxel size
#     restored_voxel_grid.origin = center - (voxel_grid.origin * scale)  # Restore origin
#     for voxel in restored_voxels:
#         restored_voxel_grid.add_voxel(voxel)

#     return restored_voxel_grid


# def camera_sphere_preprocess(model):
#     min_bound = model.get_min_bound()
#     max_bound = model.get_max_bound()
#     center = min_bound + (max_bound - min_bound) / 2.0
#     scale = np.linalg.norm(max_bound - min_bound) / 2.0 # 0.5* space diagonal
#     vertices = np.asarray(model.vertices) #verstices of the mesh
#     vertices -= center
#     vertices *= 3.0 # make the camera sphere several times larger than the object mesh
#     model.vertices = o3d.utility.Vector3dVector(vertices / scale)
#     return model