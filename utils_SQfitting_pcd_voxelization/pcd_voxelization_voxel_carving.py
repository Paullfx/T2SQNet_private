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
    return model


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

# voxel carving method
def voxel_carving(mesh,
                  output_filename,
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
        color=[1.0, 0.7, 0.0])

    # rescale geometry and align the object (e.g. handleless cup) with the camera sphere
    camera_sphere = preprocess(camera_sphere)
    mesh = preprocess(mesh)

    # setup visualizer to render depthmaps
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    pcd_agg = o3d.geometry.PointCloud()
    centers_pts = np.zeros((len(camera_sphere.vertices), 3))
    for cid, xyz in enumerate(camera_sphere.vertices): # enumerate through all the vertices of the camera sphere to get camera poses
        # get new camera pose
        trans = get_extrinsic(xyz)
        param.extrinsic = trans
        c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())
        centers_pts[cid, :] = c[:3] # cid very likely index from 0 to . raw cid, and all columns. c[:3] include the first three elements of 4-dim c
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True) # 

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

    print(f"Saving voxel_carving to {output_filename}")
    o3d.io.write_voxel_grid(output_filename, voxel_carving)


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

    return voxel_carving_surface, voxel_carving, voxel_surface

### load the data 

scene_id = "tableware_5_12"

mesh_folder = "./data_cubic_mesh"
mesh_file_path = os.path.join(mesh_folder, f"{scene_id}_cubic_mesh.ply")
mesh = o3d.io.read_triangle_mesh(mesh_file_path)
#add a print statement to check the mesh
print(mesh)
o3d.visualization.draw_geometries([mesh])

output_folder = "./voxel_carving_results"
os.makedirs(output_folder, exist_ok=True)
output_filename = os.path.join(output_folder,f"{scene_id}_voxel_carving.ply")

camera_path = os.path.join(".", "sphere.ply")
visualization = True
cubic_size = 2.0
voxel_resolution = 128.0


### run the voxel carving method ###
voxel_grid, voxel_carving, voxel_surface = voxel_carving(
    mesh, output_filename, camera_path, cubic_size, voxel_resolution)

### visualize the results ###
print("surface voxels")
print(voxel_surface)
o3d.visualization.draw_geometries([voxel_surface])

print("carved voxels")
print(voxel_carving)
o3d.visualization.draw_geometries([voxel_carving])

print("combined voxels (carved + surface)")
print(voxel_grid)
o3d.visualization.draw_geometries([voxel_grid])