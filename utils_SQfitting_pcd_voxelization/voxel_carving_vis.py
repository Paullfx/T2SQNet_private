import open3d as o3d
import numpy as np
import os
import torch
import math
import pickle
from omegaconf import OmegaConf



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
    scale = np.linalg.norm(max_bound - min_bound) / 2.0 # 0.5* space diagonal of camera sphere
    
    vertices = np.asarray(model.vertices) #verstices of the mesh
    vertices -= center
    model.vertices = o3d.utility.Vector3dVector(vertices)
    #model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model, center, scale

def get_bbox(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    bbox = [center[0], center[1], center[2], (max_bound[0] - min_bound[0])/2, (max_bound[1] - min_bound[1])/2, (max_bound[2] - min_bound[2])/2]
    return bbox

def bbox2marginal_max(bbox, marginal_bbox_size, max_bbox_size):
    marginal_bbox = np.concatenate(
			(
				bbox[0:2], 
				np.array([bbox[2] - bbox[5] + marginal_bbox_size[2]]),
				marginal_bbox_size), 
			axis=0
		)
    max_bbox = np.concatenate(
			(
				bbox[0:2], 
				np.array([bbox[2] - bbox[5] + max_bbox_size[2]]),
				max_bbox_size), 
			axis=0
		)
    return marginal_bbox, max_bbox

def restore_voxel_grid(voxel_grid, center, scale): # not used
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

def voxel_carving(mesh, 
                  camera_path,
                  marginal_bbox,
                  voxel_size,
                  w=320,
                  h=240,
                  use_depth=True,
                  surface_method='pointcloud'):
    mesh.compute_vertex_normals()
    camera_sphere = o3d.io.read_triangle_mesh(camera_path)

    # setup dense voxel grid
    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        origin=[-marginal_bbox[3],-marginal_bbox[4],-marginal_bbox[5]], # because the pre-generated camera path is centered at the origin 
        color=[0.7,0.7,0.7],
        voxel_size=voxel_size,
        width=marginal_bbox[3] * 2,
        height=marginal_bbox[4] * 2,
        depth=marginal_bbox[5] * 2,
    )

    # rescale geometry and align the object (e.g. handleless cup) with the camera sphere
    camera_sphere,camera_center, camera_scale = preprocess(camera_sphere)
    #print("camera_center :", camera_center)
    # #print("camera_scale :", camera_scale)
    #print("Original mesh size:", mesh.get_max_bound() - mesh.get_min_bound())
    mesh, mesh_center, mesh_scale = preprocess(mesh)
    # print("mesh_center :", mesh_center)
    # print("mesh_scale :", mesh_scale)
    camera_centers = np.zeros((len(camera_sphere.vertices), 3))

     # # Visualize the camera poses
    # camera_pcd = o3d.geometry.PointCloud()
    # camera_pcd.points = o3d.utility.Vector3dVector(camera_centers)

    # setup visualizer to render depthmaps
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False) # set visible to False if you don't want to see the rendering
    vis.add_geometry(mesh) # shift mesh to origin
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
            voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param, keep_voxels_outside_image = True)
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
            voxel_size=voxel_size,
            min_bound=(-marginal_bbox[3], -marginal_bbox[4], -marginal_bbox[5]),
            max_bound=(marginal_bbox[3], marginal_bbox[4], marginal_bbox[5]))
    elif surface_method == 'mesh':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=voxel_size,
            min_bound=(-marginal_bbox[3], -marginal_bbox[4], -marginal_bbox[5]),
            max_bound=(marginal_bbox[3], marginal_bbox[4], marginal_bbox[5]))
    else:
        raise Exception('invalid surface method')
    voxel_carving_with_surface = voxel_surface + voxel_carving
    print("Original dense voxel hull size:", voxel_carving.get_max_bound() - voxel_carving.get_min_bound())
    # voxel_carving_with_surface = restore_voxel_grid(voxel_carving_with_surface, mesh_center, mesh_scale)
    # voxel_carving = restore_voxel_grid(voxel_carving, mesh_center, mesh_scale)
    # voxel_surface = restore_voxel_grid(voxel_surface, mesh_center, mesh_scale)
    
    print("Restored dense voxel carving size:", voxel_carving.get_max_bound() - voxel_carving.get_min_bound())
    print("Restored dense voxel carving + surface size:", voxel_carving_with_surface.get_max_bound() - voxel_carving_with_surface.get_min_bound())
    xyz_min = voxel_carving_with_surface.get_min_bound()
    xyz_max = voxel_carving_with_surface.get_max_bound()
    whd_bbox =[0.5*(xyz_min[0]+xyz_max[0]),0.5*(xyz_min[1]+xyz_max[1]),0.5*(xyz_min[2]+xyz_max[2]),0.5 *(xyz_max[0]-xyz_min[0]),0.5*(xyz_max[1]-xyz_min[1]),0.5*(xyz_max[2]-xyz_min[2])]
    print(whd_bbox)

    return voxel_carving_with_surface, voxel_carving, voxel_surface, bbox


def raw_voxel2vox(voxel_grid, marginal_bbox, voxel_size):
    w = round(marginal_bbox[3] * 2 / voxel_size)
    h = round(marginal_bbox[4] * 2 / voxel_size)
    d = round(marginal_bbox[5] * 2 / voxel_size)

    voxels = voxel_grid.get_voxels()
    try:
        list_indices = list(vx.grid_index for vx in voxels)
    except:
        print("voxel_grid.get_voxels() failed")
    indices = np.stack(list_indices)
    indices_tensor = torch.from_numpy(indices).long()
    vox = torch.zeros(w, h, d)
    vox[
        indices_tensor[:, 0], 
        indices_tensor[:, 1], 
        indices_tensor[:, 2]
    ] = 1
    vox = vox.to(torch.bool)
    
    return vox

def get_bounds(object_bbox, max_bbox, marginal_bbox, voxel_size):

    w = round(marginal_bbox[3] * 2 / voxel_size)
    h = round(marginal_bbox[4] * 2 / voxel_size)
    d = round(marginal_bbox[5] * 2 / voxel_size)
    w_min1 = math.floor(w * (marginal_bbox[3] - max_bbox[3]) / (2 * marginal_bbox[3]))
    w_max1 = math.ceil(w * (marginal_bbox[3] + max_bbox[3]) / (2 * marginal_bbox[3]))
    h_min1 = math.floor(h * (marginal_bbox[4] - max_bbox[4]) / (2 * marginal_bbox[4]))
    h_max1 = math.ceil(h * (marginal_bbox[4] + max_bbox[4]) / (2 * marginal_bbox[4]))
    d_min1 = 0
    d_max1 = math.ceil(d * (2 * max_bbox[5]) / (2 * marginal_bbox[5])) - 1
    bound1 = [w_min1, w_max1, h_min1, h_max1, d_min1, d_max1]

    w_min2 = math.floor(w * (marginal_bbox[3] - object_bbox[3]) / (2 * marginal_bbox[3]))
    w_max2 = math.ceil(w * (marginal_bbox[3] + object_bbox[3]) / (2 * marginal_bbox[3]))
    h_min2 = math.floor(h * (marginal_bbox[4] - object_bbox[4]) / (2 * marginal_bbox[4]))
    h_max2 = math.ceil(h * (marginal_bbox[4] + object_bbox[4]) / (2 * marginal_bbox[4]))
    d_min2 = 0
    d_max2 = math.ceil(d * (2 * object_bbox[5]) / (2 * marginal_bbox[5])) - 1
    bound2 = [w_min2, w_max2, h_min2, h_max2, d_min2, d_max2]

    return bound1, bound2

def orientation_transform():
    pass
    
def vox_augmentation(raw_voxel, bound1, bound2):

    raw_voxel = raw_voxel.float()
    w_min1, w_max1, h_min1, h_max1, d_min1, d_max1 = bound1
    w_min2, w_max2, h_min2, h_max2, d_min2, d_max2 = bound2

    inside_voxel = torch.zeros_like(raw_voxel).fill_(0.)
    inside_voxel[w_min2:w_max2, h_min2:h_max2, d_min2:d_max2] = 1.
    voxel = torch.stack([raw_voxel, inside_voxel])
    voxel = voxel[:, w_min1:w_max1, h_min1:h_max1, d_min1:d_max1]

    return voxel

def bbox2o3d (bbox):

    bbox = np.array(bbox)
    bbox_min = bbox[:3] - bbox[3:]
    bbox_max = bbox[:3] + bbox[3:]
    obj_bbox = o3d.geometry.AxisAlignedBoundingBox()
    obj_bbox.min_bound = bbox_min
    obj_bbox.max_bound = bbox_max

    return obj_bbox
# def mesh2voxelhull(mesh, camera_path, yml_path)

#     dataset_args = OmegaConf.load(yml_path)
#     marginal_bbox_size_dict = dataset_args.marginal_bbox_size
#     max_bbox_size_dict = dataset_args.max_bbox_size
#     voxel_size_dict = dataset_args.voxel_size



if __name__ == "__main__":
    ### load the data 

    scene_id = "tableware_6_1" # "tableware_6_1" # "tableware_5_12"
    object_class = "Laptop" # "Laptop" # "HandlessCup"


    mesh_folder = "./data_cubic_mesh"
    mesh_file_path = os.path.join(mesh_folder, f"{scene_id}_cubic_mesh.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    print("Original mesh size:", mesh.get_max_bound() - mesh.get_min_bound())
    original_mesh = mesh
    #add a print statement to check the mesh
    print(mesh)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    print(" The x, y, z axis are rendered as red, green, and blue arrows.")
    # o3d.visualization.draw_geometries([mesh,origin_frame])

    output_folder = "./results_voxel_carving"
    scene_subfolder = os.path.join(output_folder, scene_id)
    os.makedirs(scene_subfolder, exist_ok=True)
    voxel_grid_filename = os.path.join(scene_subfolder, f"{scene_id}_voxel_grid_2.ply")
    voxel_carving_filename = os.path.join(scene_subfolder, f"{scene_id}_voxel_carving_2.ply")
    voxel_surface_filename = os.path.join(scene_subfolder, f"{scene_id}_voxel_surface_2.ply")

    camera_path = os.path.join(".", "sphere2.ply") #sphere.ply
    visualization = True
    #cubic_size = 2.0
    #voxel_resolution = 128.0

    ######################### yml config load, bbox ##########

    bbox = get_bbox(mesh)

    yml_path = os.path.join('configs', 'voxelize_config.yml')
    dataset_args = OmegaConf.load(yml_path)
    marginal_bbox_size_dict = dataset_args.marginal_bbox_size
    max_bbox_size_dict = dataset_args.max_bbox_size
    voxel_size_dict = dataset_args.voxel_size

    marginal_bbox_size = torch.tensor(marginal_bbox_size_dict[object_class])
    max_bbox_size = torch.tensor(max_bbox_size_dict[object_class])
    voxel_size = voxel_size_dict[object_class]
    voxel_scale = torch.tensor(voxel_size).unsqueeze(0)

    bbox = get_bbox(mesh)
    marginal_bbox, max_bbox = bbox2marginal_max(bbox, marginal_bbox_size, max_bbox_size)

    marginal_bbox_o3d = bbox2o3d(marginal_bbox)
    marginal_bbox_o3d.color = (1, 0, 0)  # red

    max_bbox_o3d = bbox2o3d(max_bbox)
    max_bbox_o3d.color = (0, 1, 0)  # green

    object_bbox_o3d = bbox2o3d(bbox)
    object_bbox_o3d.color = (0, 0, 1)  # blue

    o3d.visualization.draw_geometries([original_mesh,origin_frame,object_bbox_o3d,max_bbox_o3d,marginal_bbox_o3d],window_name="Original Mesh")

    ### run the voxel carving method ###
    voxel_grid, voxel_carving, voxel_surface,bbox = voxel_carving(
        mesh, camera_path, marginal_bbox, voxel_size)
    


    marginal_center = (marginal_bbox_o3d.min_bound + marginal_bbox_o3d.max_bound) / 2
    max_center = (max_bbox_o3d.min_bound + max_bbox_o3d.max_bound) / 2
    object_center = (object_bbox_o3d.min_bound + object_bbox_o3d.max_bound) / 2

    marginal_bbox_o3d.translate(-marginal_center)
    max_bbox_o3d.translate(-max_center)
    object_bbox_o3d.translate(-object_center)

    o3d.visualization.draw_geometries([voxel_surface,origin_frame,object_bbox_o3d,max_bbox_o3d,marginal_bbox_o3d],window_name="Voxel Surface")
    o3d.visualization.draw_geometries([voxel_carving,origin_frame,object_bbox_o3d,max_bbox_o3d,marginal_bbox_o3d], window_name="Voxel Carving")
    o3d.visualization.draw_geometries([voxel_grid,origin_frame,object_bbox_o3d,max_bbox_o3d,marginal_bbox_o3d], window_name="Combined Voxel Hull")

    vox = raw_voxel2vox(voxel_carving, marginal_bbox, voxel_size)

    bound1, bound2 = get_bounds(bbox, max_bbox, marginal_bbox, voxel_size)

    voxel = vox_augmentation(vox,bound1, bound2)

    
    ### save
    voxel_info ={
        "voxel": voxel,
        "voxel_scale": voxel_scale,
        "bbox": bbox
    }

    print("done")

    output_folder = f"./data_test/{scene_id}"
    output_file = os.path.join(output_folder, f"{scene_id}_voxel_info.pkl")
    os.makedirs(output_folder, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(
            voxel_info, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    # # store the voxel grid 
    # o3d.io.write_voxel_grid(voxel_grid_filename, voxel_grid)
    # o3d.io.write_voxel_grid(voxel_carving_filename, voxel_carving)
    # o3d.io.write_voxel_grid(voxel_surface_filename, voxel_surface)

    # print(f"Voxel grids saved to {scene_subfolder}")

    # ### visualize the results ###
    # print("surface voxels")
    # print(voxel_surface)
    # #o3d.visualization.draw_geometries([voxel_surface])
    # o3d.visualization.draw_geometries([voxel_surface,origin_frame],window_name="Voxel Surface")

    # print("carved voxels")
    # print(voxel_carving)
    # #o3d.visualization.draw_geometries([voxel_carving])
    # 3d.visualization.draw_geometries([voxel_carving,origin_frame], window_name="Voxel Carving")

    # print("combined voxels (carved + surface)")
    # print(voxel_grid)
    # #o3d.visualization.draw_geometries([voxel_grid])
    # o3d.visualization.draw_geometries([voxel_grid,origin_frame], window_name="Combined Voxel Hull")

