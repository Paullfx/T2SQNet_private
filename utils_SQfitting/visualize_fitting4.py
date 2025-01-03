import pickle
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import get_cmap
import os

def grid_to_bbox_coordinates(i, j, k, bbox, voxel_size):

    x_actual = bbox[0] - bbox[3] + i * voxel_size + 0.5 * voxel_size
    y_actual = bbox[1] - bbox[4] + j * voxel_size + 0.5 * voxel_size
    z_actual = bbox[2] - bbox[5] + k * voxel_size + 0.5 * voxel_size
    return x_actual, y_actual, z_actual

def visualize_voxels_with_open3d_single(voxel_dict):
    """
    Visualize a single voxel dictionary using Open3D, displaying both parts of the voxel data in separate windows.

    Parameters:
        voxel_dict (dict): A dictionary containing:
                          - voxel_hull['voxel']: Tensor data (2*w*h*d).
                          - voxel_hull['voxel_scale']: Scale or metadata for the voxels.
        exp_index (str): The experiment index for labeling voxel data sources.
    """
    # Define paths
    #exp_index = "scene_id_default"# "tableware_3_9" #scene_id_default
    input_dir = f'./intermediates/{exp_index}/bboxes_cls'
    results_dir = f'./intermediates/{exp_index}/results'

    bboxes_path = os.path.join(input_dir, 'bboxes.pkl')
    cls_path = os.path.join(input_dir, 'cls.pkl')
    results_path = os.path.join(results_dir, 'results.pkl')

    # Load class names
    class_names = []
    if os.path.exists(cls_path):
        with open(cls_path, 'rb') as f:
            class_names = pickle.load(f)
    else:
        print(f"Error: Class file not found at {cls_path}.")

    # Load bounding boxes
    bboxes = []
    if os.path.exists(bboxes_path):
        with open(bboxes_path, 'rb') as f:
            bboxes = pickle.load(f)
    else:
        print(f"Error: Bounding box file not found at {bboxes_path}.")

    bbox = bboxes[3] # bboxes[i] for the next object
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.cpu().numpy()
    # Load results
    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print("Results loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at {results_path}. Please check the path and try again.")
        return

    # Extract point cloud data
    sq_results = results[3][0]  # the list of detected objects
    number_of_points = 1000
    points = []
    points.append(results[3][0][3].get_point_cloud(number_of_points=number_of_points)) #results[3][0][i] for the next object 
    # for obj in sq_results:
    #     points.append(obj.get_point_cloud(number_of_points=number_of_points))  # Use the get_point_cloud function
    #     print(type(obj))
    if isinstance(points, list) and len(points) == 1 and isinstance(points[0], np.ndarray):
        points = points[0]
    else:
        raise ValueError

    if isinstance(voxel_dict, dict):
        # Extract voxel data and scale
        voxel_data = voxel_dict['voxel']
        voxel_scale = voxel_dict['voxel_scale']

        if isinstance(voxel_data, torch.Tensor):
            # Convert to NumPy format
            voxel_data = voxel_data.cpu().numpy()

            # Validate dimensions
            assert voxel_data.shape[0] == 2, "The first dimension of voxel data must be 2"
            _, w, h, d = voxel_data.shape

            # Separate the two parts of the voxel data
            first_part = voxel_data[0]  # Shape: w x h x d
            second_part = voxel_data[1]  # Shape: w x h x d

            # Get non-zero voxel coordinates
            first_voxel_coords = np.transpose(np.nonzero(first_part))  # Shape: (N1, 3)
            second_voxel_coords = np.transpose(np.nonzero(second_part))  # Shape: (N2, 3)

            # Downsample voxel data (e.g., take every 10th point)
            first_voxel_coords = first_voxel_coords[::20]
            second_voxel_coords = second_voxel_coords[::20]

            # marginal box
            marginal_bbox_size = np.array([0.0750041242892148, 0.07501678063491438, 0.17886416966013752])
            marginal_bbox = np.concatenate(
                (
            bbox[0:2], 
            np.array([bbox[2] - bbox[5] + marginal_bbox_size[2]]),
            marginal_bbox_size
                ), 
                axis=0
            )

            # Define bbox and voxel size (example values, replace with actual data)
            voxel_size = 0.0036  # Replace with voxel_size according to tableware class

            # Transform voxel coordinates to bbox coordinates
            first_voxel_coords_transformed = np.array([
                grid_to_bbox_coordinates(i, j, k, marginal_bbox, voxel_size) 
                for i, j, k in first_voxel_coords
            ])
            # second_voxel_coords_transformed = np.array([
            #     grid_to_bbox_coordinates(i, j, k, marginal_bbox, voxel_size) 
            #     for i, j, k in second_voxel_coords
            # ])


            # Matplotlib voxel and point cloud visualization
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Ensure equal scaling of the axes
            # all_coords = np.vstack((first_voxel_coords, second_voxel_coords, pcd_coords))
            all_coords = np.vstack((first_voxel_coords_transformed, points))
            max_range = (all_coords.max(axis=0) - all_coords.min(axis=0)).max() / 2.0
            mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
            mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
            mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            # Plot first part of transformed coordinates
            ax.scatter(first_voxel_coords_transformed[:, 0], first_voxel_coords_transformed[:, 1], first_voxel_coords_transformed[:, 2],
                       color='blue', marker='x', label='First Part transformed')


            # # Plot first part
            # ax.scatter(first_voxel_coords[:, 0], first_voxel_coords[:, 1], first_voxel_coords[:, 2],
            #            color='red', marker='o', label='First Part')

            # # Plot second part
            # ax.scatter(second_voxel_coords[:, 0], second_voxel_coords[:, 1], second_voxel_coords[:, 2],
            #            color='green', marker='^', label='Second Part')

            # Plot point cloud data
            ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color='black',
            marker='D',
            label='Point Cloud'
            )
            # for i, obj in enumerate(points):
            #     x, y, z = obj[:, 0], obj[:, 1], obj[:, 2]
            #     ax.scatter(x, y, z, color='black', marker='.', label=f"T2SQNet: {class_names[i]}", s=5)

            plt.legend()
            plt.show()

        else:
            print(f"Skipping non-Tensor voxel data, type: {type(voxel_data)}")
    else:
        print(f"Invalid voxel data, type: {type(voxel_dict)}")

# Define the experiment index
exp_index = "scene_id_default"  # scene_id_default
file_path = f'./intermediates/{exp_index}/object_list/object_list.pkl'

# Load the object list
with open(file_path, 'rb') as f:
    obj_list = pickle.load(f)

# Visualize the first voxel dictionary from obj_list[1]
visualize_voxels_with_open3d_single(obj_list[1][3]) # ob_list[1][i] for the next object
