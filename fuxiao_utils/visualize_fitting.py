import pickle
import torch
import open3d as o3d
import numpy as np

def visualize_voxels_with_open3d_single(voxel_dict, exp_index):
    """
    Visualize a single voxel dictionary using Open3D, displaying both parts of the voxel data in separate windows.

    Parameters:
        voxel_dict (dict): A dictionary containing:
                          - voxel_hull['voxel']: Tensor data (2*w*h*d).
                          - voxel_hull['voxel_scale']: Scale or metadata for the voxels.
        exp_index (str): The experiment index for labeling voxel data sources.
    """
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
            first_voxel_coords = first_voxel_coords[::2]
            second_voxel_coords = second_voxel_coords[::2]

            # Convert to float64
            first_voxel_coords = first_voxel_coords.astype(np.float64)
            second_voxel_coords = second_voxel_coords.astype(np.float64)

            # Generate colors dynamically based on x, y, and z coordinates
            def generate_colors(coords):
                if coords.shape[0] > 0:
                    x_vals, y_vals, z_vals = coords[:, 0], coords[:, 1], coords[:, 2]
                    min_x, max_x = x_vals.min(), x_vals.max()
                    min_y, max_y = y_vals.min(), y_vals.max()
                    min_z, max_z = z_vals.min(), z_vals.max()

                    normalized_x = (x_vals - min_x) / (max_x - min_x)
                    normalized_y = (y_vals - min_y) / (max_y - min_y)
                    normalized_z = (z_vals - min_z) / (max_z - min_z)

                    # Combine x, y, and z into RGB channels
                    colors = np.zeros((coords.shape[0], 3))
                    colors[:, 0] = normalized_x  # Red increases with x
                    colors[:, 1] = normalized_y  # Green increases with y
                    colors[:, 2] = normalized_z  # Blue increases with z
                    return colors
                else:
                    return np.zeros((0, 3))

            first_colors = generate_colors(first_voxel_coords)
            second_colors = generate_colors(second_voxel_coords)

            # Create point cloud objects
            first_voxel_grid = o3d.geometry.PointCloud()
            first_voxel_grid.points = o3d.utility.Vector3dVector(first_voxel_coords)
            first_voxel_grid.colors = o3d.utility.Vector3dVector(first_colors)

            second_voxel_grid = o3d.geometry.PointCloud()
            second_voxel_grid.points = o3d.utility.Vector3dVector(second_voxel_coords)
            second_voxel_grid.colors = o3d.utility.Vector3dVector(second_colors)

            # Print information
            print(f"Visualizing voxel from experiment {exp_index}, scale: {voxel_scale}")

            # Visualize the first part in a separate window
            o3d.visualization.draw_geometries([first_voxel_grid],
                                              window_name=f'Voxel Hull - Part 1 - Experiment {exp_index}')

            # Visualize the second part in a separate window
            # o3d.visualization.draw_geometries([second_voxel_grid], window_name=f'Voxel Hull - Part 2 - Experiment {exp_index}')
        else:
            print(f"Skipping non-Tensor voxel data, type: {type(voxel_data)}")
    else:
        print(f"Invalid voxel data, type: {type(voxel_dict)}")

# Define the experiment index
exp_index = "scene_id_default"  # Example experiment index
file_path = f'./intermediates/{exp_index}/object_list/object_list.pkl'

# Load the object list
with open(file_path, 'rb') as f:
    obj_list = pickle.load(f)

# Visualize the first voxel dictionary from obj_list[1]
visualize_voxels_with_open3d_single(obj_list[1][0], exp_index)
