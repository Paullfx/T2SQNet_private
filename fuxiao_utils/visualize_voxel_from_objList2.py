import pickle
import torch
import open3d as o3d
import numpy as np

def visualize_voxels_with_open3d(voxel_list, exp_index):
    """
    Visualize voxel data using Open3D, displaying both parts of each voxel_data in separate windows.

    Parameters:
        voxel_list (list): A list of dictionaries, each containing:
                           - voxel_hull['voxel']: Tensor data (2*w*h*d).
                           - voxel_hull['voxel_scale']: Scale or metadata for the voxels.
        exp_index (str): The experiment index for labeling voxel data sources.
    """
    for i, voxel_dict in enumerate(voxel_list):
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

                # Create point cloud objects
                first_voxel_grid = o3d.geometry.PointCloud()
                first_voxel_grid.points = o3d.utility.Vector3dVector(first_voxel_coords)

                second_voxel_grid = o3d.geometry.PointCloud()
                second_voxel_grid.points = o3d.utility.Vector3dVector(second_voxel_coords)

                # Assign colors to the point clouds
                first_voxel_grid.paint_uniform_color([0.0, 0.5, 1.0])  # Blue
                second_voxel_grid.paint_uniform_color([1.0, 0.5, 0.0])  # Orange

                # Print information
                print(f"Visualizing voxel {i + 1}, from experiment {exp_index}, scale: {voxel_scale}")
                
                # print the tableware class
                #table_class = 
                # Visualize the first part in a separate window
                o3d.visualization.draw_geometries([first_voxel_grid],
                                                  window_name=f'Voxel Hull {i + 1} - Part 1 - Experiment {exp_index} - Class')
                # Visualize the second part in a separate window
                #o3d.visualization.draw_geometries([second_voxel_grid], window_name=f'Voxel Hull {i + 1} - Part 2 - Experiment {exp_index}')
            else:
                print(f"Skipping non-Tensor voxel data, index: {i}, type: {type(voxel_data)}")
        else:
            print(f"Invalid voxel data, index: {i}, type: {type(voxel_dict)}")


# Define the experiment index
exp_index = "scene_id_default"  # Example experiment index
file_path = f'./intermediates/{exp_index}/object_list/object_list.pkl'

# Load the object list
with open(file_path, 'rb') as f:
    obj_list = pickle.load(f)

# print all the tabelware classes
for i in range(len(obj_list[0])):
    print (type(obj_list[0][i]))

# Call the function to visualize
visualize_voxels_with_open3d(obj_list[1], exp_index)

