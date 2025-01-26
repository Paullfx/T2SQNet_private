import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_voxels_with_matplotlib(voxel_list, exp_index):
    """
    Visualize voxel data using Matplotlib, displaying both parts of each voxel_data in separate plots.

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

                # Print the shape of the voxel data
                print(f"Voxel {i + 1} shape: {voxel_data.shape}")

                # Validate dimensions
                assert voxel_data.shape[0] == 2, "The first dimension of voxel data must be 2"
                _, w, h, d = voxel_data.shape

                # Separate the two parts of the voxel data
                first_part = voxel_data[0]  # Shape: w x h x d
                second_part = voxel_data[1]  # Shape: w x h x d

                # Get non-zero voxel coordinates
                first_voxel_coords = np.transpose(np.nonzero(first_part))  # Shape: (N1, 3)
                second_voxel_coords = np.transpose(np.nonzero(second_part))  # Shape: (N2, 3)

                # Calculate bounding box for each part
                first_bbox_min = first_voxel_coords.min(axis=0)
                first_bbox_max = first_voxel_coords.max(axis=0)
                second_bbox_min = second_voxel_coords.min(axis=0)
                second_bbox_max = second_voxel_coords.max(axis=0)

                print(f"Voxel {i + 1} - Part 1 BBox: Min {first_bbox_min}, Max {first_bbox_max}")
                print(f"Voxel {i + 1} - Part 2 BBox: Min {second_bbox_min}, Max {second_bbox_max}")

                # Downsample voxel data (e.g., take every 2nd point)
                first_voxel_coords = first_voxel_coords[::2]
                second_voxel_coords = second_voxel_coords[::2]

                # Convert to float64
                first_voxel_coords = first_voxel_coords.astype(np.float64)
                second_voxel_coords = second_voxel_coords.astype(np.float64)

                # Print information
                print(f"Visualizing voxel {i + 1}, from experiment {exp_index}, scale: {voxel_scale}")

                # Create Matplotlib 3D scatter plot for the first part
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111, projection='3d')
                ax1.scatter(first_voxel_coords[:, 0], first_voxel_coords[:, 1], first_voxel_coords[:, 2], 
                            c='blue', marker='o', s=1, label='Part 1')
                ax1.set_title(f'Voxel Hull {i + 1} - Part 1 - Experiment {exp_index}')
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
                ax1.legend()
                plt.show()

                # Create Matplotlib 3D scatter plot for the second part
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111, projection='3d')
                ax2.scatter(second_voxel_coords[:, 0], second_voxel_coords[:, 1], second_voxel_coords[:, 2], 
                            c='orange', marker='o', s=1, label='Part 2')
                ax2.set_title(f'Voxel Hull {i + 1} - Part 2 - Experiment {exp_index}')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('Z')
                ax2.legend()
                plt.show()

            else:
                print(f"Skipping non-Tensor voxel data, index: {i}, type: {type(voxel_data)}")
        else:
            print(f"Invalid voxel data, index: {i}, type: {type(voxel_dict)}")

# Define the experiment index
exp_index = "pybullet_table_1_2"  # Example experiment index # exp_index = "pybullet_table_1_2" # "pybullet_single_HandlessCup"
file_path = f'./intermediates/{exp_index}/object_list/object_list.pkl'

# Load the object list
with open(file_path, 'rb') as f:
    obj_list = pickle.load(f)

# Print all the tableware classes
for i in range(len(obj_list[0])):
    print(type(obj_list[0][i]))

# Call the function to visualize
visualize_voxels_with_matplotlib(obj_list[1], exp_index)
