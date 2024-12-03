import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_saved_occupancy(file_path):
    """
    Visualize the saved occupancy grid from a .npy file.
    Args:
        file_path (str): Path to the saved occupancy.npy file.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load the occupancy grid
    occupancy = np.load(file_path)
    print(f"Loaded occupancy grid with shape: {occupancy.shape}")

    # Extract coordinates of occupied voxels
    occupied_coords = np.argwhere(occupancy > 0)  # Get indices of occupied voxels
    x, y, z = occupied_coords[:, 0], occupied_coords[:, 1], occupied_coords[:, 2]

    # Plot the occupied voxels
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, alpha=0.5, c='blue')  # Customize size, transparency, and color

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("3D Occupancy Grid Visualization")
    plt.show()

# Define the path to the saved occupancy.npy file
file_path = './intermediates/scene_id_default/voxel/occupancy.npy'

# Call the visualization function
visualize_saved_occupancy(file_path)
