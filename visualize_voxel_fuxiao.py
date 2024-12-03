import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def visualize_voxels(voxel_list):
    """
    Visualize the list of voxel tensors in a single 3D plot.

    Args:
        voxel_list (list): List of voxel dictionaries containing 'voxel' tensors.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'green', 'blue', 'orange']  # Different colors for each voxel
    for idx, voxel_info in enumerate(voxel_list):
        # Extract voxel data
        voxel = voxel_info['voxel'].cpu().numpy()  # Convert to NumPy if tensor
        
        # Ensure the voxel is 3-dimensional
        if len(voxel.shape) == 4 and voxel.shape[0] == 1:
            voxel = voxel.squeeze(0)  # Remove batch dimension if present
        elif len(voxel.shape) != 3:
            raise ValueError(f"Unexpected voxel shape: {voxel.shape}")

        voxel_shape = voxel.shape

        # Create a grid for the voxel
        x, y, z = np.meshgrid(
            np.arange(voxel_shape[0]),
            np.arange(voxel_shape[1]),
            np.arange(voxel_shape[2]),
            indexing='ij'  # Ensure the axes match the voxel's shape
        )
        
        # Plot only non-zero parts of the voxel
        non_zero_indices = voxel > 0  # Assuming non-zero values are occupied
        ax.scatter(
            x[non_zero_indices], 
            y[non_zero_indices], 
            z[non_zero_indices], 
            c=colors[idx % len(colors)], 
            label=f'Voxel {idx}',
            alpha=0.7
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Voxel Visualization")
    plt.show()

# Load the object_list from the file
with open('./intermediates/scene_id_default/object_list/object_list.pkl', 'rb') as f:
    obj_list = pickle.load(f)

visualize_voxels(obj_list[1])
