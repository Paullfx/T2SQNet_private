
import open3d as o3d
import numpy as np
import os

def load_voxel(file_path):
    """
    Load a .ply voxel file.
    
    Parameters:
        file_path (str): Path to the .ply voxel file.
    
    Returns:
        voxel_grid: The loaded voxel grid.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading voxel grid from: {file_path}")
    voxel_grid = o3d.io.read_voxel_grid(file_path)
    print("Voxel grid loaded successfully.")
    return voxel_grid

def analyze_voxel(voxel_grid):
    """
    Analyze the size and resolution of the voxel grid.
    
    Parameters:
        voxel_grid: The voxel grid to analyze.
    
    Returns:
        dict: A dictionary containing the size and resolution of the voxel grid.
    """
    # Calculate the bounding box
    min_bound = voxel_grid.get_min_bound()
    max_bound = voxel_grid.get_max_bound()
    size = max_bound - min_bound

    # Get voxel size
    voxel_size = voxel_grid.voxel_size

    # Calculate resolution
    resolution = np.round(size / voxel_size).astype(int)

    print("\nVoxel Hull Analysis:")
    print(f"Size in X dimension: {size[0]:.4f} (meters)")
    print(f"Size in Y dimension: {size[1]:.4f} (meters)")
    print(f"Size in Z dimension: {size[2]:.4f} (meters)")
    print(f"Voxel size: {voxel_size:.4f} (meters)")
    print(f"Resolution in X dimension: {resolution[0]} voxels")
    print(f"Resolution in Y dimension: {resolution[1]} voxels")
    print(f"Resolution in Z dimension: {resolution[2]} voxels")

    return {
        "size": {"X": size[0], "Y": size[1], "Z": size[2]},
        "voxel_size": voxel_size,
        "resolution": {"X": resolution[0], "Y": resolution[1], "Z": resolution[2]}
    }

def visualize_voxel(voxel_grid, title="Voxel Grid Visualization"):
    """
    Visualize the voxel grid.
    
    Parameters:
        voxel_grid: The voxel grid to visualize.
        title (str): Title of the visualization window.
    """
    print("\nVisualizing the Voxel Grid...")
    o3d.visualization.draw_geometries([voxel_grid], window_name=title)

if __name__ == "__main__":
    # path "./voxel_carving_results/tableware_5_12_voxel_carving.ply"
    scene_id = "tableware_5_12"
    output_folder = "./voxel_carving_results"
    voxel_file_path = os.path.join(output_folder,f"{scene_id}_voxel_carving.ply")

    
    try:
        # Load voxel grid
        voxel_grid = load_voxel(voxel_file_path)

        # Analyze voxel grid
        voxel_size = analyze_voxel(voxel_grid)

        # Visualize voxel grid
        visualize_voxel(voxel_grid)

    except Exception as e:
        print(f"An error occurred: {e}")
