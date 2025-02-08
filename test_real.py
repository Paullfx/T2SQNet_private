import os
import shutil
from pcd2grid import pcd2grid



################# Block 1: voxelization #########################

# copy the real pcd data extracted by ConceptGraph pipeline
# creat a folder ./data_real, and copy paste the tableware_6_1_laptop_denposed.ply into the folder
# Define the source and destination paths
source_path = '/path/to/tableware_6_1_laptop_denposed.ply'
destination_folder = './data_real'
destination_path = os.path.join(destination_folder, 'tableware_6_1_laptop_denposed.ply')

os.makedirs(destination_folder, exist_ok=True)
shutil.copyfile(source_path, destination_path)
print(f"point cloud copied to {destination_path}") # debug

# voxelize the pcd
# use the functions in ./utils_SQfitting_pcd_voxelization/pcd_voxelization_cubic_mesh.py
    # input to the function: pc_path, voxel_size, voxel_save_path
# load pcd_path
# visualize pc_path
voxel_grid = pcd2grid(pc, voxel_size)
# save voxel_grid
# visualize voxel grid

voxel_mesh_grid = grid2meshgrid(voxel grid)
# save the voxel_mesh_grid
# visualize the voxel_mesh_grid

################# Block 2: voxel carving #########################
# Todo (draft on voxel_carving_vis.py)
        #   sphere.ply, r =1 # ref: generate_pybullet.py
    # center_shift = pcd_center - [0,0,0]
    # create dense (marginal, center = 0) # ref genrate_voxelized.py 179

    # check training input center&shift #ref: notion test_real.py # ref:in def train_step of voxel_head.py, line 170

# Good to have
    # try replace mesh input with normal voxelization
    # visualize and compare the mesh.vertical normal 
bbox = get_bbox(pc)

max_bbox = get_max_bbox(bbox, max_box_size)

marginal_bbox = get_marginal_bbox(bbox, marginal_box_size)

raw_voxel = voxel_carving (voxel_mesh_grid, voxel_scale, marginal_bbox)

# visualize raw_voxel



inside_voxel = 

# visualize inside_voxel

input_voxel = stack [raw_voxel, inside_voxel]

################### Block 3: inference with parameter prediction ######################

# load cfg


# instantiate TSQPipeline class

obj_info = tsqnet.param_predictors[object_idx](voxel.unsqueeze(0), voxel_scale).squeeze()

# reconstruct the laptop object using Tableware class

# visualize the reconstructed object and the true pcd