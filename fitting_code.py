from models.pipelines import TSQPipeline
from omegaconf import OmegaConf
#from visualize_fitting import visualize_voxels_with_open3d_single
import yaml
import torch
import numpy as np
import os
import pickle

if __name__ == "__main__":

    # load cfg
    with open('t2sqnet_config.yml') as f:
        t2sqnet_cfg = yaml.safe_load(f)

    # instantiate TSQPipeline class
    tsqnet = TSQPipeline(
                    bbox_model_path=t2sqnet_cfg["bbox_model_path"],
                    bbox_config_path=t2sqnet_cfg["bbox_config_path"],
                    param_model_paths=t2sqnet_cfg["param_model_paths"],
                    param_config_paths=t2sqnet_cfg["param_config_paths"],
                    voxel_data_config_path=t2sqnet_cfg["voxel_data_config_path"],
                    device=t2sqnet_cfg["device"],
                    dummy_data_paths=t2sqnet_cfg["dummy_data_paths"],
                    num_augs=t2sqnet_cfg["num_augs"],
                    debug_mode=t2sqnet_cfg["debug_mode"]
    )

    # prepare data input
    object_idx = 3 #     "WineGlass" : 0, "Bowl" : 1, "Bottle" : 2, "BeerBottle" : 3,
    # "HandlessCup" : 4, "Mug" : 5, "Dish" : 6

    device = torch.device(t2sqnet_cfg["device"])

    # Load the voxel 

    # Define path flexibly with experiment index
    exp_index = "scene_id_default"  # Example experiment index
    file_path = f'./intermediates/{exp_index}/object_list/object_list.pkl'

    # Define Path fixed


    # Load the object list
    with open(file_path, 'rb') as f:
        obj_list = pickle.load(f)
    # print all the tabelware classes
    for i in range(len(obj_list[0])):
        print (type(obj_list[0][i]))
    
    voxel = obj_list[1][0]['voxel']#output of voxel_carving, 

    # load voxel_size
    # Approach 1: load voxel_scale from obj_list
    voxel_scale = obj_list[1][0]['voxel_scale']
    # # Approach 2:load from yml
    # voxel_data_config = OmegaConf.load(voxel_data_config_path)
    # voxel_scale = voxel_data_config['voxel_size'] #torch.tensor([0.01], device=device)
    # # Approach 3: fix the voxel_scale to one specific tableware class
    # voxel_scale = torch.tensor([0.001832966443807953], device=device) # voxel_size from voxelize_config.yml

    # param_predictor
    obj_info = tsqnet.param_predictors[object_idx](voxel.unsqueeze(0), voxel_scale).squeeze()

    # save the obj_info in ./intermediates/{exp_index}/obj_info/obj_info.pkl
    output_dir_obj_info = f'./intermediates/{exp_index}/obj_info'  
    if not os.path.exists(output_dir_obj_info):
        os.makedirs(output_dir_obj_info)
    with open(os.path.join(output_dir_obj_info, 'obj_info.pkl'), 'wb') as f:
        pickle.dump(obj_info, f)

