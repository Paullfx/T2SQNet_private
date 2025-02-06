from models.pipelines import TSQPipeline
from omegaconf import OmegaConf
#from visualize_fitting import visualize_voxels_with_open3d_single
import yaml
import torch
import numpy as np
import os
import pickle
import open3d as o3d
from utils_SQfitting.visualize_voxel_from_objList2 import visualize_voxels_with_open3d
from tablewarenet.tableware import *

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
    object_class = "HandlessCup"
    #object_idx = 4 #     "WineGlass" : 0, "Bowl" : 1, "Bottle" : 2, "BeerBottle" : 3,
    # "HandlessCup" : 4, "Mug" : 5, "Dish" : 6
    object_idx = name_to_idx[object_class]

    device = torch.device('cuda:0')

####################################### Load the voxel and voxel scale from simulation

    # Define path flexibly with experiment index
    exp_index = "pybullet_single_HandlessCup"  # Example experiment index
    file_path = f'./intermediates/{exp_index}/object_list/object_list.pkl'
    # Load the object list
    with open(file_path, 'rb') as f:
        obj_list = pickle.load(f)
    ## print all the tabelware classes
    # for i in range(len(obj_list[0])):
    #     print (type(obj_list[0][i]))
    # the saved obj_list contains object_list(superquadric parameters) and voxel_info (results of voxel carving, including voxel and voxel_scale)
    voxel = obj_list[1][0]['voxel']
    visualize_voxels_with_open3d(obj_list[1], exp_index)
    # tableware = obj_list[0] 


    # load voxel_size
    # Approach 1: load voxel_scale from obj_list
    voxel_scale = obj_list[1][0]['voxel_scale']
    # print(voxel_scale)
    # # Approach 2:load from yml
    # voxel_data_config = OmegaConf.load(voxel_data_config_path)
    # voxel_scale = voxel_data_config['voxel_size'] #torch.tensor([0.01], device=device)
    # # Approach 3: fix the voxel_scale to one specific tableware class
    # voxel_scale = torch.tensor([0.001832966443807953], device=device) # voxel_size from voxelize_config.yml

    # Extract the bbox from simulation

    bbox_file_path = f'./intermediates/{exp_index}/bboxes_cls/bboxes.pkl'
    with open(bbox_file_path, 'rb') as f:
        bboxes = pickle.load(f)
    bbox = bboxes[0] # default the first object




    ################################ End of loading voxel from sim

    ################################ Voxel from real data


    ################################ End of loading voxel from real data

    ################################ param_predictor
    obj_info = tsqnet.param_predictors[object_idx](voxel.unsqueeze(0), voxel_scale).squeeze()
    print (obj_info)

    obj_list =[]

    pose = torch.eye(4).to(device)
    pose[0:3, 3] = obj_info[0:3] + bbox[0:3]# translation term, the bbox here is the true bbox
    pose[2, 3] -= bbox[5]
    angle = torch.atan2(obj_info[4], obj_info[3]) # aarctan (a/b)
    pose[0, 0] = torch.cos(angle) # typical rotation matrix
    pose[0, 1] = -torch.sin(angle)
    pose[1, 0] = torch.sin(angle)
    pose[1, 1] = torch.cos(angle)


    # To do: a loop for several objects
    obj = name_to_class[object_class](

    )
    ##########################store the inferred obj_info


    # # save the obj_info in ./intermediates/{exp_index}/obj_info/obj_info.pkl
    # output_dir_param = f'./intermediates/{exp_index}/param'  
    # if not os.path.exists(output_dir_param):
    #     os.makedirs(output_dir_param)
    # with open(os.path.join(output_dir_param, 'param.pkl'), 'wb') as f:
    #     pickle.dump(obj_info, f)

