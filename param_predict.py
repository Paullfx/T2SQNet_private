
import torch
from models.pipelines import TSQPipeline
from omegaconf import OmegaConf

config_path = "./t2sqnet_param_predict.yml"
object_idx = 3 
# 	0: "WineGlass",
# 	1: "Bowl",
# 	2: "Bottle",
# 	3: "BeerBottle",
# 	4: "HandlessCup",
# 	5: "Mug",
# 	6: "Dish"





cfg = OmegaConf.load(config_path)

pipleine = TSQPipeline(
    bbox_model_path=cfg.bbox_model_path, #irrelevant to param_predictor
    bbox_config_path=cfg.bbox_config_path, # irrelevant to param_predictor
    param_model_paths=cfg.param_model_paths, # weights of pretrained neural network
    param_config_paths=cfg.param_config_paths,# model architecture
    voxel_data_config_path=cfg.voxel_data_config_path, # For each tableware class, extract its according voxel size, max_bbox_size, marginal_bbox_size
    device=cfg.device,
    dummy_data_paths=None,
    num_augs=5,
    debug_mode=False,
)



obj_info = pipleine.param_predictors[object_idx](voxel.unsqueeze(0), voxel_scale).squeeze()