import open3d as o3d
import os
import torch
from models.pipelines import TSQPipeline
import yaml



# load ply file of cup from /home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/tableware_4_6/exps/exp_default/tableware_4_6_bowl_denoised.ply
tableware_ply_path = "/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/tableware_4_6/exps/exp_default/tableware_4_6_bowl_denoised.ply"
if not os.path.exists(tableware_ply_path):
    raise FileNotFoundError(f"File not found: {tableware_ply_path}")


def main():

#obj_info = self.param_predictors[object_idx](voxel.unsqueeze(0), voxel_scale).squeeze().

#testtesttest

if __name__ == "__main__":
    main()

