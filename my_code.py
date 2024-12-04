from models.pipelines import TSQPipeline
import yaml
import torch
import numpy as np
import os
import pickle
from data_pre import load_all_data

if __name__ == "__main__":

    with open('t2sqnet_config.yml') as f:
        t2sqnet_cfg = yaml.safe_load(f)
    
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
    
    # imgs = torch.randint(0,255,(7, 3, 240, 320)) # n * 3 * H * W

    # camera_intr = np.array([[302.85014343,   0.        , 160.        ],
    #                             [  0.        , 302.85014343, 120.        ],
    #                             [  0.        ,   0.        ,   1.        ]])

    # camera_params = {'camera_image_size': torch.tensor([240, 320]),
    #                  'projection_matrices': torch.randn(7, 3, 4),
    #                  'camera_intr': [camera_intr]*7, 
    #                  'camera_pose': [np.eye(4)]*7}

    imgs, camera_params = load_all_data()

    # print(imgs, camera_params)

    results = tsqnet.forward(
				imgs=imgs, 
                camera_params=camera_params, 
				output_all=True
			)
    
    #save the results in './intermediates/scene_id_default/results', results is 4-dim Tuples
    output_dir_results = './intermediates/scene_id_default/results'
    if not os.path.exists(output_dir_results):
        os.makedirs(output_dir_results)
    with open(os.path.join(output_dir_results, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)



    # sq_results = results[3][0] #the list of tablewares
    # print("Nuber of objects detected: ", len(sq_results))
    # for idx,object in enumerate(sq_results):
    #     print(idx, "\tObject Name:", object.name, "\n\tObject Params:", object.params)

    # print("All inferred superquadrics printed")

    # number_of_points = 1000

    # points = []
    # for object in sq_results:
    #     points.append(object.get_point_cloud(number_of_points=number_of_points)) #use the get_point_cloud function from class Tablewaree()
        
    #     #points.append(object.get_differentiable_point_cloud())
    #     print(type(object))

    # #points = np.stack(points)

    # print("Done!")
    # np.save("pc_test_tableware_3_9.npy", points)

