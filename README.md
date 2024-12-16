<div align="center">

# <b>T<sup>2</sup>SQNet</b> <br> Transparent Tableware SuperQuadric Network

### Conference on Robot Learning (CoRL) 2024

Young Hun Kim*, 
[Seungyeon Kim*](https://seungyeon-k.github.io/)<sup>1</sup>, 
[Yonghyeon Lee](https://www.gabe-yhlee.com/)<sup>2</sup>, and 
[Frank C. Park](https://sites.google.com/robotics.snu.ac.kr/fcp/)<sup>1</sup>
<br>
<sup>1</sup>Seoul National University, <sup>2</sup>Korea Institute For Advanced Study, 
<br>
<sup>*</sup> Equal Contribution

[Project Page](https://t2sqnet.github.io/) | [Paper](https://openreview.net/pdf?id=M0JtsLuhEE) | Video 

</div>

> TL;DR: This paper proposes a novel framework for recognizing and manipulating partially observed transparent tableware objects.

## Running on LSY working staton (Fuxiao)
- In LSY working station, conda venv T2, branch fuxiao-desktop for stage 1.1 (Running the pretrained T2 pipeline), branch fuxiao-fitting for stage 1.2  (Run the superquadric-fitting module fo T2 on the segmented point cloud from ConceptGraph pipeline)
## How to analyse the intermediate results of T2SQNet in simulation (Fuxiao)
- Run the control.py in debug mode. Add comfig.  Add a breakpoint before the control part of the section (e.g. line 289 in controller.py)，more details will be added...
- Analyse and visualize the intermediate/scene_id_default, more details will be added...
## How to analyse the intermediate results of T2SQNet with real data (Fuxiao)
- The file ./data_pre_cg.py is for the data-processing of conceptgraph data. In ./data_pre_cg.py, give the source_path of .pkl.gz (e.g. '/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/tableware_2_2/exps/exp_default/pcd_exp_default.pkl.gz'). This stores the segmented pointcloud outputted by Conceptgraph pipeline
- The file ./data_pre.py is for preparing the color images and camera poses as the input for the T2SQNet pipeline. In ./data_pre.py, firstly give the dataset_root (e.g. Path("/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external") and the scene_id (experiment index like "tableware_1_13") that you want to analyse. Give the acoording camera extrinsics (note that the camera extrinsics should fit the image size accordingly). Secondly, because the pretrained T2SQNet takes seven images as input, you need to select the seven images and give their selected_indices (e.g. ["000000", "000005", "000017", "000062", "000073", "000088", "000098"])
- The file ./my_code.py calls the functions of T2SQNet model and load the prtrained model weights from t2sqnet_config.yml. Running the my_code.py script will generate a folder with name "scene_id_default" in the folder ./intermediates. Various intermediate results can be found in this folder. Remember to rename it with the according scene_id such that it won't be covered by next experiment
- ./visualize_CG_T2_fuxiao.py is for plotting CG pcd, T2 pcd, and camera pose. ./visualize_only_bbox_pc.py is for plotting the bbox and the pcd of the fitted superquadrics../visualize_voxel_from_objList_copy.py for plotting the visual hull in form of voxels.
## Compare the visual hull voxel to the fitted superquadric point cloud (Fuxiao)

## Superquadric fitting with segmented point cloud from ConceptGraph pipeline(Fuxiao)
- Pointcloud sampling
  - Conda activate ros_cg. One terminal for running ros publisher "ros2 run sai_orbbec sai_publisher". Modify the "scene_id" in ros_tableware.yaml to e.g. "tableware_4_9". Then open another terminal, cd ./conceptgraph/slam, run "python3 ros_rerun_sai_T2.py".
  - Data processing: firstly run fuxiao_open3d_tableware.py in /home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/scripts/fuxiao_PC/drafts/fuxiao_open3d_tableware_test.py. This will generate a .ply file in 'concept-graphs/conceptgraph/dataset/external/{exp_id}/exps/exp/default' for the specific tableware e.g. "cup". Remember to modify this specific object to read the tableware from the point cloud accordingly. Secondly run tableware_process.py to denoise, this will  generate a "{exp_id}_bowl_denoised.ply" file. After experimenting, the parameters for denoise are setted "voxel_down_pcd.remove_radius_outlier(nb_points=200, radius=0.2)". 
## Preview
<I><b>Sequential Decluttering (Left):</b> T<sup>2</sup>SQNet-based method succeeds in sequentially grasping the objects without re-recognition, while avoiding collisions with other objects and the environment. </I>

<I><b>Target Retrieval (Right)</b>: T<sup>2</sup>SQNet-based method also successfully rearranges the surrounding objects and finally retrieves an initially non-graspable target object (e.g., wineglass).</I>

<div class="imgCollage">
    <span style="width: 50%">
        <img src="./assets/sd.gif"
            alt="sequential decluttering"
            width="375"/>
    </span>
    <span style="width: 50%">
        <img src="./assets/tr.gif"
            alt="target retrieval"
            width="375"/>
    </span>
</div>

## Requirements
### Environment
The project has been under a standard Anaconda environment with CUDA 11.8. To install all of our dependencies, simply run
```shell
conda create -n t2sqnet python=3.10.14
conda activate t2sqnet
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git@05c386ee95b26a8ec8398bebddf70ffb8ddd3faf
```

### Blender Setting
If you want to generate a dataset (i.e., render RGB images of transparent objects), set up Blender by following these steps:
1. Download the ``blender-4.0.2-linux-x64.tar.xz`` file from [here](https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz).
2. Unzip the downloaded file:
```shell
tar -xvf blender-4.0.2-linux-x64.tar.xz
```
3. Rename ``blender-4.0.2-linux-x64/4.0/python`` to ``blender-4.0.2-linux-x64/4.0/_python``:
```shell
mv /path/to/blender-4.0.2-linux-x64/4.0/python /path/to/blender-4.0.2-linux-x64/4.0/_python
```
4. Create a symbolic link to the Python folder in the Anaconda environment set up earlier:
```shell
ln -s /path/to/anaconda3/envs/t2sqnet /path/to/blender-4.0.2-linux-x64/4.0/python
```
> **Warining:** Blender 4.0 is only compatible with Python 3.10.

## Dataset
### Interactive Tableware Object Visualization
We provide an interactive visualization tool for tableware objects composed of deformable superquadrics. To launch the visualization app, simply run:
```shell
python visualize_dataset.py
```
<div align="center">
<span style="width: 100%"><img src="assets/vis.gif" width="500"></span>
</div>
In this demo, you can adjust the tableware parameters using sliders, and the corresponding tableware objects will be visualized according to these parameters.

### Generate TablewareNet Dataset
The data generation process consists of three major parts.
1. <b>Generate Scene using PyBullet</b> (``generate_tableware_pybullet.py``). Spawn tableware object meshes in a user-defined environment (e.g., a table or shelf) within PyBullet, a physics simulator, to generate cluttered scenes.

2. <b>Render RGB images using Blender</b> (``generate_tableware_blender.py``). Render RGB images of the scenes from arbitrary camera poses using Blender, a photorealistic renderer, with transparent textures. This step also generates depth images and TSDF.

3. <b>Voxelize Objects for Training ResNet3D</b> (``generate_tableware_voxelize.py``). Obtain smoothed visual hulls of the tableware objects to train a ResNet3D-based shape prediction network.

To check the usage of the code and generate your own dataset, please refer to the following shell script.

```shell
. generate_tablewarenet.sh
# The original command "./generate_tablewarenet.sh" given by the author seems uncorrect (commented by Fuxiao)
```

Based on the shell script, each step creates a folder, and each dictionary-style ``.pkl`` data file within each folder is structured as follows.

<table>
<tr>
<th>test</th>
<th>test_processed</th>
<th>test_voxelized</th>
</tr>
<tr>
<td valign=top>

```python
{
  "mask_imgs",
  "camera",
  "objects_pose",
  "objects_class",
  "objects_param",
  "objects_diff_pc",
  "objects_bbox",
  "workspace_origin"
}
```
</td>
<td valign=top>

```python
{
  "mask_imgs",
  "camera",
  "objects_pose",
  "objects_class",
  "objects_param",
  "objects_diff_pc",
  "objects_bbox",
  "workspace_origin",
  "depth_imgs",
  "rgb_imgs",
  "tsdf",
}
```
</td>
<td valign=top>

```python
{
  "vox",
  "bound1",
  "bound2",
  "object_pose",
  "object_class",
  "object_param",
}
```
</td>
</tr>
</table>

> **Warning:** Some data generation steps do not work in server (i.e., without a connected display). If you want generate a dataset in server, try [Open3D headless rendering](http://www.open3d.org/docs/latest/tutorial/Advanced/headless_rendering.html).

### TablewareNet Dataset
Coming soon

### Generate Dataset Using TRansPose Objects
If you want to generate data with objects from the TRansPose dataset, first request the TRansPose object files from [this link](https://sites.google.com/view/transpose-dataset), place the ``.obj`` files in the ``assets/TRansPose`` folder, and then run the following shell script.
```shell
./generate_transpose.sh
```

## Model
### Training T<sup>2</sup>SQNet 
The training script is ``train.py``. 
- ``--config`` specifies a path to a configuration ``yml`` file.
- ``--logdir`` specifies a directory where the results will be saved. The default directory is ``results``.
- ``--run`` specifies a name for an experiment. The default is the current time.
- ``--device`` specifies an GPU number to use. The default is ``any``.

Training code for DETR3D-based bounding box predictor is as follows:
```shell
python train.py --config configs/detr3d.yml --entity {Y}
```
- ``Y`` represents the WandB username, as our training code currently utilizes WandB.

Training code for ResNet3D-based shape prediction network is as follows:
```shell
python train.py --config configs/voxel_head/voxel_{X}.yml --entity {Y}
```
- ``X`` is either ``BeerBottle``, ``Bottle``, ``Bowl``, ``Dish``, ``HandlessCup``, ``Mug``, or ``WineGlass``. 
- ``Y`` represents the WandB username, as our training code currently utilizes WandB.

### Pre-trained Models
Pre-trained models should be stored in the ``pretrained/`` directory. The models are provided via this [Google drive link](https://drive.google.com/file/d/17W1HUZmyv1q4k5rXpDyYhgAOH9NbY7qe/view?usp=sharing). After setup, the ``pretrained/`` directory should be organized as follows:
```
pretrained
├── bbox
│   ├── detr3d.yml
│   └── model_best.pkl
├── dummy
└── voxel
    ├── BeerBottle
    │   ├── voxel_BeerBottle.yml
    │   └── model_best_chamfer_metric.pkl
    ├── Bottle
    │   ├── voxel_Bottle.yml
    │   └── model_best_chamfer_metric.pkl
    ...
    └── WineGlass
        ├── voxel_WineGlass.yml
        └── model_best_chamfer_metric.pkl
```

> **Warning:** We found that when training with data and model parameters set to ``torch.float32`` in the ResNet-based feature pyramid network used in DETR3D, the network produced varying (and sometimes broken) outputs for the same input during inference if the batch size differed from that used during training. To address this, we added dummy data during inference to match the batch size used in training. We also confirmed that this phenomenon does not occur when using ``torch.float64``.

## Manipulation
The manipulation script with T<sup>2</sup>SQNet is provided as follows:
```shell
python control.py --config configs/control/{X}.yml
```
- ``X`` can be either ``clear_clutter`` or ``target_retrieval``. 
- In the current config files, the variable ``exp_type`` is set to ``sim``, which runs the app to manipulate objects with T<sup>2</sup>SQNet in PyBullet (using PyBullet RGB images).
- Launch the manipulation app while adjusting various variables (e.g., ``sim_type``, ``object_types``, and ``num_objects``)
- For real-world manipulation (i.e., ``exp_type: real``), the app operates via socket communication. A simple Python guideline for communicating with the server from the robot computer is as follows:
```python
from functions.communicator_client import Talker

# Connect to server
client = Talker({ip}, {port})
client.conn_server()

# Send vision data
client.send_vision(rgbs)                  # send rgb images

# Receive data from server
data = client.recv_grasp(dict_type=True)  # receive robot action
```
- Set the server’s IP and port number in the script above and in the configuration files.

## Acknowledgement
### Dataset generation
- The code for rendering RGB images in Blender is adapted from [GraspNeRF](https://github.com/PKU-EPIC/GraspNeRF).
- TRansPose objects can be obtained from the official [TRansPose](https://sites.google.com/view/transpose-dataset) website.
### T<sup>2</sup>SQNet Model
- The bounding box predictor is adapted from [DETR3D](https://github.com/WangYueFt/detr3d).
- [Language Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything) is used as the segmentation model.
### Manipulation
- [COACD](https://github.com/SarahWeiii/CoACD) is used as a convex decomposition method for physics simulation of concave objects.

## Citation
If you found this repository useful in your research, please consider citing:
```
@inproceedings{kim2024t2sqnet,
      title={T$^2$SQNet: A Recognition Model for Manipulating Partially Observed Transparent Tableware Objects},
      author={Kim, Young Hun and Kim, Seungyeon and Lee, Yonghyeon and Park, Frank C},
      booktitle={8th Annual Conference on Robot Learning}
}
```
