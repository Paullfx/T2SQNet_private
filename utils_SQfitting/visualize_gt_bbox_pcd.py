import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import get_cmap
import warnings
from omegaconf import OmegaConf


# Import custom functions
sys.path.append(os.path.abspath("/home/fuxiao/Projects/T2_private/T2SQNet_private"))
from data_pre import load_cam_pos


def draw_3d_bbox(ax, bbox, label=None, color='blue'):
    """
    Draws a single 3D bounding box on the given axes.
    :param ax: 3D Matplotlib axes.
    :param bbox: List or tensor with format [x, z, y, width, height, depth].
    :param label: Optional label to display near the bounding box.
    :param color: Color of the bounding box.
    """
    x, y, z, w, d, h = bbox # w, d, h are  half of the width depth and height
    vertices = np.array([
        [x - w, y - d, z - h],
        [x + w, y - d, z - h],
        [x + w, y + d, z - h],
        [x - w, y + d, z - h],
        [x - w, y - d, z + h],
        [x + w, y - d, z + h],
        [x + w, y + d, z + h],
        [x - w, y + d, z + h],
    ])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    for edge in edges:
        ax.plot3D(*zip(*vertices[edge]), color=color)
    if label:
        ax.text(x, y, z, label, color=color)

# Define paths
# exp_index = "scene_id_default"# "tableware_3_9" #blender_table_0_3 #"pybullet_single_BeerBottle" # "pybullet_table_1_2"
exp_index = "pybullet_single_Bowl"
input_dir = f'./intermediates/{exp_index}/bboxes_cls'
# results_dir = f'./intermediates/{exp_index}/results'
gt_dir = f'./intermediates/{exp_index}/ground_truth'
inferred_object_list_dir = f'./intermediates/{exp_index}/inferred_obj_list'
# 
bbox_info_dir = f'./intermediates/{exp_index}/bbox_info'
voxel_hull_dir = f'./intermediates/{exp_index}/object_list'

bboxes_path = os.path.join(input_dir, 'bboxes.pkl')
cls_path = os.path.join(input_dir, 'cls.pkl')
# results_path = os.path.join(results_dir, 'results.pkl')
gt_path = os.path.join(gt_dir, 'gt.pkl')
inferred_object_list_path = os.path.join(inferred_object_list_dir, 'inferred.pkl')
# 
bbox_info_path = os.path.join(bbox_info_dir, 'bbox_info.pkl')
voxel_hull_path = os.path.join(voxel_hull_dir, 'object_list.pkl')

voxel_data_config_path = "./configs/voxelize_config.yml"

def load_voxel_infos(voxel_data_config_path):
    voxel_data_config = OmegaConf.load(voxel_data_config_path)
    return {
        "voxel_size": voxel_data_config['voxel_size'],
        "max_bbox_size": voxel_data_config['max_bbox_size'],
        "marginal_bbox_size": voxel_data_config['marginal_bbox_size'],
    }

def get_bbox_size(size_dict, lable):
    return np.array(size_dict[label])

voxel_infos = load_voxel_infos(voxel_data_config_path)
max_bbox_size_all = voxel_infos["max_bbox_size"]
marginal_bbox_size_all = voxel_infos["marginal_bbox_size"]

# Load bbox_info
try:
    with open(bbox_info_path, 'rb') as f:
        bbox_info = pickle.load(f)
    print("Bounding box info loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {bbox_info_path}. Please check the path and try again.")
    exit()
bboxes = bbox_info['bboxes']
class_names = bbox_info['objects_class']

# Load inferred object list
try:
    with open(inferred_object_list_path, 'rb') as f:
        inferred = pickle.load(f)
    print("Inferred object list loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {inferred_object_list_path}. Please check the path and try again.")
    inferred = None

# Load voxel hull
with open(voxel_hull_path, 'rb') as f:
    obj_list = pickle.load(f)

# print all the tabelware classes
for i in range(len(obj_list[0])):
    print (type(obj_list[0][i]))


# # Load results
# try:
#     with open(results_path, 'rb') as f:
#         results = pickle.load(f)
#     print("Results loaded successfully.")
# except FileNotFoundError:
#     print(f"Error: File not found at {results_path}. Please check the path and try again.")
#     results = None

# # Load class names
# class_names = []
# if os.path.exists(cls_path):
#     with open(cls_path, 'rb') as f:
#         class_names = pickle.load(f)
# else:
#     print(f"Error: Class file not found at {cls_path}.")

# # Load bounding boxes
# bboxes = []
# if os.path.exists(bboxes_path):
#     with open(bboxes_path, 'rb') as f:
#         bboxes = pickle.load(f)
# else:
#     print(f"Error: Bounding box file not found at {bboxes_path}.")

# # Validate results data
# if results is None or not results:
#     print("No results to process. Exiting.")
#     exit()

positions = []
# sq_results = results[3][0]  # the list of reconstructed tablewares object using superquadric fitting parameters
sq_results = inferred
print("Number of objects detected:", len(sq_results))
for idx, obj in enumerate(sq_results):
    print(idx, "\tObject Name:", obj.name, "\n\tObject Params:", obj.params)
    positions.append(obj.SE3[:3, 3].cpu().numpy())

# Generate point clouds
number_of_points = 1000
points = []
for obj in sq_results:
    points.append(obj.get_point_cloud(number_of_points=number_of_points))  # Use the get_point_cloud function
    print("Point cloud of reconstructed tablewares loaded successfully.")
####################################################################################
############################## About the ground truth ##############################
####################################################################################

# Load gt
try:
    with open(gt_path, 'rb') as f:
        gt = pickle.load(f)
    print("Ground truth loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {gt_path}. Please check the path and try again.")
    results = None

# load individual object
position_gt = []
print("Number of ground truth objects:", len(gt))
for idx, obj in enumerate(gt):
    print(idx, "\tObject Name:", obj.name, "\n\tObject Params:", obj.params)
    position_gt.append(obj.SE3[:3,3].cpu().numpy())

# get_point_cloud from gt tableware
number_of_points = 1000
points_gt = []
for obj_gt in gt:
    points_gt.append(obj_gt.get_point_cloud(number_of_points=number_of_points))  # Use the get_point_cloud function
    print("Point cloud of ground truth loaded successfully.")
########################################### gt module done ##############################



####################################################################################
############################## Plotting reconstructed ##############################
####################################################################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# ax.set_box_aspect(1, 1, 1)

# Define colors
number_of_colors = len(points)
cmap = get_cmap("jet", number_of_colors)
colors = [cmap(i) for i in range(number_of_colors)]

# Plot reconstructed point cloud data
for i, obj in enumerate(points):
    x, y, z = obj[:, 0], obj[:, 1], obj[:, 2]
    ax.scatter(x, y, z, color= 'r', marker='.', label=f"reconstructed T2SQNet: {class_names[i]}", s=5)
    #ax.scatter(x, y, z, color=colors[i], marker='.', label=f"reconstructed T2SQNet: {class_names[i]}", s=5)

# Plot reconstructed object position   
for i, pos in enumerate(positions):
    ax.scatter(pos[0], pos[1], pos[2], color='red', marker='o', s=100, label=f"Reconstructed position: {class_names[i]}" if i == 0 else None)

# Plot gt point cloud data
for i, obj in enumerate(points_gt):
    x, y, z = obj[:, 0], obj[:, 1], obj[:, 2]
    ax.scatter(x, y, z, color= 'b', marker='x', label=f"Ground truth T2SQNet: {class_names[i]}", s=5)
    #ax.scatter(x, y, z, color=colors[i], marker='x', label=f"Ground truth T2SQNet: {class_names[i]}", s=5)

# # Plot ground truth object position
for i, pos_gt in enumerate(position_gt):
    ax.scatter(pos_gt[0], pos_gt[1], pos_gt[2], color='blue', marker='^', s=100, label=f"GT position: {class_names[i]}" if i == 0 else None)



####################################################################################
############################## Marg bbox visualization ##############################
####################################################################################

# Plot bounding boxes using precisely the same coordinate transform in the original T2SQNet code file 
# # coordinate transformation from the DETR3D network output (bbox) into the voxel hull (marginal bbox and max bbox) is directly taken from the pipeline.py file in the T2SQNet code
for bbox, label in zip(bboxes, class_names):
    if hasattr(bbox, 'cpu'):
        bbox = bbox.cpu().numpy()
    # print("bbox:", bbox)
    # print("bottom bbox", bbox[2] - bbox[5])
    # draw_3d_bbox(ax, bbox, label=label, color='blue') 
    draw_3d_bbox(ax, bbox, color='blue') # bbox is the true bbox (output from DETR3D), marginal bbox, maximal bbox

    max_bbox_size = get_bbox_size(max_bbox_size_all, label)
    marginal_bbox_size = get_bbox_size(marginal_bbox_size_all, label)
    #max_bbox_size = np.array([0.06000329943137184, 0.06001342450793149, 0.17886416966013752]) # copy paste the max_bbox_size of the according classes from voxelize_config.yml
    # here as the most simple example, there are 4 beer bottles
    max_bbox = np.concatenate(
        (
            bbox[0:2], 
            np.array([bbox[2] - bbox[5] + max_bbox_size[2]]),  
            max_bbox_size 
        ),
        axis=0
    )
    #print("max_bbox:", max_bbox)
    # draw_3d_bbox(ax, max_bbox, label="Max BBox", color='red')
    draw_3d_bbox(ax, max_bbox, color='red')

    #marginal_bbox_size = np.array([0.0750041242892148, 0.07501678063491438, 0.17886416966013752])
    marginal_bbox = np.concatenate(
        (
            bbox[0:2], 
            np.array([bbox[2] - bbox[5] + marginal_bbox_size[2]]),
            marginal_bbox_size
        ), 
        axis=0
    )
    # print("marginal_bbox:", marginal_bbox)
    # print("bottom marginal", marginal_bbox[2] - marginal_bbox[5])
    # draw_3d_bbox(ax, marginal_bbox, label="Marginal BBox", color='green')
    draw_3d_bbox(ax, marginal_bbox, color='green')

# for i, bbox in enumerate(bboxes):
#     if hasattr(bbox, 'cpu'):
#         bbox = bbox.cpu().numpy()
#     draw_3d_bbox(ax, bbox, label=f"BBox: {class_names[i]}", color='green')





# Adjust aspect ratio
def set_equal_aspect(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = np.mean(limits, axis=1)
    ranges = np.ptp(limits, axis=1)
    max_range = ranges.max() / 2
    ax.set_xlim3d([centers[0] - max_range, centers[0] + max_range])
    ax.set_ylim3d([centers[1] - max_range, centers[1] + max_range])
    ax.set_zlim3d([centers[2] - max_range, centers[2] + max_range])

set_equal_aspect(ax)


# Add legend
lgnd = plt.legend(loc='upper right', fontsize='large', handletextpad=2)

for handle in lgnd.legend_handles:
    handle._sizes = [100]

plt.gca().add_artist(lgnd)
color_legend = [
    plt.Line2D([0], [0], color='black', lw=2, label='Original BBoxes'),
    plt.Line2D([0], [0], color='red', lw=2, label='Max BBoxes'),
    plt.Line2D([0], [0], color='green', lw=2, label='Marginal BBoxes')
]
ax.legend(handles=color_legend, loc='upper left', fontsize='medium')

# Show plot
plt.show()
