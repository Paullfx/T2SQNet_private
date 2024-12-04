import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import get_cmap
from data_pre import load_cam_pos

def draw_3d_bbox(ax, bbox, label=None, color='blue'):
    """
    Draws a single 3D bounding box on the given axes.
    :param ax: 3D Matplotlib axes.
    :param bbox: List or tensor with format [x, y, z, width, height, depth].
    :param label: Optional label to display near the bounding box.
    :param color: Color of the bounding box.
    """
    x, y, z, w, h, d = bbox
    vertices = np.array([
        [x - w / 2, y - h / 2, z - d / 2],
        [x + w / 2, y - h / 2, z - d / 2],
        [x + w / 2, y + h / 2, z - d / 2],
        [x - w / 2, y + h / 2, z - d / 2],
        [x - w / 2, y - h / 2, z + d / 2],
        [x + w / 2, y - h / 2, z + d / 2],
        [x + w / 2, y + h / 2, z + d / 2],
        [x - w / 2, y + h / 2, z + d / 2],
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

# Load bounding box data
exp_index = "blender_shelf_0_5"  #blender_shelf_0_5 #scene_id_default
input_dir = './intermediates/{exp_index}/bboxes_cls'
results_dir = f'./intermediates/{exp_index}/results'

bboxes_path = os.path.join(input_dir, 'bboxes.pkl')
cls_path = os.path.join(input_dir, 'cls.pkl')
results_path = os.path.join(results_dir, 'results.pkl')

try:
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    print("Results loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {results_path}. Please check the path and try again.")

# Read the class names from cls.pkl
if os.path.exists(cls_path):
    with open(cls_path, 'rb') as f:
        cls = pickle.load(f)
    obj_names = cls
#obj_names = ["Bowl", "HandlessCup"]

if os.path.exists(bboxes_path) and os.path.exists(cls_path):
    with open(bboxes_path, 'rb') as f:
        bboxes = pickle.load(f)
    with open(cls_path, 'rb') as f:
        cls = pickle.load(f)
else:
    print("Error: Bounding box or class files not found.")
    bboxes, cls = [], []

# manually Load point cloud data
# data = np.load("pc_test_tableware_2_4.npy")

sq_results = results[3][0] #the list of tablewares
print("Nuber of objects detected: ", len(sq_results))
for idx,object in enumerate(sq_results):
    print(idx, "\tObject Name:", object.name, "\n\tObject Params:", object.params)

print("All inferred superquadrics printed")

number_of_points = 1000

points = []
for object in sq_results:
    points.append(object.get_point_cloud(number_of_points=number_of_points)) #use the get_point_cloud function from class Tablewaree()
    
    #points.append(object.get_differentiable_point_cloud())
    print(type(object))

##

# Prepare figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Define colors
number_of_colors = len(points)
cmap = get_cmap("jet", number_of_colors)
colors = [cmap(i) for i in range(number_of_colors)]


# Plot T2SQNet point cloud data
for i, obj in enumerate(points):
    x, y, z = obj[:, 0], obj[:, 1], obj[:, 2]
    ax.scatter(x, y, z, color=colors[i], marker='.', label=f"T2SQNet: {obj_names[i]}", s=5)

# Plot bounding boxes
for bbox, label in zip(bboxes, cls):
    draw_3d_bbox(ax, bbox.cpu().numpy(), label=label, color='blue')

# Add legend
lgnd = plt.legend(loc='upper right', fontsize='large', handletextpad=2)
for handle in lgnd.legend_handles:
    handle._sizes = [100]

# Show plot
plt.show()
