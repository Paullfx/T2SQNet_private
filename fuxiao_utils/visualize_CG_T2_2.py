import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import get_cmap
import warnings

# Import custom functions
from data_pre import load_cam_pos
from data_pre_cg import load_pc_cg


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


# Load bounding box data and results
exp_index = "scene_id_default"# "tableware_3_9" #scene_id_default
input_dir = './intermediates/{exp_index}/bboxes_cls'
bboxes_path = os.path.join(input_dir, 'bboxes.pkl')
cls_path = os.path.join(input_dir, 'cls.pkl')
results_dir = f'./intermediates/{exp_index}/results'
results_path = os.path.join(results_dir, 'results.pkl')

bboxes, class_labels = [], []
if os.path.exists(bboxes_path) and os.path.exists(cls_path):
    with open(bboxes_path, 'rb') as f:
        bboxes = pickle.load(f)
    with open(cls_path, 'rb') as f:
        class_labels = pickle.load(f)
else:
    print("Error: Bounding box or class files not found.")

try:
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    print("Results loaded successfully.")
    sq_results = results[3][0]  
    print(f"Number of objects in sq_results: {len(sq_results)}")
except FileNotFoundError:
    print(f"Error: File not found at {results_path}. Ensure the file path is correct.")
    sq_results = []  # default


# Generate point clouds for T2SQNet results
number_of_points = 1000
points = []
if sq_results:  # Ensure sq_results is not empty
    for obj in sq_results:
        if hasattr(obj, "get_point_cloud"):  # Ensure the object has the method
            points.append(obj.get_point_cloud(number_of_points=number_of_points))
            print(f"Generated point cloud for object: {type(obj)}")
        else:
            print(f"Warning: Object {obj} does not have 'get_point_cloud' method.")
else:
    print("Error: sq_results is empty. Ensure results are correctly loaded.")

# Generate CG point cloud data
try:
    pc_data_cg, obj_names_cg = load_pc_cg()
except FileNotFoundError as e:
    print(f"Error loading CG point cloud data: {e}")
    pc_data_cg, obj_names_cg = [], []

# Prepare figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Define colors
colour_gap = 3
number_of_colors = len(points) + len(pc_data_cg) + colour_gap
cmap = get_cmap("jet", number_of_colors)
colors = [cmap(i) for i in range(number_of_colors)]

# Object names (if not dynamically loaded)
# obj_names = ["Bowl", "Bottle", "HandlessCup"]
# Objects names from results[2]
obj_names = results[2]  # Keep all names as is
if len(points) != len(obj_names):
    raise ValueError(f"Mismatch: len(points)={len(points)}, len(obj_names)={len(obj_names)}")

# Plot T2SQNet point cloud data
for i, obj in enumerate(points):
    x, y, z = obj[:, 0], obj[:, 1], obj[:, 2]
    ax.scatter(x, y, z, color=colors[i], marker='.', label=f"T2SQNet: {obj_names[i]}", s=5)

# Plot CG point cloud data
for i, obj in enumerate(pc_data_cg):
    x, y, z = obj[:, 0], obj[:, 1], obj[:, 2]
    ax.scatter(x, y, z, marker='.', color=colors[i + len(points) + colour_gap], label=f"CG: {obj_names_cg[i]}", s=5)

# Plot camera positions
try:
    xyz, uvw = load_cam_pos()
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    ax.scatter(x, y, z, color="black", marker='o', label="Camera Positions", s=50)
    ax.quiver(x, y, z, u, v, w, color="black", length=0.05)
except FileNotFoundError as e:
    print(f"Error loading camera positions: {e}")

# Plot bounding boxes
for bbox, label in zip(bboxes, class_labels):
    if hasattr(bbox, 'cpu'):
        bbox = bbox.cpu().numpy()  # Ensure compatibility with Tensor objects
    draw_3d_bbox(ax, bbox, label=label, color='blue')

# Add legend
lgnd = plt.legend(loc='upper right', fontsize='large', handletextpad=2)
for handle in lgnd.legend_handles:
    handle._sizes = [100]

# Show plot
plt.show()
