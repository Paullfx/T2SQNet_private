import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import get_cmap
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


# Load bounding box data
input_dir = './intermediates/scene_id_default/bboxes_cls'
bboxes_path = os.path.join(input_dir, 'bboxes.pkl')
cls_path = os.path.join(input_dir, 'cls.pkl')
#results_dir = f'./intermediates/{exp_index}/results'

if os.path.exists(bboxes_path) and os.path.exists(cls_path):
    with open(bboxes_path, 'rb') as f:
        bboxes = pickle.load(f)
    with open(cls_path, 'rb') as f:
        cls = pickle.load(f)
else:
    print("Error: Bounding box or class files not found.")
    bboxes, cls = [], []

# Load point cloud data
data = np.load("pc_test_tableware_1_5.npy")
pc_data_cg, obj_names_cg = load_pc_cg()

# Prepare figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Define colors
colour_gap = 3
number_of_colors = len(data) + len(pc_data_cg) + colour_gap
#cmap = matplotlib.colormaps.get_cmap("jet", number_of_colors)
cmap = get_cmap("jet", number_of_colors)
colors = [cmap(i) for i in range(number_of_colors)]

obj_names = ["WineGlass","Bottle","HandlessCup"]

for i, obj in enumerate(data):
    x, y, z = obj[:, 0], obj[:, 1], obj[:, 2]
    ax.scatter(x, y, z, color=colors[i], marker='.', label=f"T2SQNet: {obj_names[i]}", s=5)

for i, obj in enumerate(pc_data_cg):
    x, y, z = obj[:, 0], obj[:, 1], obj[:, 2]
    ax.scatter(x, y, z, marker='.', color=colors[i + len(data) + colour_gap], label=f"CG: {obj_names_cg[i]}", s=5)

# Plot camera positions
xyz, uvw = load_cam_pos()
x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
ax.scatter(x, y, z, color="black", marker='o', label="Camera Positions", s=50)
ax.quiver(x, y, z, u, v, w, color="black", length=0.05)

# Plot bounding boxes
for bbox, label in zip(bboxes, cls):
    draw_3d_bbox(ax, bbox.cpu().numpy(), label=label, color='blue')

# Add legend
lgnd = plt.legend(loc='upper right', fontsize='large', handletextpad=2)
for handle in lgnd.legend_handles:
    handle._sizes = [100]

# Show plot
plt.show()

# Plot point cloud data


# object_list_path = './intermediates/scene_id_default/object_list/object_list.pkl'
# if os.path.exists(object_list_path):
#     with open(object_list_path, 'rb') as f:
#         object_list = pickle.load(f)
#     obj_names = object_list[2]  # Update obj_names with object classes from object_list[2]
# else:
#     warnings.warn(
#         "Warning: Object list file not found. Ensure the file exists at the specified path."
#     )
#     obj_names = None  # Set to None or handle appropriately if this leads to further processing
# # Verify the updated obj_names
# if obj_names is not None:
#     print("Updated Object Names:", obj_names)
# else:
#     print("Object names could not be loaded. Check the warning for details.")