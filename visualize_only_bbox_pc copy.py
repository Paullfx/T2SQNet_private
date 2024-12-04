import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import get_cmap
import warnings

# Import custom functions
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


# Define paths
exp_index = "scene_id_default"# "tableware_3_9" #scene_id_default
input_dir = f'./intermediates/{exp_index}/bboxes_cls'
results_dir = f'./intermediates/{exp_index}/results'

bboxes_path = os.path.join(input_dir, 'bboxes.pkl')
cls_path = os.path.join(input_dir, 'cls.pkl')
results_path = os.path.join(results_dir, 'results.pkl')

# Load results
try:
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    print("Results loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {results_path}. Please check the path and try again.")
    results = None

# Load class names
class_names = []
if os.path.exists(cls_path):
    with open(cls_path, 'rb') as f:
        class_names = pickle.load(f)
else:
    print(f"Error: Class file not found at {cls_path}.")

# Load bounding boxes
bboxes = []
if os.path.exists(bboxes_path):
    with open(bboxes_path, 'rb') as f:
        bboxes = pickle.load(f)
else:
    print(f"Error: Bounding box file not found at {bboxes_path}.")

# Validate results data
if results is None or not results:
    print("No results to process. Exiting.")
    exit()

sq_results = results[3][0]  # the list of tablewares
print("Number of objects detected:", len(sq_results))
for idx, obj in enumerate(sq_results):
    print(idx, "\tObject Name:", obj.name, "\n\tObject Params:", obj.params)

# Generate point clouds
number_of_points = 1000
points = []
for obj in sq_results:
    points.append(obj.get_point_cloud(number_of_points=number_of_points))  # Use the get_point_cloud function
    print(type(obj))

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Define colors
number_of_colors = len(points)
cmap = get_cmap("jet", number_of_colors)
colors = [cmap(i) for i in range(number_of_colors)]

# Plot point cloud data
for i, obj in enumerate(points):
    x, y, z = obj[:, 0], obj[:, 1], obj[:, 2]
    ax.scatter(x, y, z, color=colors[i], marker='.', label=f"T2SQNet: {class_names[i]}", s=5)

# Plot bounding boxes
for bbox, label in zip(bboxes, class_names):
    if hasattr(bbox, 'cpu'):
        bbox = bbox.cpu().numpy()
    draw_3d_bbox(ax, bbox, label=label, color='blue')

# Add legend
lgnd = plt.legend(loc='upper right', fontsize='large', handletextpad=2)
for handle in lgnd.legend_handles:
    handle._sizes = [100]

# Show plot
plt.show()
