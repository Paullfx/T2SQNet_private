import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_3d_bbox(ax, bbox, label=None, color='blue'):
    """
    Draws a single 3D bounding box on the given axes.
    :param ax: 3D Matplotlib axes.
    :param bbox: List or tensor with format [x, y, z, width, height, depth].
    :param label: Optional label to display near the bounding box.
    :param color: Color of the bounding box.
    """
    x, y, z, w, h, d = bbox
    # Calculate the vertices of the bounding box
    vertices = np.array([
        [x - w/2, y - h/2, z - d/2],
        [x + w/2, y - h/2, z - d/2],
        [x + w/2, y + h/2, z - d/2],
        [x - w/2, y + h/2, z - d/2],
        [x - w/2, y - h/2, z + d/2],
        [x + w/2, y - h/2, z + d/2],
        [x + w/2, y + h/2, z + d/2],
        [x - w/2, y + h/2, z + d/2],
    ])
    
    # Define the 12 edges of the bounding box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ]
    
    # Draw the edges
    for edge in edges:
        ax.plot3D(*zip(*vertices[edge]), color=color)
    
    # Optionally add a label
    if label:
        ax.text(x, y, z, label, color=color)

# Define the input directory for bbox and class files
input_dir = './intermediates/scene_id_default/bboxes_cls'

# Check if the directory exists
if not os.path.exists(input_dir):
    print(f"Error: The directory {input_dir} does not exist.")
    exit(1)

# Load and print bounding boxes
bboxes_path = os.path.join(input_dir, 'bboxes.pkl')
cls_path = os.path.join(input_dir, 'cls.pkl')

if os.path.exists(bboxes_path):
    with open(bboxes_path, 'rb') as f:
        bboxes = pickle.load(f)
    print("Bounding Boxes:")
    print(bboxes)
else:
    print(f"Error: File {bboxes_path} does not exist.")
    exit(1)

if os.path.exists(cls_path):
    with open(cls_path, 'rb') as f:
        cls = pickle.load(f)
    print("Classes:")
    print(cls)
else:
    print(f"Error: File {cls_path} does not exist.")
    exit(1)

# Visualize the bounding boxes in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each bounding box
for bbox, label in zip(bboxes, cls):
    draw_3d_bbox(ax, bbox.cpu().numpy(), label=label, color='blue')

# Set labels and aspect
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Bounding Boxes')
plt.show()
