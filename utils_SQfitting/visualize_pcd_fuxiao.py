import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Load the results.pkl file
results_file_path = './intermediates/scene_id_default/results/results.pkl'
if not os.path.exists(results_file_path):
    raise FileNotFoundError(f"File not found: {results_file_path}")

with open(results_file_path, 'rb') as f:
    results = pickle.load(f)

sq_results = results[3][0] #the list of tablewares
print("Nuber of objects detected: ", len(sq_results))
for idx,object in enumerate(sq_results):
    print(idx, "\tObject Name:", object.name, "\n\tObject Params:", object.params)



# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

colour_gap = 5
number_of_colors = len(sq_results) + colour_gap
cmap = get_cmap("jet", number_of_colors)
colors = [cmap(i) for i in range(number_of_colors)]

# Extract and plot point clouds for each detected object
for idx, obj in enumerate(sq_results):
    #print(f"Object {idx} - Params: {obj['params']}")
    point_cloud = obj.get_point_cloud(number_of_points=1000)  # Assuming this method exists
    if isinstance(point_cloud, np.ndarray) and point_cloud.shape[1] == 3:
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        ax.scatter(x, y, z, color=colors[idx], marker='.', label=f"Object {idx}", s=5)
    else:
        print(f"Invalid point cloud format for object {idx}")

# Optional: Add camera position (dummy example, replace with actual logic if needed)
# xyz = np.array([[0, 0, 0], [1, 1, 1]])  # Replace with actual camera position data
# uvw = np.array([[0, 0, 1], [0, 1, 0]])  # Replace with actual direction vectors
# ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="black", marker='o', label="Camera Positions", s=50)
# ax.quiver(xyz[:, 0], xyz[:, 1], xyz[:, 2], uvw[:, 0], uvw[:, 1], uvw[:, 2], color="black", length=0.05)

# Add legend and display
plt.legend(loc='upper right', fontsize='large', handletextpad=2)
plt.show()
