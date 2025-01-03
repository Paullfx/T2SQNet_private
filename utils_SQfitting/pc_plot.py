import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from data_pre import load_cam_pos
from data_pre_cg import load_pc_cg


data = np.load("pc_test_tableware_1_1.npy")

pc_data_cg, obj_names_cg = load_pc_cg()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

colour_gap = 3
number_of_colors = len(data) + len(pc_data_cg) + colour_gap
cmap = get_cmap("jet", number_of_colors)
colors = [cmap(i) for i in range(number_of_colors)]

#obj_names=["tableware1", "tableware2", "tableware3", "tableware4"]
obj_names=["Bowl", "Bottle","HandlessCup"]

for i,obj in enumerate(data):
    x = obj[:,0]
    y = obj[:,1]
    z = obj[:,2]
    print(i)
    ax.scatter(x, y, z, color=colors[i], marker='.', label="T2SQNet: "+obj_names[i], s=5)


pc_data_cg, obj_names_cg = load_pc_cg()
for i,obj in enumerate(pc_data_cg):
    x = obj[:,0]
    y = obj[:,1]
    z = obj[:,2]
    ax.scatter(x, y, z, marker='.', color=colors[i+len(data)+colour_gap], label="CG: "+obj_names_cg[i], s=5)

xyz,uvw = load_cam_pos()
x = xyz[:,0]
y = xyz[:,1]
z = xyz[:,2]
u = uvw[:,0]
v = uvw[:,1]
w = uvw[:,2]
ax.scatter(x, y, z, color="black", marker='o',label="Camera Positions", s=50)
ax.quiver(x, y, z, u, v, w, color="black", length=0.05)

lgnd = plt.legend(loc='upper right', fontsize='large', handletextpad=2)
for handle in lgnd.legend_handles:
    handle._sizes = [100]  # Adjust the size of the legend markers
plt.show()



 # Display the Matplotlib legend on the main thread
# def display_legend():
#     fig, ax = plt.subplots(figsize=(4, len(legend_info) * 0.5))  # Adjust size based on number of items
#     ax.axis('off')  # Turn off the axis

#     # Add a legend entry for each class
#     for i, (class_name, color) in enumerate(legend_info.items()):
#         # Draw a color bar as a horizontal rectangle
#         ax.add_patch(mpatches.Rectangle((0, 0.1*i), 2, 0.1, color=color, transform=ax.transAxes, clip_on=False))
#         # Place the class_name next to the color bar, using the color for the text
#         ax.text(0.0, 0.1*i + 0.05, f"{class_name}", va='center', ha='left', transform=ax.transAxes, fontsize=10, color="black")

#     # Adjust plot limits and show the legend figure
#     ax.set_ylim(0, len(legend_info))
#     ax.set_xlim(0, 6)
#     plt.show()
