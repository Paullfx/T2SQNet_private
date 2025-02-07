import numpy as np
import math
import open3d as o3d


marginal_bbox_size = [0.3375,0.3375,0.175]#[0.30,0.225,0.225]
max_bbox_size=[0.27,0.27,0.175]
voxel_size=0.005

object_bbox = [0.26625317083517397, -0.3973193468869188, 0.24292643232428096, 0.24187954210851526, 0.2562771339006888, 0.12093977105425763]

max_bbox = np.concatenate(
			(
				object_bbox[0:2], 
				np.array([object_bbox[2] - object_bbox[5] + max_bbox_size[2]]),
				max_bbox_size), 
			axis=0
		)
marginal_bbox = np.concatenate(
			(
				object_bbox[0:2], 
				np.array([object_bbox[2] - object_bbox[5] + marginal_bbox_size[2]]),
				marginal_bbox_size), 
			axis=0
		)


# voxel_grid_original = o3d.geometry.VoxelGrid.create_dense( # size marginal bbox with 972k voxels
#     origin=marginal_bbox[0:3] - marginal_bbox[3:6],
#     color=[0.7,0.7,0.7],
#     voxel_size=voxel_size,
#     width=marginal_bbox[3] * 2,
#     height=marginal_bbox[4] * 2,
#     depth=marginal_bbox[5] * 2,
# )
w = round(marginal_bbox[3] * 2 / voxel_size)
h = round(marginal_bbox[4] * 2 / voxel_size)
d = round(marginal_bbox[5] * 2 / voxel_size)

print("marginal bbox w*h*d:", w,h,d)
print("max bbox w*h*d:", round(max_bbox[3] * 2 / voxel_size), round(max_bbox[4] * 2 / voxel_size), round(max_bbox[5] * 2 / voxel_size))


# max bounding box
w_min1 = math.floor(w * (marginal_bbox[3] - max_bbox[3]) / (2 * marginal_bbox[3]))
w_max1 = math.ceil(w * (marginal_bbox[3] + max_bbox[3]) / (2 * marginal_bbox[3]))
h_min1 = math.floor(h * (marginal_bbox[4] - max_bbox[4]) / (2 * marginal_bbox[4]))
h_max1 = math.ceil(h * (marginal_bbox[4] + max_bbox[4]) / (2 * marginal_bbox[4]))
d_min1 = 0
d_max1 = math.ceil(d * (2 * max_bbox[5]) / (2 * marginal_bbox[5])) - 1
bound1 = [w_min1, w_max1, h_min1, h_max1, d_min1, d_max1]

# get bounding box inside voxels
w_min2 = math.floor(w * (marginal_bbox[3] - object_bbox[3]) / (2 * marginal_bbox[3]))
w_max2 = math.ceil(w * (marginal_bbox[3] + object_bbox[3]) / (2 * marginal_bbox[3]))
h_min2 = math.floor(h * (marginal_bbox[4] - object_bbox[4]) / (2 * marginal_bbox[4]))
h_max2 = math.ceil(h * (marginal_bbox[4] + object_bbox[4]) / (2 * marginal_bbox[4]))
d_min2 = 0
d_max2 = math.ceil(d * (2 * object_bbox[5]) / (2 * marginal_bbox[5])) - 1
bound2 = [w_min2, w_max2, h_min2, h_max2, d_min2, d_max2]

print("bound1:", bound1)
print("bound2:", bound2)