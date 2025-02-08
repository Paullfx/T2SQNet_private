import open3d as o3d
import numpy as np


def bbox2o3d (bbox):

    bbox = np.array(bbox)
    bbox_min = bbox[:3] - bbox[3:]
    bbox_max = bbox[:3] + bbox[3:]
    obj_bbox = o3d.geometry.AxisAlignedBoundingBox()
    obj_bbox.min_bound = bbox_min
    obj_bbox.max_bound = bbox_max

    return obj_bbox


# marginal_bbox = [0,0,0,1,1,1]

# marginal_bbox_o3d = bbox2o3d(marginal_bbox)
# marginal_bbox_o3d.color = (1, 0, 0)

# axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

# o3d.visualization.draw_geometries([marginal_bbox_o3d, axis])