import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

def pcd2grid (point_cloud, voxel_size)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud,
                                                            voxel_size=voxel_size)
    return voxel_grid

def grid2meshgrid (voxel_grid):