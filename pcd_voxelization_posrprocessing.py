import torch
from math import floor, ceil
import numpy as np

# w = floor(2 * bbox[3] / voxel_size + 0.5) 
# h = floor(2 * bbox[4] / voxel_size + 0.5)
# d = floor(2 * bbox[5] / voxel_size + 0.5)

w = floor(2 *  0.11456040273799706 / 0.001832966443807953 + 0.5) # w = 125
h = floor(2 * 0.11133156195208502 / 0.001832966443807953 + 0.5) # h = 121
d = floor(2 * 0.06027486266068399 / 0.001832966443807953 + 0.5) # d = 66

print("done")