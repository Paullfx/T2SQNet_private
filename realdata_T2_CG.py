import os
import gzip
import shutil
import pickle
import numpy as np
import torch
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from models.pipelines import TSQPipeline
import yaml

# ==== Data Preparation Functions ====

def load_pc_cg(source_path):
    file_path = os.path.splitext(source_path)[0]
    # Decompress the .gz file
    with gzip.open(source_path, 'rb') as f_in:
        with open(file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Load data from the .pkl file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    pc_data = []
    obj_names = []
    for obj in data['objects']:
        pc_data.append(obj["pcd_np"])
        obj_names.append(obj["class_name"])

    return pc_data, obj_names

def load_images(color_path, selected_indices, resize_shape=(320, 240), show_imgs=False):
    images = []
    for idx in selected_indices:
        img_path = color_path / f"{idx}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image {img_path} not found!")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    formatted_images = [
        cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR) for img in images
    ]
    if show_imgs:
        fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
        for i, img in enumerate(formatted_images):
            axs[i].imshow(img)
            axs[i].axis("off")
            axs[i].set_title(f"Img {selected_indices[i]}")
        plt.tight_layout()
        plt.show()

    tensor_images = torch.stack(
        [torch.tensor(img).permute(2, 0, 1) for img in formatted_images], dim=0
    )
    tensor_images = tensor_images.type(torch.uint8)
    return tensor_images

def load_poses(pose_path, selected_indices):
    poses = []
    for idx in selected_indices:
        pose_path_idx = pose_path / f"{idx}.npy"
        pose = np.load(pose_path_idx)
        if pose is None:
            raise FileNotFoundError(f"Pose {pose_path_idx} not found!")
        poses.append(pose)
    return poses

def load_projections(camera_poses, camera_intr):
    """
    Calculates projection matrices for given camera poses and intrinsics.
    """
    camera_projections = []
    for pose in camera_poses:
        projection = torch.from_numpy(camera_intr @ np.linalg.inv(pose)[:3])
        camera_projections.append(projection)
    return torch.stack(camera_projections)

def load_cam_pos(pose_path, selected_indices):
    camera_poses = load_poses(pose_path, selected_indices)
    xzys = []
    uvws = []
    for camera_pose in camera_poses:
        xzy = camera_pose[:3, 3]
        xzys.append(xzy)
        uvw = camera_pose[:3, 2]
        uvws.append(uvw)
    return np.stack(xzys), np.stack(uvws)

# ==== 3D Visualization Functions ====

def draw_3d_bbox(ax, bbox, label=None, color='blue'):
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


# ==== bbox ====

input_dir = './intermediates/scene_id_default/bboxes_cls'
bboxes_path = os.path.join(input_dir, 'bboxes.pkl')
cls_path = os.path.join(input_dir, 'cls.pkl')

# Check if files exist
if os.path.exists(bboxes_path) and os.path.exists(cls_path):
    with open(bboxes_path, 'rb') as f:
        bboxes = pickle.load(f)
    with open(cls_path, 'rb') as f:
        cls = pickle.load(f)
else:
    print("Error: Bounding box or class files not found.")
    bboxes, cls = [], []

# Plot bounding boxes
for bbox, label in zip(bboxes, cls):
    draw_3d_bbox(ax, bbox.cpu().numpy(), label=label, color='blue')







# ==== Main Pipeline ====

if __name__ == "__main__":
    # Load configuration for TSQPipeline
    with open('t2sqnet_config.yml') as f:
        t2sqnet_cfg = yaml.safe_load(f)
    
    tsqnet = TSQPipeline(
        bbox_model_path=t2sqnet_cfg["bbox_model_path"],
        bbox_config_path=t2sqnet_cfg["bbox_config_path"],
        param_model_paths=t2sqnet_cfg["param_model_paths"],
        param_config_paths=t2sqnet_cfg["param_config_paths"],
        voxel_data_config_path=t2sqnet_cfg["voxel_data_config_path"],
        device=t2sqnet_cfg["device"],
        dummy_data_paths=t2sqnet_cfg["dummy_data_paths"],
        num_augs=t2sqnet_cfg["num_augs"],
        debug_mode=t2sqnet_cfg["debug_mode"]
    )

    # Load data
    dataset_root = Path("/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external")
    scene_id = "tableware_1_5"
    color_path = dataset_root / scene_id / "color"
    pose_path = dataset_root / scene_id / "pose"
    source_path = dataset_root / scene_id / "exps/exp_default/pcd_exp_default.pkl.gz"

    selected_indices = ["000000", "000010", "000023", "000035", "000043", "000063", "000068"]
    imgs = load_images(color_path, selected_indices, resize_shape=(320, 240))
    camera_poses = load_poses(pose_path, selected_indices)
    camera_intr = np.array([[187.14775, 0, 159.15425],
                            [0, 249.37767, 119.91234],
                            [0, 0, 1]])
    camera_projections = load_projections(camera_poses, camera_intr)
    xyz, uvw = load_cam_pos(pose_path, selected_indices)
    pc_data_cg, obj_names_cg = load_pc_cg(source_path)

    # Prepare camera parameters
    camera_params = {
    'camera_image_size': torch.tensor([240, 320]),
    'projection_matrices': camera_projections,  # Use computed projection matrices
    'camera_intr': [camera_intr] * len(selected_indices),  # Repeat intrinsics for each image
    'camera_pose': camera_poses  # Use loaded camera poses
    }

    # Run the TSQPipeline forward method
    results = tsqnet.forward(
    imgs=imgs,
    camera_params=camera_params,
    output_all=True
    )
    sq_results = results[3][0]
    
    # 3D Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Colors
    colour_gap = 3
    number_of_colors = len(sq_results) + len(pc_data_cg) + colour_gap
    cmap = get_cmap("jet", number_of_colors)
    colors = [cmap(i) for i in range(number_of_colors)]

    # Plot superquadrics and CG data
    for i, obj in enumerate(sq_results):
        points = obj.get_point_cloud(number_of_points=1000)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        ax.scatter(x, y, z, color=colors[i], label=f"T2SQNet: {obj.name}", s=5)

    for i, obj in enumerate(pc_data_cg):
        x, y, z = obj[:, 0], obj[:, 1], obj[:, 2]
        ax.scatter(x, y, z, color=colors[i + len(sq_results) + colour_gap], label=f"CG: {obj_names_cg[i]}", s=5)

    # Plot camera positions
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="black", marker='o', label="Camera Positions", s=50)
    ax.quiver(xyz[:, 0], xyz[:, 1], xyz[:, 2], uvw[:, 0], uvw[:, 1], uvw[:, 2], color="black", length=0.05)

    # Add legend and show
    plt.legend(loc='upper right')
    ax.set_xlim([-1, 1])  # Adjust as needed
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.show()
