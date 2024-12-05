import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


#dataset_root = Path("/home/hamilton/Master_thesis/test/inputData")
dataset_root = Path("/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external")
scene_id = "tableware_2_2"

# color_path = dataset_root / scene_id / "color"
color_path = dataset_root / scene_id / "color"
pose_path = dataset_root / scene_id / "pose"
intrinsics_path = dataset_root / scene_id / "intrinsics"

selected_indices = ["000001", "000013", "000018", "000021", "000031", "000037", "000045"]

def load_images(color_path, selected_indices, resize_shape=(320, 240), show_imgs=False):

    images = []
    for idx in selected_indices:
        img_path = color_path / f"{idx}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)  # Read the image
        if img is None:
            raise FileNotFoundError(f"Image {img_path} not found!")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    formatted_images = [
        cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR) for img in images
    ]
    if(show_imgs):
        fig, axs = plt.subplots(1, 7, figsize=(15, 5))
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
        pose = np.load(pose_path_idx)  # Read the image
        if pose is None:
            raise FileNotFoundError(f"Pose {pose_path_idx} not found!")
        poses.append(pose)

    return poses

# fx=791.7353879123823
# fy=791.7353879123823
# cx=640.0
# cy=360.0
fx=187.14775
fy=249.37767
cx=159.15425
cy=119.91234
camera_intr = np.array([[fx,  0, cx],
                        [ 0, fy, cy],
                        [ 0,  0,  1]])

def load_projections(camera_poses,camera_intr):
    camera_projections=[]
    for pose in camera_poses:
        projection = torch.from_numpy(camera_intr @ np.linalg.inv(pose)[:3])
        camera_projections.append(projection)
    return torch.stack(camera_projections)


def load_all_data():
    tensor_images = load_images(color_path, selected_indices, resize_shape=(320, 240))
    camera_poses = load_poses(pose_path, selected_indices)
    camera_projections = load_projections(camera_poses,camera_intr)

    camera_params = {'camera_image_size': torch.tensor([240, 320]),
                        'projection_matrices': camera_projections,
                        'camera_intr': [camera_intr]*7, 
                        'camera_pose': camera_poses}
    
    return tensor_images, camera_params

def load_cam_pos():
    camera_poses = load_poses(pose_path, selected_indices)
    xzys = []
    uvws = []
    for camera_pose in camera_poses:
        xzy = camera_pose[:3, 3]
        xzys.append(xzy)
        uvw = camera_pose[:3, 2]
        uvws.append(uvw)
    
    return np.stack(xzys), np.stack(uvws)