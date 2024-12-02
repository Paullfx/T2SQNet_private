# Standard library imports
import os
import numpy as np
import cv2
import torch
from pathlib import Path

# ROS 2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage, CameraInfo
from geometry_msgs.msg import PoseStamped
import ros2_numpy.point_cloud2 as point_cloud2

# Local imports
from conceptgraph.utils.general_utils import scale_intrinsics

class Subscriber(Node):
    def __init__(self):
        super().__init__('subscriber')

        # Create subscriptions for color image, camera info, and pose
        self.sub_color = self.create_subscription(ROSImage, 'spectacular_ai/color_image', self.color_callback, 10)
        self.sub_info = self.create_subscription(CameraInfo, 'spectacular_ai/camera_info', self.info_callback, 10)
        self.sub_pose = self.create_subscription(PoseStamped, 'spectacular_ai/pose_image_synced', self.pose_callback, 10)

        # Initialize message storage
        self.color_msg = None
        self.camera_msg = None
        self.pose_msg = None

        # Flags for tracking received messages
        self.received_color = False
        self.received_info = False
        self.received_pose = False

    # Callback for receiving color image
    def color_callback(self, msg):
        self.color_msg = msg
        self.received_color = True

    # Callback for receiving camera info
    def info_callback(self, msg):
        self.camera_msg = msg
        self.received_info = True

    # Callback for receiving camera pose
    def pose_callback(self, msg):
        self.pose_msg = msg
        self.received_pose = True

    # Process the inputs (color image, depth, intrinsics, pose)
    def process_inputs(self, cfg, rotate=True):
        color = self._process_color(cfg, rotate)
        intrinsics = self._process_intrinsics(cfg, rotate)
        pose = self._process_pose(cfg, rotate)
        
        # Reset flags after processing
        self.received_color = False
        self.received_info = False
        self.received_pose = False

        # Return processed inputs
        return color, intrinsics, pose
    
    # Process color image
    def _process_color(self, cfg, rotate):
        color = np.array(self.color_msg.data).astype(np.uint8).reshape(cfg.true_height, cfg.true_width, 3)
        if rotate:
            color = np.rot90(color, -1)
        color = cv2.resize(
            color,
            (cfg.desired_width, cfg.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        color = torch.from_numpy(color)
        color = color.to(cfg.device).type(torch.float)
        return color

    # Process camera intrinsics
    def _process_intrinsics(self, cfg, rotate):
        K = np.array(self.camera_msg.k).reshape(3, 3)
        if rotate:
            K[0, 2], K[1, 2] = K[1, 2], K[0, 2]  # switch cx, cy
            K[0, 0], K[1, 1] = K[1, 1], K[0, 0]  # switch fx, fy
        K = torch.from_numpy(K)
        height_downsample_ratio = float(cfg.desired_height) / cfg.image_height
        width_downsample_ratio = float(cfg.desired_width) / cfg.image_width
        K = scale_intrinsics(K, height_downsample_ratio, width_downsample_ratio)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K
        intrinsics = intrinsics.to(cfg.device).type(torch.float)
        return intrinsics

    # Process camera pose
    def _process_pose(self, cfg, rotate):
        pose = np.eye(4)
        pose[:3, :3] = R.from_quat([
            self.pose_msg.pose.orientation.x,
            self.pose_msg.pose.orientation.y,
            self.pose_msg.pose.orientation.z,
            self.pose_msg.pose.orientation.w
        ]).as_matrix()
        pose[:3, 3] = np.array([
            self.pose_msg.pose.position.x,
            self.pose_msg.pose.position.y,
            self.pose_msg.pose.position.z
        ])
        if rotate:
            image_rotation = np.eye(4)
            image_rotation[:3, :3] = R.from_euler('z', -90, degrees=True).as_matrix()
            pose = pose @ image_rotation
        pose = torch.from_numpy(pose)
        pose = pose.to(cfg.device).type(torch.float)
        return pose


def main(cfg):
    rclpy.init()

    # Initialize the Subscriber node
    node = Subscriber()

    # Main loop for processing data
    while rclpy.ok():
        # Wait for all necessary data to be received
        while not (node.received_color and node.received_info and node.received_pose):
            rclpy.spin_once(node, timeout_sec=0)

        # Process the inputs (color image, intrinsics, pose)
        color_tensor, intrinsics, pose_tensor = node.process_inputs(cfg, rotate=cfg.rotate)

        # Save or process the data as needed
        color_path = Path(cfg.color_path) / f"{frame_idx:06}.png"
        cv2.imwrite(str(color_path), color_tensor.cpu().numpy())

        # Optionally save camera pose and intrinsics, or further processing

    # Cleanup and shutdown
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
