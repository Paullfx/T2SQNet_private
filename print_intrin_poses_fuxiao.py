import os
import numpy as np

# Define the directory and file names
output_dir = './intermediates/scene_id_default/camera_pose_and_intr'
camera_pose_file = os.path.join(output_dir, 'camera_pose.npy')
camera_intr_file = os.path.join(output_dir, 'camera_intr.npy')

# Load the saved data
if os.path.exists(camera_pose_file) and os.path.exists(camera_intr_file):
    # Load camera_intr and camera_pose
    camera_intr = np.load(camera_intr_file)
    camera_pose = np.load(camera_pose_file)

    # Print camera_intr
    print("Camera Intrinsics:")
    print(camera_intr)

    # Print the 7 camera poses
    print("\nCamera Poses:")
    for i, pose in enumerate(camera_pose):
        print(f"Pose {i+1}:")
        print(pose)
        print()  # Blank line for better readability
else:
    print("The required files do not exist. Please check the directory and file paths.")
