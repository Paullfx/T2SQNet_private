import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the poses as numpy arrays
poses = [
    np.array([
        [-0.5, -0.22414387, 0.83651631, -0.63780152],
        [-0.8660254, 0.12940952, -0.48296291, 0.36823488],
        [0., -0.96592583, -0.25881905, 0.33230666],
        [0., 0., 0., 1.]
    ]),
    np.array([
        [-0.34202014, -0.24321035, 0.90767337, -0.6920552],
        [-0.93969262, 0.08852133, -0.33036609, 0.25188749],
        [0., -0.96592583, -0.25881905, 0.33230666],
        [0., 0., 0., 1.]
    ]),
    np.array([
        [-0.17364818, -0.25488701, 0.95125125, -0.72528113],
        [-0.98480775, 0.04494346, -0.16773126, 0.12788663],
        [0., -0.96592583, -0.25881905, 0.33230666],
        [0., 0., 0., 1.]
    ]),
    np.array([
        [0., -0.25881905, 0.96592583, -0.73646976],
        [-1., 0., 0., 0.],
        [0., -0.96592583, -0.25881905, 0.33230666],
        [0., 0., 0., 1.]
    ]),
    np.array([
        [0.17364818, -0.25488701, 0.95125125, -0.72528113],
        [-0.98480775, -0.04494346, 0.16773126, -0.12788663],
        [0., -0.96592583, -0.25881905, 0.33230666],
        [0., 0., 0., 1.]
    ]),
    np.array([
        [0.34202014, -0.24321035, 0.90767337, -0.6920552],
        [-0.93969262, -0.08852133, 0.33036609, -0.25188749],
        [0., -0.96592583, -0.25881905, 0.33230666],
        [0., 0., 0., 1.]
    ]),
    np.array([
        [0.5, -0.22414387, 0.83651631, -0.63780152],
        [-0.8660254, -0.12940952, 0.48296291, -0.36823488],
        [0., -0.96592583, -0.25881905, 0.33230666],
        [0., 0., 0., 1.]
    ]),
]



# Extract translation and orientation (forward vector)
translations = np.array([pose[:3, 3] for pose in poses])
orientations = np.array([pose[:3, 2] for pose in poses])  # Forward vector is column 0

# Plot the translations and orientations
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(translations[:, 0], translations[:, 1], translations[:, 2], label="Trajectory", marker='o')

# Plot the camera orientations as arrows
for i, (pos, ori) in enumerate(zip(translations, orientations)):
    ax.quiver(pos[0], pos[1], pos[2], ori[0], ori[1], ori[2], length=0.01, color='r', label='Orientation' if i == 0 else None)
    ax.text(pos[0], pos[1], pos[2], f"Pose {i + 1}", color='blue')

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title("Camera Movement Trajectory with Orientations, blender_shelf_0_3")

plt.show()


# import numpy as np

# # Extract the forward vectors (third column of the rotation matrix)
# forward_vectors = np.array([pose[:3, 2] for pose in poses])

# # Compute downward tilt angles
# tilt_angles = []
# for fv in forward_vectors:
#     # Compute the angle with the global Z-axis
#     angle = np.arccos(fv[2])  # In radians
#     tilt_angles.append(np.degrees(angle))  # Convert to degrees

# # Print the downward tilt angles for each pose
# for i, angle in enumerate(tilt_angles, start=1):
#     print(f"Pose {i}: Downward Tilt Angle = {angle:.2f} degrees")
