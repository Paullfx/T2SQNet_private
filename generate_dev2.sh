# # blender setting
# BLENDER_BIN=~/blender-4.0.2-linux-x64/./blender
# BLENDER_PROJ_PATH=assets/materials/material_lib_graspnet-v2.blend

# Generate Scene using PyBullet
python generate_tableware_pybullet.py \
--data_path datasets \
--folder_name Laptop10 \
--object_types 'Laptop' \
--num_objects 1 \
--training_num 1500 \
--validation_num 200 \
--test_num 150 \
--num_cameras 36 \
--sim_type table \
--reduce_ratio 2 \
#--enable_gui \
#--debug
# num_camera 36

# # Render RGB images using Blender
# $BLENDER_BIN $BLENDER_PROJ_PATH --background \
# --python generate_tableware_blender.py \
# --- \
# ---use_blender \
# ---use_tsdf \
# ---folder_name datasets/Laptop1 \
# ---split_list training validation test \
# ---max_data_num 1 0 0

# Voxelize Objects for Training ResNet3D
python generate_tableware_voxelize.py \
--folder_name datasets/Laptop10
