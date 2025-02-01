

# Generate Scene using PyBullet
python generate_tableware_pybullet.py \
--data_path datasets \
--folder_name test \
--object_types 'Laptop' \
--num_objects 1 \
--training_num 1 \
--validation_num 0 \
--test_num 0 \
--num_cameras 36 \
--sim_type table \
--reduce_ratio 2 \
--enable_gui \
#--debug


