import os
import pickle

# Define the directory where the data is saved
input_dir = './intermediates/scene_id_default/bboxes_cls'

# Check if the directory exists
if not os.path.exists(input_dir):
    print(f"Error: The directory {input_dir} does not exist.")
    exit(1)

# Load and print bounding boxes
bboxes_path = os.path.join(input_dir, 'bboxes.pkl')
if os.path.exists(bboxes_path):
    with open(bboxes_path, 'rb') as f:
        bboxes = pickle.load(f)
    print("Bounding Boxes:")
    print(bboxes)
else:
    print(f"Error: File {bboxes_path} does not exist.")

# Load and print classes
cls_path = os.path.join(input_dir, 'cls.pkl')
if os.path.exists(cls_path):
    with open(cls_path, 'rb') as f:
        cls = pickle.load(f)
    print("Classes:")
    print(cls)
else:
    print(f"Error: File {cls_path} does not exist.")
