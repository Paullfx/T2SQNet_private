import gzip
import shutil
import pickle
import numpy as np
import os


# Path to the file
#source_path = '/home/hamilton/Master_thesis/test/inputData/tableware_1_1/exps/exp_default/pcd_exp_default.pkl.gz'
source_path = '/home/fuxiao/Projects/Orbbec/concept-graphs/conceptgraph/dataset/external/tableware_2_2/exps/exp_default/pcd_exp_default.pkl.gz'
file_path = os.path.splitext(source_path)[0]

def load_pc_cg():
    # Decompress the .gz file
    with gzip.open(source_path, 'rb') as f_in:
        with open(file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Load data from the .pkl file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    pc_data = []
    obj_names = []
    for idx, obj in enumerate(data['objects']):
        pc_data.append(obj["pcd_np"])
        obj_names.append(obj["class_name"])

    return pc_data, obj_names