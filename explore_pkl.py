import pickle


pkl_file_path = "/home/fuxiao/Projects/Orbbec/T2SQNet-public/intermediates/tableware_3_9/object_list/object_list.pkl"



with open(pkl_file_path, "rb") as f:
    data = pickle.load(f)


