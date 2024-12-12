#

import pickle

def read_pkl_keys(file_path):
    """
    Reads the keys of a pickle file.

    Args:
        file_path (str): Path to the pickle file.
    
    Returns:
        list: List of keys in the pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, dict):
                return list(data.keys())
            else:
                print("The pickle file does not contain a dictionary.")
                return None
    except Exception as e:
        print(f"Error reading the pickle file: {e}")
        return None

# Example usage:
if __name__ == "__main__":
    file_path = "/home/hamilton/Master_thesis/T2SQNet-public/datasets/test/test/0.pkl"  # Replace with your .pkl file path
    keys = read_pkl_keys(file_path)
    if keys:
        print("Keys in the pickle file:")
        for key in keys:
            print(f"- {key}")
