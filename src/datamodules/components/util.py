import pickle


def save_pkl(data: dict, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(data, f, protocol=4)
