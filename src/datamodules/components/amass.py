import os
import pickle
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from src.datamodules.components.util import save_pkl
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class AMASSDataset(Dataset):
    def __init__(self, data_dir: str, filelist_dir: str):
        self.data_dir = Path(data_dir)
        self.filelist_dir = Path(filelist_dir)

        with open(self.filelist_dir / f"file_list.pkl", "rb") as f:
            self.motion_files = pickle.load(f)

    def __len__(self):
        return len(self.motion_files)

    def __getitem__(self, idx):
        motion_path = self.data_dir / self.motion_files[idx]
        with open(motion_path, "rb") as f:
            motion_file = pickle.load(f)

        rotation_6d = motion_file["rotation_6d"]
        translation = motion_file["translation"]
        annotation = ""
        motion_length = motion_file["motion_length"]

        return rotation_6d, translation, annotation, motion_length


class AMASSProcessor:
    def __init__(
        self,
        data_dir: str = "data/amass_smplhg",
        summary_dir: str = "data/babel_summary",
        exclude_dirs: List[str] = [],
        motion_length: int = 128,
        target_fps: int = 20,
        offset: int = 5,
    ):
        self.data_dir = Path(data_dir)
        self.summary_dir = Path(summary_dir)
        self.exclude_dirs = exclude_dirs
        self.motion_length = motion_length
        self.target_fps = target_fps
        self.offset = offset
        self.num_joints = 24  # Fix 24 joints as a SMPL format

        self.setting = f"L{self.motion_length}_FPS{self.target_fps}_O{self.offset}"
        self.target_path = Path(f"data/amass_{self.setting}")

        self.save_path_train = self.target_path / "train_data"
        self.save_path_train.mkdir(exist_ok=True, parents=True)
        self.train_file_list_path = self.target_path / "train_list"
        self.train_file_list_path.mkdir(exist_ok=True, parents=True)

        self.save_path_val = self.target_path / "val_data"
        self.save_path_val.mkdir(exist_ok=True, parents=True)
        self.val_file_list_path = self.target_path / "val_list"
        self.val_file_list_path.mkdir(exist_ok=True, parents=True)

        self.motion_files = self.get_motion_files()
        self.split_dataset()

    def split_dataset(self):
        summary_files = ["train.csv", "extra_train.csv", "val.csv", "extra_val.csv"]
        assert os.listdir(self.summary_dir) == summary_files
        train_df = pd.read_csv(self.summary_dir / "train.csv")
        extra_train_df = pd.read_csv(self.summary_dir / "extra_train.csv")
        val_df = pd.read_csv(self.summary_dir / "val.csv")
        extra_val_df = pd.read_csv(self.summary_dir / "extra_val.csv")

        train = pd.concat([train_df, extra_train_df]).reset_index(drop=True)
        val = pd.concat([val_df, extra_val_df]).reset_index(drop=True)

        motion_files_list = pd.Series(self.motion_files).drop_duplicates()

        train_paths = (
            train["feat_p"]
            .drop_duplicates()
            .apply(lambda x: os.path.join(self.data_dir, "/".join(x.split("/")[1:])))
        )
        val_paths = (
            val["feat_p"]
            .drop_duplicates()
            .apply(lambda x: os.path.join(self.data_dir, "/".join(x.split("/")[1:])))
        )

        self.validation_files = val_paths.values
        self.train_files = motion_files_list[
            ~motion_files_list.isin(self.validation_files)
        ].values

        assert len(motion_files_list) == len(self.train_files) + len(
            self.validation_files
        )

    def process(self):
        train_file_list = []
        val_file_list = []

        global_step = 0
        pbar = tqdm(self.motion_files, total=len(self.motion_files))
        for motion_file in pbar:
            pbar.set_description(f"[Processing: {global_step}]")

            motion = np.load(motion_file)

            # Stage 1 Adjust framerate
            mocap_fps = motion["mocap_framerate"]
            interval = int(mocap_fps / self.target_fps)
            target_fps_motion = motion["poses"][::interval, :]
            trans = motion["trans"][::interval, :]
            adjust_len = len(target_fps_motion)
            axis_angles = target_fps_motion.reshape(adjust_len, -1, 3)[
                :, : self.num_joints, :
            ]
            rot_mat = axis_angle_to_matrix(torch.Tensor(axis_angles))
            rotation_6d = matrix_to_rotation_6d(rot_mat).numpy()

            # Stage 2 Sliding window for fixed length (self.motion_length)
            if adjust_len <= self.motion_length:
                # Skip when entire motion sequence is shorter than motion_length
                continue
            else:
                i = 0
                while i + self.motion_length <= len(target_fps_motion):
                    motion_clip = rotation_6d[i : i + self.motion_length, :, :]
                    trans_clip = trans[i : i + self.motion_length, :]
                    assert len(motion_clip) == len(trans_clip)
                    motion_sample = {
                        "fps": self.target_fps,
                        "rotation_6d": motion_clip.astype(np.float32),
                        "translation": trans_clip.astype(np.float32),
                        "motion_length": self.motion_length,
                    }
                    file_name = f"{global_step:09d}.pkl"

                    if motion_file in self.train_files:
                        save_path = self.save_path_train
                        train_file_list.append(file_name)
                    elif motion_file in self.validation_files:
                        save_path = self.save_path_val
                        val_file_list.append(file_name)
                    else:
                        ValueError("motion file does not belong to train or test list.")

                    target_path = save_path / file_name
                    save_pkl(motion_sample, target_path)
                    i += self.offset
                    global_step += 1

        with open(self.train_file_list_path / "file_list.pkl", "wb") as f:
            pickle.dump(train_file_list, f, protocol=4)

        with open(self.val_file_list_path / "file_list.pkl", "wb") as f:
            pickle.dump(val_file_list, f, protocol=4)

        print(
            f"Processed motion files are saved at {self.train_file_list_path} / Toal clips: {global_step + 1}"
        )

    def get_motion_files(self):
        datasets = os.listdir(self.data_dir)

        for exclude_dir in self.exclude_dirs:
            datasets.remove(exclude_dir) if exclude_dir in datasets else None

        motion_file_dirs = []
        print(f"[Datasets]: {datasets}")
        for dataset in datasets:
            if dataset in self.exclude_dirs:  # Do not use setaside dataset
                continue
            dataset_dir = os.path.join(self.data_dir, dataset)
            for dirpath, dirnames, filenames in os.walk(dataset_dir):
                if not dirnames:
                    for filename in filenames:
                        if filename == "shape.npz":
                            continue
                        motion_file_dirs.append(os.path.join(dirpath, filename))

        return motion_file_dirs
