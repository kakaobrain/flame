import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from tqdm.auto import tqdm

from src.datamodules.components.util import save_pkl

# FPS is set to 20FPS by Dataset Configuration.
HUMANML3D_FPS = 20
NUM_JOINTS = 24  # SMPL Default


class HumanML3DProcessor:
    def __init__(
        self,
        humanact12_dict: dict,
        split: str,
    ):

        self.humanact12_dict = humanact12_dict
        self.humanml3d_base_path = Path("data/HumanML3D")
        self.humandml3d_index_df = pd.read_csv("data/HumanML3D/HumanML3D.csv")
        self.amass_path = Path("data/amass_smplhg")
        self.annotation_path = Path("data/HumanML3D/texts")
        self.save_target_path = Path("data/HumanML3D/processed")
        self.save_target_path.mkdir(exist_ok=True, parents=True)

        self.save_data_path = self.save_target_path / f"{split}_data"
        self.save_data_path.mkdir(exist_ok=True, parents=True)
        self.save_file_list_path = self.save_target_path / f"{split}_list"
        self.save_file_list_path.mkdir(exist_ok=True, parents=True)

        with open(f"data/HumanML3D/{split}.txt", "r") as f:
            split_data_list = [x.rstrip() for x in f]
        self.split_data_list = split_data_list

    def process(self):

        global_step = 0
        file_pointer_list = []

        pbar = tqdm(
            self.humandml3d_index_df.iterrows(), total=self.humandml3d_index_df.shape[0]
        )

        for df_idx, info in pbar:
            pbar.set_description(f"[Processing: {global_step}]")
            sample_id = info["new_name"].split(".")[0]

            if sample_id not in self.split_data_list:
                continue

            data_source = "humanact12" if "humanact" in info["source_path"] else "amass"

            if data_source == "amass":
                motion_data_path = (
                    info["source_path"]
                    .replace("./pose_data", str(self.amass_path))
                    .replace(".npy", ".npz")
                )
                with open(motion_data_path, "rb") as f:
                    motion_data = np.load(f)
                    motiton_meta = {
                        "gender": motion_data["gender"],
                        "mocap_framerate": motion_data["mocap_framerate"],
                        "trans": motion_data["trans"].astype(np.float32),
                        "poses": motion_data["poses"].astype(np.float32),
                    }

                    fps_adjust_factor = int(
                        motiton_meta["mocap_framerate"] / HUMANML3D_FPS
                    )
                    trans_adjust = motiton_meta["trans"][::fps_adjust_factor]
                    poses_adjust = motiton_meta["poses"][::fps_adjust_factor]

                    start_frame = info["start_frame"]
                    end_frame = info["end_frame"]

                    trans_target = trans_adjust[start_frame:end_frame]
                    poses_target = poses_adjust[start_frame:end_frame]

                    axis_angles = poses_target.reshape(len(poses_target), -1, 3)[
                        :, :NUM_JOINTS, :
                    ]  # (N, 24, 3)

            elif data_source == "humanact12":
                humanact12_filename = info["source_path"].split("/")[-1]
                if humanact12_filename not in self.humanact12_dict.keys():
                    print(f"{humanact12_filename} does not exists.")
                    continue
                humanact_data = self.humanact12_dict[humanact12_filename]
                poses = humanact_data["poses"]
                joints3d = humanact_data["joints3D"]
                motion_length = humanact_data["motion_length"]
                trans_target = joints3d[:, 0, :]
                axis_angles = poses.reshape(motion_length, -1, 3)  # (N, 24, 3)

            rot_mat = axis_angle_to_matrix(torch.Tensor(axis_angles))
            rotation_6d = matrix_to_rotation_6d(rot_mat).numpy()

            with open(
                self.annotation_path / info["new_name"].replace(".npy", ".txt"), "r"
            ) as f:
                motion_annotations = f.readlines()

            # Post process
            ann_txt = []
            for ann in motion_annotations:
                lang_part = ann.split("#")[0]
                if lang_part[-1] != ".":
                    lang_part += "."

                assert lang_part[-1] == "."
                ann_txt.append(lang_part)

            for ann in ann_txt:
                assert len(rotation_6d) == len(trans_target)
                motion_sample = {
                    "fps": HUMANML3D_FPS,
                    "rotation_6d": rotation_6d.astype(np.float32),  # (N, 24, 6)
                    "translation": trans_target.astype(np.float32),  # (N, 3)
                    "annotation": ann,  # (str)
                    "motion_length": len(trans_target),  # int
                    "data_source": data_source,
                    "humanml3d_id": sample_id,
                }

                file_name = f"{global_step:09d}.pkl"
                file_pointer_list.append(file_name)
                global_step += 1
                save_pkl(motion_sample, self.save_data_path / file_name)

        with open(self.save_file_list_path / "file_list.pkl", "wb") as f:
            pickle.dump(file_pointer_list, f, protocol=4)
        print(f"Motion clips are saved at {self.save_data_path}")
