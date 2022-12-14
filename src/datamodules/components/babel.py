import os
from pathlib import Path
from typing import List
import numpy as np
import torch
import json
import pandas as pd
import json
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from src.datamodules.components.util import save_pkl
from tqdm.auto import tqdm
import math
from torch.utils.data import Dataset
import pickle


class BABELDataset(Dataset):
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
        annotation = motion_file["annotation"]
        motion_length = motion_file["motion_length"]

        return rotation_6d, translation, annotation, motion_length


class BABELSummarizer:
    def __init__(
        self,
        babel_dir: str = "data/babel_v1.0_release",
        babel_file: str = "train",  #  one of a ["train", "extra_train", "val", "extra_val"]
        act_cat_only: bool = False,
    ):

        self.babel_dir = Path(babel_dir)
        self.babel_file = babel_file
        self.file_list_path = self.babel_dir / f"{self.babel_file}.json"
        self.is_extra = True if babel_file.split("_")[0] == "extra" else False
        self.act_cat_only = act_cat_only

        with open(self.file_list_path, "r") as f:
            self.json_data = json.load(f)
        self.df_data = pd.read_json(self.file_list_path, orient="index").set_index(
            "babel_sid"
        )

        self.babel_summary_dir = Path("data/babel_summary")
        self.babel_summary_dir.mkdir(parents=True, exist_ok=True)

        self.seq_key = "seq_anns" if self.is_extra else "seq_ann"
        self.frame_key = "frame_anns" if self.is_extra else "frame_ann"

    def _process_seq(self):
        annotation_dict = dict()

        for i in self.json_data:

            if self.is_extra:
                labels = self.json_data[i][self.seq_key][0]["labels"]
            else:
                labels = self.json_data[i][self.seq_key]["labels"]

            for label_idx, label in enumerate(labels):
                if label["act_cat"] is None:
                    act_idx = 0
                    annotation_dict[f"{i}_{label_idx}_{act_idx}_seq"] = {
                        "feat_p": self.json_data[i]["feat_p"],
                        "raw_label": label["raw_label"],
                        "proc_label": label["proc_label"],
                        "act_cat": "",
                        "ann_type": "seq",
                        "start_t": None,
                        "end_t": None,
                    }
                    label_idx += 1
                else:
                    act_idx = 0
                    for act_cat in label["act_cat"]:
                        annotation_dict[f"{i}_{label_idx}_{act_idx}_seq"] = {
                            "feat_p": self.json_data[i]["feat_p"],
                            "raw_label": label["raw_label"],
                            "proc_label": label["proc_label"],
                            "act_cat": act_cat,
                            "ann_type": "seq",
                            "start_t": None,
                            "end_t": None,
                        }
                        act_idx += 1
                        label_idx += 1

        summary = (
            pd.DataFrame.from_dict(annotation_dict, orient="index")
            .reset_index()
            .rename(columns={"index": "ann_id"})
        )
        proc_label_subtable = summary[
            ["ann_id", "feat_p", "proc_label", "ann_type", "start_t", "end_t"]
        ].rename(columns={"proc_label": "annotation"})
        proc_label_subtable = proc_label_subtable[
            proc_label_subtable["annotation"] != ""
        ]

        actcat_label_subtable = summary[
            ["ann_id", "feat_p", "act_cat", "ann_type", "start_t", "end_t"]
        ].rename(columns={"act_cat": "annotation"})
        actcat_label_subtable = actcat_label_subtable[
            actcat_label_subtable["annotation"] != ""
        ]

        ann_integrated = (
            pd.concat([proc_label_subtable, actcat_label_subtable])
            .drop_duplicates(["feat_p", "annotation", "ann_type", "start_t", "end_t"])
            .reset_index(drop=True)
        )
        return ann_integrated

    def _process_frame(self):
        annotation_dict = dict()

        for i in self.json_data:

            if self.json_data[i][self.frame_key] is None:
                continue

            if self.is_extra:
                labels = self.json_data[i][self.frame_key][0]["labels"]
            else:
                labels = self.json_data[i][self.frame_key]["labels"]

            for label_idx, label in enumerate(labels):
                if label["act_cat"] is None:
                    act_idx = 0
                    annotation_dict[f"{i}_{label_idx}_{act_idx}_frame"] = {
                        "feat_p": self.json_data[i]["feat_p"],
                        "raw_label": label["raw_label"],
                        "proc_label": label["proc_label"],
                        "act_cat": "",
                        "ann_type": "frame",
                        "start_t": label["start_t"],
                        "end_t": label["end_t"],
                        "act_cat_original": "",
                    }
                    label_idx += 1
                else:
                    act_idx = 0
                    for act_cat in label["act_cat"]:
                        annotation_dict[f"{i}_{label_idx}_{act_idx}_frame"] = {
                            "feat_p": self.json_data[i]["feat_p"],
                            "raw_label": label["raw_label"],
                            "proc_label": label["proc_label"],
                            "act_cat": act_cat,
                            "ann_type": "frame",
                            "start_t": label["start_t"],
                            "end_t": label["end_t"],
                            "act_cat_original": act_cat,
                        }
                        act_idx += 1
                        label_idx += 1

        summary = (
            pd.DataFrame.from_dict(annotation_dict, orient="index")
            .reset_index()
            .rename(columns={"index": "ann_id"})
        )

        if not self.act_cat_only:
            proc_label_subtable = summary[
                ["ann_id", "feat_p", "proc_label", "ann_type", "start_t", "end_t"]
            ].rename(columns={"proc_label": "annotation"})
            proc_label_subtable = proc_label_subtable[
                proc_label_subtable["annotation"] != ""
            ]

            actcat_label_subtable = summary[
                ["ann_id", "feat_p", "act_cat", "ann_type", "start_t", "end_t"]
            ].rename(columns={"act_cat": "annotation"})
            actcat_label_subtable = actcat_label_subtable[
                actcat_label_subtable["annotation"] != ""
            ]
            integrated_df = pd.concat([proc_label_subtable, actcat_label_subtable])
        else:
            actcat_label_subtable = summary[
                ["ann_id", "feat_p", "act_cat_original", "ann_type", "start_t", "end_t"]
            ].rename(columns={"act_cat_original": "annotation"})
            actcat_label_subtable = actcat_label_subtable[
                actcat_label_subtable["annotation"] != ""
            ]
            integrated_df = actcat_label_subtable

        ann_integrated = integrated_df.drop_duplicates(
            ["feat_p", "annotation", "ann_type", "start_t", "end_t"]
        ).reset_index(drop=True)
        return ann_integrated

    def save_extract_summary(self):
        # seq_ann = self._process_seq()
        frame_ann = self._process_frame()
        integrated_annotation = frame_ann
        integrated_annotation.to_csv(
            self.babel_summary_dir / f"{self.babel_file}.csv", index=False
        )


class BABELProcessor:
    def __init__(
        self,
        summary_dir: str,
        summary_files: List[str],
        amass_dir: str,
        split: str = "train",
        motion_length: int = 128,
        target_fps: int = 20,
        offset: int = 15,
        suffix: str = "",
        labels: List[str] = [],
    ):

        self.summary_dir = Path(summary_dir)
        self.summary_files = summary_files
        self.amass_dir = Path(amass_dir)
        self.split = split
        self.motion_length = motion_length
        self.target_fps = target_fps
        self.offset = offset
        self.num_joints = 24
        self.labels = labels

        summary_dfs = []
        for summary_csv in summary_files:
            summary_df = pd.read_csv(self.summary_dir / summary_csv)
            summary_dfs.append(summary_df)
        self.summary_data = pd.concat(summary_dfs).reset_index(drop=True)

        self.setting = f"L{self.motion_length}_FPS{self.target_fps}_O{self.offset}"
        self.target_path = Path((f"data/babel_{self.setting}" + suffix))
        self.save_path = self.target_path / f"{split}_data"
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.file_list_path = self.target_path / f"{split}_list"
        self.file_list_path.mkdir(exist_ok=True, parents=True)

        self.is_val = True if self.split == "val" else False
        if self.is_val:
            self.test_save_path = self.target_path / f"test_data"
            self.test_save_path.mkdir(exist_ok=True, parents=True)
            self.test_file_list_path = self.target_path / f"test_list"
            self.test_file_list_path.mkdir(exist_ok=True, parents=True)

        with open(self.target_path / "labels.txt", "w") as f:
            for label in self.labels:
                f.write(f"{label}\n")

    def process(self):
        file_path_list = []
        test_file_path_list = []

        global_step = 0
        pbar = tqdm(self.summary_data.iterrows(), total=self.summary_data.shape[0])
        for ann_ind, ann_sample in pbar:
            pbar.set_description(f"[Processing: {global_step}]")

            if ann_sample["ann_type"] == "seq":  #  Use Frame-level annotation only.
                continue

            if (
                ann_sample["annotation"] == "transition"
            ):  # Skip annotation with transition
                continue

            if self.labels and (ann_sample["annotation"] not in self.labels):
                continue

            feat_p = ann_sample["feat_p"]
            amass_path = os.path.join(self.amass_dir, "/".join(feat_p.split("/")[1:]))
            assert os.path.exists(amass_path)

            with open(amass_path, "rb") as f:
                amass_motion = np.load(f)
                amass_meta = {
                    "gender": amass_motion["gender"],
                    "mocap_framerate": amass_motion["mocap_framerate"],
                    "trans": amass_motion["trans"].astype(np.float32),
                    "poses": amass_motion["poses"].astype(np.float32),
                }

            fps_adjust_factor = int(amass_meta["mocap_framerate"] / self.target_fps)

            trans_adjust = amass_meta["trans"][::fps_adjust_factor]
            poses_adjust = amass_meta["poses"][::fps_adjust_factor]
            axis_angles = poses_adjust.reshape(len(poses_adjust), -1, 3)[
                :, : self.num_joints, :
            ]
            rot_mat = axis_angle_to_matrix(torch.Tensor(axis_angles))
            rotation_6d = matrix_to_rotation_6d(rot_mat).numpy()
            assert len(trans_adjust) == len(poses_adjust) == len(rotation_6d)

            if ann_sample["ann_type"] == "frame":  # Frame label

                start_frame = math.floor(ann_sample["start_t"] * self.target_fps)
                end_frame = math.floor(ann_sample["end_t"] * self.target_fps) - 1

                assert end_frame <= len(rotation_6d) - 1

                clip_rotation = rotation_6d[start_frame:end_frame]
                clip_trans = trans_adjust[start_frame:end_frame]
                clip_length = len(clip_rotation)

                if clip_length == 0:
                    continue

                if clip_length <= self.motion_length:

                    if start_frame + self.motion_length >= len(rotation_6d):
                        continue

                    motion_clip = rotation_6d[
                        start_frame : start_frame + self.motion_length
                    ]
                    trans_clip = trans_adjust[
                        start_frame : start_frame + self.motion_length
                    ]
                    assert len(motion_clip) == len(trans_clip) == self.motion_length

                    motion_sample = {
                        "fps": self.target_fps,
                        "rotation_6d": motion_clip.astype(np.float32),
                        "translation": trans_clip.astype(np.float32),
                        "annotation": ann_sample["annotation"],
                        "motion_length": self.motion_length,
                    }

                    file_name = f"{global_step:09d}.pkl"

                    if (self.is_val) and (global_step % 5 == 0):
                        save_path = self.test_save_path / file_name
                        test_file_path_list.append(file_name)
                    else:
                        save_path = self.save_path / file_name
                        file_path_list.append(file_name)
                    save_pkl(motion_sample, save_path)
                    global_step += 1

                else:
                    frame_idx = 0
                    while frame_idx + self.motion_length <= clip_length:
                        motion_clip = clip_rotation[
                            frame_idx : frame_idx + self.motion_length,
                            :,
                            :,
                        ]
                        trans_clip = clip_trans[
                            frame_idx : frame_idx + self.motion_length,
                            :,
                        ]
                        assert len(motion_clip) == len(trans_clip) == self.motion_length
                        motion_sample = {
                            "fps": self.target_fps,
                            "rotation_6d": motion_clip.astype(np.float32),
                            "translation": trans_clip.astype(np.float32),
                            "annotation": ann_sample["annotation"],
                            "motion_length": self.motion_length,
                        }
                        file_name = f"{global_step:09d}.pkl"

                        if (self.is_val) and (global_step % 5 == 0):
                            save_path = self.test_save_path / file_name
                            test_file_path_list.append(file_name)
                        else:
                            save_path = self.save_path / file_name
                            file_path_list.append(file_name)
                        save_pkl(motion_sample, save_path)
                        frame_idx += self.offset
                        global_step += 1
            else:
                ValueError("Unknown annotation type.")

        with open(self.file_list_path / "file_list.pkl", "wb") as f:
            pickle.dump(file_path_list, f, protocol=4)

        if self.is_val:
            with open(self.test_file_list_path / "file_list.pkl", "wb") as f:
                pickle.dump(test_file_path_list, f, protocol=4)
        print(f"Babel motion clips are saved at {self.save_path}")
