import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from pytorch3d.transforms.rotation_conversions import (
    matrix_to_quaternion,
    rotation_6d_to_matrix,
)
from src.utils.misc import remove_special_characters
from src.datamodules.components.babel import BABELDataset

import json

NUM_JOINTS = 24


@hydra.main(config_path="configs/", config_name="edit_motion.yaml")
def main(config: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    reference_set = BABELDataset("", "")

    for ref_id, data in enumerate(reference_set):
        rotation_6d, translation, annotation, motion_length = data
        rotation_6d = torch.Tensor(rotation_6d).to(device)
        translation = torch.Tensor(translation).to(device)

        rotation_matrix = rotation_6d_to_matrix(
            rotation_6d.reshape(motion_length, NUM_JOINTS, 6)
        )
        quaternion = matrix_to_quaternion(
            rotation_matrix
        )  # Real-part first for quaternion representation (w, x, y, z)

        export_dict = {
            "label": annotation,
            "translation": translation.tolist(),  # (L, 3)
            "rotation_quat": quaternion.tolist(),  # (L, 24, 4)
            "rotation_6d": rotation_6d.tolist(),  # (L, 24, 6)
            "quaternion_order": "wxyz",
        }

        ref_path = Path("reference_motions")
        ref_path.mkdir(parents=True, exist_ok=True)
        label_wos = remove_special_characters(annotation)

        file_name = f"{ref_id}_L{motion_length}.json"
        with open(ref_path / file_name, "w", encoding="utf-8") as f:
            json.dump(export_dict, f, indent=4)


if __name__ == "__main__":
    main()
