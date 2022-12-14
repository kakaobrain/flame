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
from src.utils.sample_util import sample_motion
from src.models.diffusion_module import DiffusionLitModule
from scipy.ndimage import gaussian_filter1d

import json

NUM_JOINTS = 24


@hydra.main(config_path="configs/", config_name="edit_motion.yaml")
def main(config: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    reference_motion_path = config["reference_motion_path"]
    use_reference_translation = config["use_reference_translation"]
    use_smoothing = config["use_smoothing"]
    edit_prompt = config["edit_prompt"]
    guidance_scale = config["guidance_scale"]
    edit_joints_indices = list(config["edit_joints_indices"])

    with open(reference_motion_path, "r") as f:
        reference_motion = json.load(f)

    reference_motion_trs = torch.Tensor(reference_motion["translation"]).to(device)
    reference_motion_rot6d = torch.Tensor(reference_motion["rotation_6d"]).to(device)
    L, J, D = reference_motion_rot6d.shape
    reference_motion_rot6d_flat = reference_motion_rot6d.reshape(L, J * D)
    reference_motion = (
        torch.concat([reference_motion_trs, reference_motion_rot6d_flat], dim=1)[
            None, :, :
        ]
        .permute(0, 2, 1)
        .repeat(2, 1, 1)
    )  # For classifier-free guidance
    assert reference_motion.shape == (2, 147, L)

    reference_length = len(reference_motion_trs)
    ckpt_path = Path(config["ckpt_path"])

    conditioning_info = {
        "use_reference_translation": use_reference_translation,
        "edit_prompt": edit_prompt,
        "ref_trs": reference_motion_trs,
        "ref_rotation_6d": reference_motion_rot6d,
        "edit_joints_indices": edit_joints_indices,
        "motion": reference_motion,
    }

    model = DiffusionLitModule.load_from_checkpoint(ckpt_path).cuda()
    motion_dim = (
        model.net.motion_dim
    )  # 147 = translation (3) + rotation with 6D representation format (24 * 6 = 144)
    ema_model = model.ema_model.model
    ema_model.eval()

    generated_motions = sample_motion(
        sampling_texts=[edit_prompt],
        motion_lengths=[reference_length],
        sample_fn=model.diffusion.p_sample_loop,
        ema_model=ema_model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        motion_dim=motion_dim,
        guidance_scale=guidance_scale,
        progress=True,
        conditioning_info=conditioning_info,
    )  # This will output (Batch, Motion Length, Representation dim)

    if use_smoothing:
        generated_motions = torch.Tensor(
            gaussian_filter1d(generated_motions.cpu().numpy(), sigma=1, axis=1)
        )

    translation_gen = generated_motions[0, :reference_length, :3]
    rotation_6d_gen = generated_motions[0, :reference_length, 3:]
    rotation_matrix_gen = rotation_6d_to_matrix(
        rotation_6d_gen.reshape(reference_length, NUM_JOINTS, 6)
    )
    quaternion_gen = matrix_to_quaternion(
        rotation_matrix_gen
    )  # Real-part first for quaternion representation (w, x, y, z)

    if use_reference_translation:
        translation_gen = reference_motion_trs
    else:
        translation_gen = translation_gen

    export_dict = {
        "label": edit_prompt,
        "translation": translation_gen.tolist(), # (L, 3)
        "rotation_quat": quaternion_gen.tolist(),  # (L, 24, 4)
        "quaternion_order": "wxyz",
        "guidance_scale": guidance_scale,
    }

    label_wos = remove_special_characters(edit_prompt)

    file_name = f"edit_{label_wos}_L{reference_length}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(export_dict, f, indent=4)


if __name__ == "__main__":
    main()
