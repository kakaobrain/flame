from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch as th
from src.models.diffusion_module import DiffusionLitModule
from omegaconf import OmegaConf
import json
from pytorch3d.transforms.rotation_conversions import (
    matrix_to_quaternion,
    rotation_6d_to_matrix,
)
from scipy.ndimage import gaussian_filter1d
from src.utils.sample_util import sample_motion
from src.utils.misc import remove_special_characters
from src.utils.vis_util import render_video_summary


@hydra.main(config_path="configs/", config_name="t2m_sample.yaml")
def main(config: DictConfig):
    ckpt_path = Path(config["ckpt_path"])
    motion_length = config["motion_length"]
    NUM_JOINTS = 24
    guidance_scale = config["guidance_scale"]
    labels_for_gen = list(config["labels_for_gen"])
    plot_gif = config["plot_gif"]
    export_json = config["export_json"]
    use_smoothing = config["use_smoothing"]
    fps = config["fps"]
    print(OmegaConf.to_yaml(config))

    model = DiffusionLitModule.load_from_checkpoint(ckpt_path).cuda()
    motion_dim = (
        model.net.motion_dim
    )  # 147 = translation (3) + rotation with 6D representation format (24 * 6 = 144)
    ema_model = model.ema_model.model
    ema_model.eval()

    for sample_id, ann in enumerate(labels_for_gen):

        generated_motions = sample_motion(
            sampling_texts=[ann],
            motion_lengths=[motion_length],
            sample_fn=model.diffusion.p_sample_loop,
            ema_model=ema_model,
            device=th.device("cuda" if th.cuda.is_available() else "cpu"),
            motion_dim=motion_dim,
            guidance_scale=guidance_scale,
            progress=True,
        )  # This will output (Batch, Motion Length, Representation dim)

        if use_smoothing:
            generated_motions = th.Tensor(gaussian_filter1d(generated_motions.cpu().numpy(), sigma=1, axis=1))

        label_wos = remove_special_characters(ann)

        if plot_gif:
            render_video_summary(
                img_ids=[f"{label_wos}_guidance{guidance_scale}_L{motion_length}"],
                translations=generated_motions[:, :motion_length, :3],
                rotation_6ds=generated_motions[:, :motion_length, 3:],
                annotations=[ann],
                fps=fps,
                write=True,
            )

        if export_json:
            translation = generated_motions[0, :motion_length, :3]
            rotation_6d_gen = generated_motions[0, :motion_length, 3:]
            rotation_matrix_gen = rotation_6d_to_matrix(
                rotation_6d_gen.reshape(motion_length, NUM_JOINTS, 6)
            )
            quaternion_gen = matrix_to_quaternion(
                rotation_matrix_gen
            )  # Real-part first for quaternion representation (w, x, y, z)

            export_dict = {
                "label": ann,
                "translation": translation.tolist(),  # (L, 3)
                "rotation_quat": quaternion_gen.tolist(),  # (L, 24, 4)
                "quaternion_order": "wxyz",
                "guidance_scale": guidance_scale,
            }

            file_name = f"{sample_id}_{label_wos}_L{motion_length}.json"
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(export_dict, f, indent=4)


if __name__ == "__main__":
    main()
