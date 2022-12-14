import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import shutil
from pathlib import Path
from typing import List

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from body_visualizer.mesh.mesh_viewer import MeshViewer
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from hydra.utils import get_original_cwd
from pytorch3d.transforms.rotation_conversions import (
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
from tqdm.auto import tqdm


def save_image(img_ndarray: str, frame_id: int, title: str, target_dir: str):

    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = fig.gca()

    img = img_ndarray.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    plt.axis("off")
    title = f"{title}: [{frame_id}]"
    plt.title(title)
    save_fn = os.path.join(target_dir, "{:06d}.png".format(frame_id))
    plt.savefig(save_fn, bbox_inches="tight", pad_inches=0)
    plt.close()


def render_video_summary(
    img_ids: List[str],
    translations: torch.Tensor,
    rotation_6ds: torch.Tensor,
    annotations: List[str],
    fps: int = 20,
    write: bool = False,
):
    rendered_out = []
    for img_id, translation, rotation_6d, annotation in zip(
        img_ids, translations, rotation_6ds, annotations
    ):
        rendered_motion = visualize_motion(
            translation=translation,
            rotation_6d=rotation_6d,
            motion_length=len(translation),
            title=annotation,
            suffix=f"{img_id}",
            fps=fps,
            write=write,
            tqdm_disable=True,
        )
        motion_sequence = torch.Tensor(rendered_motion)[
            None,
        ]
        rendered_out.append(motion_sequence)
    video_summary = torch.cat(rendered_out).permute(0, 1, 4, 2, 3)  # (N, T, C, H, W)
    return video_summary


def visualize_motion(
    translation: torch.Tensor,
    rotation_6d: torch.Tensor,
    motion_length: int,
    title: str,
    suffix: str = "",
    fps=20,
    tmp_image_path="tmp_images/",
    conditioning_info=None,
    write=True,
    tqdm_disable=False,
):
    imw, imh = 400, 400

    device = "cuda" if torch.cuda.is_available() else "cpu"
    translation = translation.to(device)
    rotation_6d = rotation_6d.to(device)

    if write:
        if os.path.exists(tmp_image_path):
            shutil.rmtree(tmp_image_path)
        save_path = Path(tmp_image_path)
        save_path.mkdir(parents=True, exist_ok=True)

    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    # Visualization settings
    subject_gender = "neutral"
    bm_fname = os.path.join(
        get_original_cwd(), "smpl_model", subject_gender, "model.npz"
    )
    dmpl_fname = os.path.join(
        get_original_cwd(), "dmpl_model", subject_gender, "model.npz"
    )
    num_betas = 16
    num_dmpls = 8

    bm = BodyModel(
        bm_fname=bm_fname,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname,
    ).to(device)
    faces = c2c(bm.f)

    smpl_param = matrix_to_axis_angle(
        rotation_6d_to_matrix(rotation_6d.reshape(rotation_6d.size(0), -1, 6))
    )[:motion_length].reshape(motion_length, -1)

    body_params = {
        "root_orient": smpl_param[:, :3],  # controls the global root orientation
        "pose_body": smpl_param[:, 3:66],  # controls the body
        "pose_hand": smpl_param[:, 66:],  # controls the finger articulation
        "trans": translation,  # controls the global body position
    }

    body_pose_beta = bm(
        **{k: v for k, v in body_params.items() if k in ["pose_body", "betas"]}
    )

    render_imgs = []

    pbar = tqdm(range(motion_length), total=motion_length, disable=tqdm_disable)
    for fId in pbar:
        if conditioning_info is not None:
            if conditioning_info["type"] in ["prediction"]:
                cond_start = conditioning_info["range"][0]
                cond_end = conditioning_info["range"][1]
                if cond_start <= fId <= cond_end:
                    body_color = [0.0, 1.0, 0.0]
                else:
                    body_color = [0.713, 0.580, 0.345]
            elif conditioning_info["type"] in ["inbetweening"]:
                cond_range1 = conditioning_info["range"][0]
                cond_range2 = conditioning_info["range"][1]
                if (cond_range1[0] <= fId <= cond_range1[1]) or (
                    cond_range2[0] <= fId <= cond_range2[1]
                ):
                    body_color = [0.0, 1.0, 0.0]
                else:
                    body_color = [0.713, 0.580, 0.345]
            elif conditioning_info["type"] in ["completion"]:
                body_color = [0.713, 0.580, 0.345]

        else:
            body_color = [0.713, 0.580, 0.345]  # Blue
        body_mesh = trimesh.Trimesh(
            vertices=c2c(body_pose_beta.v[fId]),
            faces=faces,
            vertex_colors=np.tile(body_color, (6890, 1)),
        )
        mv.set_static_meshes([body_mesh])
        render_img = mv.render(render_wireframe=False)
        if write:
            save_image(render_img, frame_id=fId, title=title, target_dir=tmp_image_path)
        render_imgs.append(render_img)

    # Make gif from images
    if write:
        images = []
        file_names = sorted(os.listdir(save_path))
        for file_name in file_names:
            images.append(imageio.imread(save_path / file_name))
        imageio.mimwrite(f"vis_{suffix}.gif", images, format="gif", fps=fps)
    return render_imgs
