import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d.transforms.rotation_conversions import (
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
from scipy import linalg
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.mclip_module import MotionClipLitModule


def compute_metrics(output_dir: str):
    output_dir = Path(output_dir)
    samples_path = output_dir / "generated_motions"
    sample_files = os.listdir(samples_path)
    stat_path = output_dir / "stats" / "stats.pkl"

    meta_data = []
    for pkl in tqdm(sample_files):
        with open(samples_path / pkl, "rb") as f:
            meta_data.append(pickle.load(f))

    r_precision = compute_r_precision(meta_data)
    with open(stat_path, "rb") as f:
        ref_stats = pickle.load(f)
    fid = compute_fid(meta_data, ref_stats)
    # mm_distance = compute_mm_distance(meta_data)
    clip_score = compute_clip_score(meta_data)
    # ape_ave = compute_ape_ave(meta_data)
    # mid = compute_mid(meta_data)  # For MID Computation, Please refer to https://github.com/naver-ai/mid.metric, use the epsilon of 5e-4 as the author suggested.

    return {
        "r_precision": r_precision,
        "fid": fid,
        # "mm_distance": mm_distance,
        "clip_score": clip_score,
        # "mid": mid,
        # "ape_ave": ape_ave,
    }


def compute_ape_ave(meta_data):
    gen_xyz = []
    gt_xyz = []
    gen_root = []
    gt_root = []

    for meta in meta_data:
        gt_root.append(meta["gt_translation"])
        gen_root.append(meta["translation"])

        # meta["gt_joints_xyz"] -> (:,Height axis, :) (X,Z,Y)
        # meta["gt_translation"] -> (:, :, Height axis) (X,Y,Z)
        gt_xyz.append(
            meta["gt_joints_xyz"][:, :, [0, 2, 1]] + meta["gt_translation"][:, None, :]
        )
        gen_xyz.append(
            meta["gen_joints_xyz"][:, :, [0, 2, 1]] + meta["translation"][:, None, :]
        )

    ape_root_joint_error = []
    ape_global_traj_error = []
    ape_global_joint_error = []
    ape_local_joint_error = []

    ave_root_joint_error = []
    ave_global_traj_error = []
    ave_global_joint_error = []
    ave_local_joint_error = []

    for i in range(len(meta_data)):
        num_frames = len(gt_root[i])
        ape_root_joint_error.append(
            np.linalg.norm((gen_root[i] - gt_root[i]), ord=2, axis=1).sum() / num_frames
        )
        ape_global_traj_error.append(
            np.linalg.norm(
                (gen_root[i][:, [0, 1]] - gt_root[i][:, [0, 1]]), ord=1, axis=1
            ).sum()
            / num_frames
        )
        ape_global_joint_error.append(
            np.linalg.norm((gen_xyz[i] - gt_xyz[i]), ord=2, axis=2).mean(axis=1).sum()
            / num_frames
        )
        ape_local_joint_error.append(get_local_position_error(gen_xyz[i], gt_xyz[i]))

        ve_gen_root = ((gen_root[i] - gen_root[i].mean(axis=0)) ** 2).sum(axis=0) / (
            num_frames - 1
        )
        ve_gt_root = ((gt_root[i] - gt_root[i].mean(axis=0)) ** 2).sum(axis=0) / (
            num_frames - 1
        )
        ave_root_joint_error.append(np.linalg.norm(ve_gen_root - ve_gt_root, ord=2))

        ve_gen_global_traj = (
            (gen_root[i][:, [0, 1]] - gen_root[i][:, [0, 1]].mean(axis=0)) ** 2
        ).sum(axis=0) / (num_frames - 1)
        ve_gt_global_traj = (
            (gt_root[i][:, [0, 1]] - gt_root[i][:, [0, 1]].mean(axis=0)) ** 2
        ).sum(axis=0) / (num_frames - 1)
        ave_global_traj_error.append(
            np.linalg.norm(ve_gen_global_traj - ve_gt_global_traj, ord=2)
        )

        ve_gen_joint_global = ((gen_xyz[i] - gen_xyz[i].mean(axis=0)) ** 2).mean(
            axis=1
        ).sum(axis=0) / (num_frames - 1)
        ve_gt_joint_global = ((gt_xyz[i] - gt_xyz[i].mean(axis=0)) ** 2).mean(
            axis=1
        ).sum(axis=0) / (num_frames - 1)
        ave_global_joint_error.append(
            np.linalg.norm(ve_gen_joint_global - ve_gt_joint_global, ord=2)
        )

        ave_local_joint_error.append(get_local_var_error(gen_xyz[i], gt_xyz[i]))

    ape_ave_dict = {
        "ape_root_joint_error": np.mean(ape_root_joint_error),
        "ape_global_traj_error": np.mean(ape_global_traj_error),
        "ape_local_joint_error": np.mean(ape_local_joint_error),
        "ape_global_joint_error": np.mean(ape_global_joint_error),
        "ave_root_joint_error": np.mean(ave_root_joint_error),
        "ave_global_traj_error": np.mean(ave_global_traj_error),
        "ave_local_joint_error": np.mean(ave_local_joint_error),
        "ave_global_joint_error": np.mean(ave_global_joint_error),
    }
    return ape_ave_dict


def get_local_var_error(gen, gt):
    connectivity_map = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 4],
        [2, 5],
        [3, 6],
        [4, 7],
        [5, 8],
        [6, 9],
        [7, 10],
        [8, 11],
        [9, 12],
        [9, 13],
        [9, 14],
        [12, 15],
        [13, 16],
        [14, 17],
        [16, 18],
        [17, 19],
        [18, 20],
        [19, 21],
    ]

    local_var_error = []
    for con in connectivity_map:
        num_frames = len(gt)

        gen_local_joint = gen[:, con[1], :] - gen[:, con[0], :]
        gen_local_joint_mean = gen_local_joint.mean(axis=0)
        sigma_gen_joint = ((gen_local_joint - gen_local_joint_mean) ** 2).sum(
            axis=0
        ) / (num_frames - 1)

        gt_local_joint = gt[:, con[1], :] - gt[:, con[0], :]
        gt_local_joint_mean = gt_local_joint.mean(axis=0)
        sigma_gt_joint = ((gt_local_joint - gt_local_joint_mean) ** 2).sum(axis=0) / (
            num_frames - 1
        )
        local_var_error.append(np.linalg.norm(sigma_gen_joint - sigma_gt_joint, ord=2))
    local_var_error_mean = np.mean(local_var_error)
    return local_var_error_mean


def get_local_position_error(gen, gt):
    L, J, D = gen.shape
    connectivity_map = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 4],
        [2, 5],
        [3, 6],
        [4, 7],
        [5, 8],
        [6, 9],
        [7, 10],
        [8, 11],
        [9, 12],
        [9, 13],
        [9, 14],
        [12, 15],
        [13, 16],
        [14, 17],
        [16, 18],
        [17, 19],
        [18, 20],
        [19, 21],
    ]
    local_error_joints = []
    for con in connectivity_map:
        num_frames = len(gt)
        gen_joint = gen[:, con[1], :] - gen[:, con[0], :]
        gt_joint = gt[:, con[1], :] - gt[:, con[0], :]
        local_error = (
            np.linalg.norm(gen_joint - gt_joint, ord=2, axis=1).sum() / num_frames
        )
        local_error_joints.append(local_error)
    local_error_joints_mean = np.mean(local_error_joints)
    return local_error_joints_mean


def compute_mid(meta_data):
    fake_features = []
    real_features = []
    text_features = []

    for meta in meta_data:
        fake_features.append(meta["gen_motion_feature"])
        real_features.append(meta["gt_motion_feature"])
        text_features.append(meta["gen_text_feature"])

    ff_tensor = torch.Tensor(fake_features)
    rf_tensor = torch.Tensor(real_features)
    tf_tensor = torch.Tensor(text_features)

    mid = MutualInformationDivergence(features=512)
    mid.update(rf_tensor, tf_tensor, ff_tensor)
    mid_val = mid.compute()
    return mid_val.item()


def compute_mm_distance(meta_data):
    mm_distances = []
    mm_distances_norm = []
    for meta in meta_data:
        mm_distances.append(meta["mm_distance"])
        # mm_distances_norm.append(meta["mm_distance_norm"])
    return {
        "mm_distance": np.mean(mm_distances),
        # "mm_distance_norm": np.mean(mm_distances_norm),
    }


def compute_clip_score(meta_data):
    clip_scores = []
    clip_scores_norm = []
    for meta in meta_data:
        clip_scores.append(meta["clip_score"])
        # clip_scores_norm.append(meta["clip_score_norm"])
    return {
        "clip_score": np.mean(clip_scores),
        # "clip_score_norm": np.mean(clip_scores_norm),
    }


def compute_fid(meta_data, ref_stats):
    motion_features = []
    for meta in meta_data:
        motion_features.append(meta["gen_motion_feature"])

    gen_feats = np.stack(motion_features, axis=0)
    gen_mu, gen_cov = compute_mean_covariance(gen_feats)
    gt_mu, gt_cov = ref_stats["gt_mean"], ref_stats["gt_cov"]

    fid = calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)
    return fid


def compute_r_precision(meta_data: List[dict]):
    topk_cnt = {"top-1": 0, "top-2": 0, "top-3": 0}

    for meta in meta_data:
        if meta["top_3_dict"]["top-1"]:
            topk_cnt["top-1"] += 1
            topk_cnt["top-2"] += 1
            topk_cnt["top-3"] += 1
        elif meta["top_3_dict"]["top-2"]:
            topk_cnt["top-2"] += 1
            topk_cnt["top-3"] += 1
        elif meta["top_3_dict"]["top-3"]:
            topk_cnt["top-3"] += 1

    total_samples = len(meta_data)

    rp_dict = {
        "top-1": topk_cnt["top-1"] / total_samples,
        "top-2": topk_cnt["top-2"] / total_samples,
        "top-3": topk_cnt["top-3"] / total_samples,
    }
    return rp_dict


def load_mclip_model(mclip_ckpt_path: str):
    if not os.path.exists(mclip_ckpt_path):
        ValueError("Motion CLIP ckpt does not exist.")
    mclip_model = MotionClipLitModule.load_from_checkpoint(
        checkpoint_path=mclip_ckpt_path
    )
    mclip_model.eval()
    mclip = mclip_model.net
    return mclip


def compute_activations(
    data_loader: DataLoader, model: torch.nn.Module, device: torch.device
):
    model.eval()
    motion_activation_list = []
    with torch.no_grad():
        pbar = tqdm(data_loader, total=len(data_loader))
        for batch in pbar:
            pbar.set_description("Computing feature embeddings...")
            rotation_6d, translation, annotation, motion_length = batch
            N, L, J, D = rotation_6d.shape
            rotation_6d_flat = rotation_6d.reshape(N, L, J * D)
            motion_seq = (
                torch.cat([translation, rotation_6d_flat], dim=2)
                .permute(0, 2, 1)
                .to(device)
            )  # (N, C, L)
            feature_dict = model.get_features(
                motion=motion_seq,
                texts=annotation,
                motion_length=torch.LongTensor(motion_length).to(device),
            )
            gt_motion_feat = feature_dict["motion_feat"]
            motion_activation_list.append(gt_motion_feat.cpu().numpy())

    return np.concatenate(motion_activation_list, axis=0)


def compute_mean_covariance(activations: np.array):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_body_model(root_dir: str, device: torch.device):
    subject_gender = "neutral"
    bm_fname = os.path.join(root_dir, "smpl_model", subject_gender, "model.npz")
    dmpl_fname = os.path.join(root_dir, "dmpl_model", subject_gender, "model.npz")

    num_betas = 16
    num_dmpls = 8

    bm = BodyModel(
        bm_fname=bm_fname,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname,
    ).to(device)
    return bm


def get_joint_xyz(
    bm: BodyModel,
    translation: torch.Tensor,
    rotation_6d: torch.Tensor,
    motion_length: int,
    device: torch.device,
):

    translation = translation.to(device)
    rotation_6d = rotation_6d.to(device)

    smpl_param = matrix_to_axis_angle(
        rotation_6d_to_matrix(rotation_6d.reshape(rotation_6d.size(0), -1, 6))
    )[:motion_length].reshape(motion_length, -1)

    body_params = {
        "root_orient": smpl_param[:, :3],  # controls the global root orientation
        "pose_body": smpl_param[:, 3:66],  # controls the body
        "pose_hand": smpl_param[:, 66:],  # controls the finger articulation
        "trans": translation,  # controls the global body position
    }

    body_pose = bm(
        **{k: v for k, v in body_params.items() if k in ["pose_body", "betas"]}
    )
    # https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf
    valid_joints = body_pose.Jtr[:, :22, :].cpu().numpy()
    return valid_joints
