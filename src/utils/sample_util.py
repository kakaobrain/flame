from typing import List, Callable, Union
from src.models.components.gaussian_diffusion import GaussianDiffusion
from src.models.components.respace import SpacedDiffusion
from src.utils.misc import lengths_to_mask
import torch


def edit_noise(x_t: torch.Tensor, cond: torch.Tensor, cond_info: dict):

    L, J, D = cond_info["ref_rotation_6d"].shape
    B = 2  # (sampling support one-by-one)

    if cond_info["use_reference_translation"]:
        x_t[:, :3, :] = cond[:, :3, :]

    x_t_rot_part = x_t[:, 3:, :].reshape(B, J, D, L)
    ref_rot_part = cond[:, 3:, :].reshape(B, J, D, L)

    for joint_ind in range(J):
        if joint_ind not in cond_info["edit_joints_indices"]:
            x_t_rot_part[:, joint_ind, :, :] = ref_rot_part[:, joint_ind, :, :]

    return x_t


def get_sample_fn(
    strategy: str, diffusion_model: Union[SpacedDiffusion, GaussianDiffusion]
):

    if strategy == "ddpm":
        sample_fn = diffusion_model.p_sample_loop
    elif strategy == "ddim":
        sample_fn = diffusion_model.ddim_sample_loop
    else:
        ValueError("Sampling strategy does not exist.")
    return sample_fn


def sample_motion(
    sampling_texts: List[str],
    motion_lengths: List[int],
    sample_fn: Callable,
    ema_model: Union[SpacedDiffusion, GaussianDiffusion],
    device: torch.device,
    motion_dim: int = 147,
    guidance_scale: float = 8.0,
    progress=True,
    conditioning_info=None,
):
    ema_model.eval()
    num_samples = len(sampling_texts)

    # Noise setting for classifier-free generation
    N = 2 * num_samples

    conditioning_texts = sampling_texts
    unconditional_texts = [""] * len(conditioning_texts)
    sampling_texts = conditioning_texts + unconditional_texts

    model_kwargs = {"texts": sampling_texts, "conditioning_info": conditioning_info}

    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        motion_length_cond = torch.LongTensor(motion_lengths * 2).to(x_t.device)
        mask = lengths_to_mask(motion_length_cond, x_t.device)
        assert mask.all()
        model_out = ema_model(
            combined,
            motion_length=motion_length_cond,
            timesteps=ts,
            texts=kwargs["texts"],
            mask=mask,
        )
        eps, rest = (
            model_out[:, :motion_dim],
            model_out[:, motion_dim:],
        )
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    noise_shape = (N, motion_dim, motion_lengths[0])

    sampled_motion = sample_fn(
        model=model_fn,
        shape=noise_shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=model_kwargs,
        device=device,
        progress=progress,
    )[:num_samples]

    motion_vis = sampled_motion.permute(0, 2, 1)  # (N,C,L) -> (N,L,C)
    return motion_vis
