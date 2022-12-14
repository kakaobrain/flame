import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import List

import torch
import yaml
from tqdm.auto import tqdm
from hydra.utils import get_original_cwd
from pytorch_lightning import LightningModule
from scipy.ndimage import gaussian_filter1d
from torch.nn import functional as F

from src.models.components.gaussian_diffusion import get_named_beta_schedule
from src.models.components.resample import create_named_schedule_sampler
from src.models.components.respace import SpacedDiffusion, space_timesteps
from src.models.ema import EMAModel
from src.utils.eval_util import (
    compute_activations,
    compute_mean_covariance,
    compute_metrics,
    load_mclip_model,
)
from src.utils.misc import replace_annotation_with_null
from src.utils.sample_util import get_sample_fn, sample_motion
from src.utils.vis_util import render_video_summary


class DiffusionLitModule(LightningModule):
    """Improved DDPM Model"""

    def __init__(
        self,
        net: torch.nn.Module,
        diffusion_steps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
        schedule_sampler: str = "uniform",
        text_replace_prob: float = 0.25,
        lr: float = 0.0001,
        weight_decay: float = 0.0,
        ema_start: int = 1000,
        ema_update: int = 100,
        ema_decay: float = 0.99,
        sampling_strategy: str = "ddpm",
        epoch_sample_every: int = 200,
        sample_fps: int = 20,
        sampling_length: int = 200,
        sampling_texts: List[str] = [""],
        guidance_scale: float = 8.0,
    ):

        super().__init__()
        self.save_hyperparameters(logger=False)
        self.hparams.sampling_texts = list(self.hparams.sampling_texts)

        self.net = net

        self.diffusion = SpacedDiffusion(
            model=self.net,
            betas=get_named_beta_schedule(beta_schedule, diffusion_steps),
            use_timesteps=space_timesteps(diffusion_steps, [diffusion_steps]),
        )

        self.schedule_sampler = create_named_schedule_sampler(
            schedule_sampler, self.diffusion
        )
        self.ema_model = EMAModel(model=self.net, decay=self.hparams.ema_decay)

    def step_ema(self):
        if self.global_step <= self.hparams.ema_start:
            self.ema_model.set(self.net)
        else:
            self.ema_model.update(self.net)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def _step_network(self, split: str, batch, batch_idx):
        rotation_6d, translation, annotation, motion_length = batch
        N, L, J, D = rotation_6d.shape
        replaced_annotation = replace_annotation_with_null(
            annotation=annotation, replace_prob=self.hparams.text_replace_prob
        )
        rotation_6d_flat = rotation_6d.reshape(N, L, J * D)

        # Concatenate translation and rotaiton. Note that translation goes first.
        motion_seq = torch.cat([translation, rotation_6d_flat], dim=2).permute(
            0, 2, 1
        )  # (N, C, L)
        t_indices, weights = self.schedule_sampler.sample(N, self.device)
        losses = self.diffusion.compute_loss(
            motion_seq,
            motion_length=motion_length,
            t=t_indices,
            annotation=replaced_annotation,
        )
        losses["weighted_loss"] = losses["loss"] * weights
        return losses

    def training_step(self, batch, batch_idx):
        batch_size = len(batch[0])
        losses = self._step_network("train", batch, batch_idx)
        total_loss = losses["weighted_loss"].mean()
        self.log("train/loss", total_loss, batch_size=batch_size)
        self.log(
            "train/unweighted/vb", losses["vb"].mean().item(), batch_size=batch_size
        )
        self.log(
            "train/unweighted/mse", losses["mse"].mean().item(), batch_size=batch_size
        )
        self.log(
            "train/unweighted/loss", losses["loss"].mean().item(), batch_size=batch_size
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(batch[0])
        losses = self._step_network("val", batch, batch_idx)
        total_loss = losses["weighted_loss"].mean()
        self.log("val/loss", total_loss, batch_size=batch_size)
        self.log("val/unweighted/vb", losses["vb"].mean().item(), batch_size=batch_size)
        self.log(
            "val/unweighted/mse", losses["mse"].mean().item(), batch_size=batch_size
        )
        self.log(
            "val/unweighted/loss", losses["loss"].mean().item(), batch_size=batch_size
        )
        return losses

    def test_step(self, batch, batch_idx):
        batch_size = len(batch[0])
        rotation_6d, translation, annotation, motion_length = batch
        N, L, J, D = rotation_6d.shape
        rotation_6d_flat = rotation_6d.reshape(N, L, J * D)
        motion_seq = torch.cat([translation, rotation_6d_flat], dim=2).permute(
            0, 2, 1
        )  # (N, C, L)

        device_id = self.device.index
        self.ema_model.eval()
        guidance_scale = self.test_settings["guidance_scale"]
        sample_fn = get_sample_fn(
            strategy=self.hparams.sampling_strategy, diffusion_model=self.diffusion
        )

        gt_clip_out = self.mclip_model.get_features(
            motion=motion_seq,
            texts=list(annotation),
            motion_length=motion_length,
        )

        for sample_idx, (gt_rot, gt_trs, gt_ann, gt_len) in enumerate(
            zip(rotation_6d, translation, annotation, motion_length)
        ):
            generated_motion = sample_motion(
                sampling_texts=[gt_ann],
                motion_lengths=[gt_len.item()],
                sample_fn=sample_fn,
                ema_model=self.ema_model.model,
                device=self.device,
                motion_dim=self.net.motion_dim,
                guidance_scale=guidance_scale,
                progress=False,
            )

            generated_motion_smooth = torch.Tensor(
                gaussian_filter1d(generated_motion.cpu().numpy(), sigma=1, axis=1)
            ).to(self.device)

            trs = generated_motion_smooth[0, :gt_len, :3]
            rot = generated_motion_smooth[0, :gt_len, 3:]

            clip_motion_in = generated_motion_smooth.permute(0, 2, 1).repeat(
                batch_size, 1, 1
            )  # (B, C, L), copy generated motions
            text_candidates = self.test_annotations.copy()
            wrong_answers = random.sample(
                set(text_candidates) - set([gt_ann]), batch_size - 1
            )  # sample w/o replacement.
            full_candidates = [gt_ann] + wrong_answers  # index 0 is an answer.

            clip_out = self.mclip_model.get_features(
                motion=clip_motion_in,
                texts=full_candidates,
                motion_length=torch.LongTensor([gt_len.item()] * batch_size).to(
                    self.device
                ),
            )  # Real length setting
            clip_scores = F.cosine_similarity(
                clip_out["motion_feat"], clip_out["text_feat"]
            )

            # For R Precision
            euc_distance = torch.norm(
                clip_out["motion_feat"][0:1].repeat(batch_size, 1)
                - clip_out["text_feat"],
                p="fro",
                dim=1,
            )
            top3_euc = torch.topk(euc_distance, k=3, largest=False)
            top3_dict = {
                f"top-{k+1}": (0 == idx).item()
                for k, idx in enumerate(top3_euc.indices)
            }

            export_dict = {
                "given_label": gt_ann,
                "translation": trs.cpu().numpy(),
                "rotation": rot.cpu().numpy(),
                "motion_seq": generated_motion_smooth.cpu().numpy(),
                "guidance_scale": guidance_scale,
                "clip_score": clip_scores[0].item(),
                "top_3_dict": top3_dict,
                "mm_distance": euc_distance[0].item(),
                "gen_motion_feature": clip_out["motion_feat"][0].cpu().numpy(),
                "gen_motion_feature_l2norm": clip_out["motion_feat_l2_norm"][0]
                .cpu()
                .numpy(),
                "gen_text_feature": clip_out["text_feat"][0].cpu().numpy(),
                "gen_text_feature_l2norm": clip_out["text_feat_l2_norm"][0]
                .cpu()
                .numpy(),
                "gt_motion_feature": gt_clip_out["motion_feat"][sample_idx]
                .cpu()
                .numpy(),
                "gt_motion_feature_l2norm": gt_clip_out["motion_feat_l2_norm"][
                    sample_idx
                ]
                .cpu()
                .numpy(),
                "gt_text_feature": gt_clip_out["text_feat"][sample_idx].cpu().numpy(),
                "gt_text_feature_l2norm": gt_clip_out["text_feat_l2_norm"][sample_idx]
                .cpu()
                .numpy(),
            }

            file_name = f"dev_{device_id}_bid_{batch_idx}_sid_{sample_idx}.pkl"
            with open(os.path.join(self.output_dir, file_name), "wb") as f:
                pickle.dump(export_dict, f)

    def on_test_start(self):

        project_dir = Path(get_original_cwd())

        with open(project_dir / "configs" / "test.yaml") as f:
            self.test_settings = yaml.safe_load(f)

        # Load motion clip model
        mclip_ckpt = self.test_settings["clip_ckpt"]
        mclip = load_mclip_model(mclip_ckpt)
        mclip.to(self.device)
        self.mclip_model = mclip
        logging.info(f"Motion CLIP is loaded from {mclip_ckpt}")

        # Prepare dir for generated motions
        output_dir = Path(os.getcwd()) / "generated_motions"
        output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir = output_dir

        # Prepare stat cache dir
        stats_dir = Path(os.getcwd()) / "stats"
        stats_path = stats_dir / "stats.pkl"
        self.stats_path = stats_path

        motion_acts = compute_activations(
            self.trainer.datamodule.test_dataloader(), mclip, device=self.device
        )
        gt_mean, gt_cov = compute_mean_covariance(motion_acts)
        eval_stats = {
            "gt_mean": gt_mean,
            "gt_cov": gt_cov,
        }
        stats_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "wb") as f:
            pickle.dump(eval_stats, f)
        logging.info(f"stats are saved to {stats_path}")

        self.eval_stats = eval_stats

        full_test_anns = []
        pbar = tqdm(
            self.trainer.datamodule.test_dataloader(),
            total=len(self.trainer.datamodule.test_dataloader()),
        )
        for batch in pbar:
            full_test_anns.extend(batch[2])

        self.test_annotations = full_test_anns

    def on_test_end(self):
        metrics = compute_metrics(meta_dir=self.output_dir, stat_path=self.stats_path)
        result_file = Path(os.getcwd()) / "eval.json"
        with open(result_file, "w") as f:
            json.dump(metrics, f)

    def on_train_batch_end(self, outputs, batch, batch_idx):

        if self.global_step % self.hparams.ema_update == 0:
            self.step_ema()

    def on_train_epoch_end(self):

        if self.current_epoch % self.hparams.epoch_sample_every == 0:

            num_samples = len(self.hparams.sampling_texts)

            sample_fn = get_sample_fn(
                strategy=self.hparams.sampling_strategy, diffusion_model=self.diffusion
            )

            generated_motions = sample_motion(
                sampling_texts=self.hparams.sampling_texts,
                motion_lengths=[self.hparams.sampling_length] * num_samples,
                sample_fn=sample_fn,
                ema_model=self.ema_model.model,
                device=self.device,
                motion_dim=self.net.motion_dim,
                guidance_scale=self.hparams.guidance_scale,
                progress=False,
            )  # This will output (Batch, Motion Length, Representation dim)

            video_summary = render_video_summary(
                img_ids=[f"{self.global_step}_{i}" for i in range(num_samples)],
                translations=generated_motions[:, :, :3],
                rotation_6ds=generated_motions[:, :, 3:],
                annotations=self.hparams.sampling_texts,
                fps=self.hparams.sample_fps,
                write=False,
            )
            self.logger.experiment.add_video(
                f"motion", video_summary, self.global_step, fps=self.hparams.sample_fps
            )
