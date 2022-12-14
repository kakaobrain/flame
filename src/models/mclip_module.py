import torch
from pytorch_lightning import LightningModule


class MotionClipLitModule(LightningModule):
    """Motion Clip to inject motion information on CLIP"""

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.0001,
        weight_decay: float = 0.0,
    ):

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        batch_size = len(batch[0])
        loss = self._step_network("train", batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(batch[0])
        loss = self._step_network("val", batch, batch_idx)
        self.log("val/loss", loss)
        return loss

    def _step_network(self, split: str, batch, batch_idx):
        rotation_6d, translation, annotation, motion_length = batch
        N, L, J, D = rotation_6d.shape
        rotation_6d_flat = rotation_6d.reshape(N, L, J * D)
        motion_seq = torch.cat([translation, rotation_6d_flat], dim=2).permute(
            0, 2, 1
        )  # (N, C, L)
        loss = self.net.compute_loss(
            motion=motion_seq, texts=annotation, motion_length=motion_length
        )
        return loss
