from typing import Optional
import numpy as np
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
)
from src import utils

log = utils.get_logger(__name__)


@hydra.main(config_path="configs/", config_name="train.yaml")
def inspect_samples(config: DictConfig) -> Optional[float]:
    """
    It visualizes samples in training dataset.
    """

    sampling_annotation = "kick"

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup(stage=None)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    train_dataloader = datamodule.train_dataloader()

    for batch_idx, batch in enumerate(train_dataloader):
        rotation_6d, translation, annotation = batch
        annotation = np.array(annotation)
        sampling_ind = annotation == sampling_annotation

        if (sampling_ind == False).all():
            continue
        rotation_cond = rotation_6d[sampling_ind].reshape(
            rotation_6d[sampling_ind].shape[0], rotation_6d[sampling_ind].shape[1], -1
        )
        translation_cond = translation[sampling_ind]
        annotation_cond = [sampling_annotation] * len(rotation_cond)

        img_ids = [f"bid_{batch_idx}_{i}" for i in range(len(rotation_cond))]
        model.render_motions(
            img_ids=img_ids,
            translations=translation_cond,
            rotation_6ds=rotation_cond,
            annotations=annotation_cond,
            write=True,
        )


if __name__ == "__main__":
    inspect_samples()
