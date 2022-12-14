from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.datamodules.components.babel import BABELDataset
from torch.utils.data import ConcatDataset
from typing import List
from pathlib import Path
from src.utils.collate import collate_motion


class HumanML3DDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 16,
        pin_memory: bool = False,
        split_setting: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dir = Path(data_dir) / f"HumanML3D" / "processed" / f"train_data"
        self.train_file_list_dir = (
            Path(data_dir) / f"HumanML3D" / "processed" / f"train_list"
        )

        self.val_dir = Path(data_dir) / f"HumanML3D" / "processed" / f"val_data"
        self.val_file_list_dir = (
            Path(data_dir) / f"HumanML3D" / "processed" / f"val_list"
        )

        self.test_dir = Path(data_dir) / f"HumanML3D" / "processed" / f"test_data"
        self.test_file_list_dir = (
            Path(data_dir) / f"HumanML3D" / "processed" / f"test_list"
        )

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: None):

        train_dataset = BABELDataset(self.train_dir, self.train_file_list_dir)

        val_dataset = BABELDataset(self.val_dir, self.val_file_list_dir)

        test_dataset = BABELDataset(self.test_dir, self.test_file_list_dir)

        if self.hparams.split_setting == "split":
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
        elif self.hparams.split_setting == "final":
            self.train_dataset = ConcatDataset([train_dataset, val_dataset])
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
        elif self.hparams.split_setting == "demo":
            self.train_dataset = ConcatDataset(
                [train_dataset, val_dataset, test_dataset]
            )
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
        else:
            ValueError("Undefined split setting")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_motion,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_motion,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_motion,
        )
