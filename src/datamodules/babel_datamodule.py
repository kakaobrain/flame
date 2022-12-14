from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.datamodules.components.amass import AMASSDataset
from src.datamodules.components.babel import BABELDataset
from torch.utils.data import ConcatDataset
from typing import List
from pathlib import Path


class BABELDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        babel_setting: str,
        batch_size: int = 128,
        num_workers: int = 16,
        pin_memory: bool = False,
        split_setting: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.babel_train_dir = Path(data_dir) / f"babel_{babel_setting}" / f"train_data"
        self.babel_train_file_list_dir = (
            Path(data_dir) / f"babel_{babel_setting}" / f"train_list"
        )
        self.babel_val_dir = Path(data_dir) / f"babel_{babel_setting}" / f"val_data"
        self.babel_val_file_list_dir = (
            Path(data_dir) / f"babel_{babel_setting}" / f"val_list"
        )
        self.babel_test_dir = Path(data_dir) / f"babel_{babel_setting}" / f"test_data"
        self.babel_test_file_list_dir = (
            Path(data_dir) / f"babel_{babel_setting}" / f"test_list"
        )

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: None):

        train_dataset = BABELDataset(
            self.babel_train_dir, self.babel_train_file_list_dir
        )
        val_dataset = BABELDataset(self.babel_val_dir, self.babel_val_file_list_dir)
        test_dataset = BABELDataset(self.babel_test_dir, self.babel_test_file_list_dir)

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
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
        )
