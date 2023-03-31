import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CelebA
from config import DATA_DIR


class LightingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, shuffle=False)

    def _dataloader(
        self,
        dataset: Dataset,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )


def get_dataset(transform, path=None):
    if path is None:
        path = DATA_DIR
    os.makedirs(path, exist_ok=True)
    train_data = CelebA(
        root=path, split="train", download=True, transform=transform
    )
    test_data = CelebA(
        root=path, split="test", download=True, transform=transform
    )
    return train_data, test_data


def get_celeba_module(
    transform,
    path=None,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    subset_train = None,
    subset_test = None
):
    if path is None:
        path = DATA_DIR
    train_data, test_data = get_dataset(transform, path=path)
    if subset_train is not None:
        train_data = make_subset(subset_train, train_data)
    if subset_test is not None:
        test_data = make_subset(subset_test, test_data)
    return LightingDataModule(
        train_data,
        test_data,
        batch_size,
        num_workers,
        pin_memory,
    )

def make_subset(subset, data):
    print('getting subset')
    if isinstance(subset, int):
        subset = range(subset)
    data = Subset(data, subset)
    return data
