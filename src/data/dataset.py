import os
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets

from src.data.transforms import get_transforms


def get_datasets(data_dir: str):
    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=eval_transform
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, class_names