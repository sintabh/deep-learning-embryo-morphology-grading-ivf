import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms

from src.utils.multitask_dataset import MultiTaskEmbryoDataset
from src.config import RAW_DATA_DIR


RAW_IMAGE_DIR = os.path.join(RAW_DATA_DIR, "Human_Blastocyst_Dataset")


def _build_transforms(image_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
        ),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    return train_transform, eval_transform


def create_multitask_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for multi-task embryo grading.

    Train is split into train/val with stratification based on a simple
    combined label (expansion, icm, te).
    """
    if not os.path.isdir(RAW_IMAGE_DIR):
        raise FileNotFoundError(f"Raw image directory not found: {RAW_IMAGE_DIR}")

    train_transform, eval_transform = _build_transforms(image_size=image_size)

    full_train_ds = MultiTaskEmbryoDataset(split="train", transform=train_transform)
    test_ds = MultiTaskEmbryoDataset(split="test", transform=eval_transform)

    # Build a simple combined label for stratification: (exp, icm, te) -> int
    targets = []
    for _, t in full_train_ds:
        exp, icm, te = t.tolist()
        combined = exp * 100 + icm * 10 + te
        targets.append(combined)

    indices = list(range(len(full_train_ds)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=None,  # no stratification because some classes are too rare
    )

    train_subset = Subset(full_train_ds, train_idx)
    val_subset = Subset(
        MultiTaskEmbryoDataset(split="train", transform=eval_transform),
        val_idx,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader