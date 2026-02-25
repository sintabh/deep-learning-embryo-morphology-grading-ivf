import os
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.config import RAW_DATA_DIR
from src.utils.annotation_utils import load_multitask_annotations


RAW_IMAGE_DIR = os.path.join(RAW_DATA_DIR, "Human_Blastocyst_Dataset")


class MultiTaskEmbryoDataset(Dataset):
    """
    Dataset for multi-task embryo grading.

    Each sample returns:
        - image tensor
        - target tensor: [expansion_int, icm_int, te_int]
    """

    def __init__(self, split: str, transform=None) -> None:
        super().__init__()

        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split for multitask dataset: {split}")

        if not os.path.isdir(RAW_IMAGE_DIR):
            raise FileNotFoundError(f"Raw image directory not found: {RAW_IMAGE_DIR}")

        self.split = split
        self.transform = transform

        df = load_multitask_annotations(split)
        self.image_names = df["image"].tolist()
        self.expansion_labels = df["expansion_int"].tolist()
        self.icm_labels = df["icm_int"].tolist()
        self.te_labels = df["te_int"].tolist()

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.image_names[idx]
        img_path = os.path.join(RAW_IMAGE_DIR, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        exp = self.expansion_labels[idx]
        icm = self.icm_labels[idx]
        te = self.te_labels[idx]

        targets = torch.tensor([exp, icm, te], dtype=torch.long)

        return image, targets