import os
from typing import Tuple, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from src.config import TRAIN_DIR, VAL_DIR, TEST_DIR


class EmbryoImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, class_to_idx: Dict[str, int] | None = None):
        self.root_dir = root_dir
        self.transform = transform

        classes = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )

        if not classes:
            raise RuntimeError(f"No class folders found in {root_dir}")

        if class_to_idx is None:
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.samples = []
        for cls_name in classes:
            class_index = self.class_to_idx[cls_name]
            class_dir = os.path.join(root_dir, cls_name)
            for root, _, files in os.walk(class_dir):
                for f in files:
                    path = os.path.join(root, f)
                    if self._is_image_file(path):
                        self.samples.append((path, class_index))

        if not self.samples:
            raise RuntimeError(f"No image files found in {root_dir}")

    @staticmethod
    def _is_image_file(path: str) -> bool:
        ext = os.path.splitext(path.lower())[1]
        return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, target = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def create_transforms(image_size: int = 224):
    train_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, eval_transform


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    train_transform, eval_transform = create_transforms(image_size=image_size)

    train_dataset = EmbryoImageDataset(root_dir=TRAIN_DIR, transform=train_transform)
    class_to_idx = train_dataset.class_to_idx

    val_dataset = EmbryoImageDataset(
        root_dir=VAL_DIR, transform=eval_transform, class_to_idx=class_to_idx
    )
    test_dataset = EmbryoImageDataset(
        root_dir=TEST_DIR, transform=eval_transform, class_to_idx=class_to_idx
    )

    if use_weighted_sampler:
        # Compute class-balanced sampling weights for training set
        train_labels = [target for _, target in train_dataset.samples]
        num_classes = len(class_to_idx)

        label_tensor = torch.tensor(train_labels, dtype=torch.long)
        class_counts = torch.bincount(label_tensor, minlength=num_classes).float()
        class_weights = len(train_labels) / (num_classes * class_counts)

        sample_weights = class_weights[label_tensor]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, class_to_idx