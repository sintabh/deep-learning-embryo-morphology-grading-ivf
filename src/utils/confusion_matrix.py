import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.config import TEST_DIR, FIGURES_DIR, MODELS_DIR
from src.models.cnn_model import create_model
from src.utils.data_loader import EmbryoImageDataset, create_transforms


def load_checkpoint(model_name: str = "efficientnet_b0", device_str: str | None = None) -> Dict:
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(MODELS_DIR, f"best_{model_name}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    return {"checkpoint": checkpoint, "device": device}


def build_model_from_checkpoint(ckpt: Dict, device: torch.device) -> tuple[nn.Module, Dict[str, int]]:
    checkpoint = ckpt
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    class_to_idx: Dict[str, int] = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    model = create_model(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_to_idx


def create_test_loader(class_to_idx: Dict[str, int], batch_size: int = 32, num_workers: int = 4, image_size: int = 224) -> DataLoader:
    _, eval_transform = create_transforms(image_size=image_size)
    test_dataset = EmbryoImageDataset(
        root_dir=TEST_DIR,
        transform=eval_transform,
        class_to_idx=class_to_idx,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return test_loader


def evaluate_on_test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[List[int], List[int]]:
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    return all_targets, all_preds


def plot_and_save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_to_idx: Dict[str, int],
    model_name: str = "efficientnet_b0",
) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    labels = sorted(idx_to_class.keys())
    display_labels = [idx_to_class[i] for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title(f"Confusion Matrix ({model_name})")
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {out_path}")


def main(
    model_name: str = "efficientnet_b0",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    device_str: str | None = None,
) -> None:
    ckpt_and_device = load_checkpoint(model_name=model_name, device_str=device_str)
    checkpoint = ckpt_and_device["checkpoint"]
    device = ckpt_and_device["device"]

    model, class_to_idx = build_model_from_checkpoint(checkpoint, device)
    test_loader = create_test_loader(
        class_to_idx=class_to_idx,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )

    y_true, y_pred = evaluate_on_test(model, test_loader, device)
    plot_and_save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_to_idx=class_to_idx,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()