import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.config import MODELS_DIR, DEVICE, REPORTS_DIR
from src.models.multitask_model import MultiTaskEmbryoNet
from src.utils.multitask_data_loader import create_multitask_dataloaders


def _load_best_multitask_model() -> nn.Module:
    model = MultiTaskEmbryoNet(
        num_expansion_classes=5,
        num_icm_classes=4,
        num_te_classes=4,
        pretrained=False,
    ).to(DEVICE)

    ckpt_path = os.path.join(MODELS_DIR, "best_multitask_efficientnet_b0.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def _collect_predictions(
    model: nn.Module, loader: DataLoader
) -> Tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    y_exp_true, y_exp_pred = [], []
    y_icm_true, y_icm_pred = [], []
    y_te_true, y_te_pred = [], []

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)  # (N, 3)
        exp_t = targets[:, 0]
        icm_t = targets[:, 1]
        te_t = targets[:, 2]

        logits_exp, logits_icm, logits_te = model(images)

        _, exp_p = torch.max(logits_exp, dim=1)
        _, icm_p = torch.max(logits_icm, dim=1)
        _, te_p = torch.max(logits_te, dim=1)

        y_exp_true.extend(exp_t.cpu().tolist())
        y_exp_pred.extend(exp_p.cpu().tolist())

        y_icm_true.extend(icm_t.cpu().tolist())
        y_icm_pred.extend(icm_p.cpu().tolist())

        y_te_true.extend(te_t.cpu().tolist())
        y_te_pred.extend(te_p.cpu().tolist())

    return (
        y_exp_true,
        y_exp_pred,
        y_icm_true,
        y_icm_pred,
        y_te_true,
        y_te_pred,
    )


def _save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    labels: list[int],
    title: str,
    filename: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=None)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(title)
    plt.tight_layout()

    figures_dir = os.path.join(REPORTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, filename)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"{title} saved to {out_path}")


def main() -> None:
    print(f"Using device: {DEVICE}")

    # We only need the test loader here
    _, _, test_loader = create_multitask_dataloaders(
        batch_size=32,
        num_workers=4,
        image_size=224,
    )

    model = _load_best_multitask_model()

    (
        y_exp_true,
        y_exp_pred,
        y_icm_true,
        y_icm_pred,
        y_te_true,
        y_te_pred,
    ) = _collect_predictions(model, test_loader)

    # Expansion: grades 0-4
    _save_confusion_matrix(
        y_true=y_exp_true,
        y_pred=y_exp_pred,
        labels=[0, 1, 2, 3, 4],
        title="Confusion Matrix - Expansion",
        filename="confusion_multitask_expansion.png",
    )

    # ICM: grades 0-3
    _save_confusion_matrix(
        y_true=y_icm_true,
        y_pred=y_icm_pred,
        labels=[0, 1, 2, 3],
        title="Confusion Matrix - ICM",
        filename="confusion_multitask_icm.png",
    )

    # TE: grades 0-3
    _save_confusion_matrix(
        y_true=y_te_true,
        y_pred=y_te_pred,
        labels=[0, 1, 2, 3],
        title="Confusion Matrix - TE",
        filename="confusion_multitask_te.png",
    )


if __name__ == "__main__":
    main()