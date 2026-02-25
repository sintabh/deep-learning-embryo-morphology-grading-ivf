import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.config import MODELS_DIR, DEVICE
from src.models.multitask_model import MultiTaskEmbryoNet
from src.utils.multitask_data_loader import create_multitask_dataloaders


def compute_class_weights(
    loader: DataLoader,
    num_expansion_classes: int,
    num_icm_classes: int,
    num_te_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    exp_counts = torch.zeros(num_expansion_classes, dtype=torch.float32)
    icm_counts = torch.zeros(num_icm_classes, dtype=torch.float32)
    te_counts = torch.zeros(num_te_classes, dtype=torch.float32)

    with torch.no_grad():
        for _, targets in loader:
            # targets shape: (N, 3)
            exp = targets[:, 0]
            icm = targets[:, 1]
            te = targets[:, 2]

            for c in exp:
                exp_counts[int(c)] += 1.0
            for c in icm:
                icm_counts[int(c)] += 1.0
            for c in te:
                te_counts[int(c)] += 1.0

    def inverse_freq(counts: torch.Tensor) -> torch.Tensor:
        num_classes = counts.numel()
        total = counts.sum().item()
        # avoid division by zero for classes that never appear
        weights = total / (num_classes * (counts + 1e-6))
        return weights

    exp_weights = inverse_freq(exp_counts)
    icm_weights = inverse_freq(icm_counts)
    te_weights = inverse_freq(te_counts)

    max_w = 5.0
    exp_weights = torch.clamp(exp_weights, max=max_w)
    icm_weights = torch.clamp(icm_weights, max=max_w)
    te_weights = torch.clamp(te_weights, max=max_w)

    return exp_weights, icm_weights, te_weights

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_exp: nn.Module,
    criterion_icm: nn.Module,
    criterion_te: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[float, dict]:
    model.train()
    running_loss = 0.0
    n_samples = 0

    w_exp, w_icm, w_te = task_weights

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)  # shape: (N, 3)
        exp_targets = targets[:, 0]
        icm_targets = targets[:, 1]
        te_targets = targets[:, 2]

        optimizer.zero_grad()

        logits_exp, logits_icm, logits_te = model(images)

        loss_exp = criterion_exp(logits_exp, exp_targets)
        loss_icm = criterion_icm(logits_icm, icm_targets)
        loss_te = criterion_te(logits_te, te_targets)

        loss = w_exp * loss_exp + w_icm * loss_icm + w_te * loss_te
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    epoch_loss = running_loss / max(n_samples, 1)

    return epoch_loss, {"loss": epoch_loss}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_exp: nn.Module,
    criterion_icm: nn.Module,
    criterion_te: nn.Module,
    device: torch.device,
    task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[float, dict]:
    model.eval()
    running_loss = 0.0
    n_samples = 0

    correct_exp = 0
    correct_icm = 0
    correct_te = 0

    total = 0

    w_exp, w_icm, w_te = task_weights

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        exp_targets = targets[:, 0]
        icm_targets = targets[:, 1]
        te_targets = targets[:, 2]

        logits_exp, logits_icm, logits_te = model(images)

        loss_exp = criterion_exp(logits_exp, exp_targets)
        loss_icm = criterion_icm(logits_icm, icm_targets)
        loss_te = criterion_te(logits_te, te_targets)

        loss = w_exp * loss_exp + w_icm * loss_icm + w_te * loss_te

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        _, pred_exp = torch.max(logits_exp, dim=1)
        _, pred_icm = torch.max(logits_icm, dim=1)
        _, pred_te = torch.max(logits_te, dim=1)

        correct_exp += (pred_exp == exp_targets).sum().item()
        correct_icm += (pred_icm == icm_targets).sum().item()
        correct_te += (pred_te == te_targets).sum().item()

        total += batch_size

    epoch_loss = running_loss / max(n_samples, 1)
    acc_exp = correct_exp / max(total, 1)
    acc_icm = correct_icm / max(total, 1)
    acc_te = correct_te / max(total, 1)

    metrics = {
        "loss": epoch_loss,
        "acc_exp": acc_exp,
        "acc_icm": acc_icm,
        "acc_te": acc_te,
    }

    return epoch_loss, metrics


def train_multitask(
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_workers: int = 4,
    image_size: int = 224,
    task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    device = DEVICE

    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_multitask_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )

    num_expansion_classes = 5
    num_icm_classes = 4
    num_te_classes = 4

    print("Computing class weights from training data...")
    exp_w, icm_w, te_w = compute_class_weights(
        train_loader,
        num_expansion_classes=num_expansion_classes,
        num_icm_classes=num_icm_classes,
        num_te_classes=num_te_classes,
    )
    print("Expansion weights:", exp_w.tolist())
    print("ICM weights:", icm_w.tolist())
    print("TE weights:", te_w.tolist())

    model = MultiTaskEmbryoNet(
        num_expansion_classes=num_expansion_classes,
        num_icm_classes=num_icm_classes,
        num_te_classes=num_te_classes,
        pretrained=True,
    ).to(device)

    criterion_exp = nn.CrossEntropyLoss(weight=exp_w.to(device))
    criterion_icm = nn.CrossEntropyLoss(weight=icm_w.to(device))
    criterion_te = nn.CrossEntropyLoss(weight=te_w.to(device))

    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model_path = os.path.join(MODELS_DIR, "best_multitask_efficientnet_b0.pth")

    for epoch in range(1, num_epochs + 1):
        train_loss, _ = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion_exp=criterion_exp,
            criterion_icm=criterion_icm,
            criterion_te=criterion_te,
            optimizer=optimizer,
            device=device,
            task_weights=task_weights,
        )

        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion_exp=criterion_exp,
            criterion_icm=criterion_icm,
            criterion_te=criterion_te,
            device=device,
            task_weights=task_weights,
        )

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc EXP: {val_metrics['acc_exp']:.4f} | "
            f"Val Acc ICM: {val_metrics['acc_icm']:.4f} | "
            f"Val Acc TE: {val_metrics['acc_te']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(
                f"New best multitask model saved to {best_model_path} "
                f"(val_loss={best_val_loss:.4f})"
            )

    print("\nEvaluating best model on test set...")
    best_model = MultiTaskEmbryoNet(
        num_expansion_classes=num_expansion_classes,
        num_icm_classes=num_icm_classes,
        num_te_classes=num_te_classes,
        pretrained=False,
    ).to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, test_metrics = evaluate(
        model=best_model,
        loader=test_loader,
        criterion_exp=criterion_exp,
        criterion_icm=criterion_icm,
        criterion_te=criterion_te,
        device=device,
        task_weights=task_weights,
    )

    print(
        f"Test Acc EXP: {test_metrics['acc_exp']:.4f} | "
        f"Test Acc ICM: {test_metrics['acc_icm']:.4f} | "
        f"Test Acc TE: {test_metrics['acc_te']:.4f}"
    )


if __name__ == "__main__":
    train_multitask(num_epochs=8)