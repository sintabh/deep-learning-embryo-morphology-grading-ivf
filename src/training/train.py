import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from src.config import MODELS_DIR
from src.utils.data_loader import create_dataloaders
from src.models.cnn_model import create_model


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == targets).item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == targets).item()
            total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    return epoch_loss, epoch_acc


def train_model(
    model_name: str = "efficientnet_b0",
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    step_size: int = 7,
    gamma: float = 0.1,
    num_workers: int = 4,
    image_size: int = 224,
    device_str: str | None = None,
) -> Dict[str, list]:
    os.makedirs(MODELS_DIR, exist_ok=True)

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, class_to_idx = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )

    num_classes = len(class_to_idx)

    model = create_model(model_name=model_name, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, f"best_{model_name}.pth")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "model_name": model_name,
                },
                best_model_path,
            )
            print(f"New best model saved to {best_model_path} (val_acc={val_acc:.4f})")

    history_path = os.path.join(MODELS_DIR, f"training_history_{model_name}.pt")
    torch.save(history, history_path)
    print(f"Training history saved to {history_path}")

    return history


if __name__ == "__main__":
    _ = train_model()