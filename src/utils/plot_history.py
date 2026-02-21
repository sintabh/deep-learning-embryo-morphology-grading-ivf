import os
from typing import Dict

import torch
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, MODELS_DIR


def load_history(model_name: str = "efficientnet_b0") -> Dict[str, list]:
    history_path = os.path.join(MODELS_DIR, f"training_history_{model_name}.pt")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")
    history = torch.load(history_path)
    return history


def plot_accuracy(history: Dict[str, list], model_name: str = "efficientnet_b0") -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    epochs = range(1, len(train_acc) + 1)

    plt.figure()
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Validation Accuracy ({model_name})")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(FIGURES_DIR, f"accuracy_{model_name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Accuracy plot saved to {out_path}")


def plot_loss(history: Dict[str, list], model_name: str = "efficientnet_b0") -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss ({model_name})")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(FIGURES_DIR, f"loss_{model_name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Loss plot saved to {out_path}")


if __name__ == "__main__":
    model_name = "efficientnet_b0"
    history_dict = load_history(model_name=model_name)
    plot_accuracy(history_dict, model_name=model_name)
    plot_loss(history_dict, model_name=model_name)