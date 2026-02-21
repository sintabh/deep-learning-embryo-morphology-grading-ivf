import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import FIGURES_DIR, MODELS_DIR
from src.models.cnn_model import create_model
from src.utils.data_loader import EmbryoImageDataset, create_transforms


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)

        score = output[:, target_class]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze()

        cam = torch.clamp(cam, min=0)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.cpu().numpy()


def load_checkpoint(model_name: str = "efficientnet_b0", device_str: str | None = None):
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(MODELS_DIR, f"best_{model_name}.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint, device


def build_model(checkpoint: Dict, device: torch.device):
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    model = create_model(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_to_idx


def get_target_layer(model: nn.Module, model_name: str):
    if model_name.startswith("efficientnet"):
        return model.features[-1]
    elif model_name.startswith("resnet"):
        return model.layer4[-1]
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def preprocess_image(image_path: str, image_size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    _, eval_transform = create_transforms(image_size=image_size)

    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        np_img = np.array(rgb_img)

    tensor_img = eval_transform(rgb_img).unsqueeze(0)
    return tensor_img, np_img


def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = Image.fromarray(heatmap_resized).resize(
        (image.shape[1], image.shape[0])
    )
    heatmap_resized = np.array(heatmap_resized)

    colored = plt.get_cmap("jet")(heatmap_resized / 255.0)[:, :, :3]
    colored = np.uint8(255 * colored)

    overlaid = (alpha * colored + (1 - alpha) * image).astype(np.uint8)
    return overlaid


def generate_gradcam_for_image(
    image_path: str,
    model_name: str = "efficientnet_b0",
    image_size: int = 224,
    device_str: str | None = None,
) -> str:
    checkpoint, device = load_checkpoint(model_name, device_str)
    model, class_to_idx = build_model(checkpoint, device)

    input_tensor, original_image = preprocess_image(image_path, image_size)

    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    with torch.no_grad():
        output = model(input_tensor.to(device))
        pred_class = int(torch.argmax(output, dim=1).item())

    heatmap = gradcam.generate(input_tensor.to(device), pred_class)
    result = overlay_heatmap_on_image(original_image, heatmap)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f"gradcam_{model_name}.png")
    Image.fromarray(result).save(out_path)

    return out_path


if __name__ == "__main__":
    sample_path = ""  # Provide a valid image path
    if sample_path:
        out = generate_gradcam_for_image(sample_path)
        print(out)