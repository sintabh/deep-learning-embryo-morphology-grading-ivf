import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from src.config import TEST_DIR, MODELS_DIR, DEVICE
from src.models.multitask_model import MultiTaskEmbryoNet

TaskType = Literal["expansion", "icm", "te"]


def _get_sample_image() -> Path:
    """
    Pick one PNG image from the test set to visualize.

    It tries:
      1) data/processed/test/Human_Blastocyst_Dataset
      2) data/processed/test
      3) any PNG under TEST_DIR recursively
    """
    root = Path(TEST_DIR)

    candidates: list[Path] = []

    # 1) Prefer subfolder "Human_Blastocyst_Dataset" if it exists
    hb_dir = root / "Human_Blastocyst_Dataset"
    if hb_dir.exists():
        candidates = sorted(hb_dir.glob("*.png"))

    # 2) If no candidates, look directly in TEST_DIR
    if not candidates and root.exists():
        candidates = sorted(root.glob("*.png"))

    # 3) As a last resort, search recursively
    if not candidates and root.exists():
        candidates = sorted(root.rglob("*.png"))

    if not candidates:
        raise FileNotFoundError(f"No PNG images found under test dir: {root}")

    return candidates[0]


class SimpleGradCAM:
    """
    Minimal Grad-CAM implementation for MultiTaskEmbryoNet.

    It hooks into the last convolutional block of the EfficientNet backbone.
    """

    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module):
        self.model = model
        self.model.eval()

        self.target_module = target_module
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            # grad_output is a tuple with one element (dL/dA)
            self.gradients = grad_output[0].detach()

        self.target_module.register_forward_hook(forward_hook)
        self.target_module.register_backward_hook(backward_hook)

    def generate(
        self,
        image_tensor: torch.Tensor,
        task: TaskType,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a given task head and (optionally) class idx.
        """
        self.model.zero_grad()
        with torch.enable_grad():
            # MultiTaskEmbryoNet returns a tuple:
            # (expansion_logits, icm_logits, te_logits)
            logits_exp, logits_icm, logits_te = self.model(image_tensor)

            if task == "expansion":
                logits = logits_exp
            elif task == "icm":
                logits = logits_icm
            elif task == "te":
                logits = logits_te
            else:
                raise ValueError(f"Unknown task: {task}")

            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())

            score = logits[:, class_idx]

            score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # activations: [B, C, H, W]
        # gradients:   [B, C, H, W]
        grads = self.gradients
        acts = self.activations

        # Global-average-pool gradients over spatial dims
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # Weighted sum of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        cam = F.relu(cam)
        cam = cam[0, 0].cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam


def _preprocess_image(img_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load image, keep original (numpy) for overlay, and return
    a preprocessed tensor for the network.
    """
    img = Image.open(img_path).convert("RGB")
    orig = np.array(img)

    # Same preprocessing as training (ImageNet normalization, 224x224)
    img = img.resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0

    # HWC -> CHW
    img_chw = np.transpose(img_np, (2, 0, 1))

    # Normalize
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_chw = (img_chw - mean) / std

    tensor = torch.from_numpy(img_chw).unsqueeze(0)  # [1, 3, 224, 224]
    tensor = tensor.to(DEVICE, dtype=torch.float32)

    return tensor, orig


def _overlay_heatmap_on_image(
    original: np.ndarray, cam: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """
    Resize CAM to original image size and overlay as heatmap.
    """
    h, w = original.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )  # BGR

    # Convert original from RGB to BGR for OpenCV
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(heatmap, alpha, original_bgr, 1 - alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb


def main() -> None:
    print(f"Using device: {DEVICE}")

    # 1) Load sample image from test set
    img_path = _get_sample_image()
    print(f"Using test image: {img_path}")

    input_tensor, orig_img = _preprocess_image(img_path)

    # 2) Build model and load best multitask weights
    model = MultiTaskEmbryoNet(
        num_expansion_classes=5,
        num_icm_classes=4,
        num_te_classes=4,
        pretrained=False,
    )
    model.to(DEVICE)

    ckpt_path = Path(MODELS_DIR) / "best_multitask_efficientnet_b0.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # Saved as plain state_dict
        model.load_state_dict(ckpt)

    model.eval()

    # 3) Pick target layer from the EfficientNet backbone
    # Adjust this if your MultiTaskEmbryoNet uses a different attribute name
    target_module = model.backbone.features[-1]

    gradcam = SimpleGradCAM(model, target_module)

    figures_dir = Path("reports") / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 4) Generate and save CAMs for each task
    for task in ["expansion", "icm", "te"]:
        cam = gradcam.generate(input_tensor, task=task)  # auto class_idx (argmax)
        overlay = _overlay_heatmap_on_image(orig_img, cam, alpha=0.5)

        out_path = figures_dir / f"gradcam_multitask_{task}.png"
        Image.fromarray(overlay).save(out_path)
        print(f"Grad-CAM for {task} saved to: {out_path}")


if __name__ == "__main__":
    main()