import torch
import torch.nn as nn
from torchvision import models


def create_model(model_name: str = "efficientnet_b0", num_classes: int = 3):
    if model_name.lower() == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    elif model_name.lower() == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unsupported model name: {model_name}")