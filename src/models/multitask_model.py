import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MultiTaskEmbryoNet(nn.Module):
    """
    Multi-task EfficientNet-B0 backbone with three classification heads:
        - expansion_head: expansion grade (e.g. 0-4)
        - icm_head: ICM grade (e.g. 0-2)
        - te_head: TE grade (e.g. 0-2)
    """

    def __init__(
        self,
        num_expansion_classes: int = 5,
        num_icm_classes: int = 3,
        num_te_classes: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            backbone = efficientnet_b0(weights=weights)
        else:
            backbone = efficientnet_b0(weights=None)

        # Remove the original classifier
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone

        # Task-specific heads
        self.expansion_head = nn.Linear(in_features, num_expansion_classes)
        self.icm_head = nn.Linear(in_features, num_icm_classes)
        self.te_head = nn.Linear(in_features, num_te_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits_exp = self.expansion_head(features)
        logits_icm = self.icm_head(features)
        logits_te = self.te_head(features)
        return logits_exp, logits_icm, logits_te