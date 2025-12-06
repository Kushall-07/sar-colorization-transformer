import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 feature maps.

    - Freezes VGG16 weights
    - Computes L1 distance between selected feature layers of
      pred_rgb and target_rgb.

    Inputs should be normalized to [0,1]. This class internally
    applies ImageNet normalization.
    """

    def __init__(self, layers=None, weight: float = 1.0):
        super().__init__()

        if layers is None:
            # Reasonable default layers: low, mid, high-level features
            layers = ["3", "8", "15", "22"]  # indices in features list

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg
        self.selected_layers = set(layers)
        self.weight = weight

        # ImageNet normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        """
        pred_rgb, target_rgb: (B, 3, H, W), assumed in [0,1]
        """
        if pred_rgb.shape != target_rgb.shape:
            raise ValueError("pred_rgb and target_rgb must have the same shape")

        x = self._normalize(pred_rgb)
        y = self._normalize(target_rgb)

        loss = 0.0
        current_x = x
        current_y = y

        for i, layer in enumerate(self.vgg):
            current_x = layer(current_x)
            current_y = layer(current_y)

            if str(i) in self.selected_layers:
                loss += F.l1_loss(current_x, current_y)

        return self.weight * loss
