import torch
import torch.nn.functional as F


def palette_regularization_loss(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    downsample_size: int = 16,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Palette regularization by matching downsampled RGB images (global color style).

    pred_rgb:   (B, 3, H, W)  – Predicted RGB
    target_rgb: (B, 3, H, W)  – Ground-truth RGB
    downsample_size: size to which we downsample (e.g., 16x16)
    weight: scaling factor for this loss

    Returns: scalar loss
    """
    if pred_rgb.shape != target_rgb.shape:
        raise ValueError("pred_rgb and target_rgb must have the same shape")

    # Downsample both to low-res "palette" images
    pred_small = F.interpolate(
        pred_rgb, size=(downsample_size, downsample_size), mode="area"
    )
    target_small = F.interpolate(
        target_rgb, size=(downsample_size, downsample_size), mode="area"
    )

    loss = F.l1_loss(pred_small, target_small)
    return weight * loss
