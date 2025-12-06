import torch
import torch.nn as nn
import torch.nn.functional as F


def _rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to single-channel luminance using standard weights.
    rgb: (B, 3, H, W)
    returns: (B, 1, H, W)
    """
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return lum


def _sobel_filters(device):
    """
    Create Sobel kernels for x and y gradients.
    Returns weight_x, weight_y of shape (1, 1, 3, 3).
    """
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0,  0.0,  0.0],
         [1.0,  2.0,  1.0]],
        dtype=torch.float32,
        device=device,
    )

    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    return sobel_x, sobel_y


def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient magnitude using Sobel filters.
    x: (B, 1, H, W)
    returns: (B, 1, H, W)
    """
    device = x.device
    sobel_x, sobel_y = _sobel_filters(device)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    return grad_mag


def physics_guided_loss(
    sar_tensor: torch.Tensor,
    pred_rgb: torch.Tensor,
    lambda_edge: float = 1.0,
    lambda_smooth: float = 0.1,
) -> torch.Tensor:
    """
    Physics-guided loss combining:
      1. Edge-consistency between SAR and RGB luminance
      2. Smoothness regularization on RGB output

    sar_tensor: (B, 1, Hs, Ws) – SAR input (may differ in size from pred)
    pred_rgb:   (B, 3, Hp, Wp) – Predicted colorized RGB

    Returns: scalar loss
    """
    # Ensure both are in the same spatial resolution as the prediction
    B, _, Hp, Wp = pred_rgb.shape
    sar = sar_tensor

    if sar.shape[-2:] != (Hp, Wp):
        sar = F.interpolate(sar, size=(Hp, Wp), mode="bilinear", align_corners=False)

    # Convert RGB to luminance
    lum = _rgb_to_luminance(pred_rgb)

    # 1) Edge consistency: match gradient magnitudes
    sar_edges = _gradient_magnitude(sar)
    lum_edges = _gradient_magnitude(lum)
    edge_loss = F.l1_loss(lum_edges, sar_edges)

    # 2) Smoothness: encourage smooth RGB (total variation-like)
    tv_h = torch.mean(torch.abs(pred_rgb[:, :, :, 1:] - pred_rgb[:, :, :, :-1]))
    tv_v = torch.mean(torch.abs(pred_rgb[:, :, 1:, :] - pred_rgb[:, :, :-1, :]))
    smoothness_loss = tv_h + tv_v

    total = lambda_edge * edge_loss + lambda_smooth * smoothness_loss
    return total
