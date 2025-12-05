from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torchvision.transforms import functional as F


class ToTensorAndNormalize:
    """
    Transform that:
    - converts SAR (PIL, grayscale) and Optical (PIL, RGB) images to tensors
    - normalizes both with given mean/std values.

    This will be the main transform we pass into SarOpticalPairDataset.

    Notes:
        - SAR is treated as a single-channel image: shape [1, H, W]
        - Optical is treated as standard RGB: shape [3, H, W]
    """

    def __init__(
        self,
        sar_mean: float = 0.5,
        sar_std: float = 0.5,
        optical_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        optical_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        target_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            sar_mean: mean for SAR channel normalization.
            sar_std: std for SAR channel normalization.
            optical_mean: mean for each RGB channel.
            optical_std: std for each RGB channel.
            target_size: optional (H, W). If given, both SAR and Optical
                         will be resized to this size.
        """
        self.sar_mean = sar_mean
        self.sar_std = sar_std
        self.optical_mean = optical_mean
        self.optical_std = optical_std
        self.target_size = target_size

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sar_img = sample["sar"]       # PIL Image, grayscale
        optical_img = sample["optical"]  # PIL Image, RGB

        # Optional resize step
        if self.target_size is not None:
            # target_size is (H, W) but torchvision expects (H, W)
            sar_img = F.resize(sar_img, self.target_size)
            optical_img = F.resize(optical_img, self.target_size)

        # Convert to tensors in [0, 1] range
        sar_tensor = F.to_tensor(sar_img)         # shape [1, H, W]
        optical_tensor = F.to_tensor(optical_img) # shape [3, H, W]

        # Normalize
        sar_tensor = (sar_tensor - self.sar_mean) / self.sar_std

        optical_mean_tensor = torch.tensor(self.optical_mean).view(-1, 1, 1)
        optical_std_tensor = torch.tensor(self.optical_std).view(-1, 1, 1)
        optical_tensor = (optical_tensor - optical_mean_tensor) / optical_std_tensor

        sample["sar"] = sar_tensor
        sample["optical"] = optical_tensor

        return sample


class ComposeDict:
    """
    Simple Compose for dict-based samples.

    This lets us chain multiple transforms that expect and return a sample dict.

    Example:
        transform = ComposeDict([
            ToTensorAndNormalize(...),
            SomeFutureTransform(...)
        ])
    """

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample
