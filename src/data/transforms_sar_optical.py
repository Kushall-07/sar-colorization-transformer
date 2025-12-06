from typing import Tuple, Optional

from PIL import Image
import torchvision.transforms.functional as F


class ToTensorAndNormalize:
    """
    Transform that:
      1. Optionally resizes SAR + optical to a fixed size
      2. Converts them to PyTorch tensors in [0,1]

    sample: {"sar": PIL.Image, "optical": PIL.Image}
    returns: {"sar": tensor (1,H,W), "optical": tensor (3,H,W)}
    """

    def __init__(self, resize: Optional[Tuple[int, int]] = (256, 256)):
        """
        resize: (width, height) or None to keep original size
        """
        self.resize = resize

    def __call__(self, sample):
        sar: Image.Image = sample["sar"]
        optical: Image.Image = sample["optical"]

        # 1) Resize both to the same resolution if requested
        if self.resize is not None:
            # PIL expects (width, height)
            sar = sar.resize(self.resize, Image.BILINEAR)
            optical = optical.resize(self.resize, Image.BILINEAR)

        # 2) Convert to tensors in [0,1]
        sar_tensor = F.to_tensor(sar)       # (1, H, W)
        optical_tensor = F.to_tensor(optical)  # (3, H, W)

        return {"sar": sar_tensor, "optical": optical_tensor}
