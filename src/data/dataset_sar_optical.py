from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset


class SarOpticalPairDataset(Dataset):
    """
    Dataset for paired SAR + Optical images.

    This is the foundational dataset class for the SAR colorization pipeline.
    """

    def __init__(
        self,
        sar_dir: str | Path,
        optical_dir: str | Path,
        extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.sar_dir = Path(sar_dir)
        self.optical_dir = Path(optical_dir)
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.transform = transform

        if not self.sar_dir.exists():
            raise FileNotFoundError(f"SAR directory does not exist: {self.sar_dir}")
        if not self.optical_dir.exists():
            raise FileNotFoundError(f"Optical directory does not exist: {self.optical_dir}")

        self.sar_paths = sorted(
            p for p in self.sar_dir.iterdir() if p.suffix.lower() in self.extensions
        )
        self.optical_paths = sorted(
            p for p in self.optical_dir.iterdir() if p.suffix.lower() in self.extensions
        )

        if len(self.sar_paths) == 0:
            raise RuntimeError(f"No SAR images found in {self.sar_dir}")
        if len(self.optical_paths) == 0:
            raise RuntimeError(f"No optical images found in {self.optical_dir}")
        if len(self.sar_paths) != len(self.optical_paths):
            raise RuntimeError(
                f"Mismatch between SAR ({len(self.sar_paths)}) and optical ({len(self.optical_paths)}) images."
            )

    def __len__(self) -> int:
        return len(self.sar_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sar_path = self.sar_paths[idx]
        optical_path = self.optical_paths[idx]

        sar_img = Image.open(sar_path).convert("L")       # SAR is grayscale
        optical_img = Image.open(optical_path).convert("RGB")

        sample: Dict[str, Any] = {
            "sar": sar_img,
            "optical": optical_img,
            "sar_path": str(sar_path),
            "optical_path": str(optical_path),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
