import os
from pathlib import Path

from PIL import Image

# ------------ CONFIG ------------
# Folder with your normal RGB photos
INPUT_DIR = Path("raw_rgb")

# Output folders used by your project
SAR_DIR = Path("src/data/sar")
OPTICAL_DIR = Path("src/data/optical")

# Accepted input image extensions
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Optional: resize all images to this size before saving
# Set to None to keep original size
RESIZE_TO = (256, 256)  # (width, height)
# --------------------------------


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"INPUT_DIR does not exist: {INPUT_DIR}. "
            f"Create it and put some RGB images inside."
        )

    SAR_DIR.mkdir(parents=True, exist_ok=True)
    OPTICAL_DIR.mkdir(parents=True, exist_ok=True)

    files = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in EXTENSIONS]
    if not files:
        raise RuntimeError(
            f"No images found in {INPUT_DIR}. "
            f"Supported extensions: {sorted(EXTENSIONS)}"
        )

    print(f"Found {len(files)} images in {INPUT_DIR}, generating fake SAR dataset...")

    for idx, path in enumerate(sorted(files), start=1):
        img = Image.open(path).convert("RGB")

        if RESIZE_TO is not None:
            img = img.resize(RESIZE_TO, Image.BILINEAR)

        # Save RGB as optical ground truth
        out_name = path.stem + ".png"  # force .png for consistency
        optical_out = OPTICAL_DIR / out_name
        img.save(optical_out)

        # Create fake SAR by converting to grayscale
        sar_img = img.convert("L")
        sar_out = SAR_DIR / out_name
        sar_img.save(sar_out)

        print(f"[{idx}/{len(files)}] {path.name} -> {sar_out.name}, {optical_out.name}")

    print("\nDone! Fake SAR/optical pairs are in:")
    print(f"  SAR:      {SAR_DIR}")
    print(f"  OPTICAL:  {OPTICAL_DIR}")


if __name__ == "__main__":
    main()
