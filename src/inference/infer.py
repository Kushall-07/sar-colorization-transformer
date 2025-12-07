import os
import argparse

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
import numpy as np

from src.models.dual_stream_transformer import DualStreamSarColorizationTransformer


def load_sar_image(path: str, resize=(256, 256), device="cpu"):
    """
    Load a SAR grayscale image from disk and convert to tensor (1, H, W),
    then add batch dimension -> (1, 1, H, W).
    """
    img = Image.open(path).convert("L")  # ensure grayscale

    if resize is not None:
        img = img.resize(resize, Image.BILINEAR)

    # to tensor in [0,1]
    import torchvision.transforms.functional as TF

    sar_tensor = TF.to_tensor(img).unsqueeze(0).to(device)  # (1, 1, H, W)
    return sar_tensor


def save_confidence_map(conf_tensor: torch.Tensor, out_path: str):
    """
    Save a confidence map tensor as a grayscale PNG.

    conf_tensor: (1, H, W) or (H, W)
    """
    if conf_tensor.dim() == 3:
        # assume (1, H, W)
        conf_tensor = conf_tensor.squeeze(0)

    conf_np = conf_tensor.detach().cpu().numpy()

    # Normalize to [0,1]
    conf_min = conf_np.min()
    conf_max = conf_np.max()
    if conf_max > conf_min:
        conf_norm = (conf_np - conf_min) / (conf_max - conf_min)
    else:
        conf_norm = np.zeros_like(conf_np)

    img = Image.fromarray((conf_norm * 255).astype(np.uint8), mode="L")
    img.save(out_path)
    print(f"Saved confidence map to: {out_path}")


def run_inference(args):
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    # 1) Build model and load weights
    model = DualStreamSarColorizationTransformer().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2) Load SAR image
    sar_tensor = load_sar_image(args.sar_image, resize=(256, 256), device=device)

    # 3) Forward pass (no grad)
    with torch.no_grad():
        # Our model currently takes sar only and returns (rgb, confidence)
        pred_rgb, pred_conf = model(sar_tensor)

    # 4) Post-process predicted RGB
    # Ensure in [0,1]
    pred_rgb_clamped = torch.clamp(pred_rgb, 0.0, 1.0)

    # 5) Save colorized RGB
    color_out_path = os.path.join(args.output_dir, "colorized.png")
    save_image(pred_rgb_clamped, color_out_path)
    print(f"Saved colorized RGB to: {color_out_path}")

    # 6) Save confidence map (if available)
    if pred_conf is not None:
        # Expecting shape (B, 1, H, W) or (B, H, W)
        if pred_conf.dim() == 4:
            conf_map = pred_conf[0, 0]  # first batch, first channel
        elif pred_conf.dim() == 3:
            conf_map = pred_conf[0]
        else:
            conf_map = pred_conf

        conf_out_path = os.path.join(args.output_dir, "confidence.png")
        save_confidence_map(conf_map, conf_out_path)
    else:
        print("Warning: Model returned no confidence map tensor.")


def parse_args():
    parser = argparse.ArgumentParser(description="SAR Colorization Inference Script")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--sar_image",
        type=str,
        required=True,
        help="Path to the input SAR grayscale image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save the colorized image and confidence map.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
