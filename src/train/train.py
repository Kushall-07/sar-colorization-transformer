import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.data.dataset_sar_optical import SarOpticalPairDataset
from src.models.dual_stream_transformer import DualStreamSarColorizationTransformer
from src.data.transforms_sar_optical import ToTensorAndNormalize

from src.losses import (
    physics_guided_loss,
    palette_regularization_loss,
    PerceptualLoss,
)


# -------------------------------
# Configurations
# -------------------------------
class Config:
    sar_dir = "src/data/sar"          # <-- changed
    optical_dir = "src/data/optical"  # <-- changed
    batch_size = 1
    num_workers = 0
    lr = 2e-4
    num_epochs = 50
    checkpoint_dir = "checkpoints"
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


# -------------------------------
# Helper: Create Dataloader
# -------------------------------
def build_dataloaders():
    # Resize everything to 256x256 so pred & target match in size
    transform = ToTensorAndNormalize(resize=(256, 256))

    dataset = SarOpticalPairDataset(
        sar_dir=cfg.sar_dir,
        optical_dir=cfg.optical_dir,
        extensions=[".png", ".jpg"],
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return loader


# -------------------------------
# Basic Reconstruction Loss
# -------------------------------
def reconstruction_loss(pred_rgb, gt_rgb):
    # L1 pixel-wise loss
    return nn.functional.l1_loss(pred_rgb, gt_rgb)


# -------------------------------
# Combine Total Loss
# -------------------------------
def compute_total_loss(
    pred_rgb,
    gt_rgb,
    sar,
    perceptual_criterion,
    lambda_rec=1.0,
    lambda_phys=0.2,
    lambda_palette=0.1,
    lambda_perc=0.1,
):
    """
    Combines multiple loss components:
      - Reconstruction loss (L1)
      - Physics-guided loss (SAR edge + smoothness)
      - Palette regularization loss (global color statistics)
      - Perceptual loss (VGG16 features)
    """
    # --- Ensure ground-truth RGB matches the prediction size ---
    ph, pw = pred_rgb.shape[-2], pred_rgb.shape[-1]
    gh, gw = gt_rgb.shape[-2], gt_rgb.shape[-1]

    if (ph, pw) != (gh, gw):
        gt_rgb_resized = F.interpolate(
            gt_rgb, size=(ph, pw), mode="bilinear", align_corners=False
        )
    else:
        gt_rgb_resized = gt_rgb
    # ------------------------------------------------------------

    # 1) Pixel reconstruction
    loss_rec = reconstruction_loss(pred_rgb, gt_rgb_resized)

    # 2) Physics-guided: edge consistency + smoothness
    loss_phys = physics_guided_loss(sar, pred_rgb)

    # 3) Palette-level color consistency
    loss_palette = palette_regularization_loss(pred_rgb, gt_rgb_resized)

    # 4) Perceptual similarity in feature space
    loss_perc = perceptual_criterion(pred_rgb, gt_rgb_resized)

    total = (
        lambda_rec * loss_rec
        + lambda_phys * loss_phys
        + lambda_palette * loss_palette
        + lambda_perc * loss_perc
    )

    loss_dict = {
        "reconstruction": float(loss_rec.detach().cpu()),
        "physics": float(loss_phys.detach().cpu()),
        "palette": float(loss_palette.detach().cpu()),
        "perceptual": float(loss_perc.detach().cpu()),
    }

    return total, loss_dict


# -------------------------------
# Training Loop
# -------------------------------
def train():
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print("Building dataloaders...")
    train_loader = build_dataloaders()

    print(f"Using device: {cfg.device}")
    print("Initializing model and losses...")

    model = DualStreamSarColorizationTransformer().to(cfg.device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    # Perceptual loss module (VGG-based)
    perceptual_criterion = PerceptualLoss().to(cfg.device)

    print("Starting training...")

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            sar = batch["sar"].to(cfg.device)         # (B, 1, H, W)
            optical = batch["optical"].to(cfg.device) # (B, 3, H, W)

            optimizer.zero_grad()

            # Forward pass
            # Currently we only pass SAR; later you may also pass palette tokens
            pred_rgb, pred_conf = model(sar)

            # Compute total loss (multi-term)
            total_loss, loss_dict = compute_total_loss(
                pred_rgb=pred_rgb,
                gt_rgb=optical,
                sar=sar,
                perceptual_criterion=perceptual_criterion,
            )

            # Backward + update
            total_loss.backward()
            optimizer.step()

            running_loss += float(total_loss.detach().cpu())

        avg_loss = running_loss / len(train_loader)

        print(
            f"[Epoch {epoch+1}/{cfg.num_epochs}] "
            f"Total={avg_loss:.4f} | "
            f"L_rec={loss_dict['reconstruction']:.4f} | "
            f"L_phys={loss_dict['physics']:.4f} | "
            f"L_pal={loss_dict['palette']:.4f} | "
            f"L_perc={loss_dict['perceptual']:.4f}"
        )

        # Save checkpoint per epoch
        ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

    print("Training complete!")


if __name__ == "__main__":
    train()
