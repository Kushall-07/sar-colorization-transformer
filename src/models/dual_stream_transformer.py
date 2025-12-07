import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic conv block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Keeps spatial size (padding=1 for kernel_size=3).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        return x


class DualStreamSarColorizationTransformer(nn.Module):
    """
    CURRENT VERSION: Simple CNN U-Net–like baseline
    ------------------------------------------------
    - Single SAR stream encoder-decoder
    - Palette / transformer parts are placeholders, not yet used
    - Outputs:
        * rgb_out: (B, 3, H, W) in [0,1] via sigmoid
        * confidence: (B, 1, H, W) in [0,1] via sigmoid

    LATER: We will replace the internals with a real dual-stream
    transformer while keeping:
        forward(self, sar_tensor, palette_tokens=None) -> (rgb, conf)
    so that train/inference code does not need to change.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()

        # ---- SAR encoder (downsampling path) ----
        # 256x256 -> 256x256
        self.enc1 = ConvBlock(in_channels, base_channels)

        # 256x256 -> 128x128
        self.down1 = nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1
        )
        self.enc2 = ConvBlock(base_channels * 2, base_channels * 2)

        # 128x128 -> 64x64
        self.down2 = nn.Conv2d(
            base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1
        )
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)

        # ---- Decoder (upsampling path) ----
        # 64x64 -> 128x128
        self.up1 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock(base_channels * 2 + base_channels * 2, base_channels * 2)

        # 128x128 -> 256x256
        self.up2 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock(base_channels + base_channels, base_channels)

        # ---- Output heads ----
        self.rgb_head = nn.Conv2d(base_channels, 3, kernel_size=1)
        self.confidence_head = nn.Conv2d(base_channels, 1, kernel_size=1)

        # ---- Placeholders for future transformer / palette stream ----
        # These keep the interface but do nothing for now.
        self.palette_encoder = nn.Identity()
        self.fusion = nn.Identity()

    def forward(self, sar_tensor: torch.Tensor, palette_tokens=None):
        """
        sar_tensor: (B, 1, H, W), typically 256x256 after transforms.
        palette_tokens: reserved for future transformer variant (unused now).
        """

        # Encoder
        x1 = self.enc1(sar_tensor)        # (B, C, 256, 256)
        x2_in = self.down1(x1)            # (B, 2C, 128, 128)
        x2 = self.enc2(x2_in)             # (B, 2C, 128, 128)

        x3_in = self.down2(x2)            # (B, 4C, 64, 64)
        x3 = self.bottleneck(x3_in)       # (B, 4C, 64, 64)

        # Decoder with skip connections (U-Net style)
        u1 = self.up1(x3)                 # (B, 2C, 128, 128)
        # concatenate encoder features from same scale
        u1 = torch.cat([u1, x2], dim=1)   # (B, 4C, 128, 128)
        u1 = self.dec1(u1)                # (B, 2C, 128, 128)

        u2 = self.up2(u1)                 # (B, C, 256, 256)
        u2 = torch.cat([u2, x1], dim=1)   # (B, 2C, 256, 256)
        u2 = self.dec2(u2)                # (B, C, 256, 256)

        # RGB in [0,1]
        rgb_out = torch.sigmoid(self.rgb_head(u2))       # (B, 3, 256, 256)

        # Confidence in [0,1]
        confidence = torch.sigmoid(self.confidence_head(u2))  # (B, 1, 256, 256)

        return rgb_out, confidence
