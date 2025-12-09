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
    Dual-stream hybrid:
      - SAR stream: CNN encoder + transformer bottleneck + CNN decoder (U-Net style)
      - Palette stream: encodes global color information from an RGB image
      - Fusion: SAR tokens and palette tokens are concatenated and passed through
                a shared transformer encoder; SAR tokens are then reshaped back to
                a feature map for decoding.

    During TRAINING:
      forward(sar, palette_image=gt_rgb)
    During INFERENCE:
      forward(sar, palette_image=None) -> uses learnable global palette tokens.

    Output:
      - rgb_out: (B, 3, H, W) in [0,1]
      - confidence: (B, 1, H, W) in [0,1]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_layers: int = 2,
        nhead: int = 4,
        palette_token_count: int = 64,  # number of palette tokens (e.g. 8x8)
    ):
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
        self.bottleneck_conv = ConvBlock(base_channels * 4, base_channels * 4)

        # ---- Transformer bottleneck over 64x64 SAR tokens ----
        self.embed_dim = base_channels * 4  # e.g. 128 when base_channels=32

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=nhead,
            dim_feedforward=self.embed_dim * 4,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ---- Palette stream ----
        # We downsample RGB to 8x8, project to embed_dim, and treat each of the 8*8 locations as a palette token.
        self.palette_pool = nn.AdaptiveAvgPool2d((8, 8))  # (B,3,H,W) -> (B,3,8,8)
        self.palette_proj = nn.Conv2d(3, self.embed_dim, kernel_size=1)

        # Fallback learnable palette tokens for inference (no RGB available)
        self.num_palette_tokens = palette_token_count  # should match 8*8=64
        self.learnable_palette_tokens = nn.Parameter(
            torch.randn(1, self.num_palette_tokens, self.embed_dim) * 0.02
        )

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

    # -------------------------------
    # Palette encoding
    # -------------------------------
    def _encode_palette(self, palette_image: torch.Tensor, B: int) -> torch.Tensor:
        """
        palette_image: (B, 3, H, W)
        returns palette_tokens: (B, Np, C)
        """
        # Downsample to 8x8 and project to embed_dim
        x = self.palette_pool(palette_image)     # (B,3,8,8)
        x = self.palette_proj(x)                # (B,C,8,8)
        B, C, Hp, Wp = x.shape                  # Hp*Wp = Np
        tokens = x.view(B, C, Hp * Wp).permute(0, 2, 1)  # (B,Np,C)
        return tokens

    # -------------------------------
    # Forward
    # -------------------------------
    def forward(self, sar_tensor: torch.Tensor, palette_image: torch.Tensor | None = None):
        """
        sar_tensor: (B, 1, H, W), typically 256x256 after transforms.
        palette_image: (B, 3, H, W) ground-truth RGB during training,
                       or None during inference (then we use learnable palette tokens).
        """

        # ----- CNN encoder on SAR -----
        x1 = self.enc1(sar_tensor)        # (B, C, 256, 256)

        x2_in = self.down1(x1)            # (B, 2C, 128, 128)
        x2 = self.enc2(x2_in)             # (B, 2C, 128, 128)

        x3_in = self.down2(x2)            # (B, 4C, 64, 64)
        x3 = self.bottleneck_conv(x3_in)  # (B, 4C, 64, 64)

        B, C, H, W = x3.shape  # expect H=W=64

        # ----- Flatten SAR tokens -----
        sar_tokens = x3.view(B, C, H * W).permute(0, 2, 1)  # (B, Ns, C), Ns = H*W

        # ----- Palette tokens -----
        if palette_image is not None:
            palette_tokens = self._encode_palette(palette_image, B)  # (B, Np, C)
        else:
            # Use learnable global palette tokens, broadcast over batch
            palette_tokens = self.learnable_palette_tokens.expand(B, -1, -1)  # (B, Np, C)

        # ----- Fuse SAR + palette tokens -----
        fused_tokens = torch.cat([sar_tokens, palette_tokens], dim=1)  # (B, Ns+Np, C)

        # Transformer encoder over fused sequence
        fused_out = self.transformer_encoder(fused_tokens)             # (B, Ns+Np, C)

        # Take back only the SAR part (first Ns tokens) to reconstruct feature map
        Ns = H * W
        sar_tokens_out = fused_out[:, :Ns, :]                          # (B, Ns, C)

        x3_trans = sar_tokens_out.permute(0, 2, 1).view(B, C, H, W)    # (B,4C,64,64)

        # ----- CNN decoder with skip connections -----
        u1 = self.up1(x3_trans)            # (B, 2C, 128, 128)
        u1 = torch.cat([u1, x2], dim=1)    # (B, 4C, 128, 128)
        u1 = self.dec1(u1)                 # (B, 2C, 128, 128)

        u2 = self.up2(u1)                  # (B, C, 256, 256)
        u2 = torch.cat([u2, x1], dim=1)    # (B, 2C, 256, 256)
        u2 = self.dec2(u2)                 # (B, C, 256, 256)

        # ----- Outputs -----
        rgb_out = torch.sigmoid(self.rgb_head(u2))            # (B, 3, H, W)
        confidence = torch.sigmoid(self.confidence_head(u2))  # (B, 1, H, W)

        return rgb_out, confidence
