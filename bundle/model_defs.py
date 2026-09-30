"""Model architectures for the SAR oil-spill segmentation pipeline.
Defines the StandaloneHybridCNNTransformer matching the trained model weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StandaloneAttnBlock(nn.Module):
    """Pre-norm self-attention + MLP block operating on flattened feature maps."""
    def __init__(self, channels: int, num_heads: int = 8, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden_dim = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normed = self.norm1(tokens)
        attn_out, _ = self.attention(normed, normed, normed, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class StandaloneHybridCNNTransformer(nn.Module):
    """CNN encoder -> StandaloneAttnBlock x2 -> CNN bottleneck -> CNN decoder."""
    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        self.enc1 = HybridCNNBlock(in_channels, 64)
        self.enc2 = HybridCNNBlock(64, 128)
        self.enc3 = HybridCNNBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.transformer1 = StandaloneAttnBlock(256, num_heads=8, mlp_ratio=2, dropout=0.1)
        self.transformer2 = StandaloneAttnBlock(256, num_heads=8, mlp_ratio=2, dropout=0.1)
        self.bottleneck = HybridCNNBlock(256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = HybridCNNBlock(128 + 256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = HybridCNNBlock(64 + 128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.segmentation_head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor, extra_features=None) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        original_size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        x = self.pool(e3)
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.bottleneck(x)
        x = self.up2(x)
        if x.shape[-2:] != e3.shape[-2:]:
            x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, e3], dim=1))
        x = self.up1(x)
        if x.shape[-2:] != e2.shape[-2:]:
            x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, e2], dim=1))
        x = self.up0(x)
        logits = self.segmentation_head(x)
        return F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)


class EdgeGuidedHybridCNNTransformer(nn.Module):
    """Hybrid CNN-Transformer with edge-guided and uncertainty-aware learning.

    Integrates:
    - Shallow CNN feature edge prediction (E = sigmoid(Conv3x3(e1)))
    - Decoder edge refinement: F_refined = F_dec * (1 + E)
    - Uncertainty refinement head: U = sigmoid(Conv3x3(D1))
    - Feature suppression in ambiguous zones: F_unc = D1 * (1 - U)
    - Output spill logits
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        self.enc1 = HybridCNNBlock(in_channels, 64)
        self.enc2 = HybridCNNBlock(64, 128)
        self.enc3 = HybridCNNBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.transformer1 = StandaloneAttnBlock(256, num_heads=8, mlp_ratio=2, dropout=0.1)
        self.transformer2 = StandaloneAttnBlock(256, num_heads=8, mlp_ratio=2, dropout=0.1)
        self.bottleneck = HybridCNNBlock(256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = HybridCNNBlock(128 + 256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = HybridCNNBlock(64 + 128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        # Edge guidance module (shallow features F1 -> 1-channel edge map)
        self.edge_head = nn.Conv2d(64, 1, 3, padding=1)

        # Uncertainty refinement head (decoder features D1 -> 1-channel uncertainty map)
        self.uncertainty_head = nn.Conv2d(32, 1, 3, padding=1)

        # Final segmentation head
        self.segmentation_head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        original_size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        x = self.pool(e3)
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.bottleneck(x)
        x = self.up2(x)
        if x.shape[-2:] != e3.shape[-2:]:
            x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, e3], dim=1))
        x = self.up1(x)
        if x.shape[-2:] != e2.shape[-2:]:
            x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, e2], dim=1))

        # Edge prediction from shallow features e1
        edge_logits = self.edge_head(e1)
        edge_map = torch.sigmoid(edge_logits)
        if edge_map.shape[-2:] != x.shape[-2:]:
            edge_down = F.interpolate(edge_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        else:
            edge_down = edge_map

        # Edge refinement: F_refined = F_decoder * (1 + E)
        x = x * (1.0 + edge_down)
        d1 = self.up0(x)

        # Uncertainty prediction from decoder feature d1
        unc_logits = self.uncertainty_head(d1)
        uncertainty_map = torch.sigmoid(unc_logits)

        # Uncertainty refinement: F_unc = D1 * (1 - U)
        d1_refined = d1 * (1.0 - uncertainty_map)

        logits = self.segmentation_head(d1_refined)
        logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        edge_map = F.interpolate(edge_map, size=original_size, mode="bilinear", align_corners=False)
        uncertainty_map = F.interpolate(uncertainty_map, size=original_size, mode="bilinear", align_corners=False)

        if return_aux:
            return logits, edge_map, uncertainty_map
        return logits


class EdgeGuidedUncertaintyLoss(nn.Module):
    """L_total = L_Dice + L_BCE + lambda1 * L_edge + lambda2 * L_uncertainty
    Configured with lambda1 = 0.4 and lambda2 = 0.35.
    """
    def __init__(self, lambda1: float = 0.4, lambda2: float = 0.35, eps: float = 1e-6):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eps = eps
        kx = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32) / 16.0
        ky = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32) / 16.0
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, edge_map: Optional[torch.Tensor] = None,
                uncertainty_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logits.ndim == 2: logits = logits.unsqueeze(0).unsqueeze(0)
        elif logits.ndim == 3: logits = logits.unsqueeze(1)
        if target.ndim == 2: target = target.unsqueeze(0).unsqueeze(0)
        elif target.ndim == 3: target = target.unsqueeze(1)
        target = target.float()
        prob = torch.sigmoid(logits)

        # 1. BCE
        l_bce = F.binary_cross_entropy_with_logits(logits, target)

        # 2. Dice
        inter = 2.0 * (prob * target).sum(dim=(-2, -1)) + 1.0
        card = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + 1.0
        l_dice = (1.0 - (inter / card)).mean()

        # 3. Edge loss
        gx = F.conv2d(target, self.kx, padding=1); gy = F.conv2d(target, self.ky, padding=1)
        edge_gt = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).clamp(0.0, 1.0)
        if edge_map is None:
            gx_p = F.conv2d(prob, self.kx, padding=1); gy_p = F.conv2d(prob, self.ky, padding=1)
            edge_map = torch.sqrt(gx_p ** 2 + gy_p ** 2 + 1e-8).clamp(0.0, 1.0)
        l_edge = F.l1_loss(edge_map, edge_gt)

        # 4. Uncertainty loss
        if uncertainty_map is None:
            uncertainty_map = 4.0 * prob * (1.0 - prob)
        l_unc = uncertainty_map.mean()

        return l_dice + l_bce + (self.lambda1 * l_edge) + (self.lambda2 * l_unc)

