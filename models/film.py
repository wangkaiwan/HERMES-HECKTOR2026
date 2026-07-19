"""
ClinicalBERT-driven FiLM modulation.

Mirrors the FiLM implementation that produced the best HN_CU_Seg result
(0.590 mean dice, +0.024 vs no-text). One shared MLP maps a 768-d
ClinicalBERT [CLS] embedding to scale + shift parameters for each decoder stage.

Reference:
    Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class FiLMConditioner(nn.Module):
    def __init__(self,
                 text_dim: int,
                 stage_channels: List[int],
                 hidden: int = 128,
                 text_dropout_prob: float = 0.3) -> None:
        super().__init__()
        self.stage_channels = stage_channels
        out = sum(2 * c for c in stage_channels)               # scale + shift per stage
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )
        self.text_dropout_prob = text_dropout_prob

    def forward(self, text_features: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of (scale, shift) tensors per stage, each [B, C]."""
        if self.training and self.text_dropout_prob > 0:
            mask = (torch.rand(text_features.shape[0], 1, device=text_features.device)
                    > self.text_dropout_prob).float()
            text_features = text_features * mask
        params = self.mlp(text_features)
        out = []
        offset = 0
        for c in self.stage_channels:
            scale = params[:, offset:offset + c]
            shift = params[:, offset + c:offset + 2 * c]
            out.append((scale, shift))
            offset += 2 * c
        return out


def apply_film(feat: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Apply FiLM modulation to a 3D feature map [B, C, D, H, W]."""
    scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return feat * (1.0 + scale) + shift
