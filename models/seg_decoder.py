"""
Segmentation decoder for HECKTOR 2026 (multi-class: bg / GTVp / GTVn).

For variant=Base we reuse the SwinUNETR built-in decoder via VoCoSwinEncoder.forward.
This module exists so the multitask wrapper can plug in a different decoder
(e.g. a MedNeXt decoder for the secondary ensemble model) without changing
the encoder interface.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SegHead(nn.Module):
    """Thin wrapper that routes through a backbone's built-in seg head."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
