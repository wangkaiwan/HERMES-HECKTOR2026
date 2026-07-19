"""DeepSupSwinUNETR — SwinUNETR with multi-scale decoder logits + FiLM hooks.

Subclasses MONAI's SwinUNETR.  The base `forward` returns only final-resolution
logits, so we override it to ALSO return aux logits at scales 1/2, 1/4, 1/8 in
training, and to provide per-stage hooks for FiLM modulation.

Why these scales:
- 1/16 (`dec3`) is skipped — at 2 mm-iso input, the median GTVn lesion is only
  ~440 mm³ / 11 voxels.  Downsampled 16× that's <1 voxel — nearest-neighbor GT
  becomes mostly background and the aux head learns nothing useful.  nnU-Net /
  STU-Net default DS also stops at 4 levels.

Channel sizes (variant B `feature_size=48`, verified MONAI 1.3.2 use_v2=True):
    dec0 (1/2): 48     → aux_head_2
    dec1 (1/4): 96     → aux_head_4
    dec2 (1/8): 192    → aux_head_8
    dec3 (1/16): 384   (skip)
For variant L (`feature_size=96`) the multipliers are the same → 96/192/384/768.

Inference path (`multiscale=False`, the default) is byte-for-byte identical to
the base SwinUNETR forward → backward-compat with existing call sites.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinUNETR


class DeepSupSwinUNETR(SwinUNETR):
    """SwinUNETR + per-stage aux seg heads + optional FiLM injection.

    Constructor kwargs are forwarded to `SwinUNETR.__init__` unchanged; `n_seg_classes`
    is derived from `out_channels` and `feature_size` is read from kwargs to size
    the aux heads.
    """

    def __init__(self, **kwargs) -> None:
        feature_size = kwargs.get("feature_size", 48)
        n_seg_classes = kwargs.get("out_channels", 3)
        super().__init__(**kwargs)
        # 1×1×1 conv aux heads from each decoder stage's channel count → n_seg_classes.
        self.aux_head_2 = nn.Conv3d(feature_size,     n_seg_classes, kernel_size=1)
        self.aux_head_4 = nn.Conv3d(feature_size * 2, n_seg_classes, kernel_size=1)
        self.aux_head_8 = nn.Conv3d(feature_size * 4, n_seg_classes, kernel_size=1)

    def forward(
        self,
        x_in: torch.Tensor,
        multiscale: bool = False,
        film: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        """
        Args:
            x_in:        input image [B, C, D, H, W]
            multiscale:  if True, return dict {1: full_logits, 2: half, 4: quarter, 8: eighth}.
                         False → return final-res logits only (inference default,
                         backward-compatible with base SwinUNETR.forward).
            film:        optional per-stage FiLM modulation, keyed by downsample
                         factor.  Each value is `(gamma, beta)` of shape
                         [B, C_stage, 1, 1, 1] broadcastable to the decoder
                         feature.  None → no FiLM (seg-only path).  Stages
                         available: 16 (after dec3), 8 (after dec2), 4 (after
                         dec1), 2 (after dec0).
        """
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])

        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        dec3 = self.decoder5(dec4, hidden_states_out[3])     # 1/16
        if film is not None and 16 in film:
            dec3 = self._apply_film(dec3, film[16])
        dec2 = self.decoder4(dec3, enc3)                     # 1/8
        if film is not None and 8 in film:
            dec2 = self._apply_film(dec2, film[8])
        dec1 = self.decoder3(dec2, enc2)                     # 1/4
        if film is not None and 4 in film:
            dec1 = self._apply_film(dec1, film[4])
        dec0 = self.decoder2(dec1, enc1)                     # 1/2
        if film is not None and 2 in film:
            dec0 = self._apply_film(dec0, film[2])
        out_ = self.decoder1(dec0, enc0)                     # 1/1
        logits = self.out(out_)                              # final 1×1×1 head → n_classes

        if not multiscale:
            return logits

        return {
            1: logits,
            2: self.aux_head_2(dec0),
            4: self.aux_head_4(dec1),
            8: self.aux_head_8(dec2),
        }

    @staticmethod
    def _apply_film(feat: torch.Tensor,
                    gb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """FiLM:  feat * (1 + γ) + β.  Identity when γ=0, β=0."""
        gamma, beta = gb
        return feat * (1.0 + gamma) + beta
