"""
TN-staging head.

Pools encoder features inside the predicted GTVp / GTVn regions (mask-pooled
average + global-avg fallback when masks are empty), then runs two MLP heads:

    feat_p (768 from bottleneck stage) ─► MLP ─► T_stage logits (5)
    feat_n                              ─► MLP ─► N_stage logits (4)

AJCC/UICC 7th-edition T_stage classes: T1, T2, T3, T4, Tx (collapse T4a/b → T4).
N_stage classes: N0, N1, N2, N3 (N2b/N2c collapsed to N2 per challenge spec).

If the seg prediction is empty for a class, falls back to global average pool.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_global_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average `feat` [B, C, D, H, W] over voxels where `mask` [B, 1, D, H, W] > 0.

    Falls back to plain GAP for items in the batch with empty mask.
    """
    mask = mask.float()
    masked_sum = (feat * mask).flatten(2).sum(-1)              # [B, C]
    counts = mask.flatten(2).sum(-1).clamp(min=1.0)            # [B, 1]
    pooled = masked_sum / counts                                # [B, C]

    empty = (mask.flatten(2).sum(-1) < 1.0).float()             # [B, 1]
    gap = feat.flatten(2).mean(-1)                              # [B, C]
    return pooled * (1 - empty) + gap * empty


class TNStagingHead(nn.Module):
    """T- and N-staging classifiers operating on mask-pooled encoder features.

    Task #46 (2026-05-13): also accepts a clinical tabular vector (18-d by
    default) and concats it to the pooled feature before the MLP — demographics
    (age, gender), HPV status, smoking/alcohol, PS, treatment all carry
    prognostic / staging signal that image features alone miss, and clinical is
    more reliable than seg-mask-pooled image features when the seg mask is
    noisy. Set clinical_dim=0 to disable.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 256,
                 n_t_classes: int = 4,
                 n_n_classes: int = 4,
                 clinical_dim: int = 0,
                 use_clinical: bool = True) -> None:
        super().__init__()
        self.use_clinical = use_clinical and clinical_dim > 0
        self.clinical_dim = clinical_dim if self.use_clinical else 0
        in_dim = in_channels + (clinical_dim if self.use_clinical else 0)
        self.t_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_t_classes),
        )
        self.n_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_n_classes),
        )

    def forward(self,
                bottleneck: torch.Tensor,
                seg_logits: torch.Tensor,
                clinical_feat: torch.Tensor | None = None) -> dict:
        """bottleneck: encoder bottleneck [B, C, D, H, W] (already pooled to seg shape).

        seg_logits: [B, 3, D, H, W] at the same spatial resolution as bottleneck
                    (downsampled if necessary by the caller).
        clinical_feat: [B, clinical_dim] — optional; when use_clinical is False
                       OR clinical_feat is None it falls back to image-only.
        """
        prob = F.softmax(seg_logits, dim=1)
        mask_p = prob[:, 1:2]                                  # GTVp
        mask_n = prob[:, 2:3]                                  # GTVn

        feat_p = masked_global_pool(bottleneck, mask_p)
        feat_n = masked_global_pool(bottleneck, mask_n)

        if self.use_clinical and clinical_feat is not None:
            # Same clinical vector goes to both heads — T and N stage both
            # benefit from demographics / HPV / lifestyle covariates.
            feat_p = torch.cat([feat_p, clinical_feat], dim=-1)
            feat_n = torch.cat([feat_n, clinical_feat], dim=-1)

        return {
            "t_logits": self.t_head(feat_p),
            "n_logits": self.n_head(feat_n),
        }
