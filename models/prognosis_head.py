"""
Prognosis head — Cox proportional hazards risk score.

Inputs:
    - tumor-pooled features (concat of GTVp + GTVn pooled features)
    - clinical embedding (encoded tabular features OR ClinicalBERT [CLS])

Output:
    - scalar risk score per patient (higher = worse prognosis).

Loss = Cox partial likelihood (efron tie handling). See training/losses.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PrognosisHead(nn.Module):
    """Cox risk score on top of image + clinical + (optional) TN-softmax stage.

    Task #45 (2026-05-13): the prognosis head now also accepts the softmax of
    the predicted T and N logits (concatenated to the input). This mirrors the
    SIMS-LIFE 2025 design (1st prize, used HPV-softmax → prog) — for HECKTOR
    2026 the analogous handle is TN staging, which IS a prediction target.
    Set `tn_dim=0` to disable.

    Layout of `forward` input (in concat order):
        image_feat  (image_dim, e.g. 2*768 for GTVp+GTVn pooled bottleneck)
      ‖ clinical    (clinical_dim, e.g. 18) — if use_clinical
      ‖ t_softmax   (tn_dim // 2 ≈ 4 after task #36) — if use_tn
      ‖ n_softmax   (tn_dim // 2 ≈ 4) — if use_tn

    Callers (multitask_model.HECKTORMultitaskModel) wire this up.
    """

    def __init__(self,
                 image_dim: int,
                 clinical_dim: int,
                 hidden_dim: int = 256,
                 use_clinical: bool = True,
                 tn_dim: int = 0,
                 use_tn: bool = True) -> None:
        super().__init__()
        self.use_clinical = use_clinical and clinical_dim > 0
        self.use_tn = use_tn and tn_dim > 0
        self.clinical_dim = clinical_dim if self.use_clinical else 0
        self.tn_dim = tn_dim if self.use_tn else 0
        in_dim = image_dim + self.clinical_dim + self.tn_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self,
                image_feat: torch.Tensor,
                clinical_feat: torch.Tensor | None = None,
                tn_feat: torch.Tensor | None = None) -> torch.Tensor:
        parts = [image_feat]
        if self.use_clinical and clinical_feat is not None:
            parts.append(clinical_feat)
        if self.use_tn and tn_feat is not None:
            parts.append(tn_feat)
        x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return self.net(x).squeeze(-1)                         # [B]
