"""Prognosis-head selector hook for Task 3.

Returns the configured prognosis head + its training loss type tag (so the
trainer can pick the right loss function: Cox partial likelihood vs DeepSurv
vs DeepHit discrete-time likelihood vs Brier score for time-to-event).

The head contract every prog method MUST satisfy:
    - subclass `nn.Module`
    - forward(image_feat, clinical_feat=None, tn_feat=None) -> Tensor [B] (risk)
      (or [B, num_time_bins] for discrete-time methods — see method tag)
    - expose `.method: str` so the trainer can dispatch the right loss

Currently implemented:
    - cox: PrognosisHead. Linear Cox PH with image + clinical + TN-softmax input.
      Loss = Cox partial likelihood (efron tie handling). Our default.

Planned (raise NotImplementedError for now):
    - deepsurv: Faraggi–Simon neural Cox. Same Cox loss, slightly different head
      arch (deeper MLP, dropout in every layer). Easy 1-day swap if Cox underfits.
    - deephit: discrete-time multi-output. Loss = NLL over discretised time bins
      + ranking loss. Better when the proportional-hazard assumption fails.
    - discrete: piecewise-constant hazard via Brier score loss. Simplest discrete-
      time alternative to deephit; useful when the survival curve is the goal
      rather than just a risk ranking.

Rationale for keeping the hook even though we only use Cox today: the HECKTOR
2026 prognosis metric is concordance index (Harrell's c), which is rank-only.
Cox is competitive on c-index but if validation c-index plateaus far from the
top 2025 entries (best ~0.72), we want a 1-line ablation switch rather than
rewriting the trainer.
"""
from __future__ import annotations

import torch.nn as nn

from .prognosis_head import PrognosisHead


def build_prognosis_head(
    method: str = "cox",
    *,
    image_dim: int,
    clinical_dim: int,
    hidden_dim: int = 256,
    use_clinical: bool = True,
    tn_dim: int = 0,
    use_tn: bool = True,
    **kwargs,
) -> nn.Module:
    """Return a prognosis head for the named method.

    The returned module always exposes a `.method` string attribute so the
    trainer can dispatch the matching loss function in training/losses.py.
    """
    name = (method or "cox").strip().lower()
    if name == "cox":
        head = PrognosisHead(
            image_dim=image_dim,
            clinical_dim=clinical_dim,
            hidden_dim=hidden_dim,
            use_clinical=use_clinical,
            tn_dim=tn_dim,
            use_tn=use_tn,
        )
        head.method = "cox"
        return head
    if name == "deepsurv":
        raise NotImplementedError(
            "prog_method='deepsurv' is a reserved hook. Implementation: deeper MLP "
            "(e.g., 4 layers, dropout 0.3 every layer) with the same Cox partial-"
            "likelihood loss as cox. Add 'deepsurv' branch in training/losses.py."
        )
    if name == "deephit":
        raise NotImplementedError(
            "prog_method='deephit' is a reserved hook. Implementation: discretise "
            "RFS into K time bins, head outputs [B, K] logits, loss = NLL + ranking. "
            "Needs `n_time_bins` and `time_bin_edges` kwargs added to this factory."
        )
    if name == "discrete":
        raise NotImplementedError(
            "prog_method='discrete' is a reserved hook. Piecewise-constant hazard "
            "with Brier-score loss. Simpler than deephit, useful when only the "
            "survival curve is needed."
        )
    raise ValueError(
        f"Unknown prog_method '{method}'. "
        f"Choose from: cox (impl), deepsurv|deephit|discrete (planned)."
    )
