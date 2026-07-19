"""
Multitask losses for HECKTOR 2026.

    seg        — DiceCE on 3-class segmentation (bg / GTVp / GTVn)
    staging    — class-weighted cross-entropy for T and N stage
    prognosis  — Cox partial likelihood with Efron tie handling

Per-sample masks (`has_seg`, `has_staging`, `has_survival`) zero out terms when
labels are absent. Multi-task scalarisation supports either fixed weights or
learned Kendall log-variance weighting (uncertainty_weights.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Segmentation -----------------------------------------------------------
class DiceCELoss(nn.Module):
    """Dice + CE, with an optional Focal term.

    `focal_weight > 0` adds a multi-class focal loss term (SIMS-LIFE, the
    HECKTOR 2025 RFS winner, used Dice+Focal for seg "to emphasize hard tumor
    pixels overwhelmed by large background"). Default `focal_weight=0` keeps the
    plain Dice+CE behavior, so existing configs/runs are unaffected.

    total = dice_weight·(1 - meanDice) + ce_weight·CE + focal_weight·Focal
    """

    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0,
                 focal_weight: float = 0.0, focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0, ignore_bg_dice: bool = True) -> None:
        super().__init__()
        self.dw = dice_weight
        self.cw = ce_weight
        self.fw = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ignore_bg_dice = ignore_bg_dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits [B, C, ...], target [B, 1, ...] integer
        target = target.long().squeeze(1)
        ce = F.cross_entropy(logits, target, reduction="mean")

        prob = F.softmax(logits, dim=1)
        n_classes = prob.shape[1]
        target_oh = F.one_hot(target, num_classes=n_classes).movedim(-1, 1).float()
        dims = tuple(range(2, prob.dim()))

        intersect = (prob * target_oh).sum(dim=dims)
        denom = prob.sum(dim=dims) + target_oh.sum(dim=dims)
        dice = (2 * intersect + 1.0) / (denom + 1.0)
        if self.ignore_bg_dice:
            dice = dice[:, 1:]
        dice_loss = 1.0 - dice.mean()

        total = self.dw * dice_loss + self.cw * ce

        if self.fw > 0.0:
            # Multi-class focal: FL = -alpha·(1 - p_t)^gamma·log(p_t),
            # where p_t is the predicted probability of the true class.
            logp = F.log_softmax(logits, dim=1)
            logpt = (logp * target_oh).sum(dim=1)              # [B, ...]
            pt = logpt.exp()
            focal = -self.focal_alpha * (1.0 - pt).pow(self.focal_gamma) * logpt
            total = total + self.fw * focal.mean()

        return total


# --- Deep supervision wrapper (task #27) -----------------------------------
class DeepSupervisionDiceCELoss(nn.Module):
    """Wraps a base seg loss (DiceCELoss) and applies it at multiple scales.

    Args:
        base_loss:  the per-scale seg loss (e.g. DiceCELoss with Focal).
        weights:    relative weights at scales [1, 2, 4, 8] (ascending downsample).
                    Normalised to sum=1 so the total magnitude matches a single-
                    scale run; nnU-Net / STU-Net default to [1, 0.5, 0.25, 0.125]
                    which after normalisation is [0.533, 0.267, 0.133, 0.067].

    Forward:
        ms_logits:  dict {1: full, 2: 1/2, 4: 1/4, 8: 1/8} produced by
                    `DeepSupSwinUNETR.forward(multiscale=True)`.
        target:     integer label tensor [B, 1, D, H, W] at full resolution.
                    Downsampled to each scale with nearest-neighbour so labels
                    are preserved exactly.
    """

    def __init__(self, base_loss: nn.Module,
                 weights=(1.0, 0.5, 0.25, 0.125)) -> None:
        super().__init__()
        self.base = base_loss
        w = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("w", w / w.sum())                    # normalised → sum to 1

    def forward(self, ms_logits: dict, target: torch.Tensor) -> torch.Tensor:
        # Scales [1, 2, 4, 8] in ascending order so the weights line up.
        scales = sorted(ms_logits.keys())                          # [1, 2, 4, 8]
        total = ms_logits[scales[0]].new_zeros(())
        for i, s in enumerate(scales):
            if s == 1:
                tgt = target
            else:
                # F.interpolate(mode='nearest') needs float input but preserves labels.
                tgt = F.interpolate(target.float(), scale_factor=1.0 / s,
                                    mode="nearest").to(target.dtype)
            total = total + self.w[i] * self.base(ms_logits[s], tgt)
        return total


# --- Staging ----------------------------------------------------------------
class WeightedCE(nn.Module):
    def __init__(self, n_classes: int, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.n_classes = n_classes
        if class_weights is not None:
            self.register_buffer("w", class_weights.float())
        else:
            self.w = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        loss = F.cross_entropy(logits, target.long(), weight=self.w, reduction="none")
        if mask is not None:
            loss = loss * mask.float()
            denom = mask.float().sum().clamp(min=1.0)
            return loss.sum() / denom
        return loss.mean()


# --- Prognosis (Cox partial likelihood, Efron) ------------------------------
def cox_efron_loss(risk: torch.Tensor,
                   time: torch.Tensor,
                   event: torch.Tensor,
                   mask: torch.Tensor | None = None) -> torch.Tensor:
    """Numerically-stable Cox partial likelihood with Efron's tie handling.

    risk  [B]  — model output (higher = worse)
    time  [B]  — RFS days
    event [B]  — relapse indicator (1 = event, 0 = censored)
    mask  [B]  — 1 where survival label is valid, 0 otherwise
    """
    if mask is None:
        mask = torch.ones_like(event)
    valid = mask.bool() & (event >= 0)
    if valid.sum() < 2:
        return risk.new_zeros(())

    risk = risk[valid]
    time = time[valid]
    event = event[valid].float()

    # sort by descending time; risk-set is the prefix
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    log_cumsum = torch.logcumsumexp(risk, dim=0)               # [B] running logsumexp
    pll = (risk - log_cumsum) * event
    n_events = event.sum().clamp(min=1.0)
    return -pll.sum() / n_events


def sigmoid_concordance_loss(risk: torch.Tensor,
                             time: torch.Tensor,
                             event: torch.Tensor,
                             mask: torch.Tensor | None = None,
                             temperature: float = 0.1) -> torch.Tensor:
    """SurvLoss sigmoid-concordance surrogate (Kai Wang's SurvLoss project;
    verbatim from SurvLoss/USING_THE_LOSS.md §1). Smooth pairwise concordance over
    comparable pairs (i had the event, j outlived i): value ≈ 1 - C-index and
    tracks the C-index throughout training. DROP-IN for cox_efron_loss — same risk
    convention (higher = higher risk = shorter survival), same (risk,time,event,mask)
    signature. No time-grid / IPCW / head change.

    risk  [B] — model output (higher = worse)
    time  [B] — RFS days
    event [B] — relapse indicator (1 = event, 0 = censored)
    mask  [B] — 1 where survival label is valid, 0 otherwise
    temperature — sigmoid smoothing; 0.1 is the recommended default (do not tune).
    """
    risk = risk.reshape(-1)
    time = time.reshape(-1).to(risk.dtype)
    event = event.reshape(-1).to(risk.dtype)
    if mask is not None:
        valid = mask.reshape(-1).bool() & (event >= 0)
        if valid.sum() < 2:
            return risk.new_zeros(())
        risk = risk[valid]; time = time[valid]; event = event[valid]
    ti = time.unsqueeze(1); tj = time.unsqueeze(0)
    comparable = (event.unsqueeze(1) > 0.5) & (tj > ti)        # i event, j outlives i
    if comparable.sum() == 0:
        return risk.new_zeros(())
    diff = risk.unsqueeze(1) - risk.unsqueeze(0)               # risk_i - risk_j
    per_pair = torch.sigmoid(-diff / temperature)             # ->1 when mis-ordered
    return (per_pair * comparable).sum() / comparable.sum()
