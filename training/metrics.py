"""
Metrics for HECKTOR 2026 leaderboard composite.

    seg       — Aggregated Dice (per-class TP and volume sums summed across the
                whole evaluation cohort, then 2*TP/sum). This is the official
                metric (DiceAggScore) — see
                third_party/HECKTOR2026/Task/Segmentation/utils/metrics.py
    staging   — balanced accuracy + macro recall for T and N
    prognosis — Harrell c-index and IPCW c-index

Composite val score (matches the challenge weighting):
    score = 0.25 * dice_agg_mean + 0.35 * (balacc_T + balacc_N)/2 + 0.40 * cindex
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Segmentation — official Aggregated Dice
# ---------------------------------------------------------------------------

def _tensor_to_sitk(tensor: torch.Tensor, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Lazy import — SimpleITK only loaded when DiceAggScore is used."""
    import SimpleITK as sitk
    if tensor.dim() == 4:                                       # [C, D, H, W]
        tensor = tensor[0]
    arr = tensor.detach().cpu().numpy().astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


class DiceAggScore:
    """Aggregated Dice — vendored from the official HECKTOR 2026 baseline.

    The dataset-level metric: TP and volume sums are accumulated across all
    patients first, then 2*TP/sum is computed once. This is more robust than
    per-patient mean Dice when class sizes are highly variable, and is the
    metric the challenge actually leaderboards.

    Source: third_party/HECKTOR2026/Task/Segmentation/utils/metrics.py
    """

    def __init__(self, class_labels: List[int] = (1, 2)):
        self.class_labels = list(class_labels)
        self._intermediate: List[Dict[str, float]] = []

    def _compute_volumes(self, image) -> Dict[str, float]:
        import SimpleITK as sitk
        spacing = image.GetSpacing()
        voxvol = spacing[0] * spacing[1] * spacing[2]
        stats = sitk.LabelStatisticsImageFilter()
        stats.Execute(image, image)
        out = {}
        for label in self.class_labels:
            try:
                out[f"vol{label}"] = stats.GetCount(label) * voxvol
            except RuntimeError:
                out[f"vol{label}"] = 0.0
        return out

    def update(self, pred_mask, target_mask) -> None:
        import SimpleITK as sitk
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkUInt8)
        pred_mask = caster.Execute(pred_mask)
        target_mask = caster.Execute(target_mask)

        if pred_mask.GetSize() != target_mask.GetSize() or \
                np.any(np.array(pred_mask.GetSpacing()) != np.array(target_mask.GetSpacing())):
            r = sitk.ResampleImageFilter()
            r.SetReferenceImage(target_mask)
            r.SetInterpolator(sitk.sitkNearestNeighbor)
            pred_mask = r.Execute(pred_mask)

        ovl = sitk.LabelOverlapMeasuresImageFilter()
        ovl.Execute(target_mask, pred_mask)
        vol_gt = self._compute_volumes(target_mask)
        vol_pr = self._compute_volumes(pred_mask)

        row = {}
        for label in self.class_labels:
            try:
                dsc = ovl.GetDiceCoefficient(label)
            except RuntimeError:
                dsc = 0.0
            vol_sum = vol_gt.get(f"vol{label}", 0) + vol_pr.get(f"vol{label}", 0)
            row[f"TP{label}"] = dsc * vol_sum / 2
            row[f"vol_sum{label}"] = vol_sum
        self._intermediate.append(row)

    def update_from_tensors(self, pred: torch.Tensor, target: torch.Tensor,
                             spacing=(1.0, 1.0, 1.0)) -> None:
        """pred: [B, C, D, H, W] logits  target: [B, D, H, W] integer."""
        b = pred.shape[0]
        for i in range(b):
            p = pred[i]
            t = target[i]
            if p.dim() > 3:
                p = torch.argmax(torch.softmax(p, dim=0), dim=0)
            self.update(_tensor_to_sitk(p, spacing), _tensor_to_sitk(t, spacing))

    def compute(self) -> Dict[str, float]:
        if not self._intermediate:
            return {"mean": 0.0}
        out = {}
        per_class = []
        for label in self.class_labels:
            tp_sum = float(sum(r.get(f"TP{label}", 0) for r in self._intermediate))
            vs_sum = float(sum(r.get(f"vol_sum{label}", 0) for r in self._intermediate))
            if vs_sum == 0:
                v = 1.0 if tp_sum == 0 else 0.0
            else:
                v = 2.0 * tp_sum / vs_sum
            out[f"Class_{label}"] = v
            per_class.append(v)
        out["mean"] = float(np.mean(per_class)) if per_class else 0.0
        return out

    def reset(self) -> None:
        self._intermediate = []


# ── HECKTOR 2026 official metric — GTVp ranking: mean DSC across patients ────
#
# Different from DiceAggScore (which is dataset-aggregated TP/(P+G) computed
# once). Per-patient mean averages DSC across patients, so each patient counts
# equally regardless of lesion size. Empty-prediction cases (DSC=0) drag the
# mean down harshly.
#
# Convention for empty-GT / empty-pred edge cases (matches the typical HECKTOR
# implementation):
#   GT empty + pred empty  → DSC = 1.0   (correct non-detection)
#   GT empty + pred ≠ ∅    → DSC = 0.0   (false positive)
#   GT ≠ ∅   + pred empty  → DSC = 0.0   (false negative)
class MeanDicePerPatient:
    """Per-patient Dice averaged over patients (HECKTOR 2026 GTVp metric)."""

    def __init__(self, class_labels: List[int] = (1, 2)):
        self.class_labels = list(class_labels)
        self._per_patient: Dict[int, List[float]] = {c: [] for c in self.class_labels}

    def update(self, pred_mask, target_mask) -> None:
        import SimpleITK as sitk
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkUInt8)
        pred_mask = caster.Execute(pred_mask)
        target_mask = caster.Execute(target_mask)

        if pred_mask.GetSize() != target_mask.GetSize() or \
                np.any(np.array(pred_mask.GetSpacing()) != np.array(target_mask.GetSpacing())):
            r = sitk.ResampleImageFilter()
            r.SetReferenceImage(target_mask)
            r.SetInterpolator(sitk.sitkNearestNeighbor)
            pred_mask = r.Execute(pred_mask)

        gt_arr = sitk.GetArrayFromImage(target_mask)
        pr_arr = sitk.GetArrayFromImage(pred_mask)
        for c in self.class_labels:
            gt_c = (gt_arr == c)
            pr_c = (pr_arr == c)
            gt_n = int(gt_c.sum()); pr_n = int(pr_c.sum())
            if gt_n == 0 and pr_n == 0:
                dsc = 1.0                          # both empty = trivial perfect
            elif gt_n == 0 or pr_n == 0:
                dsc = 0.0                          # one empty = total failure
            else:
                inter = int((gt_c & pr_c).sum())
                dsc = 2.0 * inter / (gt_n + pr_n)
            self._per_patient[c].append(dsc)

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        per_class = []
        for c in self.class_labels:
            v = float(np.mean(self._per_patient[c])) if self._per_patient[c] else 0.0
            out[f"Class_{c}"] = v
            per_class.append(v)
        out["mean"] = float(np.mean(per_class)) if per_class else 0.0
        return out

    def reset(self) -> None:
        self._per_patient = {c: [] for c in self.class_labels}


# ── HECKTOR 2026 GTVn detection ranking: Aggregated F1 over lesion instances ──
#
# Per-patient: label connected components in GT and pred GTVn masks. Build the
# IoU matrix over candidate (pred_cc, gt_cc) pairs. A pair counts as a detection
# match iff IoU > iou_threshold (official HECKTOR 2026 spec: 0.30). Greedy
# one-to-one assignment by IoU descending ensures each GT CC and each pred CC
# contribute to at most one TP. Count TP/FP/FN per patient, then sum across the
# cohort and compute F1 once. "Aggregated" matches the DiceAgg style: pool
# counts globally before dividing.
class AggregatedDetectionF1:
    """Lesion-level F1 aggregated across patients (HECKTOR 2026 GTVn metric).

    Default `iou_threshold=0.30` matches the official challenge spec
    (https://hecktor25.grand-challenge.org/tasks-and-evaluation/). Earlier
    revision matched at any-voxel overlap (`min_overlap_voxels=1`) which was
    overly permissive and inflated F1.
    """

    def __init__(self, class_label: int = 2,
                 iou_threshold: float = 0.30,
                 connectivity: int = 3,
                 min_overlap_voxels: int | None = None):
        # connectivity=3 → 26-neighborhood (full 3D), more permissive grouping
        self.class_label = class_label
        self.iou_threshold = float(iou_threshold)
        self.connectivity = connectivity
        # `min_overlap_voxels` is retained as a deprecated, no-op kwarg so old
        # callers don't crash. The matching rule is IoU-based regardless.
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._patients = 0

    def _label_cc(self, mask: np.ndarray) -> np.ndarray:
        from scipy.ndimage import label as cc_label, generate_binary_structure
        s = generate_binary_structure(3, self.connectivity)
        labels, _ = cc_label(mask, structure=s)
        return labels

    def update(self, pred_mask, target_mask) -> None:
        import SimpleITK as sitk
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkUInt8)
        pred_mask = caster.Execute(pred_mask)
        target_mask = caster.Execute(target_mask)

        if pred_mask.GetSize() != target_mask.GetSize() or \
                np.any(np.array(pred_mask.GetSpacing()) != np.array(target_mask.GetSpacing())):
            r = sitk.ResampleImageFilter()
            r.SetReferenceImage(target_mask)
            r.SetInterpolator(sitk.sitkNearestNeighbor)
            pred_mask = r.Execute(pred_mask)

        gt_bin = (sitk.GetArrayFromImage(target_mask) == self.class_label)
        pr_bin = (sitk.GetArrayFromImage(pred_mask) == self.class_label)

        gt_cc = self._label_cc(gt_bin)
        pr_cc = self._label_cc(pr_bin)
        n_gt = int(gt_cc.max()); n_pr = int(pr_cc.max())

        tp_p = 0
        if n_gt > 0 and n_pr > 0:
            # CC sizes (exclude background label 0).
            gt_sizes = np.bincount(gt_cc.ravel(), minlength=n_gt + 1)[1:]
            pr_sizes = np.bincount(pr_cc.ravel(), minlength=n_pr + 1)[1:]

            # Intersection size per (gt_label, pr_label) pair — only pairs that
            # actually share voxels appear. O(|overlap|) memory, much cheaper
            # than an (n_gt × n_pr) dense IoU matrix when most CCs don't touch.
            both = gt_bin & pr_bin
            if both.any():
                from collections import Counter
                gt_in_overlap = gt_cc[both]
                pr_in_overlap = pr_cc[both]
                inter_counter = Counter(zip(gt_in_overlap.tolist(),
                                            pr_in_overlap.tolist()))
                # Compute IoU per candidate pair, keep those above threshold.
                candidates = []
                for (g, p), inter in inter_counter.items():
                    union = int(gt_sizes[g - 1]) + int(pr_sizes[p - 1]) - inter
                    if union <= 0:
                        continue
                    iou = inter / union
                    if iou > self.iou_threshold:
                        candidates.append((iou, g, p))
                # Greedy one-to-one assignment by IoU descending.
                candidates.sort(reverse=True)
                gt_taken = set()
                pr_taken = set()
                for iou, g, p in candidates:
                    if g in gt_taken or p in pr_taken:
                        continue
                    gt_taken.add(g)
                    pr_taken.add(p)
                tp_p = len(gt_taken)
        fn_p = n_gt - tp_p                 # GT CCs missed by every pred CC
        fp_p = n_pr - tp_p                 # pred CCs that didn't match a GT CC
        self._tp += tp_p
        self._fn += fn_p
        self._fp += fp_p
        self._patients += 1

    def compute(self) -> Dict[str, float]:
        tp, fp, fn = self._tp, self._fp, self._fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "tp": tp, "fp": fp, "fn": fn,
            "n_patients": self._patients,
        }

    def reset(self) -> None:
        self._tp = self._fp = self._fn = 0
        self._patients = 0


def dice_per_class(pred: torch.Tensor, target: torch.Tensor, n_classes: int = 3) -> dict:
    """Per-patient classical Dice — kept for fast train-time logging only.

    Use DiceAggScore for the leaderboard-equivalent metric.
    """
    out = {}
    pred = pred.argmax(dim=1) if pred.dim() == target.dim() + 1 else pred
    target_squeezed = target.squeeze(1) if target.dim() == pred.dim() + 1 else target
    for c in range(1, n_classes):
        p = (pred == c).float()
        t = (target_squeezed == c).float()
        inter = (p * t).sum().item()
        denom = p.sum().item() + t.sum().item()
        out[c] = (2 * inter + 1e-6) / (denom + 1e-6)
    return out


# --- Staging ----------------------------------------------------------------
def balanced_accuracy(logits: torch.Tensor, target: torch.Tensor,
                      n_classes: int) -> float:
    pred = logits.argmax(dim=-1)
    recalls = []
    for c in range(n_classes):
        m = target == c
        if m.sum() == 0:
            continue
        recalls.append((pred[m] == c).float().mean().item())
    return float(np.mean(recalls)) if recalls else 0.0


# --- Prognosis --------------------------------------------------------------
def harrell_cindex(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Standard Harrell C-index — concordance among comparable pairs."""
    n = len(risk)
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if time[i] < time[j] and event[i] == 1:
                den += 1
                if risk[i] > risk[j]:
                    num += 1
                elif risk[i] == risk[j]:
                    num += 0.5
    return num / den if den > 0 else 0.0


def composite_score(dice_agg_mean: float, balacc_t: float, balacc_n: float,
                    cindex: float) -> float:
    """Reproduce the leaderboard weighting: 25% seg + 35% staging + 40% prognosis."""
    return (0.25 * dice_agg_mean
            + 0.35 * 0.5 * (balacc_t + balacc_n)
            + 0.40 * cindex)
