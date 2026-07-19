"""MEDAI-hybrid Task 2 (+ optional Task 3) trainer — port of the other-server
patch96 ResNet18 + MaskBranch + Clinical model.

Pipeline:
  1. Read CT, PET (raw native), predicted mask (CC-filtered, native).
  2. Resample CT/PET to 1mm iso (linear), mask to 1mm iso (nearest).
  3. Apply locked CC filter (mm³ 1000/500 + top-2/8) on resampled mask.
  4. Compute centroid of GTVp ∪ GTVn voxels (fallback: volume center).
  5. Pre-crop 116³ around centroid (margin for ±10-voxel train-time jitter).
     Save to /data/kwang/medai_patch_cache/<pid>.pt (~16 GB total).
  6. Train: random ±10 jitter → 96³ crop; val: centered 96³ crop.
  7. CT clip [-1000, 3000] → [0,1]; PET clip [0, 25] → [0,1]; mask 0/1/2.
  8. DualHeadFusionResNet + CE(T valid only) + CE(N).
  9. Per-fold best-balacc ckpt + per-patient val predictions CSV.

Usage:
    # one-off cache build (skips already-extracted):
    python scripts/train_medai_hybrid.py --build_cache --fold 0

    # 5-fold training (chain):
    python scripts/train_medai_hybrid.py --fold 0 --gpu 0
    python scripts/train_medai_hybrid.py --fold 1 --gpu 1
    ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label as cc_label, generate_binary_structure
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.medai_hybrid import DualHeadFusionResNet                          # noqa: E402
from training.losses import cox_efron_loss, sigmoid_concordance_loss          # noqa: E402
from training.mtlr import (make_time_bins, encode_survival,                   # noqa: E402
                            mtlr_neg_log_likelihood, mtlr_risk)
from training.metrics import harrell_cindex                                   # noqa: E402

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────
RAW = ROOT / "data/raw"
MANIFEST = ROOT / "data/manifests"
CSV_PATH = ROOT / "data/raw/HECKTOR_2026_training_data.csv"
PRED_DIR_PAT = Path(os.environ.get(
    "MEDAI_PRED_DIR_PAT",
    str(ROOT / "evaluation/results/qa_stunet_aug_fold{fold}_masks")))   # env-override for the
# 10-fold-seg retrain (set MEDAI_PRED_DIR_PAT=.../qa_10fold_oof_fold{fold}_masks). Default = old 5-fold masks.
# Env-configurable for the B (patch-size) / C (mask-mode) ablations. Defaults
# reproduce the original 96³ / 116-precrop / CC-mask pipeline exactly.
CACHE_DIR = Path(os.environ.get("MEDAI_CACHE_DIR", "/data/kwang/medai_patch_cache"))
PATCH_SIZE = int(os.environ.get("MEDAI_PATCH_SIZE", "96"))                   # model input (out) size
PRECROP_SIZE = int(os.environ.get("MEDAI_PRECROP", str(PATCH_SIZE + 20)))    # cached patch size
CACHE_MARGIN = (PRECROP_SIZE - PATCH_SIZE) // 2                              # derived (10 by default)
MASK_MODE = os.environ.get("MEDAI_MASK_MODE", "cc").strip().lower()          # cc | raw (applied at load)
# Intensity normalization (defaults reproduce the original patch pipeline).
# Ablation: align CT to the seg soft-tissue window [-200,200]; test PET variants.
CT_LO = float(os.environ.get("MEDAI_CT_LO", "-1000"))                        # seg uses -200
CT_HI = float(os.environ.get("MEDAI_CT_HI", "3000"))                         # seg uses  200
PET_NORM = os.environ.get("MEDAI_PET_NORM", "clamp25").strip().lower()       # clamp25 | clamp40 | pct
# Ordinal (squared-EMD) auxiliary loss weight on the T/N heads. 0.0 = off (plain
# CE, the locked recipe). >0 adds an EMD term that penalizes predictions farther
# from the true ordinal class (T1<T2<T3<T4, N0<N1<N2<N3). B-phase experiment.
ORDINAL_LAMBDA = float(os.environ.get("MEDAI_ORDINAL_LAMBDA", "0.0"))
# Which heads get the ordinal EMD term: both | n | t. The sweep showed EMD helps
# N (genuine ordinal nodal burden) but hurts the weaker T head → default to N-only
# is NOT set here (back-compat); experiments pass MEDAI_ORDINAL_TARGET explicitly.
ORDINAL_TARGET = os.environ.get("MEDAI_ORDINAL_TARGET", "both").strip().lower()
# RFS survival loss: 'cox' (default, Efron partial likelihood) | 'mtlr'
# (discrete-time Multi-Task Logistic Regression). MTLR head outputs MTLR_BINS-1
# phi scores; bins are quantiles of the training event times.
RFS_LOSS = os.environ.get("MEDAI_RFS_LOSS", "cox").strip().lower()
MTLR_BINS = int(os.environ.get("MEDAI_MTLR_BINS", "10"))
CCF_MIN_MM3 = {1: 1000.0, 2: 500.0}
CCF_TOPN = {1: 2, 2: 8}

T_MAP = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T4A": 3, "T4B": 3, "TX": 4}
N_MAP = {"N0": 0, "N1": 1, "N2": 2, "N2A": 2, "N2B": 2, "N2C": 2, "N3": 3}
T_LABELS = ["T1", "T2", "T3", "T4", "Tx"]
N_LABELS = ["N0", "N1", "N2", "N3"]


def soft_emd_loss(logits: torch.Tensor, target: torch.Tensor,
                  ordinal_max: int | None = None, ignore_index: int = -1) -> torch.Tensor:
    """Squared Earth-Mover's-Distance between the softmax CDF and the target CDF.

    Encodes ordinality: a prediction farther (in class index) from the true class
    is penalized more. Only samples with target != ignore_index (and, if given,
    target in [0, ordinal_max]) contribute — for T this excludes the non-ordinal
    Tx class (index 4) while T1..T4 (0..3) get the EMD term. Used as an auxiliary
    regularizer ON TOP of cross-entropy."""
    valid = target != ignore_index
    if ordinal_max is not None:
        valid = valid & (target >= 0) & (target <= ordinal_max)
    if int(valid.sum()) == 0:
        return logits.sum() * 0.0
    lg = logits[valid]
    tg = target[valid]
    p = torch.softmax(lg, dim=-1)
    cdf_p = torch.cumsum(p, dim=-1)
    onehot = torch.zeros_like(p)
    onehot.scatter_(1, tg.unsqueeze(1), 1.0)
    cdf_t = torch.cumsum(onehot, dim=-1)
    return ((cdf_p - cdf_t) ** 2).sum(-1).mean()


def _extract_risk(out: dict) -> np.ndarray:
    """Per-patient scalar risk from a model output dict, loss-agnostic.
    cox → out['risk']; mtlr → expected-event-bin via mtlr_risk(phi)."""
    if RFS_LOSS == "mtlr":
        return mtlr_risk(out["mtlr_phi"].float()).detach().cpu().numpy()
    return out["risk"].squeeze(-1).detach().cpu().numpy()


CLINICAL_COLS = [
    "age_z", "gender_male",
    "hpv_pos", "hpv_neg", "hpv_unk",
    "smoker_yes", "smoker_no", "smoker_missing",
    "drinker_yes", "drinker_no", "drinker_missing",
    "ps_0", "ps_1", "ps_high", "ps_missing",
    "treatment_rt", "treatment_other", "treatment_missing",
]
CLIN_DIM = len(CLINICAL_COLS)


# ── Clinical encoding (matches docker/inference.py) ────────────────────────

def _encode_clinical(row: pd.Series) -> np.ndarray:
    age = row.get("Age")
    age_z = (float(age) - 60.0) / 12.0 if pd.notna(age) else 0.0
    g = row.get("Gender")
    gender_male = 1.0 if pd.notna(g) and float(g) == 1.0 else 0.0
    hpv_v = row.get("HPV Status")
    if pd.isna(hpv_v):
        hpv_pos = hpv_neg = 0.0; hpv_unk = 1.0
    else:
        v = float(hpv_v)
        hpv_pos, hpv_neg = float(v == 1.0), float(v == 0.0)
        hpv_unk = float(hpv_pos + hpv_neg == 0)
    smoker_v = row.get("Tobacco Consumption")
    if pd.isna(smoker_v):
        smoker_yes = smoker_no = 0.0; smoker_missing = 1.0
    else:
        v = float(smoker_v)
        smoker_yes, smoker_no, smoker_missing = float(v == 1.0), float(v == 0.0), 0.0
    drinker_v = row.get("Alcohol Consumption")
    if pd.isna(drinker_v):
        drinker_yes = drinker_no = 0.0; drinker_missing = 1.0
    else:
        v = float(drinker_v)
        drinker_yes, drinker_no, drinker_missing = float(v == 1.0), float(v == 0.0), 0.0
    ps_v = row.get("Performance Status")
    if pd.isna(ps_v):
        ps_0 = ps_1 = ps_high = 0.0; ps_missing = 1.0
    else:
        v = float(ps_v)
        ps_0, ps_1, ps_high, ps_missing = float(v == 0.0), float(v == 1.0), float(v >= 2.0), 0.0
    tx_v = row.get("Treatment")
    if pd.isna(tx_v):
        treatment_rt = treatment_other = 0.0; treatment_missing = 1.0
    else:
        v = float(tx_v)
        treatment_rt, treatment_other, treatment_missing = float(v == 1.0), float(v == 0.0), 0.0
    return np.array([age_z, gender_male, hpv_pos, hpv_neg, hpv_unk,
                      smoker_yes, smoker_no, smoker_missing,
                      drinker_yes, drinker_no, drinker_missing,
                      ps_0, ps_1, ps_high, ps_missing,
                      treatment_rt, treatment_other, treatment_missing],
                     dtype=np.float32)


# ── Resampling + CC-filter ─────────────────────────────────────────────────

def _resample_to_iso(img: sitk.Image, spacing: float, is_label: bool) -> sitk.Image:
    in_sp = img.GetSpacing()
    in_sz = img.GetSize()
    out_sp = (spacing, spacing, spacing)
    out_sz = [int(round(s * in_sp[i] / spacing)) for i, s in enumerate(in_sz)]
    rsl = sitk.ResampleImageFilter()
    rsl.SetOutputSpacing(out_sp)
    rsl.SetSize(out_sz)
    rsl.SetOutputDirection(img.GetDirection())
    rsl.SetOutputOrigin(img.GetOrigin())
    rsl.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return rsl.Execute(img)


def _apply_cc_filter(arr: np.ndarray, voxvol_mm3: float, struct) -> np.ndarray:
    out = arr.copy()
    for cls, min_mm3 in CCF_MIN_MM3.items():
        keep_n = CCF_TOPN[cls]
        mask = (arr == cls)
        if not mask.any():
            continue
        labels, n_cc = cc_label(mask, structure=struct)
        if n_cc == 0:
            continue
        sizes = np.bincount(labels.flat, minlength=n_cc + 1)
        survivors = []
        for k in range(1, n_cc + 1):
            if sizes[k] * voxvol_mm3 < min_mm3:
                out[labels == k] = 0
            else:
                survivors.append((k, int(sizes[k])))
        if keep_n > 0 and len(survivors) > keep_n:
            survivors.sort(key=lambda kv: -kv[1])
            for k, _ in survivors[keep_n:]:
                out[labels == k] = 0
    return out


# ── Pre-crop cache builder ─────────────────────────────────────────────────

def _find_pred_mask(pid: str) -> Path | None:
    for f in range(5):
        p = Path(str(PRED_DIR_PAT).format(fold=f)) / f"fold{f}_{pid}.mha"
        if p.exists():
            return p
    return None


def build_one_patient(pid: str, force: bool = False) -> bool:
    out = CACHE_DIR / f"{pid}.pt"
    if out.exists() and not force:
        return True
    ct_p = RAW / pid / f"{pid}__CT.nii.gz"
    pt_p = RAW / pid / f"{pid}__PT.nii.gz"
    pred_p = _find_pred_mask(pid)
    if not (ct_p.exists() and pt_p.exists() and pred_p):
        return False

    ct_raw = sitk.ReadImage(str(ct_p))
    pt_raw = sitk.ReadImage(str(pt_p))
    pred_raw = sitk.ReadImage(str(pred_p))

    # Resample CT to 1mm iso, then align PET + pred to that grid.
    ct = _resample_to_iso(ct_raw, 1.0, is_label=False)
    rsl = sitk.ResampleImageFilter()
    rsl.SetReferenceImage(ct)
    rsl.SetInterpolator(sitk.sitkLinear)
    pt = rsl.Execute(pt_raw)
    rsl.SetInterpolator(sitk.sitkNearestNeighbor)
    pred = rsl.Execute(pred_raw)

    ct_arr = sitk.GetArrayFromImage(ct).astype(np.float32)
    pt_arr = sitk.GetArrayFromImage(pt).astype(np.float32)
    pred_arr = sitk.GetArrayFromImage(pred).astype(np.uint8)

    # CC filter ONLY to compute a clean centroid; STORE the RAW mask so the
    # load-time MASK_MODE can choose raw vs CC (the C ablation). With MASK_MODE=cc
    # (default) the same filter is re-applied at load → identical to the old cache.
    struct = generate_binary_structure(3, 3)
    pred_cc = _apply_cc_filter(pred_arr, 1.0, struct)

    # Centroid (mean of GTVp ∪ GTVn voxels); fall back to volume center.
    mask_fg = pred_cc > 0
    if mask_fg.any():
        idx = np.argwhere(mask_fg)
        centroid = idx.mean(axis=0)                                    # (z, y, x)
    else:
        centroid = np.array([ct_arr.shape[0] // 2, ct_arr.shape[1] // 2,
                              ct_arr.shape[2] // 2], dtype=np.float64)

    H = PRECROP_SIZE
    half = H // 2
    # Pad if needed so we can crop H centered at centroid.
    pad = [(half, half), (half, half), (half, half)]
    ct_p_arr = np.pad(ct_arr, pad, mode="constant", constant_values=-1024)
    pt_p_arr = np.pad(pt_arr, pad, mode="constant", constant_values=0)
    pr_p_arr = np.pad(pred_arr, pad, mode="constant", constant_values=0)
    cz, cy, cx = (centroid + np.array([half, half, half])).round().astype(int)
    # Clip centers so the crop fits inside padded volume
    cz = int(np.clip(cz, half, ct_p_arr.shape[0] - half))
    cy = int(np.clip(cy, half, ct_p_arr.shape[1] - half))
    cx = int(np.clip(cx, half, ct_p_arr.shape[2] - half))
    sl = (slice(cz - half, cz + half),
          slice(cy - half, cy + half),
          slice(cx - half, cx + half))
    # .copy() materialises the 116³ patch — without it, the slice is a view
    # of the (potentially gigantic) padded volume; torch.save then dumps the
    # entire underlying storage → 393 MB files for whole-body CHUV/CHUP CTs.
    ct_patch = np.ascontiguousarray(ct_p_arr[sl])
    pt_patch = np.ascontiguousarray(pt_p_arr[sl])
    pr_patch = np.ascontiguousarray(pr_p_arr[sl])

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "ct": torch.from_numpy(ct_patch).float(),                       # raw HU
        "pt": torch.from_numpy(pt_patch).float(),                       # raw SUV
        "mask": torch.from_numpy(pr_patch).to(torch.uint8),             # 0/1/2
    }, out)
    return True


def build_cache(pids: list[str], force: bool = False) -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_fail = 0
    t0 = time.time()
    for i, pid in enumerate(pids):
        try:
            if build_one_patient(pid, force):
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:                                         # noqa: BLE001
            n_fail += 1
            print(f"  fail: {pid}: {e}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  cache {i + 1}/{len(pids)}  ok={n_ok}  fail={n_fail}  "
                  f"elapsed={(time.time()-t0)/60:.1f} min", flush=True)
    return n_ok


# ── Dataset ─────────────────────────────────────────────────────────────────

def _augment_patch(image: torch.Tensor, mask: torch.Tensor) -> tuple:
    """Conservative 3D augmentation applied to (image, mask) post-crop.

    All transforms are p=0.5 independently. Kept gentle since 782-patient
    training set is small and gross deformations break anatomical priors.

    image: (C=2, D, H, W) float32 in [0, 1]
    mask:  (1, D, H, W) integer 0/1/2

    Augmentations:
      1. LR flip (axis=W, last axis) — head/neck has ~bilateral symmetry,
         GTVp/GTVn stage labels are LR-invariant.
      2. Per-channel brightness shift ±0.05.
      3. Per-channel gamma ∈ [0.85, 1.15] (clamped to [0,1] before/after).
      4. Small rotation ±10° around the superior-inferior (Z) axis ONLY.
         Z-axis rotation preserves the head's natural orientation.

    NOT used (intentionally):
      - flip along AP / IS (would invert anatomy)
      - large rotations (>15°)
      - elastic deformation
      - PET channel dropout
    """
    # 1. LR flip
    if np.random.rand() < 0.5:
        image = torch.flip(image, dims=[-1])
        mask  = torch.flip(mask,  dims=[-1])
    # 2. Brightness shift (per channel, additive in [0, 1] domain)
    if np.random.rand() < 0.5:
        delta = (torch.rand(2) - 0.5) * 0.10                            # ±0.05
        image = (image + delta.view(2, 1, 1, 1)).clamp(0.0, 1.0)
    # 3. Gamma per channel
    if np.random.rand() < 0.5:
        g = 0.85 + torch.rand(2) * 0.30                                  # [0.85, 1.15]
        image = image.clamp(min=1e-6).pow(g.view(2, 1, 1, 1)).clamp(0.0, 1.0)
    # 4. Small rotation around Z (= depth) axis. Both image and mask rotated.
    if np.random.rand() < 0.5:
        angle = float((np.random.rand() - 0.5) * 20.0)                   # ±10°
        # torchvision rotate is 2D; use scipy for 3D-around-Z (rotate H,W plane)
        import scipy.ndimage as ndi
        img_arr = image.numpy()
        msk_arr = mask.numpy().astype(np.float32)
        # axes=(2,3) means rotate H/W plane (= axial slice rotation, z stays as depth)
        img_r = ndi.rotate(img_arr, angle, axes=(2, 3), reshape=False,
                             order=1, mode="constant", cval=0.0)
        msk_r = ndi.rotate(msk_arr, angle, axes=(1, 2), reshape=False,
                             order=0, mode="constant", cval=0.0)
        image = torch.from_numpy(img_r).clamp(0.0, 1.0)
        mask  = torch.from_numpy(msk_r).round().to(torch.long)
    return image, mask


def _affine_crop_raw(ct, pt, mask, out_size, in_size, angle_deg, jitter):
    """Augment-THEN-crop: sample the OUT_SIZE^3 output directly from the larger
    IN_SIZE^3 pre-crop in a single affine resample (axial rotation + translation),
    with edge ('nearest') border — so corners are filled with REAL anatomy instead
    of the zero-padding that the old crop-then-rotate produced. Operates on RAW
    (un-normalised) tensors; jitter = (jz, jy, jx) voxels of translation.
    """
    import scipy.ndimage as ndi
    a = np.deg2rad(angle_deg); c = float(np.cos(a)); sn = float(np.sin(a))
    M = np.array([[1.0, 0.0, 0.0],
                  [0.0, c, -sn],
                  [0.0, sn, c]], dtype=np.float64)              # rotate (y, x) = axial plane
    ic = (in_size - 1) / 2.0; oc = (out_size - 1) / 2.0
    offset = (np.array([ic, ic, ic]) - M @ np.array([oc, oc, oc])
              + np.asarray(jitter, dtype=np.float64))
    def warp(arr, order):
        return ndi.affine_transform(arr, M, offset=offset,
                                    output_shape=(out_size, out_size, out_size),
                                    order=order, mode="nearest")
    ctn = warp(ct.numpy().astype(np.float32), 1)
    ptn = warp(pt.numpy().astype(np.float32), 1)
    mkn = warp(mask.numpy().astype(np.float32), 0)
    return (torch.from_numpy(ctn).float(),
            torch.from_numpy(ptn).float(),
            torch.from_numpy(np.rint(mkn)).to(torch.uint8))


def _intensity_flip_aug(image, mask):
    """Non-spatial augs safe to apply post-crop (no corner artifacts): LR flip +
    per-channel brightness ±0.05 + gamma [0.85, 1.15]. Rotation/translation are
    handled earlier by _affine_crop_raw."""
    if np.random.rand() < 0.5:
        image = torch.flip(image, dims=[-1]); mask = torch.flip(mask, dims=[-1])
    if np.random.rand() < 0.5:
        delta = (torch.rand(2) - 0.5) * 0.10
        image = (image + delta.view(2, 1, 1, 1)).clamp(0.0, 1.0)
    if np.random.rand() < 0.5:
        g = 0.85 + torch.rand(2) * 0.30
        image = image.clamp(min=1e-6).pow(g.view(2, 1, 1, 1)).clamp(0.0, 1.0)
    return image, mask


def _norm_ct(t: torch.Tensor) -> torch.Tensor:
    """CT → [0,1] over the configured HU window (default [-1000,3000]; seg uses
    [-200,200])."""
    return (t.clamp(min=CT_LO, max=CT_HI) - CT_LO) / (CT_HI - CT_LO)


def _norm_pet(t: torch.Tensor) -> torch.Tensor:
    """PET → [0,1]. clamp25/clamp40 keep ABSOLUTE SUV (prognostic); pct = per-patch
    percentile [0.5,99.5] (seg-style, scale-invariant but discards absolute SUV)."""
    if PET_NORM == "pct":
        v = t.flatten()
        lo = torch.quantile(v, 0.005); hi = torch.quantile(v, 0.995)
        return ((t - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
    if PET_NORM == "clamp40":
        return t.clamp(min=0.0, max=40.0) / 40.0
    return t.clamp(min=0.0, max=25.0) / 25.0


class MedAIPatchDataset(Dataset):
    def __init__(self, pids: list[str], clin_df: pd.DataFrame,
                 csv_df: pd.DataFrame, train: bool, jitter: int = 10,
                 augment: bool = False, affine_aug: bool = False):
        self.pids = [p for p in pids if (CACHE_DIR / f"{p}.pt").exists()]
        self.clin = clin_df.loc[self.pids].values.astype(np.float32)
        self.train = train
        self.jitter = jitter
        self.augment = augment
        self.affine_aug = affine_aug
        # Labels
        t = []; n = []; rfs = []; rel = []; surv_ok = []
        for p in self.pids:
            row = csv_df.loc[p]
            ts = str(row.get("T-stage")).strip().upper() if pd.notna(row.get("T-stage")) else ""
            ns = str(row.get("N-stage")).strip().upper() if pd.notna(row.get("N-stage")) else ""
            t.append(T_MAP.get(ts, -1))
            n.append(N_MAP.get(ns, -1))
            rfs_v = row.get("RFS")
            rel_v = row.get("Relapse")
            if pd.notna(rfs_v) and pd.notna(rel_v):
                rfs.append(float(rfs_v)); rel.append(float(rel_v)); surv_ok.append(True)
            else:
                rfs.append(0.0); rel.append(0.0); surv_ok.append(False)
        self.t = np.array(t, dtype=np.int64)
        self.n = np.array(n, dtype=np.int64)
        self.rfs = np.array(rfs, dtype=np.float32)
        self.rel = np.array(rel, dtype=np.float32)
        self.surv_ok = np.array(surv_ok, dtype=bool)

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, idx: int) -> dict:
        pid = self.pids[idx]
        d = torch.load(CACHE_DIR / f"{pid}.pt")
        ct, pt, mask = d["ct"], d["pt"], d["mask"]                     # (PRECROP^3) each
        if MASK_MODE == "cc":
            # Re-apply the CC filter on the full pre-crop mask (no-op if the cache
            # already stored a CC mask). MASK_MODE=raw skips this → keeps small/
            # spurious components (the C ablation).
            m = _apply_cc_filter(mask.numpy().astype(np.uint8), 1.0,
                                 generate_binary_structure(3, 3))
            mask = torch.from_numpy(m).to(mask.dtype)
        if self.train and self.augment and self.affine_aug:
            # NEW: augment-then-crop. Rotation (±10° axial) + translation jitter
            # are composed into ONE affine resample from the 116^3 pre-crop down
            # to PATCH_SIZE, edge-filled (no zero corners). Then non-spatial augs.
            if self.jitter > 0:
                jz = np.random.randint(-self.jitter, self.jitter + 1)
                jy = np.random.randint(-self.jitter, self.jitter + 1)
                jx = np.random.randint(-self.jitter, self.jitter + 1)
            else:
                jz = jy = jx = 0
            angle = float((np.random.rand() - 0.5) * 20.0)             # ±10°
            ct_c, pt_c, mask_c = _affine_crop_raw(ct, pt, mask, PATCH_SIZE,
                                                  PRECROP_SIZE, angle, (jz, jy, jx))
            ct = _norm_ct(ct_c)
            pt = _norm_pet(pt_c)
            mask = mask_c.to(torch.long)
            image = torch.stack([ct, pt], dim=0)
            mask = mask.unsqueeze(0)
            image, mask = _intensity_flip_aug(image, mask)
            return {
                "patient_id": pid,
                "image": image,
                "mask": mask,
                "clinical": torch.from_numpy(self.clin[idx]),
                "t_gt": int(self.t[idx]),
                "n_gt": int(self.n[idx]),
                "rfs_days": float(self.rfs[idx]),
                "relapse": float(self.rel[idx]),
                "surv_ok": bool(self.surv_ok[idx]),
            }
        # ORIGINAL path (slice crop + optional old post-crop _augment_patch).
        # Sample crop offset
        if self.train and self.jitter > 0:
            jz = np.random.randint(-self.jitter, self.jitter + 1)
            jy = np.random.randint(-self.jitter, self.jitter + 1)
            jx = np.random.randint(-self.jitter, self.jitter + 1)
        else:
            jz = jy = jx = 0
        ofs = CACHE_MARGIN
        # Clamp offsets to [0, PRECROP-PATCH] so the slice is ALWAYS exactly
        # PATCH_SIZE — jitter can exceed CACHE_MARGIN when the margin is small
        # (e.g. PATCH=160/PRECROP=176 → margin 8 < jitter 10), which otherwise
        # clips the last axis to <PATCH_SIZE and breaks batch stacking.
        maxo = PRECROP_SIZE - PATCH_SIZE
        z0 = min(max(ofs + jz, 0), maxo)
        y0 = min(max(ofs + jy, 0), maxo)
        x0 = min(max(ofs + jx, 0), maxo)
        sl = (slice(z0, z0 + PATCH_SIZE),
              slice(y0, y0 + PATCH_SIZE),
              slice(x0, x0 + PATCH_SIZE))
        ct = _norm_ct(ct[sl])                                           # [0,1]
        pt = _norm_pet(pt[sl])                                          # [0,1]
        mask = mask[sl].to(torch.long)
        image = torch.stack([ct, pt], dim=0)                            # (2, P, P, P)
        mask = mask.unsqueeze(0)                                        # (1, P, P, P)
        if self.train and self.augment:
            image, mask = _augment_patch(image, mask)
        return {
            "patient_id": pid,
            "image": image,
            "mask": mask,
            "clinical": torch.from_numpy(self.clin[idx]),
            "t_gt": int(self.t[idx]),
            "n_gt": int(self.n[idx]),
            "rfs_days": float(self.rfs[idx]),
            "relapse": float(self.rel[idx]),
            "surv_ok": bool(self.surv_ok[idx]),
        }


# ── Training ────────────────────────────────────────────────────────────────

def _load_tn_predictions(tn_dir: Path) -> dict[str, np.ndarray]:
    """Load fold-disjoint TN softmax predictions.
    5-fold mode: scans tn_dir/fold{0..4}/predictions.csv.
    3/1/1 mode: also loads tn_dir/val*_test*/predictions.csv (val) and
                  test_predictions.csv (test), giving OOF coverage for the
                  val + test patients while the model trained on the other 3.
    Returns {pid: 9-d ndarray} concatenating [t_softmax_0..4, n_softmax_0..3].
    """
    rows = []
    for f in range(5):
        p = tn_dir / f"fold{f}" / "predictions.csv"
        if p.exists():
            rows.append(pd.read_csv(p))
    # 3/1/1 layouts: val{V}_test{T}/predictions.csv + test_predictions.csv
    for sub in tn_dir.glob("val*_test*"):
        for fn in ("predictions.csv", "test_predictions.csv"):
            p = sub / fn
            if p.exists():
                rows.append(pd.read_csv(p))
    if not rows:
        return {}
    df = pd.concat(rows, ignore_index=True).drop_duplicates("patient_id")
    out = {}
    t_cols = [c for c in df.columns if c.startswith("t_softmax_")]
    n_cols = [c for c in df.columns if c.startswith("n_softmax_")]
    cols = t_cols + n_cols
    for _, r in df.iterrows():
        out[str(r["patient_id"])] = np.array([float(r[c]) for c in cols],
                                              dtype=np.float32)
    return out


def train_fold(fold: int, epochs: int, batch_size: int, lr: float,
               device: torch.device, out_dir: Path, n_workers: int = 4,
               patience: int = 20, rfs_mode: str = "none",
               rfs_warmup: int = 30, detach_tn: bool = True,
               wandb_group: str = "medai_hybrid_patch96",
               tn_predictions_dir: Path | None = None,
               augment: bool = False,
               affine_aug: bool = False,
               test_fold: int | None = None) -> dict:
    """rfs_mode = 'none' (Task 2 only) | 'rfs_only' (no T/N loss) | 'triplehead'

    If tn_predictions_dir is given, the fold-disjoint TN softmax (9-d) is
    appended to the 18-d clinical → 27-d clinical input. Lets RFS-only use
    Task 2 stage predictions as a prognostic prior (TN stage is a strong RFS
    predictor in HN cancer)."""
    df = pd.read_csv(CSV_PATH)
    df["PatientID"] = df["PatientID"].astype(str).str.strip()
    df = df.set_index("PatientID")
    val_pids = pd.read_csv(MANIFEST / f"val_fold{fold}.csv")["patient_id"]\
                  .astype(str).str.strip().tolist()
    if test_fold is not None:
        test_pids = pd.read_csv(MANIFEST / f"val_fold{test_fold}.csv")["patient_id"]\
                       .astype(str).str.strip().tolist()
    else:
        test_pids = []
    exclude = set(val_pids) | set(test_pids)
    train_pids = [p for p in df.index if p not in exclude]

    # Filter to patients with cached patch + valid labels (need at least one of T/N).
    def _has_lab(p):
        if p not in df.index: return False
        row = df.loc[p]
        ts = str(row.get("T-stage")).strip().upper() if pd.notna(row.get("T-stage")) else ""
        ns = str(row.get("N-stage")).strip().upper() if pd.notna(row.get("N-stage")) else ""
        return ts in T_MAP or ns in N_MAP

    train_pids = [p for p in train_pids if _has_lab(p) and (CACHE_DIR / f"{p}.pt").exists()]
    val_pids = [p for p in val_pids if _has_lab(p) and (CACHE_DIR / f"{p}.pt").exists()]
    test_pids = [p for p in test_pids if _has_lab(p) and (CACHE_DIR / f"{p}.pt").exists()]
    if test_fold is not None:
        print(f"3v1t (val={fold}, test={test_fold}): "
              f"train n={len(train_pids)}, val n={len(val_pids)}, test n={len(test_pids)}",
              flush=True)
    else:
        print(f"fold{fold}: train n={len(train_pids)}, val n={len(val_pids)}", flush=True)

    clin_df = pd.DataFrame({p: dict(zip(CLINICAL_COLS, _encode_clinical(df.loc[p])))
                             for p in df.index}).T

    # Optional: append fold-disjoint TN softmax to clinical features.
    extra_dim = 0
    if tn_predictions_dir is not None and tn_predictions_dir.exists():
        tn_pred = _load_tn_predictions(tn_predictions_dir)
        if tn_pred:
            extra_dim = len(next(iter(tn_pred.values())))
            tn_cols = [f"tn_softmax_{i}" for i in range(extra_dim)]
            tn_rows = []
            for pid in clin_df.index:
                vec = tn_pred.get(pid, np.zeros(extra_dim, dtype=np.float32))
                tn_rows.append(dict(zip(tn_cols, vec.tolist())))
            tn_df = pd.DataFrame(tn_rows, index=clin_df.index)
            clin_df = pd.concat([clin_df, tn_df], axis=1)
            print(f"[tn_predictions] loaded {len(tn_pred)} patients, "
                  f"clinical_dim {CLIN_DIM} → {CLIN_DIM + extra_dim}", flush=True)

    train_ds = MedAIPatchDataset(train_pids, clin_df, df, train=True,
                                   augment=augment, affine_aug=affine_aug)
    val_ds = MedAIPatchDataset(val_pids, clin_df, df, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=n_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=n_workers, pin_memory=True)

    with_rfs = rfs_mode in ("rfs_only", "triplehead")
    full_clin_dim = CLIN_DIM + extra_dim
    # MTLR time bins from TRAIN event times (cox path ignores these).
    mtlr_bins_t = None
    if with_rfs and RFS_LOSS in ("mtlr", "both"):
        ok = train_ds.surv_ok
        mtlr_bins_t = make_time_bins(train_ds.rfs[ok], train_ds.rel[ok],
                                      num_bins=MTLR_BINS).to(device)
        print(f"MTLR: {MTLR_BINS} bins, {mtlr_bins_t.numel()} cut points = "
              f"{mtlr_bins_t.cpu().numpy().round(1).tolist()}", flush=True)
    model = DualHeadFusionResNet(image_channels=2, clinical_dim=full_clin_dim,
                                  n_t_classes=len(T_LABELS),
                                  n_n_classes=len(N_LABELS),
                                  hidden_dim=128, dropout=0.1,
                                  mask_branch=True, with_rfs=with_rfs,
                                  rfs_loss=RFS_LOSS, mtlr_bins=MTLR_BINS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    print(f"rfs_mode={rfs_mode} (with_rfs={with_rfs}, detach_tn={detach_tn}, "
          f"warmup={rfs_warmup}, rfs_loss={RFS_LOSS})", flush=True)

    # wandb logging (offline-safe — uses existing project)
    try:
        import wandb
        wandb_run = wandb.init(project="hecktor2026",
                                 name=f"{wandb_group}_fold{fold}",
                                 group=wandb_group,
                                 config={
                                     "fold": fold, "epochs": epochs,
                                     "batch_size": batch_size, "lr": lr,
                                     "patience": patience,
                                     "patch_size": PATCH_SIZE,
                                     "n_train": len(train_pids),
                                     "n_val": len(val_pids),
                                     "rfs_mode": rfs_mode,
                                     "rfs_warmup": rfs_warmup,
                                     "detach_tn": detach_tn,
                                     "arch": "DualHeadFusionResNet+MaskBranch",
                                 },
                                 reinit=True)
    except Exception as e:                                              # noqa: BLE001
        print(f"  [wandb] disabled: {e}", flush=True)
        wandb_run = None

    best_score = -1.0
    best_state = None
    best_epoch = -1
    wait = 0

    for ep in range(epochs):
        model.train()
        tr_loss_t = 0.0; tr_loss_n = 0.0; tr_loss_cox = 0.0; n_tr = 0
        # RFS weight schedule:
        #   warmup=0  → w_rfs = 1.0 always (RFS-only — Cox is the main signal)
        #   warmup>0  → 0 for ep < warmup; linear 0→1 over the next 10 epochs after
        if with_rfs:
            if ep < rfs_warmup:
                w_rfs = 0.0
            elif rfs_warmup == 0:
                w_rfs = 1.0
            else:
                w_rfs = min(1.0, (ep - rfs_warmup + 1) / 10.0)
        else:
            w_rfs = 0.0
        w_tn = 0.0 if rfs_mode == "rfs_only" else 1.0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            clin = batch["clinical"].to(device, non_blocking=True)
            t_gt = batch["t_gt"].to(device, non_blocking=True)
            n_gt = batch["n_gt"].to(device, non_blocking=True)
            out = model(img, mask, clin,
                         detach_tn_for_rfs=detach_tn) if with_rfs \
                  else model(img, mask, clin)
            loss_t = ce(out["t_logits"], t_gt)
            loss_n = ce(out["n_logits"], n_gt)
            if ORDINAL_LAMBDA > 0:
                # ordinal EMD aux term: T1..T4 ordinal (idx 0..3), Tx (idx4) excluded
                # via ordinal_max=3; N0..N3 all ordinal. TARGET selects which heads.
                if ORDINAL_TARGET in ("both", "t"):
                    loss_t = loss_t + ORDINAL_LAMBDA * soft_emd_loss(out["t_logits"], t_gt, ordinal_max=3)
                if ORDINAL_TARGET in ("both", "n"):
                    loss_n = loss_n + ORDINAL_LAMBDA * soft_emd_loss(out["n_logits"], n_gt)
            loss = w_tn * (loss_t + loss_n)
            if with_rfs and w_rfs > 0:
                rfs_days = batch["rfs_days"].to(device, non_blocking=True)
                rel = batch["relapse"].to(device, non_blocking=True)
                surv_ok = batch["surv_ok"].to(device, non_blocking=True)
                # Cox loss can't run under fp16 autocast (logcumsumexp grad).
                # Since the model isn't using autocast here it's already fp32.
                lc = out[next(iter(out))].sum() * 0.0               # zero placeholder
                if RFS_LOSS in ("cox", "both"):
                    lc = cox_efron_loss(out["risk"].squeeze(-1).float(),
                                         rfs_days, rel, surv_ok)
                if RFS_LOSS == "sigmoid":                            # SurvLoss drop-in
                    lc = sigmoid_concordance_loss(out["risk"].squeeze(-1).float(),
                                                  rfs_days, rel, surv_ok)
                if RFS_LOSS in ("mtlr", "both"):
                    valid = surv_ok.bool() & (rel >= 0)
                    if int(valid.sum()) >= 2:
                        tgt = encode_survival(rfs_days[valid].float(),
                                               rel[valid].float(), mtlr_bins_t)
                        lm = mtlr_neg_log_likelihood(out["mtlr_phi"][valid].float(), tgt)
                    else:
                        lm = out["mtlr_phi"].sum() * 0.0
                    lc = lc + lm                                    # 'both': cox + mtlr
                loss = loss + w_rfs * lc
                tr_loss_cox += float(lc.item()) * img.size(0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss_t += float(loss_t.item()) * img.size(0)
            tr_loss_n += float(loss_n.item()) * img.size(0)
            n_tr += img.size(0)
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        tr_loss_t /= max(n_tr, 1)
        tr_loss_n /= max(n_tr, 1)
        tr_loss_cox /= max(n_tr, 1)
        tr_loss = w_tn * (tr_loss_t + tr_loss_n) + w_rfs * tr_loss_cox

        # Val (compute losses + accuracies + c-index over pooled risks)
        model.eval()
        val_loss_t = 0.0; val_loss_n = 0.0; n_va = 0
        t_pred_all = []; t_gt_all = []
        n_pred_all = []; n_gt_all = []
        risk_all = []; time_all = []; event_all = []; surv_ok_all = []
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                clin = batch["clinical"].to(device, non_blocking=True)
                t_gt = batch["t_gt"].to(device, non_blocking=True)
                n_gt = batch["n_gt"].to(device, non_blocking=True)
                out = model(img, mask, clin,
                             detach_tn_for_rfs=detach_tn) if with_rfs \
                      else model(img, mask, clin)
                l_t = ce(out["t_logits"], t_gt)
                l_n = ce(out["n_logits"], n_gt)
                val_loss_t += float(l_t.item()) * img.size(0)
                val_loss_n += float(l_n.item()) * img.size(0)
                n_va += img.size(0)
                tp = out["t_logits"].argmax(-1).cpu().numpy()
                np_ = out["n_logits"].argmax(-1).cpu().numpy()
                t_gt_all.extend(batch["t_gt"].numpy().tolist())
                n_gt_all.extend(batch["n_gt"].numpy().tolist())
                t_pred_all.extend(tp.tolist())
                n_pred_all.extend(np_.tolist())
                if with_rfs:
                    risk_v = (mtlr_risk(out["mtlr_phi"].float()) if RFS_LOSS == "mtlr"
                              else out["risk"].squeeze(-1))
                    risk_all.extend(risk_v.cpu().numpy().tolist())
                    time_all.extend(batch["rfs_days"].numpy().tolist())
                    event_all.extend(batch["relapse"].numpy().tolist())
                    surv_ok_all.extend(batch["surv_ok"].numpy().tolist() if hasattr(batch["surv_ok"], "numpy") else list(batch["surv_ok"]))
        val_loss_t /= max(n_va, 1)
        val_loss_n /= max(n_va, 1)
        val_loss = val_loss_t + val_loss_n
        # balacc (skip -1)
        t_mask_ok = [g >= 0 for g in t_gt_all]
        n_mask_ok = [g >= 0 for g in n_gt_all]
        t_ba = balanced_accuracy_score(np.array(t_gt_all)[t_mask_ok], np.array(t_pred_all)[t_mask_ok]) \
                 if any(t_mask_ok) else 0.0
        n_ba = balanced_accuracy_score(np.array(n_gt_all)[n_mask_ok], np.array(n_pred_all)[n_mask_ok]) \
                 if any(n_mask_ok) else 0.0
        # C-index on pooled val risks
        c_idx = 0.0
        if with_rfs and risk_all:
            mask_s = np.array(surv_ok_all, dtype=bool)
            if mask_s.sum() >= 2:
                c_idx = float(harrell_cindex(np.array(risk_all)[mask_s],
                                              np.array(time_all)[mask_s],
                                              np.array(event_all)[mask_s]))
        # Score: weighted by mode
        if rfs_mode == "rfs_only":
            score = c_idx
        elif rfs_mode == "triplehead":
            score = (t_ba + n_ba + c_idx) / 3.0
        else:
            score = 0.5 * (t_ba + n_ba)

        if wandb_run is not None:
            log_d = {
                "epoch": ep,
                "lr": cur_lr,
                "train/loss": tr_loss, "train/loss_t": tr_loss_t, "train/loss_n": tr_loss_n,
                "val/loss": val_loss, "val/loss_t": val_loss_t, "val/loss_n": val_loss_n,
                "val/balacc_t": t_ba, "val/balacc_n": n_ba, "val/score": score,
                "train/w_rfs": w_rfs, "train/w_tn": w_tn,
            }
            if with_rfs:
                log_d["train/loss_cox"] = tr_loss_cox
                log_d["val/cindex"] = c_idx
            wandb_run.log(log_d)
        print(f"  ep{ep:>3}  tr_loss={tr_loss:.4f} (t={tr_loss_t:.3f}/n={tr_loss_n:.3f}"
              f"/cox={tr_loss_cox:.3f}, w_rfs={w_rfs:.2f})  "
              f"val_loss={val_loss:.4f} (t={val_loss_t:.3f}/n={val_loss_n:.3f})  "
              f"val_t={t_ba:.4f}  val_n={n_ba:.4f}  cindex={c_idx:.4f}  score={score:.4f}  "
              f"best={best_score:.4f}@ep{best_epoch}", flush=True)

        if score > best_score:
            best_score = score
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [early-stop] no improvement in {patience} epochs",
                      flush=True)
                break

    if wandb_run is not None:
        wandb_run.summary["best/score"] = best_score
        wandb_run.summary["best/epoch"] = best_epoch
        wandb_run.finish()

    # Save best ckpt + dump per-patient val predictions with best model
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "best.ckpt")
    model.load_state_dict(best_state)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            clin = batch["clinical"].to(device, non_blocking=True)
            out = model(img, mask, clin,
                         detach_tn_for_rfs=detach_tn) if with_rfs \
                  else model(img, mask, clin)
            t_soft = F.softmax(out["t_logits"], dim=-1).cpu().numpy()
            n_soft = F.softmax(out["n_logits"], dim=-1).cpu().numpy()
            risk_batch = (_extract_risk(out) if with_rfs else None)
            risk_mtlr_batch = (mtlr_risk(out["mtlr_phi"].float()).detach().cpu().numpy()
                               if (with_rfs and RFS_LOSS == "both") else None)
            for i, pid in enumerate(batch["patient_id"]):
                r = {"patient_id": pid}
                if with_rfs:
                    r["deep_risk"] = float(risk_batch[i])
                    if risk_mtlr_batch is not None:
                        r["deep_risk_mtlr"] = float(risk_mtlr_batch[i])
                for j in range(t_soft.shape[1]): r[f"t_softmax_{j}"] = float(t_soft[i, j])
                for j in range(n_soft.shape[1]): r[f"n_softmax_{j}"] = float(n_soft[i, j])
                r["t_pred"] = T_LABELS[int(t_soft[i].argmax())]
                r["n_pred"] = N_LABELS[int(n_soft[i].argmax())]
                rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)

    # 3/1/1 mode: also dump per-patient predictions on the held-out test fold
    if test_fold is not None and len(test_pids) > 0:
        test_ds = MedAIPatchDataset(test_pids, clin_df, df, train=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=n_workers, pin_memory=True)
        test_rows = []
        with torch.no_grad():
            for batch in test_loader:
                img = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                clin = batch["clinical"].to(device, non_blocking=True)
                out = model(img, mask, clin,
                             detach_tn_for_rfs=detach_tn) if with_rfs \
                      else model(img, mask, clin)
                t_soft = F.softmax(out["t_logits"], dim=-1).cpu().numpy()
                n_soft = F.softmax(out["n_logits"], dim=-1).cpu().numpy()
                risk_batch = (_extract_risk(out) if with_rfs else None)
                risk_mtlr_batch = (mtlr_risk(out["mtlr_phi"].float()).detach().cpu().numpy()
                                   if (with_rfs and RFS_LOSS == "both") else None)
                for i, pid in enumerate(batch["patient_id"]):
                    r = {"patient_id": pid}
                    if with_rfs:
                        r["deep_risk"] = float(risk_batch[i])
                        if risk_mtlr_batch is not None:
                            r["deep_risk_mtlr"] = float(risk_mtlr_batch[i])
                    for j in range(t_soft.shape[1]): r[f"t_softmax_{j}"] = float(t_soft[i, j])
                    for j in range(n_soft.shape[1]): r[f"n_softmax_{j}"] = float(n_soft[i, j])
                    r["t_pred"] = T_LABELS[int(t_soft[i].argmax())]
                    r["n_pred"] = N_LABELS[int(n_soft[i].argmax())]
                    test_rows.append(r)
        pd.DataFrame(test_rows).to_csv(out_dir / "test_predictions.csv", index=False)
        print(f"  test_fold{test_fold} predictions written (n={len(test_rows)})",
              flush=True)

    print(f"  fold{fold} done — best score {best_score:.4f} @ ep{best_epoch}", flush=True)
    return {"best_score": float(best_score), "best_epoch": int(best_epoch),
            "n_train": len(train_pids), "n_val": len(val_pids),
            "n_test": len(test_pids), "test_fold": test_fold}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=False)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--n_workers", type=int, default=4)
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Default: predictions/medai_hybrid_patch96 or _{rfs_mode}")
    ap.add_argument("--rfs_mode", type=str, default="none",
                    choices=["none", "rfs_only", "triplehead", "rfs_with_tn"],
                    help="Task 2 only (none) | RFS-only Cox head | TripleHead T+N+Cox "
                         "| RFS-only + fold-disjoint TN softmax as extra clinical input")
    ap.add_argument("--tn_predictions_dir", type=Path,
                    default=ROOT / "predictions/medai_hybrid_patch96",
                    help="Source of pre-trained Task 2 TN softmax (used by rfs_with_tn)")
    ap.add_argument("--rfs_warmup", type=int, default=30,
                    help="Number of epochs to keep RFS weight = 0 (MEDAI v4 default 30)")
    ap.add_argument("--no_detach_tn", action="store_true",
                    help="Disable stop-gradient from Cox loss into T/N softmax")
    ap.add_argument("--augment", action="store_true",
                    help="Enable train-time augmentation (LR flip, brightness, "
                         "gamma, ±10° Z-axis rotation)")
    ap.add_argument("--affine_aug", action="store_true",
                    help="Use augment-THEN-crop: compose rotation+translation into "
                         "one affine resample from the 116³ pre-crop (edge-filled, "
                         "no zero corners). Requires --augment. Distinct run name.")
    ap.add_argument("--build_cache", action="store_true",
                    help="Build the 116³ patch cache for all 782 patients and exit")
    ap.add_argument("--cache_force", action="store_true")
    ap.add_argument("--test_fold", type=int, default=None,
                    help="3/1/1 mode: hold this fold out from training, evaluate "
                         "best.ckpt on it after training. Writes test_predictions.csv.")
    a = ap.parse_args()
    if a.out_dir is None:
        suffix = "" if a.rfs_mode == "none" else f"_{a.rfs_mode}"
        aug_suffix = ("_affaug" if a.affine_aug else "_aug") if a.augment else ""
        split_suffix = f"_v{a.fold}t{a.test_fold}" if a.test_fold is not None else ""
        a.out_dir = ROOT / f"predictions/medai_hybrid_patch96{suffix}{aug_suffix}{split_suffix}"

    df = pd.read_csv(CSV_PATH)
    df["PatientID"] = df["PatientID"].astype(str).str.strip()
    all_pids = df["PatientID"].tolist()

    if a.build_cache:
        print(f"=== building cache for {len(all_pids)} patients → {CACHE_DIR} ===",
              flush=True)
        n_ok = build_cache(all_pids, force=a.cache_force)
        print(f"=== DONE: {n_ok}/{len(all_pids)} cached ===", flush=True)
        return

    assert a.fold is not None, "--fold is required (unless --build_cache)"
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    # Ensure cache exists for all patients before training
    missing = [p for p in all_pids if not (CACHE_DIR / f"{p}.pt").exists()]
    if missing:
        print(f"cache incomplete: {len(missing)} patients missing — "
              f"run with --build_cache first.", flush=True)
        sys.exit(1)

    fold_subdir = f"fold{a.fold}" if a.test_fold is None else \
                  f"val{a.fold}_test{a.test_fold}"
    out = a.out_dir / fold_subdir
    suffix = "" if a.rfs_mode == "none" else f"_{a.rfs_mode}"
    aug_suffix = ("_affaug" if a.affine_aug else "_aug") if a.augment else ""
    wandb_group = f"medai_hybrid_patch96{suffix}{aug_suffix}"
    # rfs_with_tn: treat training as rfs_only (no T/N loss) but inject TN
    # softmax into clinical input. Map to internal rfs_only mode.
    effective_rfs_mode = "rfs_only" if a.rfs_mode == "rfs_with_tn" else a.rfs_mode
    tn_dir = a.tn_predictions_dir if a.rfs_mode == "rfs_with_tn" else None
    if a.rfs_mode == "rfs_only" and not (a.rfs_warmup == 0):
        a.rfs_warmup = 0  # ensure RFS-only uses warmup=0
    if a.test_fold is not None:
        wandb_group = f"{wandb_group}_v{a.fold}t{a.test_fold}"
    result = train_fold(a.fold, a.epochs, a.batch_size, a.lr, device,
                          out, a.n_workers, a.patience,
                          rfs_mode=effective_rfs_mode,
                          rfs_warmup=a.rfs_warmup,
                          detach_tn=(not a.no_detach_tn),
                          wandb_group=wandb_group,
                          tn_predictions_dir=tn_dir,
                          augment=a.augment,
                          affine_aug=a.affine_aug,
                          test_fold=a.test_fold)
    json.dump(result, open(out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
