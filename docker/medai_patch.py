"""
Docker-side patch extractor + medai_hybrid loader for Task 2/3.

Mirrors the training-side patch construction (scripts/train_medai_hybrid.py:
build_one_patient + MedAIPatchDataset.__getitem__) but works on in-memory
inputs (no /data cache) and starts from the docker pipeline's outputs:
  - raw CT path + raw PET path
  - segmentation array in ORIGINAL CT space (uint8, 0/1/2)

Output: 4D image tensor (2, 96, 96, 96) + 4D mask tensor (1, 96, 96, 96).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn

# Intensity normalization — MUST match scripts/train_medai_hybrid.py. Defaults
# reproduce the original patch pipeline; override via env to match an aligned
# (CT [-200,200], PET clamp40/pct) deployed model.
CT_LO = float(os.environ.get("MEDAI_CT_LO", "-1000"))
CT_HI = float(os.environ.get("MEDAI_CT_HI", "3000"))
PET_NORM = os.environ.get("MEDAI_PET_NORM", "clamp25").strip().lower()


def _norm_ct(t: torch.Tensor) -> torch.Tensor:
    return (t.clamp(min=CT_LO, max=CT_HI) - CT_LO) / (CT_HI - CT_LO)


def _norm_pet(t: torch.Tensor) -> torch.Tensor:
    if PET_NORM == "pct":
        v = t.flatten()
        lo = torch.quantile(v, 0.005); hi = torch.quantile(v, 0.995)
        return ((t - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
    if PET_NORM == "clamp40":
        return t.clamp(min=0.0, max=40.0) / 40.0
    return t.clamp(min=0.0, max=25.0) / 25.0
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as cc_label

PATCH_SIZE = 96
CACHE_MARGIN = 10
PRECROP_SIZE = PATCH_SIZE + 2 * CACHE_MARGIN  # 116

# CC filter thresholds (= training-time cache builder, see
# scripts/train_medai_hybrid.py:CCF_MIN_MM3). Locked per per-class sweep
# (evaluation/results/cc_sweep_perclass_5fold.md): GTVp ≥1000 mm³, GTVn ≥500 mm³.
CCF_MIN_MM3 = {1: 1000, 2: 500}
CCF_TOPN = {1: 0, 2: 0}  # no top-N cap


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


# ── ROI-crop RAM fix (2026-06-16) ─────────────────────────────────────────────
# The full-volume 1 mm resample of CT/PET (whole-body) was the dominant TN-stage RAM
# spike (~+5 GB). The medai patches are only 96³/112³ around the mask centroid, so we
# resample just an ROI box onto a phase-locked sub-region of the FULL 1 mm grid — sitk
# samples each input at its true geometry, so the sample points are an exact subset of the
# full-volume path's grid. Linear/NN interpolation (1-voxel support) → a small native
# margin makes every patch voxel BYTE-IDENTICAL to the full-volume path (verified).
_PATCH_MARGIN_MM = 64.0    # 1 mm output sub-grid padding ≥ max half-patch (112/2 = 56 mm)
_INPUT_MARGIN_MM = 72.0    # native crop padding ≥ output margin + interp support


def _full_iso_ref(img: sitk.Image, spacing: float) -> sitk.Image:
    """Cheap dummy carrying the FULL iso grid geometry the full-volume path lands on."""
    in_sp, in_sz = img.GetSpacing(), img.GetSize()
    new_size = [int(round(s * in_sp[i] / spacing)) for i, s in enumerate(in_sz)]
    ref = sitk.Image(new_size, sitk.sitkUInt8)
    ref.SetSpacing((spacing, spacing, spacing))
    ref.SetDirection(img.GetDirection())
    ref.SetOrigin(img.GetOrigin())
    return ref


def _world_box_region(img: sitk.Image, w_lo, w_hi, margin_mm: float):
    """Index (start, size) of img's sub-region covering [w_lo-margin, w_hi+margin]."""
    lo = np.asarray(w_lo, float) - margin_mm
    hi = np.asarray(w_hi, float) + margin_mm
    idxs = []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                idxs.append(img.TransformPhysicalPointToContinuousIndex((float(x), float(y), float(z))))
    idxs = np.asarray(idxs)
    sz = np.asarray(img.GetSize())
    i0 = np.clip(np.floor(idxs.min(0)).astype(int), 0, sz - 1)
    i1 = np.clip(np.ceil(idxs.max(0)).astype(int), 1, sz)
    return [int(v) for v in i0], [int(v) for v in (i1 - i0)]


def _mask_world_bbox(seg_arr_zyx: np.ndarray, native_ref: sitk.Image):
    """World (x,y,z) min/max corner of the nonzero mask, via the native geometry."""
    nz = np.argwhere(seg_arr_zyx > 0)                       # (z,y,x)
    if nz.size == 0:
        return None
    lo_zyx, hi_zyx = nz.min(0), nz.max(0)
    pts = []
    for z in (int(lo_zyx[0]), int(hi_zyx[0])):
        for y in (int(lo_zyx[1]), int(hi_zyx[1])):
            for x in (int(lo_zyx[2]), int(hi_zyx[2])):
                pts.append(native_ref.TransformIndexToPhysicalPoint((x, y, z)))
    pts = np.asarray(pts)
    return pts.min(0), pts.max(0)


def _resample_to_ref(img: sitk.Image, ref: sitk.Image, is_label: bool) -> sitk.Image:
    rsl = sitk.ResampleImageFilter()
    rsl.SetReferenceImage(ref)
    rsl.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return rsl.Execute(img)


def _prepare_iso(ct_path: str, pet_path: str, seg_array: np.ndarray):
    """Resample CT/PET/seg to 1mm-iso and compute the crop centroid.

    Returns (ct_arr, pt_arr, pred_raw, pred_cc, centroid) at 1mm. `pred_raw` is
    the UN-filtered predicted mask (the v6 deep model's mask channel — small
    components carry staging/prognosis signal); `pred_cc` is the CC-filtered mask
    used ONLY to locate the centroid (matches the training cache: CC for centroid,
    RAW for the channel). The heavy resampling runs once and is reused across
    patch sizes (multi-scale).
    """
    ct_raw = sitk.ReadImage(str(ct_path))
    pt_raw = sitk.ReadImage(str(pet_path))

    seg_sitk = sitk.GetImageFromArray(seg_array.astype(np.uint8))
    seg_sitk.SetOrigin(ct_raw.GetOrigin())
    seg_sitk.SetSpacing(ct_raw.GetSpacing())
    seg_sitk.SetDirection(ct_raw.GetDirection())

    # ROI-crop RAM fix: resample only a mask-centred box onto a phase-locked sub-region of
    # the full 1 mm grid (byte-identical to the full-volume path; see helper notes). Falls
    # back to the full-volume resample if the mask is empty.
    bbox = _mask_world_bbox(seg_array, ct_raw)
    if bbox is None:
        ct = _resample_to_iso(ct_raw, 1.0, is_label=False)
        rsl = sitk.ResampleImageFilter()
        rsl.SetReferenceImage(ct); rsl.SetInterpolator(sitk.sitkLinear)
        pt = rsl.Execute(pt_raw)
        rsl.SetInterpolator(sitk.sitkNearestNeighbor)
        pred = rsl.Execute(seg_sitk)
    else:
        w_lo, w_hi = bbox
        full_ref = _full_iso_ref(ct_raw, 1.0)
        roi_idx, roi_sz = _world_box_region(full_ref, w_lo, w_hi, _PATCH_MARGIN_MM)
        sub_ref = sitk.RegionOfInterest(full_ref, roi_sz, roi_idx)   # phase-locked 1 mm sub-grid
        c_idx, c_sz = _world_box_region(ct_raw, w_lo, w_hi, _INPUT_MARGIN_MM)
        ct_crop = sitk.RegionOfInterest(ct_raw, c_sz, c_idx)
        seg_crop = sitk.RegionOfInterest(seg_sitk, c_sz, c_idx)
        ct = _resample_to_ref(ct_crop, sub_ref, is_label=False)      # linear
        pt = _resample_to_ref(pt_raw, sub_ref, is_label=False)       # linear (PET native → sub-grid)
        pred = _resample_to_ref(seg_crop, sub_ref, is_label=True)    # NN

    ct_arr = sitk.GetArrayFromImage(ct).astype(np.float32)
    pt_arr = sitk.GetArrayFromImage(pt).astype(np.float32)
    pred_raw = sitk.GetArrayFromImage(pred).astype(np.uint8)

    # CC filter at 1mm-iso → centroid only (training: CC for centroid, RAW channel).
    struct = generate_binary_structure(3, 3)
    pred_cc = _apply_cc_filter(pred_raw, 1.0, struct)
    mask_fg = pred_cc > 0
    if mask_fg.any():
        centroid = np.argwhere(mask_fg).mean(axis=0)            # (z, y, x)
    else:
        centroid = np.array([ct_arr.shape[0] // 2, ct_arr.shape[1] // 2,
                              ct_arr.shape[2] // 2], dtype=np.float64)
    return ct_arr, pt_arr, pred_raw, pred_cc, centroid


def _crop_patch(ct_arr, pt_arr, pred_arr, centroid,
                patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Centroid-centred crop of size `patch_size`³ (no jitter at inference =
    the training val-time slice), then normalize. Returns (2,P,P,P) image +
    (1,P,P,P) long mask."""
    half = patch_size // 2
    pad = [(half, half)] * 3
    ct_p = np.pad(ct_arr, pad, mode="constant", constant_values=-1024)
    pt_p = np.pad(pt_arr, pad, mode="constant", constant_values=0)
    pr_p = np.pad(pred_arr, pad, mode="constant", constant_values=0)
    cz, cy, cx = (centroid + np.array([half, half, half])).round().astype(int)
    cz = int(np.clip(cz, half, ct_p.shape[0] - half))
    cy = int(np.clip(cy, half, ct_p.shape[1] - half))
    cx = int(np.clip(cx, half, ct_p.shape[2] - half))
    sl = (slice(cz - half, cz + half),
          slice(cy - half, cy + half),
          slice(cx - half, cx + half))
    ct_c = _norm_ct(torch.from_numpy(np.ascontiguousarray(ct_p[sl])).float())
    pt_c = _norm_pet(torch.from_numpy(np.ascontiguousarray(pt_p[sl])).float())
    mask_c = torch.from_numpy(np.ascontiguousarray(pr_p[sl])).to(torch.long).unsqueeze(0)
    image = torch.stack([ct_c, pt_c], dim=0)                   # (2, P, P, P)
    return image, mask_c


def build_medai_patch(ct_path: str, pet_path: str, seg_array: np.ndarray,
                       patch_size: int = 96,
                       raw_mask: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-scale patch. Defaults (96³, CC mask) reproduce the v5 behavior;
    v6 passes raw_mask=True via build_medai_patches_multiscale."""
    ct_arr, pt_arr, pred_raw, pred_cc, centroid = _prepare_iso(ct_path, pet_path, seg_array)
    pred_for_mask = pred_raw if raw_mask else pred_cc
    return _crop_patch(ct_arr, pt_arr, pred_for_mask, centroid, patch_size)


def build_medai_patches_multiscale(ct_path: str, pet_path: str, seg_array: np.ndarray,
                                    patch_sizes=(96, 112), raw_mask: bool = True
                                    ) -> "dict[int, tuple[torch.Tensor, torch.Tensor]]":
    """v6: build {96³, 112³} patches in one resample pass, RAW mask channel.
    Returns {patch_size: (image (2,P,P,P), mask (1,P,P,P))}."""
    ct_arr, pt_arr, pred_raw, pred_cc, centroid = _prepare_iso(ct_path, pet_path, seg_array)
    pred_for_mask = pred_raw if raw_mask else pred_cc
    return {ps: _crop_patch(ct_arr, pt_arr, pred_for_mask, centroid, ps) for ps in patch_sizes}


# ───────────────── Model loader ──────────────────────────────────────────────

def load_medai_model(ckpt_path: Path, device: torch.device,
                      clinical_dim: int = 18, with_rfs: bool = False) -> nn.Module:
    """Build DualHeadFusionResNet and load the ckpt state_dict.

    Args:
        ckpt_path: path to .ckpt file (flat state_dict, not Lightning).
        clinical_dim: input dim for clinical features. 18 = base. The rfs_with_tn
                      variant uses 27 (= 18 + 9 TN softmax) but we don't ship that
                      variant in docker.
        with_rfs: True for the rfs_only model (has Cox head); False for TN-only.
    """
    from models.medai_hybrid import DualHeadFusionResNet
    model = DualHeadFusionResNet(image_channels=2, clinical_dim=clinical_dim,
                                   n_t_classes=5, n_n_classes=4,
                                   hidden_dim=128, dropout=0.1,
                                   mask_branch=True, with_rfs=with_rfs)
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[medai] {ckpt_path.name}: {len(missing)} missing keys "
              f"e.g. {missing[:3]}", flush=True)
    if unexpected:
        print(f"[medai] {ckpt_path.name}: {len(unexpected)} unexpected keys "
              f"e.g. {unexpected[:3]}", flush=True)
    model.to(device).eval()
    return model
