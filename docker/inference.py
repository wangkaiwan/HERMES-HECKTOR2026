"""
HECKTOR 2026 Challenge — Inference Entry Point.

Single-patient invocation per the docker-template branch of
BioMedIA-MBZUAI/HECKTOR2026.

Output paths/format follow the docker-template branch Task/inference.py
(confirmed by the HECKTOR 2026 committee, 2026-05-18 — that file is the
authoritative submission scaffold; the README and main branch are reference
only):

    /output/images/head-neck-tumor-segmentation/output.mha
        uint8 mask at original CT spacing/origin; labels 0=bg, 1=GTVp, 2=GTVn
    /output/t-stage.json   — a bare JSON string, e.g. "T2"
    /output/n-stage.json   — a bare JSON string, e.g. "N1"
    /output/rfs.json       — a bare JSON float (higher score = higher risk)

Model weights live at /opt/ml/model/ (mounted at runtime — uploaded to Grand
Challenge as a separate Model tarball, NOT baked into the image). Locally,
mirror the official template by placing weights under docker/model/ — the
do_test_run.sh script mounts that to /opt/ml/model.

Local test:
    ./do_test_run.sh
Save image + model:
    ./do_save.sh
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F

# Project code is copied into /opt/app/ at build time (see Dockerfile).
sys.path.insert(0, "/opt/app")

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")

# Task #36 (2026-05-13): T1-T4 per challenge spec. Phase 1 fold-0 ckpt's
# tn_head was random-init (Phase 1 is seg-only), so this map only takes
# effect once a Phase 3 multitask ckpt is loaded. Garbage T values from the
# seg-only ckpt remain garbage either way.
T_LABEL = {0: "T1", 1: "T2", 2: "T3", 3: "T4"}
N_LABEL = {0: "N0", 1: "N1", 2: "N2", 3: "N3"}

# After 2026-05-13 redesign: cache_volume=(160,160,192) in configs/default.yaml
# means the test pipeline produces images at EXACTLY (2,160,160,192). The model
# is FCN (SwinUNETR encoder + decoder), so we do ONE full-volume forward —
# no sliding window, no overlap stitching, no TN/Prog center-crop. All three
# tasks share the same encoder forward.
#   Coverage: 320×320×384 mm (at 2mm iso) — covers 100% of training GT bboxes
#             (max 180×114×150 mm) anchored at body top per KeepTopZd.
#   VRAM (5090 fp16): peak 7.5 GB / reserved 10.7 GB → safe on T4 16 GB.
#   Time (5090 fp16): seg 170 ms + TN 80 ms + Prog 80 ms ≈ 330 ms / case.
ROI_SIZE = (160, 160, 192)                                 # ≡ input size; single forward

# Test-time augmentation. Controlled by env var TTA — DEFAULT IS the full 6-TTA
# (LR-flip + AP-flip + IS-flip + γ=0.9 + γ=1.1) so the production submission
# gets all in-distribution test-time perturbations automatically. Modes:
#   "off" / "none" / "0" / "" / "orig"  — 1 pass, no TTA
#   "lr"                                — 2 passes: orig + LR-flip
#   "flips"                             — 4 passes: orig + LR + AP + IS
#   "gamma"                             — 3 passes: orig + γ=0.9 + γ=1.1
#   "lr+gamma"                          — 4 passes: orig + LR + γ=0.9 + γ=1.1
#   "flips+gamma" / "full" / "6tta"     — 6 passes: orig + LR + AP + IS + γ=0.9 + γ=1.1
#
# Rationale:
#   - All three axis-flips: training augmentation already includes
#     RandFlipd(prob=0.5) on every spatial axis (X / Y / Z), so the model has
#     seen all 2³=8 flip combinations during training. AP and IS flips ARE
#     in-distribution despite being anatomically directional (trachea ant. vs
#     spine post., etc.) — the model learned to recognise either orientation.
#   - Gamma jitter (γ=0.9/1.1): training also uses RandAdjustContrast γ=0.7-1.5,
#     so γ shifts are well within the training distribution.
#
# Sequential passes keep VRAM peak unchanged; wall-clock scales linearly. On
# T4 with 5-fold ensemble: 6-TTA ≈ 270 s/case (well under 1200 s budget).
_TTA_DEFAULT = "full"
_TTA_OFF_VALUES = {"off", "none", "0", "false", "no", "orig", ""}
TTA_MODE = os.environ.get("TTA", _TTA_DEFAULT).strip().lower()

# Class-confidence post-processing. Many HECKTOR patients have only GTVp OR
# only GTVn — when the model produces a handful of stray FP voxels for the
# absent class, that class's dice drops to 0.0 and drags the per-patient mean.
# We reject a class entirely if its predicted volume is too small or its
# max-confidence too low. Both thresholds are tunable via env vars.
#   CC_MIN_VOX     : min # voxels of a class to keep (in (160,160,192) space,
#                    voxel = 2×2×2 mm = 8 mm³, so 30 vox = 240 mm³ = 0.24 cc)
#   CC_MIN_MAXPROB : min max softmax prob for that class anywhere in the volume
# Set CC_MIN_VOX=0 and CC_MIN_MAXPROB=0 to disable (or use CC=off).
def _cc_thresholds() -> tuple[int, float]:
    # Default DISABLED (2026-05-13): empirically MIN_MAXPROB=0.55 was too strict
    # after 4-TTA averaging (softmax max gets softened to ~0.6-0.8 from ~0.85+),
    # killing real predictions on borderline patients (e.g., MDA-211 dice 0.94 →
    # 0.44 because its GTVn voxels lost confidence below threshold post-TTA).
    # The "single-class FP" failure mode I expected this to fix turned out to be
    # rare; the loss from killing real predictions outweighed any gain. Leave the
    # env-var knobs alive in case Phase-3 multitask model needs tighter post-
    # processing — just opt-in via `CC=on CC_MIN_VOX=N CC_MIN_MAXPROB=p`.
    if os.environ.get("CC", "off").strip().lower() in _TTA_OFF_VALUES:
        return 0, 0.0
    try:
        min_vox = int(os.environ.get("CC_MIN_VOX", "30"))
    except ValueError:
        min_vox = 30
    try:
        min_p = float(os.environ.get("CC_MIN_MAXPROB", "0.55"))
    except ValueError:
        min_p = 0.55
    return min_vox, min_p


# ── Connected-component (CC) filter ─────────────────────────────────────────
# HECKTOR 2026 GTVn ranking includes an Aggregated F1-detection metric that
# operates on connected-component instances of the predicted GTVn mask. Stray
# small FP CCs (say 1-3 voxels at the boundary of a true lesion, or a noisy
# blob far from any GT lesion) each become FP lesions and hurt F1 directly.
#
# This filter labels CCs of each foreground class and zeros out any CC whose
# size is below CC_MIN_VOX (default 10 voxels = 80 mm³ at 2mm iso, well below
# the smallest clinical GTVn). Cheaper and more F1-friendly than the softmax-
# threshold approach in _apply_class_confidence.
def _ccfilter_thresholds() -> dict[int, int]:
    """Returns {class_label: min_voxels_to_keep}. Empty dict = disabled."""
    if os.environ.get("CCF", "on").strip().lower() in _TTA_OFF_VALUES:
        return {}
    # Defaults locked from 5-fold per-class CC-filter sweep (2026-05-31):
    #   GTVp=1000 mm³, GTVn=500 mm³ → mean borda 0.7228 (+0.0106 over no-filter)
    # Filter runs in 1mm-iso voxel space, so N voxels = N mm³ exactly.
    # Override via env vars CCF_GTVP_MIN_VOX / CCF_GTVN_MIN_VOX if needed.
    try:
        gtvp_min = int(os.environ.get("CCF_GTVP_MIN_VOX", "1000"))
    except ValueError:
        gtvp_min = 1000
    try:
        gtvn_min = int(os.environ.get("CCF_GTVN_MIN_VOX", "500"))
    except ValueError:
        gtvn_min = 500
    return {1: gtvp_min, 2: gtvn_min}


def _topn_keep_counts() -> dict[int, int]:
    """Returns {class_label: max_CCs_to_keep_by_volume}. 0 / negative = disabled.

    Defensive top-N keep filter layered on top of the mm³ filter. Caps from GT
    distribution of HECKTOR 2026 training cohort (n=782):
      - GTVp: max 2 CCs in any patient (top-2 covers 100%)
      - GTVn: max 13 CCs in any patient; top-8 covers 99.1%
    The 5-fold top-N sweep (2026-05-31) showed top-2/top-8 is a true no-op on
    training data after the mm³ filter (no patient predicts >2 GTVp or >8 GTVn
    surviving CCs), but kept as a guard against pathological test-time inputs.
    """
    try:
        gtvp_top = int(os.environ.get("CCF_GTVP_TOPN", "2"))
    except ValueError:
        gtvp_top = 2
    try:
        gtvn_top = int(os.environ.get("CCF_GTVN_TOPN", "8"))
    except ValueError:
        gtvn_top = 8
    return {1: gtvp_top, 2: gtvn_top}


def _apply_cc_filter(pred: np.ndarray) -> np.ndarray:
    """Drop CC instances smaller than the per-class voxel threshold,
    then keep only the volume top-N largest survivors per class.

    pred: (X, Y, Z) uint8 argmax — classes {0=bg, 1=GTVp, 2=GTVn}.
    """
    thresholds = _ccfilter_thresholds()
    topn = _topn_keep_counts()
    if not thresholds:
        return pred
    from scipy.ndimage import label as cc_label, generate_binary_structure
    struct = generate_binary_structure(3, 3)                   # 26-connectivity
    out = pred.copy()
    for cls, min_vox in thresholds.items():
        keep_n = topn.get(cls, 0)
        mask = (pred == cls)
        if not mask.any():
            continue
        labels, n_cc = cc_label(mask, structure=struct)
        if n_cc == 0:
            continue
        # Count voxels per CC (label 0 = bg, skip).
        sizes = np.bincount(labels.flat, minlength=n_cc + 1)
        # Phase 1: mm³ filter — collect survivor (cc_id, size) pairs.
        survivors = []
        n_drop_mm3 = 0; vox_drop_mm3 = 0
        for cc_id in range(1, n_cc + 1):
            if min_vox > 0 and sizes[cc_id] < min_vox:
                out[labels == cc_id] = 0
                n_drop_mm3 += 1
                vox_drop_mm3 += int(sizes[cc_id])
            else:
                survivors.append((cc_id, int(sizes[cc_id])))
        # Phase 2: top-N keep — sort survivors by volume desc, drop the rest.
        n_drop_topn = 0; vox_drop_topn = 0
        if keep_n > 0 and len(survivors) > keep_n:
            survivors.sort(key=lambda kv: -kv[1])
            for cc_id, sz in survivors[keep_n:]:
                out[labels == cc_id] = 0
                n_drop_topn += 1
                vox_drop_topn += sz
        if n_drop_mm3 or n_drop_topn:
            print(f"[inference] CCF class {cls}: mm³ dropped {n_drop_mm3}/{n_cc} CCs "
                  f"({vox_drop_mm3} vox, thr={min_vox}); "
                  f"top-N dropped {n_drop_topn} more CCs ({vox_drop_topn} vox, "
                  f"keep={keep_n})",
                  file=sys.stderr, flush=True)
    return out


def _suv_gate_gtvn(seg_array: np.ndarray, ct_path, pet_path,
                   suv_thr: float | None = None) -> np.ndarray:
    """S3 post-proc (v7.1): drop predicted GTVn (class 2) connected components whose
    PEAK SUV is below `suv_thr` (default 2.5, the clinical malignancy cutoff).

    Validated on 782 OOF patients (evaluation/results/s3_suv_gate_validation.md):
    GTVn F1agg 0.686 -> 0.699 (+0.013, precision-driven), GTVn DSCagg -0.0008
    (negligible), GTVp unchanged. False-positive GTVn nodes are SUV-cold (peak
    median 4.1) vs real nodes (9.3); the few TP nodes dropped are tiny.

    `seg_array` is at the native CT grid (post back-projection). `pet_path` is the
    co-registered PET (SUV, already on the CT grid). Env override: SUV_GATE_GTVN
    (float threshold; 'off'/0 disables). Applied to output.mha ONLY — Task 2/3
    keep the ungated mask (the gain was validated for segmentation only).
    """
    if suv_thr is None:
        v = os.environ.get("SUV_GATE_GTVN", "2.5").strip().lower()
        if v in _TTA_OFF_VALUES or v == "0":
            return seg_array
        suv_thr = float(v)
    if suv_thr <= 0 or (seg_array == 2).sum() == 0:
        return seg_array
    from scipy.ndimage import label as cc_label, generate_binary_structure
    ref = sitk.ReadImage(str(ct_path))
    seg_img = sitk.GetImageFromArray(seg_array.astype(np.uint8)); seg_img.CopyInformation(ref)
    rsl = sitk.ResampleImageFilter()
    rsl.SetReferenceImage(seg_img); rsl.SetInterpolator(sitk.sitkLinear)
    suv = sitk.GetArrayFromImage(rsl.Execute(sitk.ReadImage(str(pet_path))))
    out = seg_array.copy()
    struct = generate_binary_structure(3, 3)
    labels, n_cc = cc_label(seg_array == 2, structure=struct)
    n_drop = 0
    for cc_id in range(1, n_cc + 1):
        cc = labels == cc_id
        if float(suv[cc].max()) < suv_thr:
            out[cc] = 0
            n_drop += 1
    if n_drop:
        print(f"[inference] SUV-gate: dropped {n_drop}/{n_cc} GTVn CC "
              f"(peak SUV < {suv_thr})", flush=True)
    return out


def _apply_class_confidence(pred: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """If predicted class-c volume or max prob below threshold, reset to bg.

    pred:  (X, Y, Z) uint8 argmax
    probs: (3, X, Y, Z) softmax probs (averaged over models + TTA)
    """
    min_vox, min_p = _cc_thresholds()
    if min_vox <= 0 and min_p <= 0.0:
        return pred
    for c in (1, 2):
        mask_c = (pred == c)
        n_vox = int(mask_c.sum())
        if n_vox == 0:
            continue
        max_prob_c = float(probs[c][mask_c].max())
        drop_for_vox = n_vox < min_vox
        drop_for_prob = max_prob_c < min_p
        if drop_for_vox or drop_for_prob:
            print(f"[inference] CC: drop class {c} "
                  f"(vox={n_vox}, max_prob={max_prob_c:.3f}, "
                  f"vox_thr={min_vox}, prob_thr={min_p}, "
                  f"drop_reason={'small_volume' if drop_for_vox else 'low_confidence'})",
                  file=sys.stderr, flush=True)
            pred[mask_c] = 0
    return pred


def _tta_passes() -> list[tuple[str, int | None, float]]:
    """Returns list of (label, flip_dim, gamma) tuples for the active TTA mode.

    flip_dim: tensor dim to flip (None = no flip):
        dim 2 = L↔R (after Orientationd("RAS") + EnsureChannelFirst, shape
                (B, C, X=L→R, Y=P→A, Z=I→S))
        dim 3 = P↔A
        dim 4 = I↔S
    gamma: gamma-correction exponent applied to the [0,1] image before fwd
           (1.0 = no change). For γ<1 image gets brighter, γ>1 darker.

    Modes (TTA env var):
        off                            → [orig]                                          (1 pass)
        lr                             → [orig, LR]                                      (2 passes)
        flips                          → [orig, LR, AP, IS]                              (4 passes)
        gamma                          → [orig, γ=0.9, γ=1.1]                            (3 passes)
        lr+gamma                       → [orig, LR, γ=0.9, γ=1.1]                        (4 passes)
        flips+gamma / full / 6tta      → [orig, LR, AP, IS, γ=0.9, γ=1.1]                (6 passes)
    """
    passes: list[tuple[str, int | None, float]] = [("orig", None, 1.0)]
    mode = TTA_MODE
    if mode in _TTA_OFF_VALUES:
        return passes
    # Bit-flags for which augmentations to apply, derived from the mode string.
    do_lr     = mode in {"lr", "lrflip", "lr+gamma", "flips", "flips+gamma", "full", "6tta"}
    do_ap     = mode in {"flips", "flips+gamma", "full", "6tta"}
    do_is     = mode in {"flips", "flips+gamma", "full", "6tta"}
    do_gamma  = mode in {"gamma", "lr+gamma", "flips+gamma", "full", "6tta"}
    if not (do_lr or do_ap or do_is or do_gamma):
        # Unknown mode → log loudly but fall back to off so we don't silently
        # mispredict at submission time.
        print(f"[inference] WARN unknown TTA mode '{TTA_MODE}', falling back to no TTA",
              file=sys.stderr, flush=True)
        return passes
    if do_lr:
        passes.append(("lr", 2, 1.0))                          # dim 2 = L→R
    if do_ap:
        passes.append(("ap", 3, 1.0))                          # dim 3 = P→A
    if do_is:
        passes.append(("is", 4, 1.0))                          # dim 4 = I→S
    if do_gamma:
        passes.append(("gamma_lo", None, 0.9))                 # brighter (γ<1)
        passes.append(("gamma_hi", None, 1.1))                 # darker (γ>1)
    return passes


def _apply_tta_input(image: torch.Tensor, flip_dim, gamma: float) -> torch.Tensor:
    """Build a TTA-augmented copy of the input image for one forward pass.

    image: [1, 2, X, Y, Z], values in [0, 1] (ScaleIntensityRanged output).
    """
    out = image
    if flip_dim is not None:
        out = torch.flip(out, dims=[flip_dim])
    if gamma != 1.0:
        # Both CT and PT channels are in [0, 1] from the test transform; fp16
        # autocast can introduce tiny <0 values, so clamp first to avoid NaN
        # from pow(-x, 0.9).
        out = out.clamp(0.0, 1.0).pow(gamma)
    return out


# Back-compat alias used by older code paths (replaced by _tta_passes).
_tta_flips = _tta_passes


# ─── I/O ─────────────────────────────────────────────────────────────────────

def _first_image(directory: Path) -> Path:
    files = (
        glob(str(directory / "*.mha"))
        + glob(str(directory / "*.nii.gz"))
        + glob(str(directory / "*.tif"))
        + glob(str(directory / "*.tiff"))
    )
    if not files:
        raise FileNotFoundError(f"no image in {directory}")
    return Path(files[0])


def _coregister_pet_to_ct(ct_path, pet_path) -> str:
    """Resample PET onto the CT voxel grid so the two model-input channels are
    spatially aligned. Returns a path to the aligned PET (a temp file), or the
    original path unchanged when CT and PET already share a grid.

    WHY (root cause of the 2026-06-09 sanity-check Dice=0.19 collapse):
    The released HECKTOR training data ships CT+PET ALREADY co-registered on a
    common grid (verified: CHUM-001 CT/PT/GT all 512x512x90 @ 0.977/3.307, same
    origin). The challenge PLATFORM instead feeds RAW images with CT and PET on
    DIFFERENT native grids (different origin/size/spacing; same world direction)
    — this is exactly what the official preprocess.py:resample_images() exists to
    fix. Our pipeline Spacingd's CT and PET INDEPENDENTLY then PETROICropd applies
    PET-derived voxel indices to the CT; if the two are not on a shared grid the
    channels desynchronise and segmentation collapses.

    Verified locally on a simulated raw pair (PET on a coarse shifted grid):
        current code .................. mean DSCagg 0.000
        + this PET->CT co-registration  mean DSCagg 0.794  (== common-grid 0.793)
    Resampling PET onto the CT grid (rather than the official 1mm/identity/
    intersection grid) reproduces the EXACT training preprocessing, because the
    training common-grid IS the CT's native grid with PET resampled onto it — so
    output.mha also stays on the native CT grid with no extra back-projection.

    No-op when grids already match, so it cannot perturb the common-grid local QA.
    """
    ct = sitk.ReadImage(str(ct_path))
    pet = sitk.ReadImage(str(pet_path))
    same = (
        tuple(ct.GetSize()) == tuple(pet.GetSize())
        and np.allclose(ct.GetSpacing(), pet.GetSpacing(), atol=1e-4)
        and np.allclose(ct.GetOrigin(), pet.GetOrigin(), atol=1e-3)
        and np.allclose(ct.GetDirection(), pet.GetDirection(), atol=1e-4)
    )
    if same:
        print("[inference] CT/PET already share a grid — skip PET co-registration",
              file=sys.stderr, flush=True)
        return str(pet_path)
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ct)                      # PET -> CT origin/size/spacing/direction
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(0.0)
    pet_on_ct = rs.Execute(pet)
    out = str(Path(tempfile.gettempdir()) / "pet_on_ct.nii.gz")
    sitk.WriteImage(pet_on_ct, out)
    print(f"[inference] co-registered PET {tuple(pet.GetSize())} "
          f"sp={tuple(round(x,2) for x in pet.GetSpacing())} -> CT grid "
          f"{tuple(ct.GetSize())} sp={tuple(round(x,2) for x in ct.GetSpacing())}",
          file=sys.stderr, flush=True)
    return out


def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[inference] WARN: could not read {p} ({e}) — using empty EHR "
              f"(encode_ehr fills missing fields with defaults)", flush=True)
        return {}


def _write_json(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def _write_segmentation(directory: Path, array: np.ndarray, ref_path: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    ref = sitk.ReadImage(str(ref_path))
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    img = sitk.GetImageFromArray(array.astype(np.uint8))
    img.CopyInformation(ref)
    sitk.WriteImage(img, str(directory / "output.mha"), useCompression=True)


# ─── EHR encoding ────────────────────────────────────────────────────────────
# Matches data/build_clinical_features.encode_tabular but operates directly on
# the ehr.json fields available at test time. Keeps the model input shape (18-d)
# identical to training.

def _ehr_num(v):
    """Parse an EHR field to float, returning None for anything not a real number:
    None, missing, NaN/inf, empty string, or non-numeric. Routes such values to each
    field's 'missing'/'unknown' default below instead of poisoning the tensor with NaN
    (a present-but-NaN field used to slip past `is None` → float(nan) → NaN clinical
    input → garbage T/N/RFS) or crashing on float('')/float('n/a'). 2026-06-16."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else f


def encode_ehr(ehr: dict) -> torch.Tensor:
    def _hpv(v):
        v = _ehr_num(v)
        if v is None: return "unknown"
        if v == 1.0: return "positive"
        if v == 0.0: return "negative"
        return "unknown"

    def _bin(v):
        v = _ehr_num(v)
        if v is None: return None
        if v == 1.0: return "yes"
        if v == 0.0: return "no"
        return None

    age = _ehr_num(ehr.get("Age"))
    age_z = (age - 60.0) / 12.0 if age is not None else 0.0

    g = _ehr_num(ehr.get("Gender"))
    gender_male = 1.0 if g is not None and g == 1.0 else 0.0

    hpv = _hpv(ehr.get("HPV Status"))
    hpv_pos, hpv_neg, hpv_unk = float(hpv == "positive"), float(hpv == "negative"), float(hpv == "unknown")

    smoker = _bin(ehr.get("Tobacco Consumption"))
    smoker_yes, smoker_no, smoker_missing = float(smoker == "yes"), float(smoker == "no"), float(smoker is None)

    drinker = _bin(ehr.get("Alcohol Consumption"))
    drinker_yes, drinker_no, drinker_missing = float(drinker == "yes"), float(drinker == "no"), float(drinker is None)

    ps = _ehr_num(ehr.get("Performance Status"))
    if ps is None:
        ps_0 = ps_1 = ps_high = 0.0
        ps_missing = 1.0
    else:
        v = ps
        ps_0, ps_1, ps_high, ps_missing = float(v == 0.0), float(v == 1.0), float(v >= 2.0), 0.0

    tx = _ehr_num(ehr.get("Treatment"))
    if tx is None:
        treatment_rt = treatment_other = 0.0
        treatment_missing = 1.0
    else:
        v = tx
        # FIX (2026-06-03): training _encode_clinical maps Treatment=1.0 → rt
        # and Treatment=0.0 → other. The previous docker encoding had this
        # swapped, which silently broke any model expecting training-convention
        # clinical input (including the new medai_hybrid Task 2/3 models).
        treatment_rt = float(v == 1.0)
        treatment_other = float(v == 0.0)
        treatment_missing = 0.0

    out = torch.tensor([
        age_z, gender_male,
        hpv_pos, hpv_neg, hpv_unk,
        smoker_yes, smoker_no, smoker_missing,
        drinker_yes, drinker_no, drinker_missing,
        ps_0, ps_1, ps_high, ps_missing,
        treatment_rt, treatment_other, treatment_missing,
    ], dtype=torch.float32)
    # final safety: never hand a non-finite clinical vector to the heads.
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ─── Model load + preprocessing ──────────────────────────────────────────────

_MODEL_CACHE = {}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_models() -> list:
    """Load all checkpoints under /opt/ml/model/seg_fold{i}.ckpt for ensembling.

    Falls back to /opt/ml/model/best.ckpt if no fold-specific files exist.
    """
    if _MODEL_CACHE:
        return _MODEL_CACHE["models"]

    from omegaconf import OmegaConf
    from training.trainer import HECKTORLightningModule

    cfg_path = MODEL_PATH / "config.yaml"
    if not cfg_path.exists():
        cfg_path = Path("/opt/app/configs/multitask_text.yaml")
    cfg_base = Path("/opt/app/configs/default.yaml")
    cfg = OmegaConf.merge(OmegaConf.load(cfg_base), OmegaConf.load(cfg_path))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["model"]["pretrained_weights"] = None                  # weights come from ckpt

    # Exclude medai_hybrid ckpts (those are loaded by _load_medai_models, not
    # this seg ensemble path). Match only `seg_fold*.ckpt`. NO cap — load all
    # 15 STU-Net top ckpts (3 per fold × 5 folds) for the locked seg ensemble.
    ckpts = sorted([p for p in MODEL_PATH.glob("seg_fold*.ckpt")
                     if not p.name.startswith("medai_")])
    if not ckpts:
        raise FileNotFoundError(f"no seg_fold*.ckpt under {MODEL_PATH}")

    models = []
    for c in ckpts:                                            # full 15-ckpt ensemble
        # Build the LightningModule with the CURRENT model architecture (which
        # may have head changes vs the ckpt — e.g. Phase 1 seg-only ckpt has
        # TN/Prog heads with old shapes from before tasks #36 / #45 / #46). We
        # use load_state_dict with manual shape-filter rather than
        # `load_from_checkpoint(strict=False)`, because PyTorch's strict=False
        # only ignores missing/extra keys, NOT size mismatches — those always
        # raise. Filtered keys (mismatched heads) stay at their fresh random init,
        # which is fine for fold-0 Phase 1 (heads were random anyway).
        ckpt = torch.load(str(c), map_location="cpu", weights_only=False)
        lm = HECKTORLightningModule(cfg=cfg)
        model_sd = lm.state_dict()
        ckpt_sd = ckpt.get("state_dict", ckpt)
        compatible = {}
        skipped = []
        for k, v in ckpt_sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                compatible[k] = v
            else:
                skipped.append((k, tuple(v.shape) if hasattr(v, "shape") else None,
                                tuple(model_sd[k].shape) if k in model_sd else None))
        missing, unexpected = lm.load_state_dict(compatible, strict=False)
        if skipped:
            print(f"[inference] {c.name}: skipped {len(skipped)} shape-mismatched keys "
                  f"(fresh init): e.g. {skipped[0]}", flush=True)
        if missing:
            print(f"[inference] {c.name}: {len(missing)} missing keys "
                  f"(fresh init): e.g. {missing[:3]}", flush=True)
        if unexpected:
            print(f"[inference] {c.name}: {len(unexpected)} unexpected keys "
                  f"(dropped): e.g. {unexpected[:3]}", flush=True)
        lm.to(_device()).eval()
        models.append(lm.model)
    _MODEL_CACHE["models"] = models
    _MODEL_CACHE["cfg"] = cfg
    return models


def _preprocess(ct_path: Path, pt_path: Path) -> dict:
    """Apply the test-time transforms to one CT+PT pair. Returns sample dict."""
    from data.transforms import get_test_transforms
    cfg = _MODEL_CACHE["cfg"]
    tfm = get_test_transforms(cfg["data"])
    sample = tfm({"ct": str(ct_path), "pt": str(pt_path),
                  "patient_id": "X", "has_pet": True})
    return sample


# ─── Subtask implementations ─────────────────────────────────────────────────

def _vram(label: str) -> None:
    """Profile: report peak-allocated and reserved GPU memory since the last call.
    Goes to stderr so the QA script's docker_stderr.log captures it. Lets us
    pinpoint which subtask spikes VRAM for T4-fit planning."""
    if not torch.cuda.is_available():
        return
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    reserved_mb = torch.cuda.memory_reserved() / 1024 ** 2
    alloc_mb = torch.cuda.memory_allocated() / 1024 ** 2
    import sys as _sys
    print(f"  [VRAM {label:>20s}] now_alloc={alloc_mb:6.0f}MB  "
          f"peak={peak_mb:6.0f}MB  reserved={reserved_mb:6.0f}MB",
          flush=True, file=_sys.stderr)
    torch.cuda.reset_peak_memory_stats()


def _seg_setup():
    """Load cfg + list seg ckpt PATHS (caches cfg for _preprocess). Does NOT build
    the 15 models — they're built one-at-a-time in run_segmentation (load-forward-free)
    so the container never holds >1 multitask module in RAM. Loading all 15 at once
    OOM-killed the Grand Challenge container (2026-06-16, confirmed via GC log: died
    while loading seg_fold14)."""
    if "cfg" in _MODEL_CACHE and "seg_ckpts" in _MODEL_CACHE:
        return _MODEL_CACHE["cfg"], _MODEL_CACHE["seg_ckpts"]
    from omegaconf import OmegaConf
    cfg_path = MODEL_PATH / "config.yaml"
    if not cfg_path.exists():
        cfg_path = Path("/opt/app/configs/multitask_text.yaml")
    cfg_base = Path("/opt/app/configs/default.yaml")
    cfg = OmegaConf.merge(OmegaConf.load(cfg_base), OmegaConf.load(cfg_path))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["model"]["pretrained_weights"] = None                  # weights come from ckpt
    ckpts = sorted([p for p in MODEL_PATH.glob("seg_fold*.ckpt")
                     if not p.name.startswith("medai_")])
    if not ckpts:
        raise FileNotFoundError(f"no seg_fold*.ckpt under {MODEL_PATH}")
    _MODEL_CACHE["cfg"] = cfg
    _MODEL_CACHE["seg_ckpts"] = ckpts
    return cfg, ckpts


def _build_seg_model(ckpt_path, cfg):
    """Build ONE seg model from a ckpt (shape-filtered load), move to device, return
    the inner nn.Module. Caller MUST `del` + empty_cache after the forward so the
    next ckpt reuses the freed memory (load-forward-free streaming)."""
    from training.trainer import HECKTORLightningModule
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    lm = HECKTORLightningModule(cfg=cfg)
    model_sd = lm.state_dict()
    ckpt_sd = ckpt.get("state_dict", ckpt)
    compatible = {k: v for k, v in ckpt_sd.items()
                  if k in model_sd and v.shape == model_sd[k].shape}
    n_skip = len(ckpt_sd) - len(compatible)
    lm.load_state_dict(compatible, strict=False)
    if n_skip:
        print(f"[inference] {ckpt_path.name}: {n_skip} shape-mismatched keys "
              f"left at fresh init", flush=True)
    del ckpt, ckpt_sd, compatible, model_sd
    lm.to(_device()).eval()
    return lm.model                                            # inner model; lm wrapper gc'd


def run_segmentation(ct_path: str, pet_path: str, ehr: dict) -> np.ndarray:
    """Single forward seg ensemble; returns uint8 array at the cached spatial size.

    LOAD-FORWARD-FREE streaming (2026-06-16): builds one ckpt's model, runs all TTA
    passes accumulating the softmax, then frees it before the next ckpt — so peak RAM
    is ~1 multitask module, not 15. (Holding all 15 OOM-killed the GC container.)
    Mathematically identical to the old all-at-once ensemble (softmax avg is associative).
    """
    import gc
    _vram("seg:enter")
    cfg, ckpts = _seg_setup()
    sample = _preprocess(Path(ct_path), Path(pet_path))
    image = sample["image"].unsqueeze(0).to(_device())          # [1, 2, 160, 160, 192]

    passes = _tta_passes()                                       # [(label, flip_dim, gamma)]
    seg_sum = None
    n_passes = 0
    n_ok = 0
    # ROBUSTNESS (2026-07-09): per-ckpt fault isolation. A single ckpt that
    # OOMs / NaNs / fails to load must NOT sink the whole case (which would
    # cascade an empty mask + random T/N/RFS into ALL three tasks). We isolate
    # each ckpt: a failure is logged and skipped, and we ensemble whatever
    # succeeded. As long as ≥1 ckpt runs, the segmentation is a real prediction;
    # only if EVERY ckpt fails do we emit a safe empty mask (below) rather than
    # crash. A ckpt is merged only after ALL its TTA passes succeed (so a
    # mid-TTA failure never leaves a partially-weighted model in the average).
    for c in ckpts:                                             # ← stream: 1 model at a time
        try:
            m = _build_seg_model(c, cfg)
            ckpt_sum = None
            ckpt_passes = 0
            for tta_label, flip_dim, gamma in passes:
                inp = _apply_tta_input(image, flip_dim, gamma)
                # fp16 autocast halves activation memory — keeps T4 16 GiB safe.
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = m.encoder(inp)                      # full STU-Net fwd
                prob = F.softmax(logits.float(), dim=1)
                # Un-flip the prediction back into the original orientation before
                # accumulating — otherwise the L/R-swapped probs would average
                # against the original probs and cancel anatomical asymmetries.
                # Gamma passes don't require un-flipping (no spatial transform).
                if flip_dim is not None:
                    prob = torch.flip(prob, dims=[flip_dim])
                ckpt_sum = prob if ckpt_sum is None else ckpt_sum + prob
                ckpt_passes += 1
                del inp, logits, prob
            # merge only after the whole ckpt succeeded
            seg_sum = ckpt_sum if seg_sum is None else seg_sum + ckpt_sum
            n_passes += ckpt_passes
            n_ok += 1
            del ckpt_sum
        except Exception as e:
            print(f"[seg] WARN: ckpt {c.name} failed ({type(e).__name__}: {e}) — "
                  f"skipping, ensembling the {n_ok} that succeeded so far",
                  flush=True)
        finally:
            try:
                del m                                           # free this model before next
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        _vram(f"seg:freed_{c.name}")

    # Total-failure fallback: no ckpt produced any output. Emit an empty mask at
    # the CT grid so the container still writes VALID output for this case (the
    # downstream tasks then hit their own empty-seg fallbacks: T2/N2, RFS 0.0)
    # instead of the container erroring out with no output at all.
    if seg_sum is None or n_passes == 0:
        print("[seg] ERROR: ALL seg ckpts failed — emitting empty-mask fallback so "
              "the case still produces valid output", flush=True)
        import SimpleITK as _sitk
        ref = _sitk.ReadImage(str(ct_path))
        return np.zeros(_sitk.GetArrayFromImage(ref).shape, dtype=np.uint8)
    print(f"[seg] ensembled {n_ok}/{len(ckpts)} ckpts ({n_passes} TTA passes)", flush=True)
    seg_avg = seg_sum / max(n_passes, 1)

    _vram("seg:after_fwd")
    pred = seg_avg.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    # Connected-component filter: drop tiny CC instances (helps GTVn F1-detection).
    pred = _apply_cc_filter(pred)
    # Class-confidence post-processing (default OFF since 2026-05-13): zero out
    # an entire class if predicted volume/max-prob is too low. See _cc_thresholds.
    pred = _apply_class_confidence(pred, seg_avg[0].cpu().numpy())
    # Cache the averaged probs + the MetaTensor affine for TN/Prog reuse — they
    # share the same encoder forward but only need the bottleneck features.
    _MODEL_CACHE["seg_probs"] = seg_avg.detach()
    _MODEL_CACHE["image_affine"] = sample["image"].affine
    del seg_sum
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _vram("seg:after_empty_cache")

    # Map back to the original CT spatial grid using the MetaTensor's RAS affine
    pred = _resample_to_reference(pred, ct_reference_path=Path(ct_path),
                                  image_affine=sample["image"].affine)
    del seg_avg
    return pred


def _bottleneck_and_mask(model, image, seg_probs):
    """Encoder-only forward + downsample seg_probs to bottleneck grid.

    Image is already at inference target size (160,160,192) — no center-crop.
    Returns (bottleneck [B, C, d, h, w], mask_p, mask_n) at bottleneck resolution.
    """
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        feats = model.encoder.encode(image)                     # only swinViT trunk, no decoder
        bottleneck = feats[-1].float()                           # cast back for the head MLPs
    del feats
    seg_at_bn = F.interpolate(seg_probs, size=bottleneck.shape[2:],
                              mode="trilinear", align_corners=False)
    mask_p = seg_at_bn[:, 1:2]                                  # GTVp prob channel
    mask_n = seg_at_bn[:, 2:3]                                  # GTVn prob channel
    return bottleneck, mask_p, mask_n


def _load_medai_models() -> dict:
    """Load the v6 multi-scale triplehead ensemble: medai_p{96,112}_fold*.ckpt
    (5 folds each, all DualHeadFusionResNet with T+N+RFS heads, trained with
    augment-then-crop + RAW mask). Returns {96: [m0..m4], 112: [m0..m4]}.
    """
    if "medai" in _MODEL_CACHE:
        return _MODEL_CACHE["medai"]
    # In docker, inference.py + medai_patch.py both live in /opt/app/.
    # On host, both live in docker/. Either way, add this file's directory.
    import sys as _sys
    here = str(Path(__file__).resolve().parent)
    if here not in _sys.path:
        _sys.path.insert(0, here)
    from medai_patch import load_medai_model
    dev = _device()
    medai = {}
    for ps in (96, 112):
        ckpts = sorted(MODEL_PATH.glob(f"medai_p{ps}_fold*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No medai_p{ps}_fold*.ckpt in {MODEL_PATH}")
        medai[ps] = [load_medai_model(c, dev, clinical_dim=18, with_rfs=True)
                      for c in ckpts]
        print(f"[medai] p{ps}: loaded {len(medai[ps])} fold ckpts", flush=True)
    _MODEL_CACHE["medai"] = medai
    return medai


def _load_medai_rfs_models() -> list:
    """v10: second RFS deep expert — rfs_only_aug ensemble (medai_rfs_fold*.ckpt,
    p96 only). Equal-weight member of the 3-expert RFS ensemble. Returns [] if the
    ckpts are absent (backward compat → falls back to the 2-expert path)."""
    if "medai_rfs" in _MODEL_CACHE:
        return _MODEL_CACHE["medai_rfs"]
    import sys as _sys
    here = str(Path(__file__).resolve().parent)
    if here not in _sys.path:
        _sys.path.insert(0, here)
    from medai_patch import load_medai_model
    dev = _device()
    ckpts = sorted(MODEL_PATH.glob("medai_rfs_fold*.ckpt"))
    models = [load_medai_model(c, dev, clinical_dim=18, with_rfs=True) for c in ckpts]
    print(f"[medai_rfs] loaded {len(models)} rfs-expert ckpts", flush=True)
    _MODEL_CACHE["medai_rfs"] = models
    return models


def _load_medai_rfsaff_models() -> list:
    """v11: FOURTH RFS deep expert — rfs_only + affine-aug ensemble
    (medai_rfsaff_fold*.ckpt, p96). Adds ensemble diversity: honest OOF 4-expert
    0.6988→0.7102 (higher AND lower per-fold variance). Returns [] if absent."""
    if "medai_rfsaff" in _MODEL_CACHE:
        return _MODEL_CACHE["medai_rfsaff"]
    import sys as _sys
    here = str(Path(__file__).resolve().parent)
    if here not in _sys.path:
        _sys.path.insert(0, here)
    from medai_patch import load_medai_model
    dev = _device()
    ckpts = sorted(MODEL_PATH.glob("medai_rfsaff_fold*.ckpt"))
    models = [load_medai_model(c, dev, clinical_dim=18, with_rfs=True) for c in ckpts]
    print(f"[medai_rfsaff] loaded {len(models)} rfsaff-expert ckpts", flush=True)
    _MODEL_CACHE["medai_rfsaff"] = models
    return models


def _load_medai_rfssig_models() -> list:
    """v15: FIFTH RFS deep expert — rfs_only trained with the SIGMOID-concordance
    loss (SurvLoss) instead of cox (medai_rfssig_fold*.ckpt, p96). Honest OOF: the
    strongest+most-stable single deep expert (solo 0.7127 ±0.0225 vs cox rfs10
    0.6933 ±0.0366); adding it 4→5 experts = 0.7107→0.7147 (+lower variance).
    Returns [] if absent (graceful equal4 fallback)."""
    if "medai_rfssig" in _MODEL_CACHE:
        return _MODEL_CACHE["medai_rfssig"]
    import sys as _sys
    here = str(Path(__file__).resolve().parent)
    if here not in _sys.path:
        _sys.path.insert(0, here)
    from medai_patch import load_medai_model
    dev = _device()
    ckpts = sorted(MODEL_PATH.glob("medai_rfssig_fold*.ckpt"))
    models = [load_medai_model(c, dev, clinical_dim=18, with_rfs=True) for c in ckpts]
    print(f"[medai_rfssig] loaded {len(models)} rfssig-expert ckpts", flush=True)
    _MODEL_CACHE["medai_rfssig"] = models
    return models


# Indices to drop when going from encode_ehr's 18-d output to the 13-d LogReg
# clinical block. These are the redundant "baseline" one-hot columns (each
# group sums to 1, so dropping one breaks the perfect linear dep that caused
# Cox/LogReg convergence to fail).
_CLIN_KEEP_IDX = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16]  # 13 of 18


def _radio_bundle() -> dict:
    """Load (once) docker/model/task23_radio.joblib — the full-data refit
    Route-B classifiers (LASSO-Cox RFS, LogReg N, LogReg T) + RFS z-norm stats."""
    if "radio_bundle" not in _MODEL_CACHE:
        import joblib
        _MODEL_CACHE["radio_bundle"] = joblib.load(MODEL_PATH / "task23_radio.joblib")
    return _MODEL_CACHE["radio_bundle"]


def _radio_feature_frame(ct_path: str, pet_path: str, seg_array: np.ndarray,
                         clin18: torch.Tensor) -> pd.DataFrame:
    """Assemble the 1-row (clin13 ++ radio30) feature frame in the joblib's
    column order, for the Route-B classifiers. Caches per-case (the seg array
    and clinical vector are identical across the TN and prognosis calls)."""
    if "radio_X" in _MODEL_CACHE:
        return _MODEL_CACHE["radio_X"]
    b = _radio_bundle()
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from radio_extract import extract_radio_vector
    clin13 = clin18.detach().cpu().numpy()[0, _CLIN_KEEP_IDX]            # (13,)
    # E (2026-06-16): radiomics on a degenerate/empty ROI can throw or return NaN.
    # Guard so it degrades to a zero vector instead of failing the case.
    try:
        radio_vec = extract_radio_vector(ct_path, pet_path, seg_array,
                                         b["feature_order"]["radio"])    # (30,)
        radio_vec = np.nan_to_num(np.asarray(radio_vec, dtype=float),
                                  nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        print(f"[inference] WARN: radiomics extraction failed ({e}) — zero radio vector",
              flush=True)
        radio_vec = np.zeros(len(b["feature_order"]["radio"]), dtype=float)
    cols = b["feature_order"]["clin"] + b["feature_order"]["radio"]
    X = pd.DataFrame([np.concatenate([clin13, radio_vec])], columns=cols)
    _MODEL_CACHE["radio_X"] = X
    return X


def _radio_cox_risk(X: pd.DataFrame) -> float:
    """Route-B RFS risk = log partial hazard of the full-data LASSO-Cox."""
    b = _radio_bundle()["rfs"]
    Xz = (X[b["keep"]] - b["mu"]) / b["sd"]
    return float(np.log(b["cph"].predict_partial_hazard(Xz).values[0] + 1e-9))


def _nodal_geom_features(seg_array: np.ndarray, ct_path: str) -> dict:
    """Clinically-aligned GTVn nodal-geometry features for N-staging (v13).

    Computed on the CC-filtered `seg` (the SAME array N-staging receives — 26-conn,
    ≥500 mm³ GTVn CCs, no SUV-gate), byte-matching the OOF extraction
    (scripts/n_nodal_geom_probe.py:nodal_geom): seg is (z,y,x) sitk order, spacing
    from the CT (sx,sy,sz), so vv=sx*sy*sz and per-axis extent pairs [sz,sy,sx].
    Returns dict keyed by the 7 feature names (largest node count + size encode the
    N-stage definition that radio30 lacked)."""
    from scipy.ndimage import label as cc_label, generate_binary_structure
    sx, sy, sz = sitk.ReadImage(str(ct_path)).GetSpacing()
    vv = float(sx * sy * sz)
    zero = {"n_ccs": 0.0, "largest_ml": 0.0, "largest_frac": 0.0,
            "largest_maxdim": 0.0, "total_ml": 0.0, "log_largest": 0.0, "log_total": 0.0}
    m = (seg_array == 2)
    if not m.any():
        return zero
    struct = generate_binary_structure(3, 3)                       # 26-connectivity
    lab, n = cc_label(m, structure=struct)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    vols = sizes[1:] * vv / 1000.0                                 # ml per CC
    keep = vols >= 500.0 / 1000.0                                  # ≥500 mm³ (deploy GTVn thr)
    keep_ids = np.arange(1, n + 1)[keep]; vols = vols[keep]
    if vols.size == 0:
        return zero
    big = keep_ids[int(np.argmax(vols))]
    coords = np.argwhere(lab == big)
    extent_mm = (coords.max(0) - coords.min(0) + 1) * np.array([sz, sy, sx])
    largest = float(vols.max()); total = float(vols.sum())
    return {"n_ccs": float(vols.size), "largest_ml": largest,
            "largest_frac": float(largest / total), "largest_maxdim": float(extent_mm.max()),
            "total_ml": total, "log_largest": float(np.log1p(largest)),
            "log_total": float(np.log1p(total))}


def _gtvp_geom_features(seg_array: np.ndarray, ct_path: str) -> dict:
    """Primary-tumor (GTVp) geometry for T-staging (v14): the largest-diameter size
    criterion that radio30 lacked. Byte-matches the OOF extraction
    (scripts/eval_t_geom_oof.py:gtvp_geom): class 1, ≥1000 mm³ CC, (z,y,x) seg with
    CT spacing. Keys: p_n_ccs, p_vol_ml, p_maxdim, p_logvol."""
    from scipy.ndimage import label as cc_label, generate_binary_structure
    sx, sy, sz = sitk.ReadImage(str(ct_path)).GetSpacing()
    vv = float(sx * sy * sz)
    zero = {"p_n_ccs": 0.0, "p_vol_ml": 0.0, "p_maxdim": 0.0, "p_logvol": 0.0}
    m = (seg_array == 1)
    if not m.any():
        return zero
    struct = generate_binary_structure(3, 3)
    lab, n = cc_label(m, structure=struct)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    vols = sizes[1:] * vv / 1000.0
    keep = vols >= 1000.0 / 1000.0                                 # ≥1000 mm³ (deploy GTVp thr)
    keep_ids = np.arange(1, n + 1)[keep]; vols = vols[keep]
    if vols.size == 0:
        return zero
    big = keep_ids[int(np.argmax(vols))]
    coords = np.argwhere(lab == big)
    extent_mm = (coords.max(0) - coords.min(0) + 1) * np.array([sz, sy, sx])
    tot = float(vols.sum())
    return {"p_n_ccs": float(vols.size), "p_vol_ml": tot,
            "p_maxdim": float(extent_mm.max()), "p_logvol": float(np.log1p(tot))}


def run_tn_staging(ct_path: str, pet_path: str, ehr: dict,
                   segmentation_array: np.ndarray) -> tuple[str, str]:
    """Task 2 ensemble (v6, locked 2026-06-11): multi-scale deep triplehead
    {patch96, patch112} × 5 folds, fused with the Route-B radiomics+clinical LogReg.
      T-stage: 0.5·deep + 0.5·radio (deep T head RESURRECTED by affaug+RAW;
               honest fuse 0.564 vs radio-alone 0.457). argmax restricted to T1..T4.
      N-stage: radio-leaning soft-avg(deep n_softmax, radio LogReg proba), w_radio 0.6.

    Deep t_softmax column order is [T1,T2,T3,T4,T0] → reorder to [T0..T4] before
    fusing (deep_t_reorder in the joblib). See /tmp/fusion_new.py + LIVE_STATE.
    """
    _vram("tn:enter")
    # A (2026-06-16): empty predicted seg → radiomics/patch extraction would crash.
    # Fall back to the TRAINING MARGINAL MODES (T2: 285/782, N2: 471/782) — the
    # expected-score-optimal blind guess. (Was T2/N0; N0 is near-rarest at 87/782.)
    if int((segmentation_array > 0).sum()) == 0:
        print("[inference] WARN: empty segmentation — Task 2 default T2 / N2 (training modes)", flush=True)
        return "T2", "N2"
    medai = _load_medai_models()
    dev = _device()

    # Build multi-scale {96,112} patches with RAW mask channel (one resample pass).
    from medai_patch import build_medai_patches_multiscale
    patches = build_medai_patches_multiscale(
        ct_path, pet_path, segmentation_array, patch_sizes=(96, 112), raw_mask=True)
    clin18 = encode_ehr(ehr).unsqueeze(0).to(dev)               # (1, 18)
    _vram("tn:patch_built")

    # ---- Deep ensemble: avg T+N softmax over all 10 (2 scales × 5 folds) ----
    t_sum = None; n_sum = None; n_models = 0
    cached = {}
    with torch.no_grad():
        for ps in (96, 112):
            image, mask = patches[ps]
            image = image.unsqueeze(0).to(dev); mask = mask.unsqueeze(0).to(dev)
            cached[ps] = (image, mask)
            for m in medai[ps]:
                out = m(image, mask, clin18)
                t = F.softmax(out["t_logits"], dim=-1)
                n = F.softmax(out["n_logits"], dim=-1)
                t_sum = t if t_sum is None else t_sum + t
                n_sum = n if n_sum is None else n_sum + n
                n_models += 1
    t_deep = (t_sum / n_models)[0].detach().cpu().numpy()       # (5,) order [T1,T2,T3,T4,T0]
    n_deep = (n_sum / n_models)[0].detach().cpu().numpy()       # (4,) N0..N3
    _vram("tn:after_fwd")

    # ---- Route-B radiomics+clinical classifiers ----
    b = _radio_bundle()
    X = _radio_feature_frame(ct_path, pet_path, segmentation_array, clin18)
    fo = b["feature_order"]
    # T-stage (v14): append GTVp geometry to the radio-T frame when present (honest
    # OOF fused-T 0.4439→0.4535); else legacy clin+radio30.
    if "gtvp_geom" in fo:
        pg = _gtvp_geom_features(segmentation_array, ct_path)
        Xt = pd.DataFrame([np.concatenate([X.values[0], np.array([pg[k] for k in fo["gtvp_geom"]], float)])],
                          columns=list(X.columns) + list(fo["gtvp_geom"]))
        t_radio = b["t"]["clf"].predict_proba(b["t"]["scaler"].transform(Xt.values))[0]  # (5,) T0..T4
    else:
        t_radio = b["t"]["clf"].predict_proba(b["t"]["scaler"].transform(X.values))[0]  # (5,) T0..T4
    # N-stage (v13): clin + nodal-geometry when the joblib carries a nodal_geom block
    # (honest OOF 0.6911→0.7200); else legacy clin+radio30.
    if "nodal_geom" in fo:
        clin13 = clin18.detach().cpu().numpy()[0, _CLIN_KEEP_IDX]
        geom = _nodal_geom_features(segmentation_array, ct_path)
        n_cols = list(fo["clin"]) + list(fo["nodal_geom"])
        n_vec = np.concatenate([clin13, np.array([geom[k] for k in fo["nodal_geom"]], float)])
        Xn = pd.DataFrame([n_vec], columns=n_cols)
        n_radio = b["n"]["clf"].predict_proba(b["n"]["scaler"].transform(Xn.values))[0]  # (4,) N0..N3
    else:
        n_radio = b["n"]["clf"].predict_proba(b["n"]["scaler"].transform(X.values))[0]  # (4,) N0..N3

    # Cache the patches + clinical for run_prognosis (same case).
    _MODEL_CACHE["medai_patches"] = cached
    _MODEL_CACHE["medai_clin18"] = clin18

    # ---- T-stage: deep+radio fusion (reorder deep to T0..T4), restrict T1..T4 ----
    reorder = b.get("deep_t_reorder", [4, 0, 1, 2, 3])
    t_deep_o = t_deep[reorder]                                  # → [T0,T1,T2,T3,T4]
    wt = b["ensemble"].get("t_w_radio", 0.5)
    t_ens = (1.0 - wt) * t_deep_o + wt * t_radio                # (5,) T0..T4
    t_classes = b["classes"]["t"]                               # ['T0',..,'T4']
    valid = [i for i, c in enumerate(t_classes) if c != "T0"]
    t_idx = valid[int(np.argmax(t_ens[valid]))]
    t_label = t_classes[t_idx]

    # ---- N-stage: soft ensemble of deep + radio ----
    wn = b["ensemble"]["n_w_radio"]
    n_ens = (1.0 - wn) * n_deep + wn * n_radio                  # (4,)
    n_label = b["classes"]["n"][int(np.argmax(n_ens))]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _vram("tn:exit")
    return t_label, n_label


def run_prognosis(ct_path: str, pet_path: str, ehr: dict,
                  segmentation_array: np.ndarray, t_stage: str, n_stage: str) -> float:
    """Task 3 RFS ensemble (v6, 2026-06-11): z-avg of the multi-scale deep
    triplehead risk (10 models = {patch96,patch112}×5 folds) and the radiomics+
    clinical LASSO-Cox risk. Higher = greater recurrence risk.

    Honest 3/1/1 rotation (/tmp/fusion_new.py): deep 0.668, radio 0.645,
    z-avg fuse 0.672. z-norm uses TRAIN mean/std baked into task23_radio.joblib.
    """
    _vram("prog:enter")
    # A (2026-06-16): empty predicted seg → radiomics/patch extraction would crash.
    if int((segmentation_array > 0).sum()) == 0:
        print("[inference] WARN: empty segmentation — Task 3 default risk 0.0", flush=True)
        return 0.0
    medai = _load_medai_models()
    dev = _device()

    # Reuse the cached multi-scale patches from run_tn_staging if available.
    if "medai_patches" in _MODEL_CACHE:
        cached = _MODEL_CACHE["medai_patches"]
        clin18 = _MODEL_CACHE["medai_clin18"]
    else:
        from medai_patch import build_medai_patches_multiscale
        patches = build_medai_patches_multiscale(
            ct_path, pet_path, segmentation_array, patch_sizes=(96, 112), raw_mask=True)
        cached = {}
        for ps in (96, 112):
            image, mask = patches[ps]
            cached[ps] = (image.unsqueeze(0).to(dev), mask.unsqueeze(0).to(dev))
        clin18 = encode_ehr(ehr).unsqueeze(0).to(dev)
    _vram("prog:patch_ready")

    # ---- Deep expert: multi-scale triplehead ensemble (10 models) → mean risk ----
    risks = []
    with torch.no_grad():
        for ps in (96, 112):
            image, mask = cached[ps]
            for m in medai[ps]:
                out = m(image, mask, clin18)
                risks.append(float(out["risk"].squeeze(-1).item()))
    deep_risk = float(np.mean(risks))

    # ---- Radio expert: full-data LASSO-Cox log partial hazard ----
    X = _radio_feature_frame(ct_path, pet_path, segmentation_array, clin18)
    radio_risk = _radio_cox_risk(X)

    # ---- z-norm each via TRAIN stats ----
    b = _radio_bundle()
    zn = b["rfs_zn"]; ens = b["ensemble"]
    tri_z = (deep_risk - zn["deep_mu"]) / (zn["deep_sd"] + 1e-9)
    radio_z = (radio_risk - zn["radio_mu"]) / (zn["radio_sd"] + 1e-9)

    # Equal-weight deep+radio RFS ensemble. Honest 5-fold OOF C-index:
    #   2-expert (v9-)  0.6824 ; 3-expert (v10 equal3)  0.6988 ;
    #   4-expert (v11 equal4, +rfs_aff)  0.7102 (higher AND lower per-fold variance).
    # z-norm each expert with its OOF train stats, then average. rfs_aff/rfs are
    # p96-only (use the 96 patch). Graceful: any missing ckpt-set is skipped and
    # we average whatever loaded (>=2 experts), never crashing the case.
    mode = ens.get("rfs_mode", "")
    img96, msk96 = cached[96]
    zs = [tri_z, radio_z]

    def _mean_risk(models):
        rr = []
        with torch.no_grad():
            for m in models:
                rr.append(float(m(img96, msk96, clin18)["risk"].squeeze(-1).item()))
        return float(np.mean(rr))

    if mode in ("equal3", "equal4", "equal5") and "rfs_mu" in zn:
        rfs_models = _load_medai_rfs_models()
        if rfs_models:
            zs.append((_mean_risk(rfs_models) - zn["rfs_mu"]) / (zn["rfs_sd"] + 1e-9))
    if mode in ("equal4", "equal5") and "rfsaff_mu" in zn:
        rfsaff_models = _load_medai_rfsaff_models()
        if rfsaff_models:
            zs.append((_mean_risk(rfsaff_models) - zn["rfsaff_mu"]) / (zn["rfsaff_sd"] + 1e-9))
    if mode == "equal5" and "rfssig_mu" in zn:                # v15: SurvLoss sigmoid expert
        rfssig_models = _load_medai_rfssig_models()
        if rfssig_models:
            zs.append((_mean_risk(rfssig_models) - zn["rfssig_mu"]) / (zn["rfssig_sd"] + 1e-9))

    if len(zs) >= 3:                       # ensemble path (equal weight)
        risk = float(np.mean(zs))
    else:                                  # 2-expert fallback (old joblib / missing ckpts)
        w = ens["rfs_w_radio"]
        risk = float((1.0 - w) * tri_z + w * radio_z)
    _vram("prog:exit")
    return risk


def _UNUSED_run_prognosis_legacy(ct_path: str, pet_path: str, ehr: dict,
                                   segmentation_array: np.ndarray, t_stage: str,
                                   n_stage: str) -> float:
    """LEGACY — old SwinUNETR-encoder Cox-head path. Kept as reference; not called."""
    from models.tn_staging_head import masked_global_pool

    models = _load_models()
    seg_probs = _MODEL_CACHE.get("seg_probs")
    sample = _preprocess(Path(ct_path), Path(pet_path))
    image = sample["image"].unsqueeze(0).to(_device())
    clin_feat = encode_ehr(ehr).unsqueeze(0).to(_device())
    _vram("prog:before_fwd")

    # Task #45: prog_head also takes the TN softmax cached by run_tn_staging.
    # If TN wasn't run first (defensive), use a uniform fallback so the Linear
    # in_dim still matches.
    tn_softmax = _MODEL_CACHE.get("tn_softmax")
    if tn_softmax is None and getattr(models[0].prog_head, "use_tn", False):
        n_t = models[0].tn_head.t_head[-1].out_features
        n_n = models[0].tn_head.n_head[-1].out_features
        tn_softmax = torch.full((1, n_t + n_n), 1.0 / max(n_t, n_n), device=_device())

    passes = _tta_passes()
    risks = []
    for m in models:
        prog_use_clinical = bool(getattr(m.prog_head, "use_clinical", True))
        prog_use_tn = bool(getattr(m.prog_head, "use_tn", False))
        for tta_label, flip_dim, gamma in passes:
            img_in = _apply_tta_input(image, flip_dim, gamma)
            seg_in = seg_probs if flip_dim is None else torch.flip(seg_probs, dims=[flip_dim])
            bottleneck, mask_p, mask_n = _bottleneck_and_mask(m, img_in, seg_in)
            feat_p = masked_global_pool(bottleneck, mask_p)
            feat_n = masked_global_pool(bottleneck, mask_n)
            image_feat = torch.cat([feat_p, feat_n], dim=-1)
            risk = m.prog_head(
                image_feat,
                clin_feat if prog_use_clinical else None,
                tn_feat=tn_softmax if prog_use_tn else None,
            )
            _vram(f"prog:after_fwd_{tta_label}")
            risks.append(float(risk[0].item()))
            del bottleneck, mask_p, mask_n, feat_p, feat_n, image_feat, risk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _vram("prog:exit")
    return float(np.mean(risks))


# ─── helpers ────────────────────────────────────────────────────────────────

def _resample_to_reference(pred_arr: np.ndarray, ct_reference_path: Path,
                           image_affine) -> np.ndarray:
    """Resample the cached-spatial-size prediction back to the original CT grid.

    `image_affine` is the 4x4 RAS affine of the MetaTensor that was fed to the
    model (`sample["image"].affine`). It encodes the cumulative spatial transform
    (Spacingd 2mm + CropForegroundd + ResizeWithPadOrCropd + DivisiblePadd) so we
    can map every prediction voxel back to its true world coordinate, then resample
    onto the CT's pixel grid. The previous version only set spacing — origin and
    direction defaulted to (0,0,0)/identity, which dropped the patient outside the
    CT's bounding box and gave DiceAgg=0 against ground truth.
    """
    if hasattr(image_affine, "cpu"):
        image_affine = image_affine.cpu().numpy()
    A = np.asarray(image_affine, dtype=np.float64)
    # MONAI tensor axis order (D, H, W) -> sitk pixel order (x=W, y=H, z=D);
    # permute affine columns so it maps sitk (x,y,z) instead of MONAI (i,j,k).
    P = np.array([[0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1]], dtype=np.float64)
    A = A @ P
    # MONAI is RAS world; sitk is LPS world. Flip x,y signs.
    A = np.diag([-1.0, -1.0, 1.0, 1.0]) @ A
    origin = A[:3, 3]
    M = A[:3, :3]
    spacing = np.linalg.norm(M, axis=0)
    direction = (M / spacing).flatten().tolist()              # row-major flat for sitk

    pred_sitk = sitk.GetImageFromArray(pred_arr.astype(np.uint8))
    pred_sitk.SetOrigin(origin.tolist())
    pred_sitk.SetSpacing(spacing.tolist())
    pred_sitk.SetDirection(direction)

    ct_ref = sitk.ReadImage(str(ct_reference_path))
    rsl = sitk.ResampleImageFilter()
    rsl.SetReferenceImage(ct_ref)
    rsl.SetInterpolator(sitk.sitkNearestNeighbor)
    rsl.SetDefaultPixelValue(0)
    return sitk.GetArrayFromImage(rsl.Execute(pred_sitk)).astype(np.uint8)


# ─── entry point ─────────────────────────────────────────────────────────────

def run() -> int:
    """B (2026-06-16): every subtask is wrapped so a single failure on one case
    produces a sensible default output (and a logged traceback) instead of crashing
    the whole submission. The container always writes all 4 outputs and exits 0."""
    import traceback
    ct = _first_image(INPUT_PATH / "images/ct")
    pt = _first_image(INPUT_PATH / "images/pet")
    ehr = _load_json(INPUT_PATH / "ehr.json")

    # CRITICAL (2026-06-09): the platform feeds RAW CT/PET on DIFFERENT native
    # grids; our pipeline assumes they're co-registered (as the training data is).
    # Resample PET onto the CT grid up-front. No-op when grids already match.
    try:
        pt = _coregister_pet_to_ct(ct, pt)
    except Exception:
        print(f"[inference] WARN: PET→CT coregistration failed — using PET as-is\n"
              f"{traceback.format_exc()}", flush=True)

    # Subtask 1 — segmentation (output.mha). On failure: empty mask at CT grid.
    try:
        seg = run_segmentation(str(ct), str(pt), ehr)
        seg_out = _suv_gate_gtvn(seg, ct, pt)   # v7.1 SUV-gate on output.mha only
    except Exception:
        print(f"[inference] ERROR: segmentation failed — writing EMPTY mask\n"
              f"{traceback.format_exc()}", flush=True)
        ref = sitk.ReadImage(str(ct))
        seg = np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint8)
        seg_out = seg
    _write_segmentation(OUTPUT_PATH / "images/head-neck-tumor-segmentation", seg_out, ct)

    # Subtask 2 — T/N staging (bare JSON strings). On failure: TRAINING MARGINAL
    # MODES T2/N2 (expected-score-optimal blind guess; see run_tn_staging guard A).
    try:
        t_stage, n_stage = run_tn_staging(str(ct), str(pt), ehr, seg)
    except Exception:
        print(f"[inference] ERROR: TN staging failed — default T2 / N2 (training modes)\n"
              f"{traceback.format_exc()}", flush=True)
        t_stage, n_stage = "T2", "N2"
    _write_json(OUTPUT_PATH / "t-stage.json", t_stage)
    _write_json(OUTPUT_PATH / "n-stage.json", n_stage)

    # Subtask 3 — RFS (bare float). On failure: median risk 0.0.
    try:
        rfs = run_prognosis(str(ct), str(pt), ehr, seg, t_stage, n_stage)
    except Exception:
        print(f"[inference] ERROR: prognosis failed — default risk 0.0\n"
              f"{traceback.format_exc()}", flush=True)
        rfs = 0.0
    # SIGN FIX (2026-06-30): run_prognosis returns a RISK (higher = higher hazard =
    # shorter RFS). The GC C-index for this challenge scores rfs.json as if HIGHER =
    # LONGER survival, so our correct risk graded as 1-C (validation showed 0.3286 =
    # 1 - 0.6714, our OOF). NEGATE so higher = longer survival → C flips 0.329 → 0.671
    # on the same set. Confirmed by TingYi's two submissions summing to 1.0000.
    _write_json(OUTPUT_PATH / "rfs.json", float(-rfs))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
