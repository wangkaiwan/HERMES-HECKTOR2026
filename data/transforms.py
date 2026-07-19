"""
MONAI transform pipelines for HECKTOR 2026.

Adapted from HN_CU_Seg/data/transforms.py with these simplifications:
    - drop the optional PCT third modality (HECKTOR has only CT + PT)
    - drop missing-modality handling (both CT and PT are always present)
    - 3-class label (0=bg, 1=GTVp, 2=GTVn) — use RandCropByLabelClassesd
      with ratios matching the official baseline (bg, GTVp, GTVn)
    - propagate patient_id, t_stage, n_stage, relapse, rfs_days, and
      has_seg/has_staging/has_survival flags through SelectItemsd

Two-phase pipeline (cached via PersistentDataset):
    get_train_preprocess_transforms  — deterministic: load → orient → 2mm → clip → crop → pad
    get_train_augmentation_transforms — random patch + flips + intensity jitter +
                                        optional PET drop (`pet_drop_prob`)

CT clip: [-200, 200] HU → [0, 1].
PT: percentile-clip 0.5/99.5 → [0, 1] (default), or absolute SUV `pt_clip_range` if provided.
Target spacing: 2 mm isotropic.
Final padded volume: 200×200×200 (cap on cache size).
"""
from __future__ import annotations

import numpy as np
import torch
from monai.transforms import (
    BorderPad,
    Compose,
    ConcatItemsd,
    CopyItemsd,
    CropForegroundd,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandomizableTransform,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSimulateLowResolutiond,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SelectItemsd,
    Spacingd,
    SpatialCrop,
)


_CACHE_VOLUME = (200, 200, 200)               # padded size in cache (per HN_CU_Seg)
_TARGET_SPACING = (2.0, 2.0, 2.0)             # mm isotropic
# Keys that travel through the transform pipeline (read from the manifest
# and must survive SelectItemsd at the end of augmentation). Note that
# text_features and clinical_feat are NOT here — those are injected by
# data.dataset._TwoPhaseDataset.__getitem__ AFTER transforms run.
_PASSTHROUGH_KEYS = [
    "patient_id", "center", "has_pet",
    "t_stage", "n_stage", "relapse", "rfs_days",
    "has_seg", "has_staging", "has_survival",
]


# ── Custom transforms ─────────────────────────────────────────────────────────

class Contiguousd(MapTransform):
    """
    Force every keyed tensor to be contiguous (its own storage = view size).

    MONAI's Spacingd / CropForegroundd / ResizeWithPadOrCropd can leave the
    output tensor as a non-contiguous VIEW into a much bigger underlying
    storage. PersistentDataset pickles the underlying storage, not the view —
    we observed cache files 8× larger than the logical tensor size for some
    patients. A single .contiguous() at the end of preprocessing breaks this
    by reallocating storage to match the view exactly.
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            v = d.get(key)
            if not isinstance(v, torch.Tensor):
                continue
            if not v.is_contiguous():
                v = v.contiguous()
            # A tensor can be CONTIGUOUS yet still be a view into a much larger
            # underlying storage (e.g. a crop whose strides happen to match) —
            # .contiguous() is then a no-op and does NOT shrink storage, so
            # torch.save still pickles the full oversized buffer (observed 8×
            # cache bloat on whole-body CTs via the nnU-Net CT-norm path). clone()
            # forces a right-sized allocation. Guard so we only pay it when needed.
            try:
                if v.untyped_storage().nbytes() > v.numel() * v.element_size():
                    v = v.clone()
            except Exception:
                v = v.contiguous().clone()
            d[key] = v
        return d


class KeepTopZd(MapTransform):
    """Restrict the z (axis -1, RAS +z = superior = head) extent to `target_z`
    voxels, biasing toward the **top** of the volume.

    After `Orientationd(axcodes="RAS")`, MONAI volumes have shape (C, X, Y, Z)
    where the last spatial axis is I→S. The head sits at high z indices for
    head-and-neck or whole-body scans.

    - If current z > target_z: crop FROM THE BOTTOM (keep the top `target_z`
      slices). This preserves the head on whole-body CTs where `CropForegroundd`
      returned a body bbox extending into chest/abdomen.
    - If current z < target_z: PAD AT THE BOTTOM (low-z, inferior end) so the
      data stays at the top of the canvas. The subsequent `ResizeWithPadOrCropd`
      is then a no-op in z and only needs to handle x/y center crop/pad.

    Insert between `CropForegroundd` and `ResizeWithPadOrCropd`. Apply to the
    same keys (`ct`, `pt`, and `label` when present) so the GT mask gets the
    same crop as the image — critical to keep training-time DiceAgg honest.

    Diagnosis & rationale: `evaluation/results/QA_FOLD0_DIAGNOSIS.md`.
    """

    def __init__(self, keys, target_z: int, pad_mode: str = "constant",
                 allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.target_z = int(target_z)
        self.pad_mode = pad_mode

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            t = d[key]
            shape = t.shape  # (C, X, Y, Z)
            cx, cy, cz = shape[-3], shape[-2], shape[-1]
            if cz == self.target_z:
                continue
            if cz > self.target_z:
                # Crop along z: keep slices [cz - target_z : cz] (the top).
                # MetaTensor's `__getitem__` does NOT update the affine origin
                # for slice offsets in MONAI 1.3.2 (verified empirically). We
                # use SpatialCrop, which IS meta-aware and shifts origin by
                # roi_start * spacing * direction.
                cropper = SpatialCrop(
                    roi_start=[0, 0, cz - self.target_z],
                    roi_end=[cx, cy, cz],
                )
                d[key] = cropper(t)
            else:
                # Pad at the inferior (low-z) end. BorderPad is the MetaTensor-
                # aware version of torch.nn.functional.pad — it updates the
                # MONAI affine so the new slice-0 maps to a lower z in physical
                # space, which is essential for `_resample_to_reference` in
                # docker inference. spatial_border for 3 spatial dims is laid
                # out as (x_low, x_high, y_low, y_high, z_low, z_high); we pad
                # only at z_low.
                pad_amount = self.target_z - cz
                padder = BorderPad(
                    spatial_border=[0, 0, 0, 0, pad_amount, 0],
                    mode=self.pad_mode,
                )
                d[key] = padder(t)
        return d


class KeepSuperiorMMd(MapTransform):
    """Pre-resample coarse crop: keep only the superior `keep_mm` of Z (RAS +z =
    superior = head), in NATIVE space, BEFORE Spacingd.

    Why: whole-body CTs (e.g. CHUV/Lausanne, FOV ~1095mm in Z) blow up the 1mm
    resample intermediate to ~5e8 voxels (~10GB host RAM) BEFORE PETROICropd
    shrinks them — which OOM-kills the docker container under any memory
    pressure. Dropping everything below ~keep_mm from the head apex cuts the
    resample cost dramatically while keeping the entire head-neck region intact
    (HN tumors + nodes are all superior; below the upper mediastinum is
    irrelevant). No-op for scans already shorter than keep_mm (normal HN ~300mm).

    Runs after Orientationd(RAS), before Spacingd. Reads z-spacing from the
    MetaTensor affine, keeps the top `keep_mm/z_spacing` voxels via SpatialCrop
    (meta-aware → shifts affine origin correctly, required for docker
    back-projection). CT + PT share the native grid (verified 2026-05-21) so the
    same keep_vox crops both identically.
    """

    def __init__(self, keys, keep_mm: float = 450.0, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.keep_mm = float(keep_mm)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            t = d[key]
            aff = getattr(t, "affine", None)
            z_spacing = float(torch.linalg.norm(aff[:3, 2])) if aff is not None else 1.0
            cx, cy, cz = t.shape[-3], t.shape[-2], t.shape[-1]
            keep_vox = int(np.ceil(self.keep_mm / max(z_spacing, 1e-6)))
            if cz <= keep_vox:
                continue                                       # already short enough
            cropper = SpatialCrop(roi_start=[0, 0, cz - keep_vox], roi_end=[cx, cy, cz])
            d[key] = cropper(t)
        return d


class RandInvertGammad(MapTransform, RandomizableTransform):
    """nnU-Net's gamma transform applied to the INVERTED image (p=0.1 default).
    Normalize per-key to [0,1] using its own min/max, invert (x→1-x), apply gamma,
    invert back, denormalize. Operates per-key so CT and PT each get an
    independent random gamma. Skips keys with zero range.
    """

    def __init__(self, keys, prob: float = 0.1, gamma=(0.7, 1.5)):
        MapTransform.__init__(self, keys=keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.gamma_range = (float(gamma[0]), float(gamma[1]))

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d
        for key in self.key_iterator(d):
            t = d[key]
            gamma = float(self.R.uniform(*self.gamma_range))
            lo = float(t.min()); hi = float(t.max()); rng = hi - lo
            if rng <= 0:
                continue
            x = (t - lo) / rng                       # [0, 1]
            x = 1.0 - x                              # invert
            x = torch.clamp(x, min=0.0) ** gamma     # gamma
            x = 1.0 - x                              # invert back
            d[key] = x * rng + lo                    # denormalize
        return d


class RandContrastAroundMeand(MapTransform, RandomizableTransform):
    """nnU-Net ContrastAugmentationTransform (p=0.15 default). Scales contrast
    around the per-key mean: x' = (x - mean) * factor + mean, factor sampled
    from `factor_range`. If `preserve_range=True` (nnU-Net default), clamps the
    output to the original [min, max] so CT/PET stay in their physical window
    (avoids the contrast aug nuking ScaleIntensityRanged's normalization).
    """

    def __init__(self, keys, prob: float = 0.15,
                 factor_range=(0.75, 1.25), preserve_range: bool = True):
        MapTransform.__init__(self, keys=keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.factor_range = (float(factor_range[0]), float(factor_range[1]))
        self.preserve_range = bool(preserve_range)

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d
        for key in self.key_iterator(d):
            t = d[key]
            factor = float(self.R.uniform(*self.factor_range))
            mn = float(t.mean())
            if self.preserve_range:
                lo, hi = float(t.min()), float(t.max())
            t = (t - mn) * factor + mn
            if self.preserve_range:
                t = torch.clamp(t, min=lo, max=hi)
            d[key] = t
        return d


class RandDropPETd(MapTransform, RandomizableTransform):
    """
    Randomly zero out the PET channel of an already-concatenated CT+PT image.

    HECKTOR test set may include CT-only edge cases; teaching the model to
    handle a zeroed PT channel preserves robustness.
    """

    def __init__(self, image_key: str = "image", pet_channel=1, prob: float = 0.1):
        MapTransform.__init__(self, keys=[image_key])
        RandomizableTransform.__init__(self, prob=prob)
        # pet_channel may be an int or a list of ints. For the 3-channel S5 input
        # (CT, PET-percentile, PET-SUV) BOTH PET channels must be zeroed together —
        # zeroing only one leaves the other as a signal the model could cheat on.
        self.pet_channels = [pet_channel] if isinstance(pet_channel, int) else list(pet_channel)

    def randomize(self, data=None):
        super().randomize(None)

    def __call__(self, data):
        self.randomize()
        if not self._do_transform:
            return data
        d = dict(data)
        for key in self.keys:
            img = d[key]
            for ch in self.pet_channels:
                img[ch] = torch.zeros_like(img[ch])
            d[key] = img
        return d


# ── PET intensity transform selector ──────────────────────────────────────────

class PETROICropd(MapTransform):
    """PET-intensity ROI crop — port of HECKTOR2025-MEDAI (2025 seg winner)
    `get_roi_center` + `crop_neck_region_sitk` (task #28, V3).

    Crops ct/pt/label to a fixed box centred on the largest high-uptake region
    in the SUPERIOR part of the PET.  Runs AFTER Orientationd(axcodes="RAS") +
    Spacingd, on co-registered MetaTensors of shape (C, X, Y, Z) where +Z =
    superior (head).  HECKTOR CT/PET/GT share an identical grid (verified
    2026-05-21), so one box index set crops every key.

    Run this on the RAW (pre-normalisation) PET so the z-score thresholding
    sees true SUV contrast (the winner finds the ROI before normalising).  A
    following ResizeWithPadOrCropd pads to exactly crop_size when the box hits
    an image boundary.

    Algorithm (verbatim from the winner):
      - take the top `z_top_fraction` slices by z (default keeps upper 25%)
      - z-score normalise that block, threshold > `z_score_threshold`
      - largest 26-connected component → its centroid is the crop centre
      - fall back to the geometric centre of the top block if nothing exceeds
    """

    def __init__(self, keys, pet_key: str = "pt", crop_size=(192, 192, 320),
                 z_top_fraction: float = 0.75, z_score_threshold: float = 1.0):
        super().__init__(keys)
        self.pet_key = pet_key
        self.crop_size = np.asarray(crop_size, dtype=int)
        self.z_top_fraction = float(z_top_fraction)
        self.z_score_threshold = float(z_score_threshold)

    def _roi_center(self, pet_xyz: np.ndarray) -> np.ndarray:
        from scipy.ndimage import label as cc_label, generate_binary_structure
        shape = np.asarray(pet_xyz.shape)
        z_start = int(self.z_top_fraction * shape[2])
        top = pet_xyz[..., z_start:]
        mask = ((top - top.mean()) / (top.std() + 1e-8)) > self.z_score_threshold
        if not mask.any():
            center_in_top = (np.asarray(top.shape) / 2).astype(int)
        else:
            lab, n = cc_label(mask, structure=generate_binary_structure(3, 3))
            if n > 0:
                sizes = np.bincount(lab.ravel())[1:]            # ignore background
                comp_idx = np.argwhere(lab == (int(np.argmax(sizes)) + 1))
            else:
                comp_idx = np.argwhere(mask)
            center_in_top = np.mean(comp_idx, axis=0)
        return (center_in_top + np.array([0, 0, z_start])).astype(int)

    def __call__(self, data):
        d = dict(data)
        pet = d[self.pet_key]                                   # (C, X, Y, Z)
        arr = pet[0]
        pet_xyz = np.asarray(arr.detach().cpu()) if hasattr(arr, "detach") else np.asarray(arr)
        center = self._roi_center(pet_xyz)
        img_shape = np.asarray(pet_xyz.shape)
        start = np.clip(center - self.crop_size // 2, 0, img_shape)
        end = np.clip(start + self.crop_size, 0, img_shape)
        start = np.maximum(end - self.crop_size, 0)             # guard undersized images
        sl = (slice(None),
              slice(int(start[0]), int(end[0])),
              slice(int(start[1]), int(end[1])),
              slice(int(start[2]), int(end[2])))
        # MONAI MetaTensor's __getitem__ does NOT update .affine on plain slice
        # indexing — origin stays at the PRE-CROP image origin. That's invisible
        # for training-val (GT is cropped identically, dice computed in the
        # cropped frame), but breaks docker submission: `_resample_to_reference`
        # uses the (wrong) origin to back-project the prediction into the CT's
        # original world grid, landing the prediction far from GT — especially
        # GTVp, which sits at the periphery of the PET-ROI crop. Diagnosed
        # 2026-05-26 after STU-Net bs1/bs2 docker QA showed GTVp meanDSC drop
        # 0.685 → 0.475 vs training-val while GTVn stayed close.
        import torch as _torch
        for key in self.key_iterator(d):
            tensor = d[key]
            cropped = tensor[sl]
            aff = getattr(cropped, "affine", None)
            if aff is not None:
                start_vec = [float(start[0]), float(start[1]), float(start[2])]
                if isinstance(aff, _torch.Tensor):
                    aff_new = aff.clone()
                    v = _torch.as_tensor(start_vec, dtype=aff.dtype, device=aff.device)
                    aff_new[:3, 3] = aff_new[:3, 3] + aff_new[:3, :3] @ v
                else:
                    aff_new = np.array(aff, copy=True)
                    v = np.asarray(start_vec, dtype=aff_new.dtype)
                    aff_new[:3, 3] = aff_new[:3, 3] + aff_new[:3, :3] @ v
                cropped.affine = aff_new
            d[key] = cropped
        return d


class NativePetRoiPreCropd(MapTransform):
    """INFERENCE-ONLY (2026-06-16): crop ct/pt to a margin'd box centred on the PET-ROI
    centroid in NATIVE space, BEFORE Spacingd — so only the ROI is resampled to 1 mm
    instead of the whole (possibly whole-body) volume. Kills the ~10 GB host-RAM spike
    that OOM-killed the Grand Challenge container.

    Computes the SAME centroid the full-volume PETROICropd would (identical z_top_fraction
    semantics on the full native volume → same physical point), then crops a box of
    crop_size(mm) × margin centred there. The downstream centred ResizeWithPadOrCropd(
    crop_size) at 1 mm then lands the identical final crop, so the model input is
    unchanged (verify Dice≈1 vs the full-volume path). Use INSTEAD of PETROICropd.
    Updates the MetaTensor affine on crop (see PETROICropd's affine-bug note)."""

    def __init__(self, keys, pet_key="pt", crop_size=(192, 192, 320),
                 target_spacing=(1.0, 1.0, 1.0), z_top_fraction=0.75,
                 z_score_threshold=1.0, margin=1.4):
        super().__init__(keys)
        self.pet_key = pet_key
        self.crop_mm = np.asarray(crop_size, float) * np.asarray(target_spacing, float)
        self.ztf = float(z_top_fraction); self.zst = float(z_score_threshold)
        self.margin = float(margin)

    def _roi_center(self, pet_xyz):                               # identical to PETROICropd
        from scipy.ndimage import label as cc_label, generate_binary_structure
        shape = np.asarray(pet_xyz.shape)
        z_start = int(self.ztf * shape[2])
        top = pet_xyz[..., z_start:]
        mask = ((top - top.mean()) / (top.std() + 1e-8)) > self.zst
        if not mask.any():
            c = (np.asarray(top.shape) / 2).astype(int)
        else:
            lab, n = cc_label(mask, structure=generate_binary_structure(3, 3))
            if n > 0:
                sizes = np.bincount(lab.ravel())[1:]
                comp_idx = np.argwhere(lab == (int(np.argmax(sizes)) + 1))
            else:
                comp_idx = np.argwhere(mask)
            c = np.mean(comp_idx, axis=0)
        return (c + np.array([0, 0, z_start])).astype(int)

    def __call__(self, data):
        import torch as _torch
        d = dict(data)
        pet = d[self.pet_key]
        arr = pet[0]
        pet_xyz = np.asarray(arr.detach().cpu()) if hasattr(arr, "detach") else np.asarray(arr)
        center = self._roi_center(pet_xyz)
        aff = getattr(pet, "affine", None)
        if aff is not None:
            A = np.asarray(aff.detach().cpu() if hasattr(aff, "detach") else aff, dtype=float)
            sp = np.linalg.norm(A[:3, :3], axis=0)                # native mm/voxel (x,y,z)
        else:
            sp = np.array([1.0, 1.0, 1.0])
        box = np.ceil(self.crop_mm * self.margin / np.maximum(sp, 1e-6)).astype(int)
        shape = np.asarray(pet_xyz.shape)
        start = np.clip(center - box // 2, 0, shape)
        end = np.clip(start + box, 0, shape)
        start = np.maximum(end - box, 0)
        # Keep the FULL Z extent: the downstream PETROICropd uses z_top_fraction (top
        # 25% by Z) to find the centroid — cropping Z here would change that reference
        # and shift the crop. Only XY is pre-cropped (the main resample-RAM hog); the
        # unchanged PETROICropd then reproduces the identical centroid + final crop.
        start[2] = 0; end[2] = int(shape[2])
        sl = (slice(None), slice(int(start[0]), int(end[0])),
              slice(int(start[1]), int(end[1])), slice(int(start[2]), int(end[2])))
        for key in self.key_iterator(d):
            t = d[key]; cropped = t[sl]
            a = getattr(cropped, "affine", None)
            if a is not None:
                v = [float(start[0]), float(start[1]), float(start[2])]
                if isinstance(a, _torch.Tensor):
                    an = a.clone(); vv = _torch.as_tensor(v, dtype=a.dtype, device=a.device)
                    an[:3, 3] = an[:3, 3] + an[:3, :3] @ vv
                else:
                    an = np.array(a, copy=True); an[:3, 3] = an[:3, 3] + an[:3, :3] @ np.asarray(v, dtype=an.dtype)
                cropped.affine = an
            d[key] = cropped
        return d


class PhaseLockedRoiResampled(MapTransform):
    """INFERENCE-ONLY (2026-06-16): resample ct/pt to target spacing DIRECTLY onto a
    sub-block of the full-volume Spacingd output grid, covering a margin'd box around the
    PET-ROI centroid (full Z preserved). Because MONAI Spacingd preserves the affine
    origin, the full-volume 1 mm grid samples at `origin + j·pixdim`; this transform
    resamples the NATIVE volume onto an integer-offset sub-grid `origin + (j0+i)·pixdim`,
    so its samples are an EXACT subset of the full-volume grid → output is BYTE-IDENTICAL
    to the full-volume path over the ROI (verified max|diff|=0, corr=1.000000), with NO
    sub-voxel grid-phase shift (the naive native-crop-then-Spacingd path shifted the grid
    by the crop-origin fraction → cost ~0.022 GTVp meanDSC / ~0.015 borda; this does not).

    Never materialises the whole 1 mm volume → caps the host-RAM spike that OOM-killed
    the Grand Challenge container. REPLACES Spacingd (do NOT also run Spacingd); the
    downstream PETROICropd re-finds the identical centroid on this phase-locked sub-block
    and lands the identical final crop. See NativePetRoiPreCropd for the (rejected) naive
    variant kept only for A/B comparison."""

    def __init__(self, keys, pet_key="pt", crop_size=(192, 192, 320),
                 target_spacing=(1.0, 1.0, 1.0), z_top_fraction=0.75,
                 z_score_threshold=1.0, margin=1.4):
        super().__init__(keys)
        self.pet_key = pet_key
        self.target_spacing = np.asarray(target_spacing, float)
        self.crop_vox = np.asarray(crop_size, int)                     # FINAL crop size (voxels)
        self.ztf = float(z_top_fraction)
        self.zst = float(z_score_threshold)
        self.margin = float(margin)                                    # unused (kept for API parity)

    def _center_in_top(self, top_xyz):
        """PETROICropd._roi_center's centroid math, on an already-extracted top-Z block."""
        from scipy.ndimage import label as cc_label, generate_binary_structure
        mask = ((top_xyz - top_xyz.mean()) / (top_xyz.std() + 1e-8)) > self.zst
        if not mask.any():
            return (np.asarray(top_xyz.shape) / 2).astype(int)
        lab, n = cc_label(mask, structure=generate_binary_structure(3, 3))
        if n > 0:
            sizes = np.bincount(lab.ravel())[1:]
            comp_idx = np.argwhere(lab == (int(np.argmax(sizes)) + 1))
        else:
            comp_idx = np.argwhere(mask)
        return np.mean(comp_idx, axis=0)

    def __call__(self, data):
        import torch as _torch
        from monai.transforms import SpatialResample
        d = dict(data)
        pet = d[self.pet_key]
        A = np.asarray(pet.affine.detach().cpu() if hasattr(pet.affine, "detach")
                       else pet.affine, dtype=float)
        sp_nat = np.linalg.norm(A[:3, :3], axis=0)                     # native mm/voxel (x,y,z)
        # full-volume Spacingd output grid: SAME origin, direction scaled to target spacing.
        A_out = A.copy()
        for a in range(3):
            A_out[:3, a] = A[:3, a] / max(sp_nat[a], 1e-8) * self.target_spacing[a]
        origin = A_out[:3, 3]
        Aout3 = A_out[:3, :3]
        nat_shape = np.asarray(pet.shape[1:])                          # (X,Y,Z)
        # full-volume Spacingd output size — use MONAI's OWN shape computation so the
        # boundary-clip in the crop arithmetic matches the full-volume path EXACTLY (a
        # hand formula mis-rounds the .5 cases by 1 → 1-voxel crop shift on whole-body
        # scans whose tumor sits near the Z boundary; cost ~20% voxel mismatch on CHUP-001).
        from monai.data.utils import compute_shape_offset
        full_shape = np.asarray(compute_shape_offset(nat_shape, A, A_out)[0], dtype=int)
        sr = SpatialResample(mode="bilinear")
        # ── EXACT centroid: resample ONLY the top-Z block of PET to the full-XY 1mm grid
        # (the sole region PETROICropd's z-score uses) and run the identical centroid math.
        # This reproduces the full-volume PETROICropd centroid BYTE-FOR-BYTE (eliminates the
        # ≤2-voxel native-vs-resampled rounding gap) while never resampling the whole volume.
        z_start = int(self.ztf * full_shape[2])
        top_size = [int(full_shape[0]), int(full_shape[1]), int(full_shape[2] - z_start)]
        dst_top = A_out.copy()
        dst_top[:3, 3] = origin + Aout3 @ np.array([0.0, 0.0, float(z_start)])
        top = sr(pet, dst_affine=_torch.as_tensor(dst_top, dtype=_torch.float64),
                 spatial_size=top_size)
        top_xyz = np.asarray(top[0].detach().cpu())
        c_full = (self._center_in_top(top_xyz) + np.array([0, 0, z_start])).astype(int)
        del top, top_xyz
        # reproduce PETROICropd's EXACT box arithmetic on the full 1mm grid, so the
        # resampled crop region is the same one the full-volume path would emit.
        crop = self.crop_vox
        start = np.clip(c_full - crop // 2, 0, full_shape)
        end = np.clip(start + crop, 0, full_shape)
        start = np.maximum(end - crop, 0)
        size = [int(v) for v in (end - start)]
        dst = A_out.copy()
        dst[:3, 3] = origin + Aout3 @ start.astype(float)              # crop-region world origin
        dst_t = _torch.as_tensor(dst, dtype=_torch.float64)
        for key in self.key_iterator(d):
            d[key] = sr(d[key], dst_affine=dst_t, spatial_size=size)
        return d


def _pet_intensity_transform(dcfg: dict):
    """Absolute SUV clip if `pt_clip_range` set; otherwise percentile clip."""
    if "pt_clip_range" in dcfg:
        lo, hi = dcfg["pt_clip_range"]
        return ScaleIntensityRanged(
            keys=["pt"], a_min=lo, a_max=hi,
            b_min=0.0, b_max=1.0, clip=True,
        )
    pct = dcfg.get("pt_clip_percentile", [0.5, 99.5])
    return ScaleIntensityRangePercentilesd(
        keys=["pt"], lower=pct[0], upper=pct[1],
        b_min=0.0, b_max=1.0, clip=True,
    )


def _ct_intensity_transform(dcfg: dict) -> list:
    """CT normalization as a FLAT list of transforms (never a nested Compose).

    'window' (default, fixed HU window→[0,1]) or 'nnunet' (S9: foreground
    0.5/99.5-percentile clip + per-image z-score, the nnU-Net scheme both 2025
    seg winners use).

    ⚠️ MUST return a flat list spliced into the preprocess (`*_ct_intensity_transform`),
    NOT a Compose: MONAI PersistentDataset caches the deterministic transform PREFIX
    and treats a nested Compose as an opaque break point — caching the RAW cropped
    volume (a huge non-contiguous view) before normalization/Contiguousd, which
    ballooned the on-disk cache ~18× on whole-body CTs. Flat transforms cache cleanly.
    """
    mode = str(dcfg.get("ct_norm", "window")).lower()
    if mode == "nnunet":
        # clip to per-image 0.5/99.5 percentile then z-score over nonzero voxels.
        return [
            ScaleIntensityRangePercentilesd(
                keys=["ct"], lower=0.5, upper=99.5,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            NormalizeIntensityd(keys=["ct"], nonzero=True, channel_wise=True),
        ]
    ct_clip = dcfg["ct_clip_range"]
    return [ScaleIntensityRanged(
        keys=["ct"], a_min=ct_clip[0], a_max=ct_clip[1],
        b_min=0.0, b_max=1.0, clip=True,
    )]


def _pt_suv_transform(dcfg: dict):
    """S5 3rd channel: ABSOLUTE-SUV-normalized PET (on the `pt_suv` key). Default
    clip [0, suv_max=25]→[0,1] — preserves absolute avidity (vs the percentile
    `pt` channel which is scale-invariant). Override via `pt_suv_clip`."""
    lo, hi = dcfg.get("pt_suv_clip", [0.0, 25.0])
    return ScaleIntensityRanged(
        keys=["pt_suv"], a_min=lo, a_max=hi,
        b_min=0.0, b_max=1.0, clip=True,
    )


def _use_suv_channel(dcfg: dict) -> bool:
    """S5: enabled when the config requests a 3rd absolute-SUV PET channel."""
    return bool(dcfg.get("pt_suv_channel", False))


def _image_keys(dcfg: dict) -> list[str]:
    """Channels concatenated into `image`: [ct, pt] (+ pt_suv when S5 on)."""
    return ["ct", "pt", "pt_suv"] if _use_suv_channel(dcfg) else ["ct", "pt"]


def _ct_foreground(x):
    return x > 0


# ── Phase 1: deterministic preprocessing (cacheable) ──────────────────────────

def get_preprocess_transforms(dcfg: dict) -> Compose:
    """Cacheable deterministic preprocessing — shared by train + val + prepare.

    Output sample dict has separate `ct`, `pt`, `label` MetaTensors at the
    cache_volume size. ConcatItemsd happens later (in post-cache transforms)
    so the cache stays small and re-usable.
    """
    return _get_preprocess_transforms(dcfg)


# Back-compat alias (other code paths still import this name).
def get_train_preprocess_transforms(dcfg: dict) -> Compose:
    """Load + orient + resample + clip + crop + pad. Output kept as separate ct/pt/label."""
    ct_clip = dcfg["ct_clip_range"]
    cache_size = tuple(dcfg.get("cache_volume", _CACHE_VOLUME))
    spacing = tuple(dcfg.get("target_spacing", _TARGET_SPACING))

    return _get_preprocess_transforms(dcfg)


def _get_preprocess_transforms(dcfg: dict) -> Compose:
    ct_clip = dcfg["ct_clip_range"]
    cache_size = tuple(dcfg.get("cache_volume", _CACHE_VOLUME))
    spacing = tuple(dcfg.get("target_spacing", _TARGET_SPACING))
    # roi_crop: "body" (default, V1/V2 = CropForeground + KeepTopZ) | "pet" (V3 =
    # PET-intensity ROI centroid crop, the 2025 winner's method). See task #28.
    roi_crop = str(dcfg.get("roi_crop", "body")).lower()

    head = [
        LoadImaged(keys=["ct", "pt", "label"]),
        EnsureChannelFirstd(keys=["ct", "pt", "label"]),
        Orientationd(keys=["ct", "pt", "label"], axcodes="RAS"),
        Spacingd(
            keys=["ct", "pt", "label"],
            pixdim=spacing,
            mode=["bilinear", "bilinear", "nearest"],
        ),
    ]

    # S5: a 3rd absolute-SUV PET channel `pt_suv` (a copy of the cropped raw PET,
    # normalised by absolute SUV instead of percentile). All spatial keys include
    # it so it stays aligned; `_pt_suv_transform` normalises it after the copy.
    suv = _use_suv_channel(dcfg)
    spat = ["ct", "pt", "pt_suv", "label"] if suv else ["ct", "pt", "label"]
    suv_steps = [CopyItemsd(keys=["pt"], names=["pt_suv"]), _pt_suv_transform(dcfg)] if suv else []

    if roi_crop == "pet":
        # V3: crop on RAW PET contrast FIRST (z-score ROI finding sees true SUV),
        # then normalise the cropped volume, then pad to exactly cache_size.
        body = [
            PETROICropd(
                keys=["ct", "pt", "label"], pet_key="pt", crop_size=cache_size,
                z_top_fraction=dcfg.get("roi_z_top_fraction", 0.75),
                z_score_threshold=dcfg.get("roi_z_score_threshold", 1.0),
            ),
            # copy RAW cropped PET → pt_suv BEFORE the percentile norm touches `pt`.
            *suv_steps,
            *_ct_intensity_transform(dcfg),
            _pet_intensity_transform(dcfg),
            ResizeWithPadOrCropd(keys=spat, spatial_size=cache_size),
        ]
    else:
        # V1/V2: body-foreground crop + top-Z bias + centered pad/crop.
        body = [
            *_ct_intensity_transform(dcfg),
            *suv_steps,
            _pet_intensity_transform(dcfg),
            CropForegroundd(
                keys=spat,
                source_key="ct",
                select_fn=_ct_foreground,
                margin=10,
                allow_smaller=True,
            ),
            # Z-axis top-biased BEFORE the centered ResizeWithPadOrCropd so whole-body
            # CTs (CHUP/CHUV) keep the head (top of bbox in RAS +z) instead of being
            # center-cropped into chest/abdomen. See evaluation/results/QA_FOLD0_DIAGNOSIS.md.
            KeepTopZd(keys=spat, target_z=cache_size[2]),
            ResizeWithPadOrCropd(keys=spat, spatial_size=cache_size),
        ]

    # CRITICAL: see docstring on Contiguousd above.
    return Compose(head + body + [Contiguousd(keys=spat)])


# ── Phase 2: random augmentation (per epoch) ──────────────────────────────────

def _augmentation_pipeline(roi_size: tuple, pos_neg: list, pet_drop_prob: float,
                            sample_strategy: str = "label_classes",
                            aug_profile: str = "default",
                            use_suv: bool = False) -> list:
    """Returns the list of random transforms applied after preprocessing.

    aug_profile (added 2026-05-29):
      - "default" : current pipeline (rotation ±15°, scale [0.9,1.1],
        brightness-mult [0.9,1.1], no gamma-invert, no nnU-Net-contrast)
      - "nnunet"  : match 2025 winner's nnU-Net default DA — widen rotation to
        ±30°, scale to [0.75,1.25], brightness-mult to [0.75,1.25]; ADD
        gamma-invert (p=0.1) + nnU-Net ContrastAugmentation (p=0.15). Keep our
        extras (elastic p=0.3, PET-drop p=0.1, lowres-sim p=0.25).
    """
    nnunet = (str(aug_profile).lower() == "nnunet")
    # nnunet widens brightness-mult / rotation / scale; default uses the
    # narrower historical values.
    BRIGHT_MULT = 0.25 if nnunet else 0.1                          # ±0.25 = [0.75,1.25]
    ROTATE_RAD  = 0.52 if nnunet else 0.26                         # 0.52 rad ≈ ±30°
    SCALE_DEV   = 0.25 if nnunet else 0.1                          # ±0.25 → [0.75,1.25]

    # S5: pt_suv (absolute-SUV PET) flows through the same spatial transforms.
    spat_keys = ["ct", "pt", "pt_suv", "label"] if use_suv else ["ct", "pt", "label"]
    img_keys = ["ct", "pt", "pt_suv"] if use_suv else ["ct", "pt"]
    mods = ("ct", "pt", "pt_suv") if use_suv else ("ct", "pt")
    if sample_strategy == "label_classes":
        # Match official baseline ratios (bg, GTVp, GTVn) → biased toward foreground
        sampler = RandCropByLabelClassesd(
            keys=spat_keys,
            label_key="label",
            spatial_size=roi_size,
            ratios=[0.1, 0.45, 0.45],
            num_classes=3,
            num_samples=1,
            allow_smaller=True,
            warn=False,
        )
    else:
        sampler = RandCropByPosNegLabeld(
            keys=spat_keys,
            label_key="label",
            spatial_size=roi_size,
            pos=pos_neg[0], neg=pos_neg[1],
            num_samples=1,
            image_key="ct",
            image_threshold=0,
        )

    # ── Per-modality intensity / acquisition-quality augmentation ──────────
    # Applied to CT and PET as SEPARATE transform instances (one per modality)
    # so each is randomised INDEPENDENTLY. CT (HU) and PET (SUV) are physically
    # independent acquisitions — uncorrelated noise, native resolution, and
    # intensity calibration. Running these before ConcatItemsd, while `ct` and
    # `pt` are still separate keys, avoids the previous behavior where one
    # shared random parameter hit both channels identically.
    intensity_augs = []
    for m in mods:
        intensity_augs += [
            RandGaussianNoised(keys=[m], prob=0.20, mean=0.0, std=0.05),
            RandGaussianSmoothd(
                keys=[m], prob=0.20,
                sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5),
            ),
            # Low-resolution simulation (nnU-Net default DA — the only gap we had
            # vs SOTA augmentation). Random downsample→upsample teaches
            # robustness to scanner-resolution variation; HECKTOR mixes 1-3 mm CT
            # / ~1-5 mm PET and the 2026 test set has unseen centers (CHU Brest /
            # UAE / public). Per-modality because CT and PET have independent
            # native resolution.
            RandSimulateLowResolutiond(keys=[m], prob=0.25, zoom_range=(0.5, 1.0)),
            # brightness multiplicative — factors=0.1 in default, 0.25 in nnunet
            RandScaleIntensityd(keys=[m], prob=0.30, factors=BRIGHT_MULT),
            RandShiftIntensityd(keys=[m], prob=0.30, offsets=0.1),
            RandAdjustContrastd(keys=[m], prob=0.30, gamma=(0.7, 1.5)),
        ]
        if nnunet:
            # nnU-Net default DA adds: gamma applied to the INVERTED image
            # (p=0.1) and contrast scaling around the mean (p=0.15). Both
            # per-modality so CT/PET get independent random parameters.
            intensity_augs += [
                RandInvertGammad(keys=[m], prob=0.1, gamma=(0.7, 1.5)),
                RandContrastAroundMeand(keys=[m], prob=0.15,
                                         factor_range=(0.75, 1.25),
                                         preserve_range=True),
            ]

    return [
        sampler,
        # intensity/quality aug runs per-modality BEFORE concat (see above)
        *intensity_augs,
        ConcatItemsd(keys=img_keys, name="image", dim=0),
        # drop ALL PET channels together: [1] for CT+PET, [1,2] for CT+PET+PET-SUV (S5)
        RandDropPETd(image_key="image", pet_channel=([1, 2] if use_suv else 1),
                     prob=pet_drop_prob),
        # Spatial aug stays joint on image+label (must keep CT/PET/label aligned).
        # Probabilities bumped 2026-05-15: affine 0.3→0.5, elastic 0.2→0.3 — the
        # old values were conservative vs nnU-Net default; geometric aug is
        # high-value and cheap.
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=[ROTATE_RAD, ROTATE_RAD, ROTATE_RAD],  # default 0.26 (~±15°), nnunet 0.52 (~±30°)
            scale_range=[SCALE_DEV, SCALE_DEV, SCALE_DEV],       # default 0.1 ([0.9,1.1]), nnunet 0.25 ([0.75,1.25])
            translate_range=[10, 10, 10],
            mode=["bilinear", "nearest"],
            padding_mode="zeros",
        ),
        Rand3DElasticd(
            keys=["image", "label"],
            prob=0.3,
            sigma_range=(5, 7),
            magnitude_range=(50, 150),
            mode=["bilinear", "nearest"],
            padding_mode="zeros",
        ),
        SelectItemsd(keys=["image", "label", *_PASSTHROUGH_KEYS]),
        EnsureTyped(keys=["image", "label"]),
    ]


def get_train_augmentation_transforms(dcfg: dict) -> Compose:
    """Random augmentation — applied AFTER the cached preprocess output.

    Includes ConcatItemsd (CT, PT → image), patch crop, flips, intensity jitter.
    """
    roi_size = tuple(dcfg["roi_size"])
    pos_neg = dcfg.get("pos_neg_ratio", [3, 1])
    drop_prob = dcfg.get("pet_drop_prob", 0.1)
    strategy = dcfg.get("sample_strategy", "label_classes")
    aug_profile = dcfg.get("aug_profile", "default")
    return Compose(_augmentation_pipeline(roi_size, pos_neg, drop_prob, strategy,
                                          aug_profile, use_suv=_use_suv_channel(dcfg)))


def get_val_postcache_transforms(dcfg: dict) -> Compose:
    """Deterministic post-cache transform for val/test — runs on every iter.

    Concatenates CT + PT into `image`, pads to a multiple of 32 (SwinUNETR
    requirement), and tightens types. The padded size is the smallest
    multiple of 32 ≥ cache_volume; for cache_volume=200 this is 224.
    """
    return Compose([
        ConcatItemsd(keys=_image_keys(dcfg), name="image", dim=0),
        DivisiblePadd(keys=["image", "label"], k=32, mode="constant"),
        SelectItemsd(keys=["image", "label", *_PASSTHROUGH_KEYS]),
        EnsureTyped(keys=["image", "label"]),
    ])


# ── Full pipeline (no caching — used when cache_dir is unset) ─────────────────

def get_train_transforms(dcfg: dict) -> Compose:
    pre = get_train_preprocess_transforms(dcfg)
    aug = get_train_augmentation_transforms(dcfg)
    return Compose([*pre.transforms, *aug.transforms])


# ── Validation pipeline ───────────────────────────────────────────────────────

def get_val_transforms(dcfg: dict) -> Compose:
    """Full val pipeline = preprocess + post-cache. Uses the same cacheable
    preprocess as train, then concat + DivisiblePadd to a SwinUNETR-friendly size."""
    pre = _get_preprocess_transforms(dcfg)
    post = get_val_postcache_transforms(dcfg)
    return Compose([*pre.transforms, *post.transforms])


# ── Test pipeline (no labels) ─────────────────────────────────────────────────

def get_test_transforms(dcfg: dict) -> Compose:
    """Test-time transforms (no labels). Must MIRROR `_get_preprocess_transforms`
    branching on `roi_crop` so docker inference sees the same input distribution
    the model trained on.

    Bug history (2026-05-26): previous version ignored `roi_crop` entirely and
    always did V1/V2 body-foreground + KeepTopZ crop, regardless of whether the
    model trained with PETROICropd. STU-Net bs1 docker QA hit GTVp meanDSC 0.45
    vs train-val 0.69 because the model was given a body-centered crop at
    inference time when it had only ever seen PET-centroid-centered crops during
    training. Also: PETROICropd MUST run on RAW PET (before any intensity
    normalisation) — the z-score ROI finding needs the true SUV contrast to find
    the tumor centroid.
    """
    ct_clip = dcfg["ct_clip_range"]
    spacing = tuple(dcfg.get("target_spacing", _TARGET_SPACING))
    cache_size = tuple(dcfg.get("cache_volume", _CACHE_VOLUME))
    roi_crop = str(dcfg.get("roi_crop", "body")).lower()

    # Optional pre-resample coarse Z-crop (kills the whole-body 1mm-resample
    # memory spike). Set data.pre_resample_keep_mm (e.g. 450) to enable; unset =
    # no-op (current behavior). Must run BEFORE Spacingd.
    keep_mm = dcfg.get("pre_resample_keep_mm", None)
    # Optional pre-resample PET-ROI box crop (2026-06-16): crops to a margin'd box
    # around the PET-ROI centroid in NATIVE space BEFORE Spacingd, so only the ROI is
    # resampled to 1mm (not the whole volume) → caps the ~10GB host-RAM spike that OOM-
    # killed the GC container. INFERENCE-ONLY (training cache path unchanged). When on,
    # it REPLACES the post-Spacingd PETROICropd (centering is done by this + the centred
    # ResizeWithPadOrCropd). Output verified ≈ identical to the full-volume path.
    # pre_resample_petroi:
    #   "locked" → PhaseLockedRoiResampled (BYTE-EXACT to full-vol path; SHIP THIS)
    #   truthy   → NativePetRoiPreCropd + Spacingd (naive; sub-voxel shift, −0.015 borda)
    #   false    → full-volume Spacingd (original; 10GB RAM spike, OOMs GC)
    pre_petroi_val = dcfg.get("pre_resample_petroi", False)
    pre_petroi_mode = (str(pre_petroi_val).lower()
                       if pre_petroi_val not in (False, None) else "")
    pre_locked = pre_petroi_mode in ("locked", "phaselock", "exact")
    pre_naive = bool(pre_petroi_val) and not pre_locked

    head = [
        # Submission inputs are .mha (Grand Challenge convention). MONAI's auto-
        # reader detection picks PydicomReader for .mha and chokes on the missing
        # DICM header — force ITKReader, which is .mha's native format and also
        # reads .nii.gz cleanly.
        LoadImaged(keys=["ct", "pt"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["ct", "pt"]),
        Orientationd(keys=["ct", "pt"], axcodes="RAS"),
    ]
    if keep_mm is not None:
        head.append(KeepSuperiorMMd(keys=["ct", "pt"], keep_mm=float(keep_mm)))
    if pre_locked:
        # phase-locked resample IS the 1mm resample (no separate Spacingd).
        head.append(PhaseLockedRoiResampled(
            keys=["ct", "pt"], pet_key="pt", crop_size=cache_size, target_spacing=spacing,
            z_top_fraction=dcfg.get("roi_z_top_fraction", 0.75),
            z_score_threshold=dcfg.get("roi_z_score_threshold", 1.0),
            margin=float(dcfg.get("pre_resample_margin", 1.4))))
    else:
        if pre_naive:
            head.append(NativePetRoiPreCropd(
                keys=["ct", "pt"], pet_key="pt", crop_size=cache_size, target_spacing=spacing,
                z_top_fraction=dcfg.get("roi_z_top_fraction", 0.75),
                z_score_threshold=dcfg.get("roi_z_score_threshold", 1.0),
                margin=float(dcfg.get("pre_resample_margin", 1.4))))
        head.append(Spacingd(keys=["ct", "pt"], pixdim=spacing, mode=["bilinear", "bilinear"]))

    if roi_crop == "pet":
        # V3 / STU-Net path — PET-ROI centroid crop, exact mirror of
        # _get_preprocess_transforms (training cache) ordering: crop FIRST on
        # raw PET, then normalise. PETROICropd is ALWAYS kept; the optional XY-only
        # NativePetRoiPreCropd above just shrinks the resample input (full Z preserved
        # so PETROICropd's z_top_fraction + centroid are unchanged → identical crop).
        body = [
            # locked path: PhaseLockedRoiResampled already produced the exact final crop
            # region (centroid from full-FOV native PET) — PETROICropd would re-find a
            # DIFFERENT centroid on the cropped FOV (FOV-dependent z-score stats) and shift
            # the crop, so it is SKIPPED here. Non-locked paths keep it.
            *([] if pre_locked else [PETROICropd(
                keys=["ct", "pt"], pet_key="pt", crop_size=cache_size,
                z_top_fraction=dcfg.get("roi_z_top_fraction", 0.75),
                z_score_threshold=dcfg.get("roi_z_score_threshold", 1.0),
            )]),
            ScaleIntensityRanged(
                keys=["ct"], a_min=ct_clip[0], a_max=ct_clip[1],
                b_min=0.0, b_max=1.0, clip=True,
            ),
            _pet_intensity_transform(dcfg),
            ResizeWithPadOrCropd(keys=["ct", "pt"], spatial_size=cache_size),
        ]
    else:
        # V1/V2 body-foreground crop (default for older configs).
        body = [
            ScaleIntensityRanged(
                keys=["ct"], a_min=ct_clip[0], a_max=ct_clip[1],
                b_min=0.0, b_max=1.0, clip=True,
            ),
            _pet_intensity_transform(dcfg),
            CropForegroundd(
                keys=["ct", "pt"], source_key="ct",
                select_fn=_ct_foreground, margin=10, allow_smaller=True,
            ),
            # Top-bias the z axis (RAS +z = superior = head) so whole-body CTs
            # don't lose their head/neck region to the centered crop below.
            KeepTopZd(keys=["ct", "pt"], target_z=cache_size[2]),
            ResizeWithPadOrCropd(keys=["ct", "pt"], spatial_size=cache_size),
        ]

    tail = [
        # NOTE: test/sliding-window path does NOT yet create pt_suv (S5) — wire it
        # here before deploying a 3-channel model to docker. Train/val screen uses
        # get_val_transforms (which DOES support S5), so this is fine for now.
        ConcatItemsd(keys=["ct", "pt"], name="image", dim=0),
        # SwinUNETR needs spatial dims divisible by 32 — V1/V2 cache_volume 200
        # isn't (200/32=6.25). V3 cache_volume 192/320 IS /32, so this is a no-op
        # for V3 but kept for back-compat.
        DivisiblePadd(keys=["image"], k=32, mode="constant"),
        SelectItemsd(keys=["image", "patient_id", "has_pet"]),
        EnsureTyped(keys=["image"]),
    ]

    return Compose(head + body + tail)
