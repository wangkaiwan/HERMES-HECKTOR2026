"""Docker-side radiomics extractor for Task 2/3 Route-B classifiers.

Self-contained port of scripts/extract_radiomics.py::extract_patient +
scripts/extract_radiomics_predicted.py CC-filter, using ONLY numpy / SciPy /
SimpleITK (already in the docker image — NO pyradiomics, no scripts/ import).

Produces the 30-d basic first-order/shape feature vector that the refit
classifiers in docker/model/task23_radio.joblib were trained on. Features are
computed on the PREDICTED segmentation (the same mask the docker pipeline emits),
after the locked CC filter (GTVp ≥1000 mm³ / GTVn ≥500 mm³, top-2 / top-8),
resampled to 2 mm isotropic with ABSOLUTE HU/SUV intensities (NOT the per-volume
NN normalisation) — exactly as the offline training extraction did.

Public API:
    extract_radio_vector(ct_path, pet_path, seg_array_ct_space, radio_names)
        -> np.ndarray ordered to match `radio_names`
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as cc_label

TARGET_SPACING = (2.0, 2.0, 2.0)  # mm — matches training extraction

# ROI-crop margins (2026-06-16 RAM fix): radiomics features are first-order/shape
# computed INSIDE the mask, so resampling only an ROI box around the seg mask gives a
# BYTE-IDENTICAL feature vector while never resampling the whole (whole-body) volume —
# which spiked host RAM ~+5 GB (the next OOM after the seg-resample fix). The output is
# a phase-locked sub-region of the FULL 2 mm grid (sitk samples each input at its true
# geometry), so the 2 mm sample points are an exact subset of the full-volume path's.
# Margins are generous so every mask voxel keeps full BSpline support (order-3 ≈ 2 input
# voxels) → exactness verified on all QA patients.
# INPUT margin is in VOXELS, not mm: the BSpline pre-filter is a recursive filter whose
# boundary influence decays per INPUT voxel (~0.17^d for cubic). 16 voxels → ~1e-13, so
# mask-voxel intensities match the full-volume resample to machine precision regardless of
# slice spacing. (An mm margin under-covers coarse-Z scans → ~1e-4 error; that was the bug.)
_INPUT_MARGIN_VOX = 16     # native-voxel crop padding around the mask bbox (BSpline support)
_OUTPUT_MARGIN_MM = 10.0   # 2 mm output sub-grid padding around the mask bbox

# Locked CC filter (= scripts/extract_radiomics_predicted.py and the docker seg
# post-processing): mm³ minimum then volume top-N keep, per class.
CCF_MIN_MM3 = {1: 1000, 2: 500}
CCF_TOPN = {1: 2, 2: 8}


def _apply_cc_filter(pred: np.ndarray, voxvol_mm3: float, struct) -> np.ndarray:
    out = pred.copy()
    for cls, min_mm3 in CCF_MIN_MM3.items():
        keep_n = CCF_TOPN[cls]
        mask = (pred == cls)
        if not mask.any():
            continue
        labels, n_cc = cc_label(mask, structure=struct)
        if n_cc == 0:
            continue
        sizes = np.bincount(labels.flat, minlength=n_cc + 1)
        survivors: list[tuple[int, int]] = []
        for cc_id in range(1, n_cc + 1):
            if sizes[cc_id] * voxvol_mm3 < min_mm3:
                out[labels == cc_id] = 0
            else:
                survivors.append((cc_id, int(sizes[cc_id])))
        if keep_n > 0 and len(survivors) > keep_n:
            survivors.sort(key=lambda kv: -kv[1])
            for cc_id, _ in survivors[keep_n:]:
                out[labels == cc_id] = 0
    return out


def _resample_to_iso(img: sitk.Image, is_label: bool) -> sitk.Image:
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()
    new_size = [int(round(s * o / n))
                for s, o, n in zip(old_size, old_spacing, TARGET_SPACING)]
    rsl = sitk.ResampleImageFilter()
    rsl.SetOutputSpacing(TARGET_SPACING)
    rsl.SetSize(new_size)
    rsl.SetOutputDirection(img.GetDirection())
    rsl.SetOutputOrigin(img.GetOrigin())
    rsl.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return rsl.Execute(img)


def _resample_to_ref(img: sitk.Image, ref: sitk.Image, is_label: bool) -> sitk.Image:
    """Resample `img` (using its true geometry) onto the grid of `ref` — same
    interpolators as _resample_to_iso. `ref` is a phase-locked sub-region of the full
    2 mm grid, so the output sample points are an exact subset of the full-volume path."""
    rsl = sitk.ResampleImageFilter()
    rsl.SetReferenceImage(ref)
    rsl.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return rsl.Execute(img)


def _full_iso_ref(img: sitk.Image) -> sitk.Image:
    """Cheap dummy image carrying the FULL 2 mm-iso grid geometry of `img` (no heavy
    intensity data). The full-volume path's _resample_to_iso lands on exactly this grid,
    so any RegionOfInterest of it is phase-locked to that path."""
    old_spacing, old_size = img.GetSpacing(), img.GetSize()
    new_size = [int(round(s * o / n))
                for s, o, n in zip(old_size, old_spacing, TARGET_SPACING)]
    ref = sitk.Image(new_size, sitk.sitkUInt8)
    ref.SetSpacing(TARGET_SPACING)
    ref.SetDirection(img.GetDirection())
    ref.SetOrigin(img.GetOrigin())
    return ref


def _world_box_region(img: sitk.Image, world_lo, world_hi, margin_mm: float):
    """Index (start, size) of the sub-region of `img` covering the world box
    [world_lo-margin, world_hi+margin]. Uses all 8 corners so axis-flipping directions
    are handled. Clipped to the image bounds."""
    lo = np.asarray(world_lo, float) - margin_mm
    hi = np.asarray(world_hi, float) + margin_mm
    idxs = []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                idxs.append(img.TransformPhysicalPointToContinuousIndex((float(x), float(y), float(z))))
    idxs = np.asarray(idxs)
    size_full = np.asarray(img.GetSize())
    i0 = np.clip(np.floor(idxs.min(0)).astype(int), 0, size_full - 1)
    i1 = np.clip(np.ceil(idxs.max(0)).astype(int), 1, size_full)
    return [int(v) for v in i0], [int(v) for v in (i1 - i0)]


def _mask_world_bbox(seg_arr_zyx: np.ndarray, native_ref: sitk.Image):
    """World-space (x,y,z) min/max corner of the nonzero mask, via the native geometry."""
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


def _stats_in_roi(arr: np.ndarray, mask: np.ndarray) -> dict:
    v = arr[mask]
    if v.size == 0:
        return dict.fromkeys(["mean", "std", "max", "min", "median", "p90"], 0.0)
    return {
        "mean":   float(v.mean()),
        "std":    float(v.std()),
        "max":    float(v.max()),
        "min":    float(v.min()),
        "median": float(np.median(v)),
        "p90":    float(np.percentile(v, 90)),
    }


def _feature_dict(ct_path: str, pet_path: str,
                  seg_array_ct_space: np.ndarray) -> dict[str, float]:
    """Mirror extract_radiomics.extract_patient, but the label comes from the
    in-memory predicted seg array (already on the CT grid)."""
    ct_raw = sitk.ReadImage(str(ct_path))

    # Wrap the predicted seg (CT-space, z,y,x) as a SimpleITK image on the CT grid.
    seg_sitk = sitk.GetImageFromArray(seg_array_ct_space.astype(np.uint8))
    seg_sitk.CopyInformation(ct_raw)

    # Re-apply the locked CC filter at the CT's NATIVE voxel volume — guarantees
    # the radiomics mask is identical to the offline training extraction
    # regardless of where the seg pipeline applied its own filter.
    sx, sy, sz = ct_raw.GetSpacing()
    voxvol = float(sx * sy * sz)
    struct = generate_binary_structure(3, 3)
    seg_arr = _apply_cc_filter(sitk.GetArrayFromImage(seg_sitk).astype(np.uint8),
                               voxvol, struct)
    seg_sitk = sitk.GetImageFromArray(seg_arr)
    seg_sitk.CopyInformation(ct_raw)

    # ── ROI crop (2026-06-16 RAM fix): resample only an ROI box around the mask, onto a
    # phase-locked sub-region of the FULL 2 mm grid → byte-identical features, ~5 GB less
    # host RAM. Falls back to the full-volume path if the mask is empty.
    bbox = _mask_world_bbox(seg_arr, ct_raw)
    if bbox is None:
        ct_img = _resample_to_iso(ct_raw, is_label=False)
        rsl = sitk.ResampleImageFilter()
        rsl.SetReferenceImage(ct_raw); rsl.SetInterpolator(sitk.sitkLinear)
        pt_on_ct = rsl.Execute(sitk.ReadImage(str(pet_path)))
        pt_img = _resample_to_iso(pt_on_ct, is_label=False)
        lb_img = _resample_to_iso(seg_sitk, is_label=True)
    else:
        w_lo, w_hi = bbox
        full_ref = _full_iso_ref(ct_raw)
        roi_idx, roi_sz = _world_box_region(full_ref, w_lo, w_hi, _OUTPUT_MARGIN_MM)
        sub_ref = sitk.RegionOfInterest(full_ref, roi_sz, roi_idx)   # phase-locked 2 mm sub-grid
        # crop the heavy native inputs to the mask box + a VOXEL margin (BSpline support is
        # per-voxel) so pre-filtering runs over the ROI, not the whole body, with no boundary
        # bias at mask voxels. seg_arr is (z,y,x); RegionOfInterest index/size are (x,y,z).
        nz = np.argwhere(seg_arr > 0)
        lo_zyx = np.maximum(nz.min(0) - _INPUT_MARGIN_VOX, 0)
        hi_zyx = np.minimum(nz.max(0) + 1 + _INPUT_MARGIN_VOX, seg_arr.shape)
        c_idx = [int(lo_zyx[2]), int(lo_zyx[1]), int(lo_zyx[0])]
        c_sz = [int(hi_zyx[2] - lo_zyx[2]), int(hi_zyx[1] - lo_zyx[1]), int(hi_zyx[0] - lo_zyx[0])]
        ct_crop = sitk.RegionOfInterest(ct_raw, c_sz, c_idx)
        seg_crop = sitk.RegionOfInterest(seg_sitk, c_sz, c_idx)
        # PET → cropped CT grid (linear), exactly mirroring the full path's pt_on_ct.
        rsl = sitk.ResampleImageFilter()
        rsl.SetReferenceImage(ct_crop); rsl.SetInterpolator(sitk.sitkLinear)
        pt_on_crop = rsl.Execute(sitk.ReadImage(str(pet_path)))
        ct_img = _resample_to_ref(ct_crop, sub_ref, is_label=False)
        pt_img = _resample_to_ref(pt_on_crop, sub_ref, is_label=False)
        lb_img = _resample_to_ref(seg_crop, sub_ref, is_label=True)

    voxel_ml = TARGET_SPACING[0] * TARGET_SPACING[1] * TARGET_SPACING[2] / 1000.0
    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    pt_arr = sitk.GetArrayFromImage(pt_img).astype(np.float32)
    lb_arr = sitk.GetArrayFromImage(lb_img).astype(np.uint8)

    feats: dict[str, float] = {}
    for cls, name in [(1, "p"), (2, "n")]:
        mask = lb_arr == cls
        n_vox = int(mask.sum())
        feats[f"gtv{name}_present"] = float(n_vox > 0)
        feats[f"gtv{name}_vol_ml"] = float(n_vox * voxel_ml)
        for k, val in _stats_in_roi(ct_arr, mask).items():
            feats[f"gtv{name}_ct_{k}"] = val
        for k, val in _stats_in_roi(pt_arr, mask).items():
            feats[f"gtv{name}_pt_{k}"] = val
        feats[f"gtv{name}_pt_tlg"] = (feats[f"gtv{name}_pt_mean"]
                                      * feats[f"gtv{name}_vol_ml"])
    return feats


def extract_radio_vector(ct_path: str, pet_path: str,
                         seg_array_ct_space: np.ndarray,
                         radio_names: list[str]) -> np.ndarray:
    """Return the 30-d radiomics vector ordered to match `radio_names`."""
    feats = _feature_dict(ct_path, pet_path, seg_array_ct_space)
    return np.array([feats[n] for n in radio_names], dtype=np.float64)
