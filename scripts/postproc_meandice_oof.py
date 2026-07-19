"""Compare seg post-processing options against the REAL 2026 Task-1 metric
(Mean Dice over GTVp+GTVn), computed BOTH per-patient and aggregated, on the
10-fold OOF predicted masks vs ground truth.

Answers: our CC-filter vs the 2025-winner nnU-Net "keep-largest-CC" vs raw.
Purely offline on saved masks — no docker / GPU / PET.

Usage:
  python scripts/postproc_meandice_oof.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label, generate_binary_structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.metrics import DiceAggScore, MeanDicePerPatient  # noqa: E402


def preload(fold_dirs, raw_dir, struct):
    cases = []
    for d in fold_dirs:
        for mf in sorted(Path(d).glob("fold*_*.mha")):
            pid = mf.stem.split("_", 1)[1]
            gt = raw_dir / pid / f"{pid}.nii.gz"
            if not gt.exists():
                continue
            ps = sitk.ReadImage(str(mf))
            gs = sitk.ReadImage(str(gt))
            sx, sy, sz = ps.GetSpacing()
            voxvol = float(sx * sy * sz)
            pred = sitk.GetArrayFromImage(ps).astype(np.uint8)
            gt_arr = sitk.GetArrayFromImage(gs).astype(np.uint8)
            if pred.shape != gt_arr.shape:
                continue
            fg = (pred > 0) | (gt_arr > 0)
            if fg.any():
                c = np.argwhere(fg)
                lo = np.maximum(c.min(0) - 2, 0)
                hi = np.minimum(c.max(0) + 3, pred.shape)
                sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
                pred = np.ascontiguousarray(pred[sl])
                gt_arr = np.ascontiguousarray(gt_arr[sl])
            cc = {}
            for cls in (1, 2):
                m = (pred == cls)
                if m.any():
                    lab, n = cc_label(m, structure=struct)
                    sizes = np.bincount(lab.ravel(), minlength=n + 1)
                    cc[cls] = (lab.astype(np.int32), sizes)
                else:
                    cc[cls] = (None, None)
            cases.append({"pid": pid, "pred": pred, "gt": gt_arr,
                          "voxvol": voxvol, "cc": cc})
    return cases


def postproc(case, mode):
    """Return a processed copy of the pred array under `mode`."""
    out = case["pred"].copy()
    voxvol = case["voxvol"]

    def _thr(cls, thr_mm3):
        lab, sizes = case["cc"][cls]
        if lab is None:
            return
        for k in range(1, sizes.size):
            if sizes[k] * voxvol < thr_mm3:
                out[lab == k] = 0

    def _keep_largest(cls):
        lab, sizes = case["cc"][cls]
        if lab is None:
            return
        if sizes.size <= 2:
            return  # 0 or 1 CC
        keep = int(np.argmax(sizes[1:]) + 1)
        out[(lab != keep) & (lab != 0) & (case["pred"] == cls)] = 0

    if mode == "raw":
        pass
    elif mode == "ours_cc":                 # GTVp 1000 / GTVn 500 mm^3
        _thr(1, 1000.0); _thr(2, 500.0)
    elif mode == "keep_largest_both":       # winner nnU-Net, both classes
        _keep_largest(1); _keep_largest(2)
    elif mode == "keep_largest_gtvp":       # largest GTVp only, GTVn untouched
        _keep_largest(1)
    elif mode.startswith("gpgn_"):          # custom (GTVp A / GTVn B) mm^3
        _a, _b = mode.split("_")[1:3]
        _thr(1, float(_a)); _thr(2, float(_b))
    elif mode.startswith("gtvn_thr_"):      # sweep: GTVn min-vol only
        _thr(2, float(mode.split("_")[-1]))
    elif mode.startswith("both_thr_"):      # GTVp 1000 + GTVn variable
        _thr(1, 1000.0); _thr(2, float(mode.split("_")[-1]))
    else:
        raise ValueError(mode)
    return out


def evaluate(cases, mode):
    dpp = MeanDicePerPatient(class_labels=(1, 2))
    dag = DiceAggScore(class_labels=(1, 2))
    for c in cases:
        out = postproc(c, mode)
        fimg = sitk.GetImageFromArray(out)
        gimg = sitk.GetImageFromArray(c["gt"])
        dpp.update(fimg, gimg)
        dag.update(fimg, gimg)
    pp = dpp.compute(); ag = dag.compute()
    pp1, pp2 = float(pp["Class_1"]), float(pp["Class_2"])
    ag1, ag2 = float(ag["Class_1"]), float(ag["Class_2"])
    return {
        "pp_gtvp": pp1, "pp_gtvn": pp2, "pp_mean": 0.5 * (pp1 + pp2),
        "ag_gtvp": ag1, "ag_gtvn": ag2, "ag_mean": 0.5 * (ag1 + ag2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold_dirs", nargs="+", type=Path,
                   default=[ROOT / f"evaluation/results/qa_10fold_oof_fold{f}_masks"
                            for f in range(5)])
    p.add_argument("--raw_dir", type=Path, default=ROOT / "data/raw")
    a = p.parse_args()

    struct = generate_binary_structure(3, 3)
    print("=== preload ===", flush=True)
    cases = preload(a.fold_dirs, a.raw_dir, struct)
    print(f"  {len(cases)} patients", flush=True)

    modes = ["ours_cc", "gpgn_200_50", "gpgn_500_100", "raw"]

    print(f"\n{'mode':22s} | {'PER-PATIENT':^26s} | {'AGGREGATED':^26s}", flush=True)
    print(f"{'':22s} | {'GTVp':>7s} {'GTVn':>7s} {'MEAN':>8s} | "
          f"{'GTVp':>7s} {'GTVn':>7s} {'MEAN':>8s}", flush=True)
    print("-" * 82, flush=True)
    results = {}
    for m in modes:
        r = evaluate(cases, m)
        results[m] = r
        print(f"{m:22s} | {r['pp_gtvp']:7.4f} {r['pp_gtvn']:7.4f} "
              f"{r['pp_mean']:8.4f} | {r['ag_gtvp']:7.4f} {r['ag_gtvn']:7.4f} "
              f"{r['ag_mean']:8.4f}", flush=True)

    best_pp = max(results, key=lambda k: results[k]["pp_mean"])
    best_ag = max(results, key=lambda k: results[k]["ag_mean"])
    print("-" * 82, flush=True)
    print(f"BEST per-patient MeanDice: {best_pp}  ({results[best_pp]['pp_mean']:.4f})", flush=True)
    print(f"BEST aggregated  MeanDice: {best_ag}  ({results[best_ag]['ag_mean']:.4f})", flush=True)
    print(f"\nDeployed (ours_cc): per-patient {results['ours_cc']['pp_mean']:.4f} | "
          f"aggregated {results['ours_cc']['ag_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
