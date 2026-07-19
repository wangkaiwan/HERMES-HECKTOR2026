"""
Pre-cache the HECKTOR 2026 release through the MONAI preprocessing pipeline.

This is a one-time sanity check + cache primer:
    1. Validate every patient has CT + PT NIfTI files (label may be absent for
       patients with no GT mask, e.g. inference-time releases).
    2. Run the preprocess transforms (data/transforms.get_train_preprocess_transforms)
       once per patient and write the result into MONAI's PersistentDataset cache.
       After this, every training epoch reads pre-resampled, pre-cropped tensors
       directly off disk — no per-iter SimpleITK overhead.

Cache layout:
    {cache_dir}/   PersistentDataset on-disk cache (pickle blobs)

Usage:
    python data/prepare_hecktor.py \\
        --raw_dir data/raw/hecktor2026_training \\
        --csv data/raw/hecktor2026_training/HECKTOR_2026_training_data.csv \\
        --cache_dir /data/kwang/hecktor2026_cache \\
        --num_workers 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from monai.data import PersistentDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.transforms import get_train_preprocess_transforms


def build_data_dicts(csv_path: Path, raw_dir: Path) -> list[dict]:
    """One sample-dict per patient with raw CT/PT/label paths + clinical labels."""
    df = pd.read_csv(csv_path)
    samples = []
    skipped = []
    for _, row in df.iterrows():
        pid = str(row["PatientID"]).strip()
        pdir = raw_dir / pid
        ct = pdir / f"{pid}__CT.nii.gz"
        pt = pdir / f"{pid}__PT.nii.gz"
        lb = pdir / f"{pid}.nii.gz"

        if not ct.exists() or not pt.exists():
            skipped.append((pid, f"missing CT or PT (CT={ct.exists()}, PT={pt.exists()})"))
            continue

        # If label is missing (inference-time) we still cache image; downstream
        # code masks the seg loss via `has_seg`.
        if not lb.exists():
            sample = {
                "ct": str(ct), "pt": str(pt),
                "label": str(ct),                          # placeholder — will be masked
                "patient_id": pid,
                "has_pet": True,
                "has_seg": False,
            }
        else:
            sample = {
                "ct": str(ct), "pt": str(pt),
                "label": str(lb),
                "patient_id": pid,
                "has_pet": True,
                "has_seg": True,
            }
        samples.append(sample)
    return samples, skipped


def main(raw_dir: Path, csv_path: Path, cache_dir: Path,
         num_workers: int = 8, ct_clip: list[int] = (-200, 200),
         target_spacing: list[float] = (2.0, 2.0, 2.0),
         pt_clip_percentile: list[float] = (0.5, 99.5),
         cache_volume: list[int] = (200, 200, 200)) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples, skipped = build_data_dicts(csv_path, raw_dir)
    print(f"prepare: {len(samples)} samples (skipped {len(skipped)})")
    if skipped:
        for pid, reason in skipped[:10]:
            print(f"  SKIP {pid}: {reason}")

    dcfg = {
        "ct_clip_range": list(ct_clip),
        "target_spacing": list(target_spacing),
        "pt_clip_percentile": list(pt_clip_percentile),
        "cache_volume": list(cache_volume),
    }
    transform = get_train_preprocess_transforms(dcfg)

    ds = PersistentDataset(data=samples, transform=transform, cache_dir=str(cache_dir))
    loader = DataLoader(ds, batch_size=1, num_workers=num_workers,
                        shuffle=False, persistent_workers=False)

    n_done = 0
    failed = []
    for sample in tqdm(loader, total=len(ds), desc="caching"):
        # Iterating through the DataLoader populates the on-disk cache.
        try:
            _ = sample["image"] if "image" in sample else sample["ct"]
            n_done += 1
        except Exception as e:                                # noqa: BLE001
            failed.append((sample.get("patient_id", "?"), str(e)))

    print(f"cached {n_done}/{len(ds)} patients to {cache_dir}")
    if failed:
        print("first failures:")
        for pid, err in failed[:5]:
            print(f"  {pid}: {err}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--cache_dir", type=Path, required=True)
    p.add_argument("--num_workers", type=int, default=8)
    a = p.parse_args()
    main(a.raw_dir, a.csv, a.cache_dir, a.num_workers)
