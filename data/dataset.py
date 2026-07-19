"""
Dataset + DataLoader builder for HECKTOR 2026 multitask training.

Design:
    Phase 1 (deterministic, cacheable) — get_train_preprocess_transforms
        load NIfTI → orient RAS → 2mm iso → clip CT, normalise PT → body crop → pad
        cached on disk via MONAI PersistentDataset (key = serialised sample dict)
    Phase 2 (random, per-iteration) — get_train_augmentation_transforms
        RandCropByLabelClassesd → ConcatItemsd → RandDropPETd → flips/affine/elastic/intensity

Sample keys (after the full pipeline):
    image          float32 [2, D, H, W]   CT (ch 0) + PT (ch 1)
    label          uint8   [1, D, H, W]   0/1/2
    patient_id     str
    center         str
    has_pet/has_seg/has_staging/has_survival : bool flags
    t_stage        long  scalar  (0..4 mapping below; -1 if missing)
    n_stage        long  scalar  (0..3; -1 if missing)
    relapse        long  scalar  (0/1; -1 if missing)
    rfs_days       float scalar
    text_features  float [768]   ClinicalBERT [CLS]  (zeros if not loaded)
    clinical_feat  float [16]    encoded tabular     (zeros if not loaded)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from monai.data import DataLoader, Dataset, PersistentDataset
import monai.data.dataset as _monai_dataset
from torch.utils.data import Dataset as TorchDataset

# PyTorch 2.6+ changed torch.load default to weights_only=True, which fails
# on MONAI's MetaTensor (not in the safe-globals list) and breaks PersistentDataset
# cache loading. Patch the torch.load that MONAI calls to keep the legacy behavior.
_orig_torch_load = _monai_dataset.torch.load
def _patched_torch_load(f, **kwargs):                          # noqa: ANN001
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, **kwargs)
_monai_dataset.torch.load = _patched_torch_load


# AJCC/UICC 7th ed.
#   T: 4 classes — T1, T2, T3, T4 (T4A/B collapsed to T4). T0 is NOT a class
#                  per challenge spec; the 3 training patients with T0 (MDA-
#                  297/386/447) are excluded from staging supervision via
#                  has_staging_t below. Task #36, 2026-05-13.
#   N: 4 classes — N0, N1, N2 (N2A/B/C collapsed to N2), N3.
T_STAGE_TO_INT = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T4A": 3, "T4B": 3}
N_STAGE_TO_INT = {"N0": 0, "N1": 1, "N2": 2, "N2A": 2, "N2B": 2, "N2C": 2, "N3": 3}


def _encode_t(s) -> tuple[int, bool]:
    """Returns (class_idx, has_valid_t).

    class_idx is a valid index (0..3) — 0 when missing/unknown/T0, which lets
    F.cross_entropy run without OOB indexing. has_valid_t is False when:
      - the value is NaN/missing,
      - the value is T0 (out of the challenge's T1-T4 spec).
    Callers AND the loss mask via the dataset's has_staging flag.
    """
    if pd.isna(s):
        return 0, False
    k = str(s).strip().upper()
    if k == "T0":
        return 0, False                              # excluded from supervision
    if k in T_STAGE_TO_INT:
        return T_STAGE_TO_INT[k], True
    return 0, False                                  # unknown string → drop


def _encode_n(s) -> tuple[int, bool]:
    """Returns (class_idx, has_valid_n). Mirrors _encode_t for symmetry."""
    if pd.isna(s):
        return 0, False
    k = str(s).strip().upper()
    if k in N_STAGE_TO_INT:
        return N_STAGE_TO_INT[k], True
    return 0, False


def build_data_dicts(manifest_df: pd.DataFrame) -> List[Dict]:
    """Convert a manifest row → MONAI sample-dict."""
    samples: List[Dict] = []
    for _, row in manifest_df.iterrows():
        pid = str(row["patient_id"])
        if not row["has_ct"] or not row["has_pet"]:
            continue                                          # nothing we can do

        # Label path may be empty (no GT mask). Use CT as a placeholder so MONAI
        # can load *something*; the trainer masks the loss via has_seg.
        label_path = row["label_path"] if row["has_seg"] else row["ct_path"]

        rfs = row.get("RFS")
        rfs_val = float(rfs) if pd.notna(rfs) else 0.0
        relapse = row.get("Relapse")
        relapse_val = int(relapse) if pd.notna(relapse) else 0

        t_idx, t_valid = _encode_t(row.get("T-stage"))
        n_idx, n_valid = _encode_n(row.get("N-stage"))
        # has_staging — flagged True only if BOTH T and N are valid for the loss.
        # The manifest's existing has_staging is union-over-(T,N); we tighten it
        # here so #36 (drop T0) takes effect without re-running the manifest.
        manifest_has_staging = bool(row["has_staging"])

        sample = {
            "ct": row["ct_path"],
            "pt": row["pt_path"],
            "label": label_path,
            "patient_id": pid,
            "center": row["center"],
            "has_pet": True,
            "has_seg": bool(row["has_seg"]),
            "has_staging": manifest_has_staging and t_valid and n_valid,
            "has_survival": bool(row["has_survival"]),
            "t_stage": t_idx,
            "n_stage": n_idx,
            "relapse": relapse_val,
            "rfs_days": rfs_val,
        }
        samples.append(sample)
    return samples


class _TwoPhaseDataset(TorchDataset):
    """Cached preprocess + per-iter augmentation, plus per-sample feature lookup."""

    def __init__(self, base_ds, aug_transforms,
                 text_features: Dict[str, torch.Tensor] | None = None,
                 clinical_features: Dict[str, torch.Tensor] | None = None,
                 text_dim: int = 768, clinical_dim: int = 18) -> None:
        self.base_ds = base_ds
        self.aug = aug_transforms
        self.text = text_features or {}
        self.clin = clinical_features or {}
        self.text_dim = text_dim
        self.clinical_dim = clinical_dim

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx):
        result = self.aug(self.base_ds[idx])
        sample = result[0] if isinstance(result, list) else result
        pid = sample.get("patient_id", "")
        sample["text_features"] = self.text.get(pid, torch.zeros(self.text_dim))
        sample["clinical_feat"] = self.clin.get(pid, torch.zeros(self.clinical_dim))
        return sample


class _ValDataset(TorchDataset):
    """Cached preprocess + deterministic post-cache transform + feature lookup."""

    def __init__(self, monai_ds, post_transforms,
                 text_features: Dict[str, torch.Tensor] | None = None,
                 clinical_features: Dict[str, torch.Tensor] | None = None,
                 text_dim: int = 768, clinical_dim: int = 18) -> None:
        self.ds = monai_ds
        self.post = post_transforms
        self.text = text_features or {}
        self.clin = clinical_features or {}
        self.text_dim = text_dim
        self.clinical_dim = clinical_dim

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        if self.post is not None:
            sample = self.post(sample)
            sample = sample[0] if isinstance(sample, list) else sample
        pid = sample.get("patient_id", "")
        sample["text_features"] = self.text.get(pid, torch.zeros(self.text_dim))
        sample["clinical_feat"] = self.clin.get(pid, torch.zeros(self.clinical_dim))
        return sample


def build_dataloaders(cfg: dict, fold: int) -> Dict[str, DataLoader]:
    """Returns {'train': DataLoader, 'val': DataLoader} for the given fold."""
    from data.transforms import (
        get_train_preprocess_transforms,
        get_train_augmentation_transforms,
        get_val_transforms,
    )

    dcfg = cfg["data"]
    manifest_dir = Path(dcfg.get("manifest_dir", "data/manifests"))
    train_df = pd.read_csv(manifest_dir / f"train_fold{fold}.csv")
    val_df = pd.read_csv(manifest_dir / f"val_fold{fold}.csv")
    train_dicts = build_data_dicts(train_df)
    val_dicts = build_data_dicts(val_df)
    print(f"fold {fold}: train={len(train_dicts)} samples, val={len(val_dicts)} samples")

    text = _maybe_load(dcfg.get("text_features_path"))
    clin = _maybe_load(dcfg.get("clinical_tabular_path"))

    cache_dir = dcfg.get("cache_dir")
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Shared on-disk cache: same preprocess pipeline for train + val.
        # Per-iter post-cache transforms (random aug for train, deterministic
        # concat+pad for val) are applied on top by the wrapper datasets and
        # are NOT cached.
        from data.transforms import (
            get_preprocess_transforms,
            get_val_postcache_transforms,
        )
        pre = get_preprocess_transforms(dcfg)
        train_aug = get_train_augmentation_transforms(dcfg)
        val_post = get_val_postcache_transforms(dcfg)

        train_base = PersistentDataset(data=train_dicts, transform=pre, cache_dir=str(cache_dir))
        train_ds = _TwoPhaseDataset(train_base, train_aug, text, clin)

        val_base = PersistentDataset(data=val_dicts, transform=pre, cache_dir=str(cache_dir))
        val_ds = _ValDataset(val_base, val_post, text, clin)
    else:
        from data.transforms import get_train_transforms
        train_ds = _TwoPhaseDataset(
            Dataset(data=train_dicts, transform=get_train_transforms(dcfg)),
            aug_transforms=lambda x: x,                       # already applied above
            text_features=text, clinical_features=clin,
        )
        val_ds = _ValDataset(
            Dataset(data=val_dicts, transform=get_val_transforms(dcfg)),
            post_transforms=None,
            text_features=text, clinical_features=clin,
        )

    nw = dcfg.get("num_workers", 8)
    bs = cfg["training"].get("batch_size", 1)

    # Hard-case oversampling — give listed "hard" patients a higher sampling
    # probability so the model sees them more often per epoch. Default OFF
    # (enabled=false) → plain shuffle, current behavior unchanged.
    # `patient_ids` is the list of hard patient_id strings; everyone else keeps
    # weight 1.0. Data-driven population: after Phase-1 5-fold, rank patients by
    # per-patient val dice and put the bottom quartile here.
    hc = dcfg.get("hard_case_oversampling", {}) or {}
    sampler = None
    use_shuffle = True
    if hc.get("enabled", False) and hc.get("patient_ids"):
        from torch.utils.data import WeightedRandomSampler
        hard = set(str(p) for p in hc["patient_ids"])
        w_hard = float(hc.get("weight", 3.0))
        pid_order = train_df["patient_id"].astype(str).tolist()   # same order as train_ds
        weights = [w_hard if pid in hard else 1.0 for pid in pid_order]
        n_hard = sum(1 for pid in pid_order if pid in hard)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        use_shuffle = False
        print(f"[oversample] hard-case oversampling ON: {n_hard}/{len(pid_order)} "
              f"patients at weight {w_hard}")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=use_shuffle,
                              sampler=sampler,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=nw, pin_memory=True,
                            persistent_workers=nw > 0)
    return {"train": train_loader, "val": val_loader}


def _maybe_load(path: Optional[str]) -> Optional[Dict[str, torch.Tensor]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return torch.load(p, weights_only=True)
