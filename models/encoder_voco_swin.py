"""
VoCo-pretrained SwinUNETR encoder + decoder.

Supports two weight sources, with a single load helper that handles both:

  1. Local checkpoint (already used in HN_CU_Seg)
        third_party/voco_pretrained/VoComni_B.pt
        load via load_voco_local

  2. Original VoCo release on HuggingFace
        repo = "Luffy503/VoCo"   (Apache-2.0)
        files = {
            (B, "ssl"):  "VoCo_B_SSL_head.pt",   220 MB
            (L, "ssl"):  "VoCo_L_SSL_head.pt",   856 MB
            (H, "ssl"):  "VoCo_H_SSL_head.pt",   3.4 GB
            (B, "omni"): "VoComni_B.pt",         299 MB
            (L, "omni"): "VoComni_L.pt",         1.17 GB
            (H, "omni"): "VoComni_H.pt",         4.65 GB
        }
        load via load_voco_hf

The state_dict adaptation (prefix stripping, swin_vit→swinViT rename, and
patch_embed inflation from 1-ch pretrained → N-ch) is the proven recipe from
HN_CU_Seg/models/voco.py. SwinUNETR is constructed with `use_v2=True` to match
the architecture VoCo was pretrained on.

Multi-scale features for downstream heads (TN-staging, prognosis pooling) are
exposed via VoCoSwinEncoder.encode().
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from monai.networks.nets import SwinUNETR


# ---------------------------------------------------------------------------
# Variant registry — feature_size + depths/heads for B / L / H from VoCo
# ---------------------------------------------------------------------------
VOCO_VARIANTS = {
    "B": dict(feature_size=48,  depths=(2, 2, 2, 2), num_heads=(3,  6, 12, 24)),
    "L": dict(feature_size=96,  depths=(2, 2, 2, 2), num_heads=(3,  6, 12, 24)),
    "H": dict(feature_size=192, depths=(2, 2, 2, 2), num_heads=(3,  6, 12, 24)),
}

VOCO_HF_REPO = "Luffy503/VoCo"
VOCO_HF_FILES = {
    ("B", "ssl"):  "VoCo_B_SSL_head.pt",
    ("L", "ssl"):  "VoCo_L_SSL_head.pt",
    ("H", "ssl"):  "VoCo_H_SSL_head.pt",
    ("B", "omni"): "VoComni_B.pt",
    ("L", "omni"): "VoComni_L.pt",
    ("H", "omni"): "VoComni_H.pt",
    ("B", "nnunet"): "VoComni_nnunet.pt",          # not a SwinUNETR — separate path
}


# ---------------------------------------------------------------------------
# SwinUNETR construction
# ---------------------------------------------------------------------------
def build_swinunetr(in_channels: int,
                    out_channels: int,
                    img_size: Tuple[int, int, int],
                    variant: str = "B",
                    use_checkpoint: bool = True,
                    drop_path_rate: float = 0.1,
                    deep_supervision: bool = False) -> SwinUNETR:
    """SwinUNETR matching one of the VoCo variants (uses MONAI v2 blocks).

    deep_supervision=True returns `DeepSupSwinUNETR` instead — same base
    architecture + per-stage aux seg heads + FiLM hooks (task #27).
    """
    cfg = VOCO_VARIANTS[variant]
    kw = dict(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=cfg["feature_size"],
        depths=cfg["depths"],
        num_heads=cfg["num_heads"],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=drop_path_rate,
        use_checkpoint=use_checkpoint,
        spatial_dims=3,
        use_v2=True,                                  # required to match VoCo pretrained blocks
    )
    if deep_supervision:
        from .decoder_swin import DeepSupSwinUNETR
        return DeepSupSwinUNETR(**kw)
    return SwinUNETR(**kw)


# ---------------------------------------------------------------------------
# State-dict adaptation (port of HN_CU_Seg/models/voco.py:_load_voco_weights)
# ---------------------------------------------------------------------------
def _extract_state_dict(raw) -> dict:
    """VoCo checkpoints wrap weights under several possible keys."""
    if isinstance(raw, dict):
        for k in ("state_dict", "network_weights", "net", "student", "model"):
            if k in raw and isinstance(raw[k], dict):
                return raw[k]
    return raw


def _strip_prefixes(state_dict: dict, prefixes: Iterable[str]) -> dict:
    """Strip wrapper prefixes (e.g. 'module.', 'backbone.') from every key.

    Only removes outer wrappers — `swin_vit.` / `swinViT.` are NOT included
    because those are part of MONAI's expected key paths; we want to preserve
    them and let `_rename_swin_vit` align casing.
    """
    sd = dict(state_dict)
    for prefix in prefixes:
        first = next(iter(sd), "")
        if first.startswith(prefix):
            sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    return sd


def _rename_swin_vit(state_dict: dict) -> dict:
    """VoCo's source uses 'swin_vit'; MONAI's SwinUNETR uses 'swinViT'."""
    return {k.replace("swin_vit", "swinViT"): v for k, v in state_dict.items()}


def _maybe_add_swinvit_prefix(state_dict: dict, model_sd: dict) -> dict:
    """Some VoCo encoder-only checkpoints store keys without the 'swinViT.' prefix.

    If the loaded keys all look like trunk-internal paths ('patch_embed.', 'layers'..)
    and the model expects them under 'swinViT.', prepend that prefix. Idempotent —
    leaves keys alone if 'swinViT.' is already present in the loaded dict.
    """
    expects_swinvit = any(k.startswith("swinViT.") for k in model_sd)
    has_swinvit = any(k.startswith("swinViT.") for k in state_dict)
    if expects_swinvit and not has_swinvit:
        trunk_keys = ("patch_embed.", "layers", "norm.", "pos_embed", "pos_drop")
        if any(any(k.startswith(p) for p in trunk_keys) for k in state_dict):
            return {f"swinViT.{k}": v for k, v in state_dict.items()}
    return state_dict


def _adapt_patch_embed(state_dict: dict, target_in_channels: int, model_sd: dict) -> dict:
    """Tile a 1-channel patch_embed kernel to N channels and renormalise.

    Mirrors HN_CU_Seg recipe: weight.repeat(1, N, 1, 1, 1) / N — each input
    channel gets an equal informed start; summing N identical inputs reproduces
    the original 1-ch activation magnitude.
    """
    key = "swinViT.patch_embed.proj.weight"
    if key in state_dict and key in model_sd:
        pt_w = state_dict[key]                                  # [C_out, 1, kD, kH, kW]
        cur_w = model_sd[key]
        if pt_w.shape != cur_w.shape and pt_w.dim() == 5 and pt_w.shape[1] == 1:
            state_dict[key] = pt_w.repeat(1, target_in_channels, 1, 1, 1) / target_in_channels
    return state_dict


def _adapt_state_dict(raw, model: nn.Module, in_channels: int) -> Tuple[dict, int, int]:
    """Full adaptation pipeline. Returns (state_dict, n_loaded, n_total)."""
    sd = _extract_state_dict(raw)
    sd = _strip_prefixes(sd, prefixes=("module.", "backbone."))
    sd = _rename_swin_vit(sd)
    cur = model.state_dict()
    sd = _maybe_add_swinvit_prefix(sd, cur)
    sd = _adapt_patch_embed(sd, in_channels, cur)

    # Keep only keys whose shapes match the current model.
    matched = {k: v for k, v in sd.items() if k in cur and v.shape == cur[k].shape}
    n_loaded, n_total = len(matched), len(cur)
    return matched, n_loaded, n_total


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------
def load_voco_local(model: SwinUNETR, weights_path: Path, in_channels: int) -> dict:
    """Load weights from a local VoCo .pt file. Returns load summary."""
    raw = torch.load(weights_path, map_location="cpu", weights_only=False)
    matched, n_loaded, n_total = _adapt_state_dict(raw, model, in_channels)
    info = model.load_state_dict(matched, strict=False)
    return {
        "source": str(weights_path),
        "loaded": n_loaded,
        "total": n_total,
        "missing": list(info.missing_keys),
        "unexpected": list(info.unexpected_keys),
    }


def load_voco_hf(model: SwinUNETR,
                 variant: str,
                 head: str,
                 in_channels: int,
                 cache_dir: Path | str | None = None,
                 token: str | None = None) -> dict:
    """Download and load VoCo weights from huggingface.co/Luffy503/VoCo.

    Args:
        model       : SwinUNETR instance (already constructed with the same variant).
        variant     : "B" | "L" | "H".
        head        : "ssl"  → VoCo_<variant>_SSL_head.pt   (self-supervised)
                      "omni" → VoComni_<variant>.pt          (omni-supervised; recommended)
        in_channels : input channel count (e.g. 2 for HECKTOR CT+PT).
        cache_dir   : where huggingface_hub caches the download. Defaults to
                      ~/.cache/huggingface (or HF_HOME if set).
        token       : optional HuggingFace access token.

    Returns:
        dict with keys: source, loaded, total, missing, unexpected.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required for load_voco_hf. "
            "Install via `pip install huggingface_hub` (already pinned in requirements.txt)."
        ) from e

    key = (variant, head)
    if key not in VOCO_HF_FILES:
        raise ValueError(
            f"Unknown VoCo variant/head combination {key}. "
            f"Available: {sorted(VOCO_HF_FILES.keys())}"
        )
    if head == "nnunet":
        raise ValueError(
            "VoComni_nnunet is an nnUNet checkpoint, not a SwinUNETR. "
            "Use it via scripts/train_nnunetv2.sh instead."
        )

    filename = VOCO_HF_FILES[key]
    cache_str = str(cache_dir) if cache_dir is not None else None
    weights_path = hf_hub_download(
        repo_id=VOCO_HF_REPO,
        filename=filename,
        cache_dir=cache_str,
        token=token,
    )
    return load_voco_local(model, Path(weights_path), in_channels)


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------
class VoCoSwinEncoder(nn.Module):
    """Wraps a SwinUNETR; exposes the multi-scale trunk features alongside seg."""

    def __init__(self,
                 in_channels: int = 2,
                 out_channels: int = 3,
                 img_size: Tuple[int, int, int] = (192, 192, 128),
                 variant: str = "B",
                 pretrained: Optional[Path | str] = None,
                 source: str = "local",                # "local" | "hf"
                 hf_head: str = "omni",                # "omni" | "ssl"
                 hf_cache_dir: Optional[Path | str] = None,
                 hf_token: Optional[str] = None,
                 use_checkpoint: bool = True,
                 drop_path_rate: float = 0.1,
                 deep_supervision: bool = False) -> None:
        super().__init__()
        self.variant = variant
        self.deep_supervision = deep_supervision
        self.net = build_swinunetr(in_channels, out_channels, img_size,
                                   variant, use_checkpoint, drop_path_rate,
                                   deep_supervision=deep_supervision)
        # Channel count of the deepest trunk feature returned by encode().
        # Used by the backbone-selector factory in multitask_model.py to size
        # the TN / prognosis heads without hardcoding 768.
        self.bottleneck_channels = VOCO_VARIANTS[variant]["feature_size"] * 16

        if pretrained is not None:
            if source == "local":
                info = load_voco_local(self.net, Path(pretrained), in_channels)
            elif source == "hf":
                info = load_voco_hf(self.net, variant, hf_head, in_channels,
                                    cache_dir=hf_cache_dir, token=hf_token)
            else:
                raise ValueError(f"Unknown source '{source}' (expected 'local' or 'hf')")

            pct = 100.0 * info["loaded"] / max(info["total"], 1)
            print(f"[VoCo {variant}/{source}] {info['loaded']}/{info['total']} keys "
                  f"({pct:.1f}%) loaded from {info['source']}; "
                  f"{len(info['missing'])} missing, {len(info['unexpected'])} unexpected")

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Multi-scale features from the Swin transformer trunk."""
        return self.net.swinViT(x, normalize=True)

    def forward(self, x: torch.Tensor,
                multiscale: bool = False,
                film: dict | None = None):
        """Return final-res seg logits by default; pass `multiscale=True` to get a
        dict of per-stage logits {1, 2, 4, 8} (only supported when
        deep_supervision=True at construction). `film` is the per-stage
        modulation dict (see `DeepSupSwinUNETR.forward`); None disables FiLM."""
        if multiscale or film is not None:
            if not self.deep_supervision:
                raise RuntimeError(
                    "multiscale / FiLM forward requires deep_supervision=True "
                    "at encoder construction."
                )
            return self.net(x, multiscale=multiscale, film=film)
        return self.net(x)
