"""STU-Net encoder wrapper for the HECKTOR seg pipeline.

Satisfies the encoder contract enforced by `models/encoder_factory.py`:
  - `forward(x)`               → final-res seg logits (back-compat with VoCoSwinEncoder)
  - `forward(x, multiscale=)`  → dict {1, 2, 4, 8} when deep_supervision=True
  - `encode(x)`                → list of multi-scale skip features (deepest last = bottleneck)
  - `.bottleneck_channels`     → int (deepest encoder feature channel count)

Pretrained-weight loading (optional): adapts the 1-channel TotalSegmentator
STU-Net weights to our 2-channel CT+PT input by tiling the first conv kernels
(`conv_blocks_context[0][0].conv1.weight` + the 1×1 skip `conv3.weight`) from
in_channels=1 → 2, divided by 2 (sum-preserving — keeps activation magnitude
at the original CT-only signal, halves contribution of each modality so a 2-ch
input with PT=0 acts like the original 1-ch CT).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn

from .stunet import STUNET_VARIANTS, build_stunet


def _adapt_state_dict_to_n_in(state_dict: dict, model_sd: dict, target_in_channels: int) -> tuple[dict, dict]:
    """Tile the first-conv kernels from 1→N input channels (TotalSegmentator pretrained
    is 1-channel CT). Mirrors `models/encoder_voco_swin._adapt_patch_embed`.
    Returns (adapted_state_dict, info)."""
    adapted = dict(state_dict)
    info = {"adapted_keys": [], "skipped_shape_mismatch": []}
    for key in ("conv_blocks_context.0.0.conv1.weight",     # first 3×3×3 input conv
                "conv_blocks_context.0.0.conv3.weight"):    # 1×1×1 residual skip
        if key not in adapted or key not in model_sd:
            continue
        w = adapted[key]
        if w.shape[1] == target_in_channels:
            continue                                         # already matches
        if w.shape[1] == 1:
            new_w = w.repeat(1, target_in_channels, 1, 1, 1) / float(target_in_channels)
            adapted[key] = new_w
            info["adapted_keys"].append((key, tuple(w.shape), tuple(new_w.shape)))
        else:
            info["skipped_shape_mismatch"].append((key, tuple(w.shape), tuple(model_sd[key].shape)))
    return adapted, info


class STUNetEncoder(nn.Module):
    """STU-Net backbone wrapper exposing the same contract as VoCoSwinEncoder."""

    def __init__(self,
                 in_channels: int = 2,
                 out_channels: int = 3,
                 img_size=(192, 192, 320),
                 variant: str = "Small",
                 pretrained: Optional[Path | str] = None,
                 deep_supervision: bool = False) -> None:
        super().__init__()
        if variant not in STUNET_VARIANTS:
            raise ValueError(f"STU-Net variant must be one of {list(STUNET_VARIANTS)}, got '{variant}'")
        self.variant = variant
        self.deep_supervision = deep_supervision

        self.net = build_stunet(in_channels=in_channels,
                                out_channels=out_channels,
                                variant=variant,
                                img_size=img_size)
        # Bottleneck = last encoder stage's output channels = dims[-1].
        self.bottleneck_channels = STUNET_VARIANTS[variant]["dims"][-1]

        if pretrained is not None:
            self._load_pretrained(Path(pretrained), in_channels)

    def _load_pretrained(self, path: Path, in_channels: int) -> None:
        if not path.exists():
            print(f"[STU-Net {self.variant}] pretrained not found at {path}; using random init", flush=True)
            return
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        # Unwrap common nesting (nnU-Net saves under 'state_dict' / 'network_weights' / 'model').
        sd = raw
        if isinstance(raw, dict):
            for k in ("state_dict", "network_weights", "net", "model"):
                if k in raw and isinstance(raw[k], dict):
                    sd = raw[k]
                    break
        # nnU-Net often prepends 'network.' or 'module.'; strip both.
        cleaned = {}
        for k, v in sd.items():
            nk = k
            for prefix in ("network.", "module."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
        model_sd = self.net.state_dict()
        cleaned, info = _adapt_state_dict_to_n_in(cleaned, model_sd, in_channels)
        # Final shape-filtered load (drops any other mismatched keys, e.g. seg head if num_classes differs).
        compatible, dropped = {}, []
        for k, v in cleaned.items():
            if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape):
                compatible[k] = v
            elif k in model_sd:
                dropped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        missing, unexpected = self.net.load_state_dict(compatible, strict=False)
        pct = 100.0 * len(compatible) / max(len(cleaned), 1)
        print(f"[STU-Net {self.variant}] loaded {len(compatible)}/{len(cleaned)} keys ({pct:.1f}%) "
              f"from {path}; adapted {len(info['adapted_keys'])} input-channel keys "
              f"(1ch→{in_channels}ch tile), dropped {len(dropped)} shape-mismatch, "
              f"{len(missing)} missing, {len(unexpected)} unexpected", flush=True)

    # ── contract ─────────────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.net.encode(x)

    def forward(self, x: torch.Tensor,
                multiscale: bool = False,
                film: Optional[dict] = None):
        # FiLM is a VoCo-pipeline concept; STU-Net doesn't have FiLM hooks. Accept
        # the kwarg for interface compatibility but only honor multiscale.
        if film is not None:
            raise NotImplementedError("STU-Net encoder does not implement FiLM modulation. "
                                       "Set use_text_conditioning=False for STU-Net configs.")
        # Return type is driven by the caller's `multiscale` kwarg, NOT by
        # self.deep_supervision — the multitask model passes `multiscale=True`
        # only in train+DS mode and expects a plain tensor everywhere else
        # (sanity check, validation, inference).
        if multiscale:
            return self.net(x, multiscale=True)
        return self.net(x)
