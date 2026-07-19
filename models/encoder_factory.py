"""Backbone-selector hook for the seg encoder.

The MultitaskModel forwards a small kwarg dict and a `seg_backbone` string;
this module returns the encoder instance + its bottleneck channel count.

The encoder contract every backbone MUST satisfy:
    - subclass `nn.Module`
    - expose `encode(x: Tensor) -> List[Tensor]` returning multi-scale features,
      with the deepest (bottleneck) feature LAST
    - expose `forward(x: Tensor) -> Tensor` returning seg logits at input resolution
    - expose `.bottleneck_channels: int` attribute

Currently implemented:
    - voco: VoCoSwinEncoder (our primary). Pretrained SSL warm-start supported.

Planned (raise NotImplementedError for now):
    - segresnet: MONAI's monai.networks.nets.SegResNet. Task #17. Drop-in: rank-5
      team MoriiHuang used SegResNet + SSIM-hard-mining (highest GTVn F1 of top-7).
    - stunet: HKUST-AIoPS STU-Net Small. Task #18. Apache-2.0; TotalSegmentator
      pretrained weights available. Rank-1 team lishancai21 used STU-Net Small × 10.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch.nn as nn

from .encoder_voco_swin import VoCoSwinEncoder


def build_encoder(
    seg_backbone: str = "voco",
    *,
    in_channels: int = 2,
    out_channels: int = 3,
    img_size: Tuple[int, int, int] = (192, 192, 128),
    # VoCo-specific
    variant: str = "B",
    pretrained: Path | str | None = None,
    source: str = "local",
    hf_head: str = "omni",
    hf_cache_dir: Path | str | None = None,
    hf_token: str | None = None,
    use_checkpoint: bool = True,
    drop_path_rate: float = 0.1,
    deep_supervision: bool = False,            # task #27 — multi-scale aux heads + FiLM hooks
    # SegResNet / STU-Net specific kwargs go below the **per-backbone gate**.
    **kwargs,
) -> nn.Module:
    """Return a seg-encoder instance for the named backbone.

    Args common to every backbone:
        in_channels, out_channels, img_size

    Args specific to a backbone are passed through; non-applicable args are
    ignored. The returned encoder MUST expose `.bottleneck_channels` so the
    head-sizing math in `HECKTORMultitaskModel.__init__` can read it.
    """
    name = (seg_backbone or "voco").strip().lower()
    if name == "voco":
        return VoCoSwinEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            variant=variant,
            pretrained=pretrained,
            source=source,
            hf_head=hf_head,
            hf_cache_dir=hf_cache_dir,
            hf_token=hf_token,
            use_checkpoint=use_checkpoint,
            drop_path_rate=drop_path_rate,
            deep_supervision=deep_supervision,
        )
    if name == "segresnet":
        raise NotImplementedError(
            "seg_backbone='segresnet' is reserved for task #17. To wire it: "
            "import monai.networks.nets.SegResNet, expose .encode() returning "
            "intermediate features (likely the encoder block outputs), .forward() "
            "returning seg logits, and set .bottleneck_channels = init_filters * 2**(num_down)."
        )
    if name == "stunet":
        from .encoder_stunet import STUNetEncoder
        return STUNetEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            variant=variant,                       # "Small" / "Base" / "Large" / "Huge"
            pretrained=pretrained,
            deep_supervision=deep_supervision,
        )
    raise ValueError(
        f"Unknown seg_backbone '{seg_backbone}'. "
        f"Choose from: voco (impl), segresnet (planned #17), stunet (planned #18)."
    )
