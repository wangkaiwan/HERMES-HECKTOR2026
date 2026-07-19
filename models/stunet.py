"""STU-Net — standalone PyTorch port from uni-medical/STU-Net (nnUNet-1.7.1/.../STUNet.py).

The original lives inside nnU-Net's framework and inherits from
`SegmentationNetwork` (which drags in nnU-Net's sliding-window inferer + a He
initializer). We need a plain `nn.Module` we can plug into our pipeline, so:

  - inherit from `nn.Module` (drops `SegmentationNetwork`'s baggage)
  - inline `InitWeights_He` as a module-level function (a/Kaiming, identical to nnU-Net)
  - keep the encoder / decoder / forward unchanged → cached pretrained state_dicts
    (uni-medical TotalSegmentator weights) load cleanly with `strict=False`
  - add a `multiscale=True` toggle to forward that returns the DS outputs as a
    `{1: full, 2: 1/2, 4: 1/4, 8: 1/8}` dict to match our DeepSupervisionDiceCELoss
    contract (drops the noisiest 1/16 aux on purpose)
  - add a `encode()` method returning the encoder skip features list (deepest =
    bottleneck) — needed if we later attach TN/Cox heads (Route C in
    [[project-task23-arch-decision]]).

Reference variants (uni-medical STU-Net repo, STUNetTrainer.py):
  Small : dims=[16,32,64,128,256,256], depth=[1,1,1,1,1,1]  (~14M params)
  Base  : dims=[32,64,128,256,512,512], depth=[1,1,1,1,1,1]  (~58M)
  Large : dims=[64,128,256,512,1024,1024], depth=[2,2,2,2,2,2]
  Huge  : depth=[3,3,3,3,3,3] of Large dims
The 2025 HECKTOR seg winner ("Less is More") used STU-Net Small + ensemble.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


STUNET_VARIANTS = {
    "Small": dict(dims=[16, 32, 64, 128, 256, 256],   depth=[1, 1, 1, 1, 1, 1]),
    "Base":  dict(dims=[32, 64, 128, 256, 512, 512],  depth=[1, 1, 1, 1, 1, 1]),
    "Large": dict(dims=[64, 128, 256, 512, 1024, 1024], depth=[2, 2, 2, 2, 2, 2]),
    "Huge":  dict(dims=[64, 128, 256, 512, 1024, 1024], depth=[3, 3, 3, 3, 3, 3]),
}


def _init_he(module: nn.Module) -> None:
    """Match nnU-Net's `InitWeights_He(neg_slope=1e-2)`."""
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, a=1e-2)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class BasicResBlock(nn.Module):
    """3-conv residual block (InstanceNorm + LeakyReLU)."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int | list = 3,
                 padding: int | list = 1, stride: int | list = 1,
                 use_1x1conv: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(out_c, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(out_c, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride) if use_1x1conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        return self.act2(y + x)


class _NearestUpsample(nn.Module):
    def __init__(self, in_c: int, out_c: int, scale, mode: str = "nearest") -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=1)
        self.scale = scale
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode))


class STUNet(nn.Module):
    """STU-Net network — standalone (no nnU-Net dependency).

    Forward modes:
      - default (`multiscale=False`): returns the final-res seg logits only.
      - `multiscale=True`: returns dict `{1: full, 2: 1/2, 4: 1/4, 8: 1/8}` (drops
        the 1/16 aux that the original returns — useless for small lesions).
    """

    def __init__(self, input_channels: int, num_classes: int,
                 dims: list[int], depth: list[int],
                 pool_op_kernel_sizes: list[list[int]],
                 conv_kernel_sizes: list[list[int]]) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dims = list(dims)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, axis=0, dtype=np.int64).tolist()

        num_pool = len(pool_op_kernel_sizes)
        assert num_pool == len(dims) - 1, f"need {len(dims)-1} pools for {len(dims)} stages, got {num_pool}"
        conv_pad_sizes = [[k // 2 for k in krnl] for krnl in conv_kernel_sizes]

        # encoder
        self.conv_blocks_context = nn.ModuleList()
        self.conv_blocks_context.append(nn.Sequential(
            BasicResBlock(input_channels, dims[0], conv_kernel_sizes[0], conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], conv_kernel_sizes[0], conv_pad_sizes[0]) for _ in range(depth[0] - 1)],
        ))
        for d in range(1, num_pool + 1):
            self.conv_blocks_context.append(nn.Sequential(
                BasicResBlock(dims[d - 1], dims[d], conv_kernel_sizes[d], conv_pad_sizes[d],
                              stride=pool_op_kernel_sizes[d - 1], use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d], conv_kernel_sizes[d], conv_pad_sizes[d]) for _ in range(depth[d] - 1)],
            ))

        # upsample + decoder
        self.upsample_layers = nn.ModuleList(
            _NearestUpsample(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u]) for u in range(num_pool)
        )
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            self.conv_blocks_localization.append(nn.Sequential(
                BasicResBlock(dims[-2 - u] * 2, dims[-2 - u],
                              conv_kernel_sizes[-2 - u], conv_pad_sizes[-2 - u], use_1x1conv=True),
                *[BasicResBlock(dims[-2 - u], dims[-2 - u],
                                conv_kernel_sizes[-2 - u], conv_pad_sizes[-2 - u]) for _ in range(depth[-2 - u] - 1)],
            ))

        # per-stage seg heads (1×1×1 conv → num_classes)
        self.seg_outputs = nn.ModuleList(
            nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1) for ds in range(num_pool)
        )

        self.apply(_init_he)

    # ─── encode-only path (skip features; bottleneck = trunk[-1]) ────────────
    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            feats.append(x)
        x = self.conv_blocks_context[-1](x)
        feats.append(x)            # bottleneck appended last
        return feats

    def forward(self, x: torch.Tensor, multiscale: bool = False):
        # encoder
        skips: list[torch.Tensor] = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
        x = self.conv_blocks_context[-1](x)
        # decoder with skip concat
        seg_outputs: list[torch.Tensor] = []
        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat([x, skips[-(u + 1)]], dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.seg_outputs[u](x))

        final = seg_outputs[-1]                                            # full-res logits
        if not multiscale:
            return final
        # Map per-decoder outputs to scale-keyed dict.  seg_outputs is ordered
        # from the COARSEST decoder (1/16 for 5-pool stack) to the FINEST (1/1).
        # Our DeepSupervisionDiceCELoss expects {1, 2, 4, 8} → take the top 4
        # (1, 1/2, 1/4, 1/8) and drop the 1/16 aux (small lesions vanish after
        # 16× downsample; matches DeepSupSwinUNETR's choice).
        ms: dict[int, torch.Tensor] = {1: final}
        # seg_outputs[-1]=1/1 already used; seg_outputs[-2]=1/2, [-3]=1/4, [-4]=1/8
        for scale, idx in [(2, -2), (4, -3), (8, -4)]:
            if -idx <= len(seg_outputs):
                ms[scale] = seg_outputs[idx]
        return ms


def build_stunet(in_channels: int, out_channels: int,
                 variant: str = "Small", img_size=None) -> STUNet:
    """Construct STU-Net with the standard 5-pool isotropic config used by the
    HECKTOR 2025 winner. `img_size` is only checked for /32 divisibility.
    """
    cfg = STUNET_VARIANTS[variant]
    # 5 isotropic 2× pools (matches the winner's setup on 1mm iso 200×200×310; we
    # use 192×192×320 which is the same /32 grid).
    pool_op_kernel_sizes = [[2, 2, 2]] * 5
    conv_kernel_sizes = [[3, 3, 3]] * 6
    if img_size is not None:
        divisor = int(np.prod(pool_op_kernel_sizes, axis=0, dtype=np.int64).max())
        for d, s in zip("xyz", img_size):
            if s % divisor != 0:
                raise ValueError(f"STU-Net img_size {img_size}: axis {d}={s} not divisible by {divisor}")
    return STUNet(
        input_channels=in_channels,
        num_classes=out_channels,
        dims=cfg["dims"],
        depth=cfg["depth"],
        pool_op_kernel_sizes=pool_op_kernel_sizes,
        conv_kernel_sizes=conv_kernel_sizes,
    )
