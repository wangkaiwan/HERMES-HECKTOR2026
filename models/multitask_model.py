"""
End-to-end multitask model: VoCo SwinUNETR + FiLM + (Seg, TN, Cox) heads.

Forward returns a dict:
    seg_logits      [B, 3, D, H, W]
    t_logits        [B, 5]
    n_logits        [B, 4]
    risk            [B]

The module accepts an optional `text_features` tensor [B, 768]; when present
and `use_text_conditioning=True`, FiLM modulation is applied to the seg
decoder stages and the bottleneck feature feeding the staging / Cox heads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from .film import FiLMConditioner, apply_film
from .tn_staging_head import TNStagingHead, masked_global_pool
from .encoder_factory import build_encoder
from .prog_factory import build_prognosis_head


class HECKTORMultitaskModel(nn.Module):
    def __init__(self,
                 in_channels: int = 2,
                 out_seg_channels: int = 3,
                 img_size: Tuple[int, int, int] = (192, 192, 128),
                 variant: str = "B",
                 pretrained: Path | None = None,
                 source: str = "local",
                 use_text: bool = True,
                 text_dim: int = 768,
                 text_dropout_prob: float = 0.3,
                 staging_hidden: int = 256,
                 prognosis_hidden: int = 256,
                 n_clinical: int = 16,
                 n_t_classes: int = 5,
                 n_n_classes: int = 4,
                 # Backbone-selector hook (default 'voco' keeps current behavior).
                 # Reserved values 'segresnet' / 'stunet' raise NotImplementedError
                 # until tasks #17 / #18 wire them. See models/encoder_factory.py.
                 seg_backbone: str = "voco",
                 # Prog-method hook (default 'cox' = current PrognosisHead).
                 # Reserved 'deepsurv' / 'deephit' / 'discrete' raise. See
                 # models/prog_factory.py.
                 prog_method: str = "cox",
                 # Task #27 — deep supervision. When True the encoder returns
                 # multi-scale logits during training (forward() puts them in
                 # 'seg_logits_ms'); the trainer then uses DeepSupervisionDiceCELoss.
                 # The aux heads are random-init; VoCo ckpt loading is unaffected.
                 deep_supervision: bool = False) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision

        self.encoder = build_encoder(
            seg_backbone=seg_backbone,
            in_channels=in_channels,
            out_channels=out_seg_channels,
            img_size=img_size,
            variant=variant,
            pretrained=pretrained,
            source=source,
            deep_supervision=deep_supervision,
        )

        # Auto-derive bottleneck channels from the encoder rather than hardcoding
        # 768 — this lets a SegResNet (default ~256) or STU-Net Small (~320)
        # swap in without changing head sizes.
        bottleneck_dim = int(self.encoder.bottleneck_channels)

        self.use_text = use_text
        if use_text:
            # FiLM is applied at 4 decoder stages (powers of feat_size). Currently
            # this assumes the VoCo variant's decoder pyramid; SegResNet/STU-Net
            # FiLM wiring is part of their integration tasks (#17/#18).
            if seg_backbone != "voco":
                raise NotImplementedError(
                    f"use_text_conditioning=True is only wired for seg_backbone='voco' "
                    f"today; got '{seg_backbone}'. Disable FiLM for non-voco backbones."
                )
            from .encoder_voco_swin import VOCO_VARIANTS
            feat_size = VOCO_VARIANTS[variant]["feature_size"]
            stage_channels = [feat_size * 16, feat_size * 8, feat_size * 4, feat_size * 2]
            self.film = FiLMConditioner(text_dim, stage_channels, text_dropout_prob=text_dropout_prob)

        # Task #46: tn_head also receives the 18-d clinical tabular concat'd to
        # the mask-pooled image features. Demographics / HPV / lifestyle carry
        # genuine staging signal and are robust when the seg mask is noisy.
        self.tn_head = TNStagingHead(in_channels=bottleneck_dim,
                                     hidden_dim=staging_hidden,
                                     n_t_classes=n_t_classes,
                                     n_n_classes=n_n_classes,
                                     clinical_dim=n_clinical,
                                     use_clinical=True)

        # Task #45: prog_head also receives softmax(t_logits) and softmax(n_logits)
        # — SIMS-LIFE 2025 1st-prize trick adapted from HPV→prog to TN→prog
        # (HPV is INPUT in 2026, TN is the prediction target).
        self.prog_head = build_prognosis_head(
            method=prog_method,
            image_dim=2 * bottleneck_dim,                           # GTVp + GTVn pools
            clinical_dim=n_clinical,
            hidden_dim=prognosis_hidden,
            use_clinical=True,
            tn_dim=n_t_classes + n_n_classes,
            use_tn=True,
        )
        # Cache the staging class counts so MultitaskModel.forward can build
        # the correctly-sized softmax concat in one place.
        self._n_t_classes = n_t_classes
        self._n_n_classes = n_n_classes

    def forward(self,
                image: torch.Tensor,
                text_features: torch.Tensor | None = None,
                clinical_feat: torch.Tensor | None = None) -> dict:
        # Task #27 — when deep_supervision is on AND we're in train mode, ask
        # the encoder for multi-scale logits.  FiLM hook is now wired in the
        # decoder (DeepSupSwinUNETR) but currently we pass film=None since the
        # seg_only v2 retrain runs with use_text=False.  Phase-3 multitask will
        # compute the per-stage gamma/beta tensors here and pass them through.
        seg_logits_ms = None
        if self.deep_supervision and self.training:
            seg_logits_ms = self.encoder(image, multiscale=True, film=None)
            seg_logits = seg_logits_ms[1]                                    # final-res
        else:
            seg_logits = self.encoder(image)

        # Encoder bottleneck for staging + prognosis pooling.
        # MONAI SwinUNETR exposes intermediate features via .swinViT — we re-run
        # the trunk only when needed. For the scaffold this is the placeholder
        # for the refactored forward that returns features alongside seg.
        feats = self.encoder.encode(image)
        bottleneck = feats[-1]                                 # [B, 16*feat, d, h, w]

        # Downsample seg_logits to bottleneck spatial resolution for masked pooling.
        seg_for_pool = torch.nn.functional.interpolate(
            seg_logits, size=bottleneck.shape[2:], mode="trilinear", align_corners=False)

        # Task #46: feed clinical_feat into tn_head along with mask-pooled bottleneck.
        tn_out = self.tn_head(bottleneck, seg_for_pool, clinical_feat=clinical_feat)

        # Prognosis input: concat of GTVp-pooled and GTVn-pooled bottleneck
        # features (image), the 18-d clinical tabular, AND softmax(t_logits) ‖
        # softmax(n_logits) — task #45, TN-softmax is a clinically-grounded
        # intermediate that should make Cox more interpretable.
        prob = torch.softmax(seg_for_pool, dim=1)
        feat_p = masked_global_pool(bottleneck, prob[:, 1:2])
        feat_n = masked_global_pool(bottleneck, prob[:, 2:3])
        image_feat = torch.cat([feat_p, feat_n], dim=-1)
        tn_softmax = torch.cat([
            torch.softmax(tn_out["t_logits"], dim=-1),
            torch.softmax(tn_out["n_logits"], dim=-1),
        ], dim=-1)
        risk = self.prog_head(image_feat, clinical_feat, tn_feat=tn_softmax)

        return {
            "seg_logits": seg_logits,
            "seg_logits_ms": seg_logits_ms,   # dict {1, 2, 4, 8} in train+DS, else None
            "t_logits": tn_out["t_logits"],
            "n_logits": tn_out["n_logits"],
            "risk": risk,
        }
