"""MEDAI-hybrid architecture ported from the team's other-server report.

DualHeadFusionResNet: 3D ResNet-18 over CT+PET (2-ch) + MaskBranch (one-hot of
predicted GTVp/GTVn mask processed by tiny CNN, fused early to conv1) +
clinical MLP fusion + T/N classification heads. Optional RFS head for joint
Task 2+3 training.

Input shape per sample: image (2, 96, 96, 96), mask (1, 96, 96, 96)
                        (mask is integer 0/1/2; we one-hot to 2 ch internally)
Clinical input: (clinical_dim,) per sample.

Output dict: {'t_logits': [B, n_t], 'n_logits': [B, n_n], 'risk': [B, 1]}
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ResNet


class MaskBranch(nn.Module):
    """Tiny CNN that processes the predicted mask one-hot and produces a
    feature map matching the backbone's conv1 output, to be added element-wise."""

    def __init__(self, out_channels: int = 64):
        super().__init__()
        # 2-channel input (one-hot of GTVp, GTVn classes)
        self.conv = nn.Sequential(
            nn.Conv3d(2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, mask_int: torch.Tensor) -> torch.Tensor:
        # mask_int: (B, 1, D, H, W) integer 0/1/2 (or float)
        m = mask_int.long().squeeze(1)                                # (B, D, H, W)
        # One-hot 3 classes, drop background → 2 channels
        oh = F.one_hot(m.clamp(min=0), num_classes=3)                 # (B, D, H, W, 3)
        oh = oh.permute(0, 4, 1, 2, 3).float()                        # (B, 3, D, H, W)
        oh = oh[:, 1:]                                                # drop bg → (B, 2, D, H, W)
        return self.conv(oh)


class _ResNet18Backbone(nn.Module):
    """MONAI ResNet wrapper that exposes conv1 features (for MaskBranch
    early fusion) AND the global pooled output (for the classifier).

    We can't simply use monai.ResNet's forward because it goes straight from
    input → conv1 → bn1 → layers → avgpool → fc. We need to inject the
    MaskBranch feature ADDITION right after conv1 (before layer1).
    """

    def __init__(self, in_channels: int = 2, base_planes: int = 32,
                 mask_branch: bool = True):
        super().__init__()
        self.net = ResNet(
            block="basic",
            layers=[2, 2, 2, 2],
            block_inplanes=[base_planes, base_planes * 2,
                            base_planes * 4, base_planes * 8],
            spatial_dims=3,
            n_input_channels=in_channels,
            act="PRELU",
        )
        # Replace the FC head with identity so .net(x) returns the pooled vector
        # before classification.
        self.feat_dim = base_planes * 8                                # 256 for base_planes=32
        self.net.fc = nn.Identity()
        self.mask_branch = MaskBranch(out_channels=base_planes) if mask_branch else None
        # The conv1 output channel in MONAI ResNet equals base_planes (= 32).
        # We add the MaskBranch feature element-wise to it.

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        n = self.net
        # Manual forward — MONAI ResNet exposes the same layer names.
        x = n.conv1(image)
        x = n.bn1(x)
        x = n.act(x) if hasattr(n, "act") else F.relu(x, inplace=True)
        if self.mask_branch is not None and mask is not None:
            m_feat = self.mask_branch(mask)                            # (B, C=base_planes, D, H, W)
            # Spatial sizes should match — conv1 stride=1 (default in monai.ResNet
            # for 3D). If they don't, interpolate.
            if m_feat.shape[2:] != x.shape[2:]:
                m_feat = F.interpolate(m_feat, size=x.shape[2:], mode="trilinear",
                                        align_corners=False)
            x = x + m_feat
        # maxpool then layers
        if hasattr(n, "maxpool"):
            x = n.maxpool(x)
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        x = n.layer4(x)
        # Global pool
        x = n.avgpool(x)
        x = torch.flatten(x, 1)                                       # (B, feat_dim)
        return x


class DualHeadFusionResNet(nn.Module):
    """Image (CT+PET) + MaskBranch + Clinical → T/N classification heads.
    Optional RFS head for joint Task 3 training."""

    def __init__(self,
                 image_channels: int = 2,
                 clinical_dim: int = 18,
                 n_t_classes: int = 5,
                 n_n_classes: int = 4,
                 hidden_dim: int = 128,
                 dropout: float = 0.1,
                 mask_branch: bool = True,
                 with_rfs: bool = False,
                 rfs_loss: str = "cox",
                 mtlr_bins: int = 10):
        super().__init__()
        self.backbone = _ResNet18Backbone(in_channels=image_channels,
                                           mask_branch=mask_branch)
        feat_dim = self.backbone.feat_dim
        self.with_rfs = with_rfs
        # rfs_loss = 'cox' → scalar risk head (1 output);
        #            'mtlr' → MTLR phi head (mtlr_bins-1 outputs).
        self.rfs_loss = rfs_loss
        self.mtlr_bins = mtlr_bins

        # Fusion: image_feat (feat_dim) + clinical (clinical_dim)
        fusion_in = feat_dim + clinical_dim
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(fusion_in),
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.t_head = nn.Linear(hidden_dim, n_t_classes)
        self.n_head = nn.Linear(hidden_dim, n_n_classes)

        if with_rfs:
            # RFS head(s): shared_feat ⊕ softmax(T) ⊕ softmax(N) ⊕ clinical → risk.
            # 'cox'  → scalar-risk head; 'mtlr' → (bins-1) phi head;
            # 'both' → BOTH heads (one forward, two risks → ensemble at inference).
            rfs_in = hidden_dim + n_t_classes + n_n_classes + clinical_dim

            def _head(out_dim):
                return nn.Sequential(
                    nn.Linear(rfs_in, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(64, out_dim),
                )
            if rfs_loss in ("cox", "both", "sigmoid"):     # sigmoid = SurvLoss, same scalar-risk head as cox
                self.rfs_head = _head(1)
            if rfs_loss in ("mtlr", "both"):
                self.mtlr_head = _head(mtlr_bins - 1)

    def forward(self, image: torch.Tensor, mask: torch.Tensor,
                clinical: torch.Tensor, detach_tn_for_rfs: bool = True
                ) -> dict:
        img_feat = self.backbone(image, mask)                        # (B, feat_dim)
        fused_in = torch.cat([img_feat, clinical], dim=-1)
        shared = self.fusion(fused_in)                               # (B, hidden_dim)
        t_logits = self.t_head(shared)
        n_logits = self.n_head(shared)
        out = {"t_logits": t_logits, "n_logits": n_logits}
        if self.with_rfs:
            t_soft = F.softmax(t_logits, dim=-1)
            n_soft = F.softmax(n_logits, dim=-1)
            if detach_tn_for_rfs:
                t_soft = t_soft.detach()
                n_soft = n_soft.detach()
            rfs_in = torch.cat([shared, t_soft, n_soft, clinical], dim=-1)
            if self.rfs_loss in ("cox", "both", "sigmoid"):
                out["risk"] = self.rfs_head(rfs_in)                 # (B, 1)
            if self.rfs_loss in ("mtlr", "both"):
                out["mtlr_phi"] = self.mtlr_head(rfs_in)            # (B, mtlr_bins-1)
        return out


__all__ = ["DualHeadFusionResNet"]
