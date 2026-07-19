"""
PyTorch Lightning multitask trainer for HECKTOR 2026.

train step:
    - one forward + backward pass on a 192³ patch
    - per-task losses combined via Kendall uncertainty weighting
    - per-sample masks (has_seg / has_staging / has_survival) zero out terms
      when labels are missing

validation step:
    - sliding-window inference at the val ROI size (192³) on the full volume
    - per-patient: update DiceAggScore (the official leaderboard metric) and
      stash staging predictions + risk scores

on_validation_epoch_end:
    - compute Aggregated Dice (mean of GTVp + GTVn)
    - compute balanced accuracy for T and N
    - compute Harrell c-index over all val patients
    - compute composite val/score = 0.25*dice + 0.35*balacc_avg + 0.40*cindex
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytorch_lightning as pl
import SimpleITK as sitk
import torch

from models.multitask_model import HECKTORMultitaskModel
from training.losses import DiceCELoss, WeightedCE, cox_efron_loss
from training.uncertainty_weights import UncertaintyWeights
from training.metrics import (
    DiceAggScore, MeanDicePerPatient, AggregatedDetectionF1,
    balanced_accuracy, harrell_cindex, composite_score,
)


class HECKTORLightningModule(pl.LightningModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        m = cfg["model"]
        self.model = HECKTORMultitaskModel(
            in_channels=m["in_channels"],
            out_seg_channels=m["out_seg_channels"],
            img_size=tuple(cfg["data"]["roi_size"]),
            variant=m.get("variant", "B"),
            pretrained=m.get("pretrained_weights"),
            source=m.get("pretrained_source", "local"),
            use_text=m.get("use_text_conditioning", False),
            text_dim=m.get("text_dim", 768),
            text_dropout_prob=m.get("text_dropout_prob", 0.3),
            staging_hidden=m["staging"]["hidden_dim"],
            n_t_classes=m["staging"]["n_t_classes"],
            n_n_classes=m["staging"]["n_n_classes"],
            prognosis_hidden=m["prognosis"]["hidden_dim"],
            n_clinical=m["prognosis"]["n_clinical"],
            # Backbone / prog-method selector hooks (default keep current behavior;
            # alternatives raise NotImplementedError until tasks #17/#18 land).
            seg_backbone=m.get("seg_backbone", "voco"),
            prog_method=m.get("prognosis", {}).get("method", "cox"),
            # Task #27 — deep supervision (aux heads at 1/2, 1/4, 1/8).
            deep_supervision=m.get("deep_supervision", False),
        )

        # Task 2/3 Route C — freeze the seg backbone (encoder = full STU-Net) and
        # train only the TN + Cox heads on the bottleneck. Caller is expected to
        # have loaded a trained seg ckpt via `--init`.
        if cfg["training"].get("freeze_backbone", False):
            for p in self.model.encoder.parameters():
                p.requires_grad_(False)
            # Keep encoder in eval mode so dropout/BN-like stats don't drift.
            # IMPORTANT: Lightning will toggle .train() at epoch start; we
            # re-pin encoder to eval each step via the train()-override below.
            self.model.encoder.eval()
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.model.parameters())
            print(f"[freeze_backbone] encoder frozen — {n_trainable:,}/{n_total:,} "
                  f"params trainable ({100.0*n_trainable/max(n_total,1):.2f}%)",
                  flush=True)

        _sl = cfg["training"]["seg_loss"]
        self.seg_loss = DiceCELoss(
            dice_weight=_sl["dice_weight"],
            ce_weight=_sl["ce_weight"],
            # Focal term added ON TOP of Dice+CE. focal_weight=0 (default) = off.
            focal_weight=_sl.get("focal_weight", 0.0),
            focal_alpha=_sl.get("focal_alpha", 0.25),
            focal_gamma=_sl.get("focal_gamma", 2.0),
        )
        # Task #27 — DeepSupervisionDiceCELoss wraps the base seg loss; only
        # activated when both the model has deep_supervision on AND the model
        # returned multi-scale logits this step (= training mode).
        if m.get("deep_supervision", False):
            from training.losses import DeepSupervisionDiceCELoss
            self.seg_loss_ds = DeepSupervisionDiceCELoss(
                self.seg_loss,
                weights=_sl.get("deep_sup_weights", [1.0, 0.5, 0.25, 0.125]),
            )
        else:
            self.seg_loss_ds = None
        # Task #38 (2026-05-13): class-weighted CE for T/N. Weights computed from
        # the full training CSV (after task #36 drops T0). Sklearn-balanced
        # formula: weight_c = N / (K * count_c). Saves ~0.02-0.04 balanced-acc on
        # the imbalanced N (60% N2) and minor on T (T2 dominant). If the staging
        # block in cfg specifies explicit `t_weights` / `n_weights` (list of K
        # floats), those override the auto-computed ones — useful for ablation.
        t_w = self._compute_class_weights(
            csv_path=cfg["data"]["csv"],
            col="T-stage",
            mapping={"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T4A": 3, "T4B": 3},
            n_classes=m["staging"]["n_t_classes"],
            override=m["staging"].get("t_weights"),
        )
        n_w = self._compute_class_weights(
            csv_path=cfg["data"]["csv"],
            col="N-stage",
            mapping={"N0": 0, "N1": 1, "N2": 2, "N2A": 2, "N2B": 2, "N2C": 2, "N3": 3},
            n_classes=m["staging"]["n_n_classes"],
            override=m["staging"].get("n_weights"),
        )
        self.t_loss = WeightedCE(n_classes=m["staging"]["n_t_classes"], class_weights=t_w)
        self.n_loss = WeightedCE(n_classes=m["staging"]["n_n_classes"], class_weights=n_w)

        self.use_uncert = cfg["training"]["uncertainty_weighting"]
        if self.use_uncert:
            self.uw = UncertaintyWeights(["seg", "t", "n", "cox"])
        else:
            w = cfg["training"]["loss_weights"]
            self.scalar_w = {"seg": w["seg"], "t": w["staging"], "n": w["staging"],
                             "cox": w["prognosis"]}

        # Validation accumulators (reset each epoch)
        self._val_dice = DiceAggScore(class_labels=(1, 2))
        # HECKTOR 2026 official metrics (different from DiceAgg used in 2025):
        #   GTVp ranking = mean DSC across patients   → MeanDicePerPatient.Class_1
        #   GTVn ranking = Borda(DiceAgg, F1-detect)  → DiceAgg.Class_2 + AggregatedDetectionF1.f1
        self._val_mean_dsc = MeanDicePerPatient(class_labels=(1, 2))
        self._val_det_f1 = AggregatedDetectionF1(class_label=2)
        self._val_t_logits: List[torch.Tensor] = []
        self._val_n_logits: List[torch.Tensor] = []
        self._val_t_target: List[torch.Tensor] = []
        self._val_n_target: List[torch.Tensor] = []
        self._val_t_mask: List[torch.Tensor] = []
        self._val_n_mask: List[torch.Tensor] = []
        self._val_risk: List[float] = []
        self._val_time: List[float] = []
        self._val_event: List[float] = []
        self._val_surv_mask: List[bool] = []

    @staticmethod
    def _compute_class_weights(csv_path: str, col: str, mapping: dict,
                                n_classes: int, override=None) -> torch.Tensor:
        """Sklearn-balanced class weights from the training CSV.

        weight_c = N_total / (K * count_c)  for each class c that appears.
        Classes with zero count get weight 1.0 (neutral) — happens if the spec
        defines a class never used in the cohort.

        `override`: if not None, a length-`n_classes` list is returned verbatim
        as a tensor (ablation knob via cfg.model.staging.{t,n}_weights).

        Falls back to uniform weights when the CSV is unreadable — this is
        the case inside the inference docker container, which doesn't ship
        the training CSV. The loss term is only used at training time, so
        uniform weights at inference are harmless.
        """
        if override is not None:
            assert len(override) == n_classes
            return torch.tensor(list(override), dtype=torch.float32)

        import pandas as _pd
        try:
            df = _pd.read_csv(csv_path)
        except (FileNotFoundError, OSError, PermissionError) as e:
            # Inference container or test environment without the training CSV.
            # The class-weight tensor only matters for the staging CE loss, which
            # is only computed during training. At inference we just need ANY
            # valid tensor of the right shape so __init__ doesn't crash.
            print(f"[trainer] _compute_class_weights: CSV unavailable ({e}) — "
                  f"falling back to uniform weights for `{col}` (inference mode)",
                  flush=True)
            return torch.ones(n_classes, dtype=torch.float32)
        if col not in df.columns:
            return torch.ones(n_classes, dtype=torch.float32)
        counts = [0] * n_classes
        for v in df[col]:
            if _pd.isna(v):
                continue
            k = str(v).strip().upper()
            if k in mapping:
                counts[mapping[k]] += 1
        total = sum(counts)
        if total == 0:
            return torch.ones(n_classes, dtype=torch.float32)
        weights = []
        for c in counts:
            weights.append(total / (n_classes * c) if c > 0 else 1.0)
        return torch.tensor(weights, dtype=torch.float32)

    # ── train ─────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        out = self.model(batch["image"],
                         text_features=batch.get("text_features"),
                         clinical_feat=batch.get("clinical_feat"))

        losses: Dict[str, torch.Tensor] = {}
        # Skip the seg loss entirely when its weight is 0 (Route C frozen-encoder
        # training): the encoder has requires_grad=False so seg_logits has no
        # grad_fn; multiplying its loss by 0 leaves a no-grad term that breaks
        # backward when it's the ONLY surviving component in a batch (e.g. one
        # patient with NaN T/N/RFS).
        seg_w = self.scalar_w.get("seg", 1.0) if not self.use_uncert else 1.0
        if seg_w > 0 and batch.get("has_seg", torch.tensor([True])).any():
            # Task #27 — if the model emitted multi-scale logits (deep_supervision
            # on, training mode), use the DS wrapper; else fall back to plain loss.
            if self.seg_loss_ds is not None and out.get("seg_logits_ms") is not None:
                losses["seg"] = self.seg_loss_ds(out["seg_logits_ms"], batch["label"])
            else:
                losses["seg"] = self.seg_loss(out["seg_logits"], batch["label"])
        if batch.get("has_staging", torch.tensor([True])).any():
            losses["t"] = self.t_loss(out["t_logits"], batch["t_stage"], batch.get("has_staging"))
            losses["n"] = self.n_loss(out["n_logits"], batch["n_stage"], batch.get("has_staging"))
        if batch.get("has_survival", torch.tensor([True])).any():
            # Cox partial likelihood uses logcumsumexp, whose backward is not
            # implemented for float16 (PyTorch limitation). With AMP + batch≥2,
            # this raises NotImplementedError. Disable AMP for the Cox compute
            # only (a single scalar — fp32 overhead negligible). bs=1 happened
            # to work because the single-sample partial likelihood is degenerate
            # and short-circuits the backward.
            with torch.cuda.amp.autocast(enabled=False):
                losses["cox"] = cox_efron_loss(out["risk"].float(), batch["rfs_days"],
                                               batch["relapse"], batch.get("has_survival"))

        if self.use_uncert:
            total = self.uw(losses)
        else:
            total = sum(self.scalar_w[k] * v for k, v in losses.items()
                        if self.scalar_w.get(k, 0.0) > 0)

        # Defensive: if EVERY loss got filtered out (e.g. a bs=1 batch where the
        # single patient has NaN T/N/RFS AND the seg loss is also disabled), tie
        # a 0-tensor to a trainable param so backward() has something to walk.
        if not isinstance(total, torch.Tensor) or total.grad_fn is None:
            # First trainable param — its 0-coef contribution preserves grad_fn.
            trainable = next((p for p in self.parameters() if p.requires_grad), None)
            if trainable is not None:
                total = trainable.sum() * 0.0

        for k, v in losses.items():
            self.log(f"train/loss_{k}", v, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/loss", total, prog_bar=True, on_step=False, on_epoch=True)
        return total

    # ── validation: sliding-window inference + DiceAgg ────────────────────────

    def on_validation_epoch_start(self) -> None:
        self._val_dice.reset()
        self._val_mean_dsc.reset()
        self._val_det_f1.reset()
        self._val_t_logits = []
        self._val_n_logits = []
        self._val_t_target = []
        self._val_n_target = []
        self._val_t_mask = []
        self._val_n_mask = []
        self._val_risk = []
        self._val_time = []
        self._val_event = []
        self._val_surv_mask = []
        # Val loss accumulators (per-batch, weighted by batch size).
        self._val_loss_seg_sum = 0.0
        self._val_loss_t_sum = 0.0
        self._val_loss_n_sum = 0.0
        self._val_loss_cox_sum = 0.0
        self._val_n_seg = 0
        self._val_n_stage = 0
        self._val_n_surv = 0

    def validation_step(self, batch, batch_idx):
        # The val transforms pad/crop every patient to data.cache_volume (200³),
        # so a single full-volume forward is feasible on a 32 GB GPU and keeps
        # all the heads in one pass. Submission-time inference uses
        # sliding_window_inference (see scripts/inference.py).
        with torch.no_grad():
            out = self.model(batch["image"],
                             text_features=batch.get("text_features"),
                             clinical_feat=batch.get("clinical_feat"))
        seg_logits = out["seg_logits"]

        pred_mask = seg_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        gt_mask = batch["label"].squeeze().detach().cpu().numpy().astype(np.uint8)
        pr_img = sitk.GetImageFromArray(pred_mask)
        gt_img = sitk.GetImageFromArray(gt_mask)
        self._val_dice.update(pr_img, gt_img)
        # HECKTOR 2026 official metrics — added 2026-05-14.
        self._val_mean_dsc.update(pr_img, gt_img)
        self._val_det_f1.update(pr_img, gt_img)

        # Stash for epoch-end metrics
        self._val_t_logits.append(out["t_logits"].detach().cpu())
        self._val_n_logits.append(out["n_logits"].detach().cpu())
        self._val_t_target.append(batch["t_stage"].detach().cpu())
        self._val_n_target.append(batch["n_stage"].detach().cpu())
        self._val_t_mask.append(batch["has_staging"].detach().cpu())
        self._val_n_mask.append(batch["has_staging"].detach().cpu())
        self._val_risk.append(out["risk"].detach().cpu().item())
        self._val_time.append(batch["rfs_days"].detach().cpu().item())
        self._val_event.append(batch["relapse"].detach().cpu().item())
        self._val_surv_mask.append(bool(batch["has_survival"].detach().cpu().item()))

        # Validation losses — mirror training_step so users can watch
        # train↔val gap (overfitting diagnostic). All under no_grad already.
        bs = int(batch["image"].shape[0])
        seg_w = self.scalar_w.get("seg", 1.0) if not self.use_uncert else 1.0
        if seg_w > 0 and batch.get("has_seg", torch.tensor([True])).any():
            try:
                lseg = self.seg_loss(seg_logits, batch["label"]).item()
                self._val_loss_seg_sum += lseg * bs; self._val_n_seg += bs
            except Exception:                                          # noqa: BLE001
                pass
        if batch.get("has_staging", torch.tensor([True])).any():
            try:
                lt = self.t_loss(out["t_logits"], batch["t_stage"],
                                 batch.get("has_staging")).item()
                ln = self.n_loss(out["n_logits"], batch["n_stage"],
                                 batch.get("has_staging")).item()
                self._val_loss_t_sum += lt * bs
                self._val_loss_n_sum += ln * bs
                self._val_n_stage += bs
            except Exception:                                          # noqa: BLE001
                pass
        # Cox loss is NOT computed per-batch in validation because val uses
        # bs=1 (data/dataset.py hardcodes) → single-sample partial likelihood
        # is degenerate (no risk pairs). It is computed ONCE in
        # on_validation_epoch_end on the accumulated risks.

    def on_validation_epoch_end(self) -> None:
        # ── Seg metrics ────────────────────────────────────────────────────────
        # We compute three distinct metrics that together cover the HECKTOR 2026
        # official Borda-of-Borda ranking. Naming uses the metric's flavor in
        # the prefix so wandb/tensorboard split them cleanly.
        #
        #   val/diceagg_class_{1,2}  : dataset-aggregated DSC (HECKTOR 2025 metric, our legacy)
        #   val/meandsc_class_{1,2}  : per-patient mean DSC  (HECKTOR 2026 GTVp official)
        #   val/f1_class_2           : aggregated F1 over GTVn lesion CCs (HECKTOR 2026 GTVn detection official)
        #   val/borda_score          : composite proxy for the 2026 Borda ranking — equally weights
        #                              the three components that drive final rank.
        diceagg = self._val_dice.compute()
        meandsc = self._val_mean_dsc.compute()
        det     = self._val_det_f1.compute()

        for k, v in diceagg.items():
            if k.lower() == "mean":
                continue
            self.log(f"val/diceagg_{k.lower()}", float(v), prog_bar=False)
        self.log("val/diceagg_mean", float(diceagg.get("mean", 0.0)), prog_bar=False)

        for k, v in meandsc.items():
            if k.lower() == "mean":
                continue
            self.log(f"val/meandsc_{k.lower()}", float(v), prog_bar=True)
        self.log("val/meandsc_mean", float(meandsc.get("mean", 0.0)), prog_bar=False)

        self.log("val/f1_class_2", float(det["f1"]), prog_bar=True)
        self.log("val/f1_precision", float(det["precision"]), prog_bar=False)
        self.log("val/f1_recall",    float(det["recall"]),    prog_bar=False)

        # HECKTOR 2026 Borda proxy: average rank-driving signals on a 0-1 scale.
        #   GTVp_part = mean DSC of class 1 (the actual 2026 GTVp metric)
        #   GTVn_part = mean(DiceAgg class 2, F1 class 2) — proxy for the GTVn Borda sub-rank
        gtvp_part = float(meandsc.get("Class_1", 0.0))
        gtvn_part = 0.5 * float(diceagg.get("Class_2", 0.0)) + 0.5 * float(det["f1"])
        borda_proxy = 0.5 * gtvp_part + 0.5 * gtvn_part
        self.log("val/borda_score", borda_proxy, prog_bar=True)

        # Legacy `val/dice_mean` kept for backward-compat with old ModelCheckpoint
        # configs (configs/seg_only.yaml still says monitor: val/dice_mean). When
        # we switch monitor to val/borda_score in a future config update, the
        # legacy alias can be retired.
        dice_mean = float(diceagg.get("mean", 0.0))
        self.log("val/dice_mean", dice_mean, prog_bar=True)

        # Staging: balanced accuracy on valid samples only
        if self._val_t_logits:
            t_logits = torch.cat(self._val_t_logits, dim=0)
            n_logits = torch.cat(self._val_n_logits, dim=0)
            t_targ = torch.cat(self._val_t_target, dim=0)
            n_targ = torch.cat(self._val_n_target, dim=0)
            t_mask = torch.cat(self._val_t_mask, dim=0).bool()
            n_mask = torch.cat(self._val_n_mask, dim=0).bool()
            balacc_t = balanced_accuracy(t_logits[t_mask], t_targ[t_mask],
                                         self.cfg["model"]["staging"]["n_t_classes"]) \
                if t_mask.any() else 0.0
            balacc_n = balanced_accuracy(n_logits[n_mask], n_targ[n_mask],
                                         self.cfg["model"]["staging"]["n_n_classes"]) \
                if n_mask.any() else 0.0
            self.log("val/balacc_t", float(balacc_t), prog_bar=False)
            self.log("val/balacc_n", float(balacc_n), prog_bar=False)
        else:
            balacc_t = balacc_n = 0.0

        # Prognosis: Harrell c-index
        if self._val_risk:
            mask = np.array(self._val_surv_mask, dtype=bool)
            if mask.sum() >= 2:
                risk = np.array(self._val_risk)[mask]
                time = np.array(self._val_time)[mask]
                event = np.array(self._val_event)[mask]
                cidx = harrell_cindex(risk, time, event)
            else:
                cidx = 0.0
        else:
            cidx = 0.0
        self.log("val/cindex", float(cidx), prog_bar=False)

        # Composite leaderboard score
        score = composite_score(dice_mean, balacc_t, balacc_n, cidx)
        self.log("val/score", float(score), prog_bar=True)

        # ── Validation losses ─────────────────────────────────────────────
        # T/N CE losses are accumulated per-batch (single-sample CE is well
        # defined). Cox is computed ONCE here over the accumulated val risks
        # (val uses bs=1 → per-batch Cox is degenerate).
        v_loss_seg = self._val_loss_seg_sum / max(self._val_n_seg, 1)
        v_loss_t = self._val_loss_t_sum / max(self._val_n_stage, 1)
        v_loss_n = self._val_loss_n_sum / max(self._val_n_stage, 1)
        v_loss_cox = 0.0
        if self._val_risk:
            mask_s = np.array(self._val_surv_mask, dtype=bool)
            if mask_s.sum() >= 2:
                try:
                    risk_t = torch.tensor(
                        np.array(self._val_risk)[mask_s], dtype=torch.float32)
                    time_t = torch.tensor(
                        np.array(self._val_time)[mask_s], dtype=torch.float32)
                    event_t = torch.tensor(
                        np.array(self._val_event)[mask_s], dtype=torch.float32)
                    surv_t = torch.ones_like(event_t, dtype=torch.bool)
                    v_loss_cox = float(cox_efron_loss(
                        risk_t, time_t, event_t, surv_t).item())
                except Exception:                                       # noqa: BLE001
                    v_loss_cox = 0.0
        if self._val_n_seg:    self.log("val/loss_seg", float(v_loss_seg), prog_bar=False)
        if self._val_n_stage:  self.log("val/loss_t",   float(v_loss_t),   prog_bar=False)
        if self._val_n_stage:  self.log("val/loss_n",   float(v_loss_n),   prog_bar=False)
        if self._val_risk:     self.log("val/loss_cox", float(v_loss_cox), prog_bar=False)
        # Total val loss using the SAME scalar weights as training_step (so
        # train/loss and val/loss are directly comparable).
        if not self.use_uncert:
            v_loss_total = (self.scalar_w["seg"]   * v_loss_seg +
                            self.scalar_w["t"]     * v_loss_t   +
                            self.scalar_w["n"]     * v_loss_n   +
                            self.scalar_w["cox"]   * v_loss_cox)
            self.log("val/loss", float(v_loss_total), prog_bar=False)

        # Echo task-level breakdown to stdout each val epoch (helps watching
        # multitask training without round-tripping through wandb).
        print(f"[val] ep={self.current_epoch} | dice={dice_mean:.4f} "
              f"| balacc_t={balacc_t:.4f} balacc_n={balacc_n:.4f} "
              f"| cindex={cidx:.4f} | score={float(score):.4f} "
              f"| val_loss seg={v_loss_seg:.3f} t={v_loss_t:.3f} "
              f"n={v_loss_n:.3f} cox={v_loss_cox:.3f}",
              flush=True)

    # ── optim ─────────────────────────────────────────────────────────────────

    def train(self, mode: bool = True):
        """Override to keep the encoder pinned in eval() when frozen."""
        super().train(mode)
        if self.cfg["training"].get("freeze_backbone", False):
            self.model.encoder.eval()
        return self

    def configure_optimizers(self):
        cfg = self.cfg["training"]
        # Only collect trainable params. With freeze_backbone=True the encoder
        # has requires_grad=False; the AdamW state dict would otherwise allocate
        # zeroed moments for those tensors which is wasteful but harmless.
        text_params = [p for n, p in self.named_parameters() if ".film." in n and p.requires_grad]
        other = [p for n, p in self.named_parameters() if ".film." not in n and p.requires_grad]
        groups: List[Dict[str, Any]] = [{"params": other, "lr": cfg["lr"]}]
        if text_params:
            groups.append({"params": text_params, "lr": cfg["text_lr"]})
        opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])

        # Task #27 — ReduceLROnPlateau is per-fold-adaptive (drops LR when borda
        # plateaus).  Diagnosis from #24: fixed cosine(300ep) + EarlyStop(50ep) meant
        # fold0/2/4 plateaued at LR still ~8-9e-5 → under-converged.  Plateau scheduler
        # fixes this. Plain cosine still default for backward compat.
        sched_name = cfg.get("scheduler", "cosine").lower()
        if sched_name == "plateau":
            p = cfg.get("scheduler_plateau", {}) or {}
            monitor = cfg.get("monitor_metric", "val/borda_score")
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max",
                factor=p.get("factor", 0.5),
                patience=p.get("patience", 5),      # in val-checks (= 10 epochs at check_val_every_2)
                threshold=cfg.get("min_delta", 1.0e-4),  # 2026-06-16: abs min improvement to count
                threshold_mode="abs",                    # 0.001 absolute (not the default 1e-4 relative)
                min_lr=p.get("min_lr", 1.0e-7),
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": monitor,
                    "interval": "epoch",
                    "frequency": cfg.get("check_val_every_n_epoch", 1),
                },
            }
        # Default: cosine.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["max_epochs"], eta_min=cfg["lr_min"])
        return {"optimizer": opt, "lr_scheduler": sched}

    # ── LR warmup (linear, step-based) ────────────────────────────────────────
    # The config field `warmup_epochs` was historically wired into our scheduler
    # but never actually applied — both `CosineAnnealingLR` and `ReduceLROnPlateau`
    # start at peak LR.  This bites Adam/AdamW (initial `v` estimate is 0 → first
    # updates can be huge) and is especially rough on the VoCo-pretrained encoder
    # because the random TN/Cox/aux heads emit large noisy gradients into the
    # backbone before warmup would normally damp them.  Manual linear warmup
    # via optimizer_step is the simplest fix that composes with both cosine and
    # ReduceLROnPlateau (SequentialLR doesn't compose with the latter cleanly).
    #
    # After global_step >= warmup_iters the override stops, leaving the main
    # scheduler to drive LR (cosine decay or plateau reductions).  Resume from
    # mid-warmup ckpts works because Lightning restores `global_step` correctly.

    def on_train_start(self) -> None:
        warmup_epochs = self.cfg["training"].get("warmup_epochs", 0)
        iters_per_epoch = int(self.trainer.num_training_batches or 0)
        self._warmup_iters = int(warmup_epochs * iters_per_epoch) if iters_per_epoch > 0 else 0
        self._base_lrs = [pg["lr"] for pg in self.trainer.optimizers[0].param_groups]
        if self._warmup_iters > 0:
            print(f"[warmup] linear LR warmup ON: {warmup_epochs} epochs × "
                  f"{iters_per_epoch} iters = {self._warmup_iters} steps; "
                  f"base LRs {[f'{lr:.1e}' for lr in self._base_lrs]}",
                  flush=True)
        else:
            print("[warmup] OFF (warmup_epochs=0 or unknown dataloader length)",
                  flush=True)

    def optimizer_step(self, *args, **kwargs):
        wi = getattr(self, "_warmup_iters", 0)
        if wi > 0 and self.trainer.global_step < wi:
            scale = (self.trainer.global_step + 1) / wi
            for pg, base in zip(self.trainer.optimizers[0].param_groups, self._base_lrs):
                pg["lr"] = base * scale
        super().optimizer_step(*args, **kwargs)
