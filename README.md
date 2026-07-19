# HERMES — HECKTOR 2026

**H**ybrid **E**nsemble for **R**adiotherapy-target segmentation, **M**alignancy staging, and **E**vent-free **S**urvival.

A single containerized algorithm for the three [HECKTOR 2026](https://hecktor26.grand-challenge.org/) subtasks on head-and-neck FDG-PET/CT:

1. **Segmentation** of the primary tumor (GTVp) and pathological lymph nodes (GTVn);
2. **T/N staging** (radiological, AJCC/UICC 7th edition);
3. **Recurrence-free survival (RFS)** prediction.

Team **AMC_HNC**. This repository releases the code for our challenge submission; see `paper/HERMES_paper_V2.8.pdf` for the accompanying paper.

---

## Method at a glance

A 10-fold **STU-Net Small** ensemble produces the segmentation; the predicted mask then drives two downstream tasks.

- **Staging** fuses a 3-D deep patch model (`DualHeadFusionResNet`), a radiomics + clinical model, and — our main staging contribution — **geometry features derived from the predicted mask** (nodal component count, largest-node size/extent, total nodal burden, primary-tumor extent) that align with the size/number axes of 7th-edition N/T staging. Replacing radiomics with geometry raises N-stage balanced accuracy by +0.030 (out-of-fold).
- **Survival** is an equal-weight ensemble of deep and clinical Cox experts, including one deep expert trained with a **concordance-tracking loss** (`training/losses.py:sigmoid_concordance_loss`) whose value approximates 1 − C-index during training.

All components were selected on honest out-of-fold cross-validation under an anti-overfitting protocol (strong regularization, equal-weight fusion, no tuning on the public validation set).

## Repository layout

```
models/     network architectures (STU-Net, DualHeadFusionResNet, heads)
training/   losses (incl. sigmoid_concordance_loss), metrics, trainer, MTLR
data/        preprocessing transforms and clinical encoding
docker/      inference.py (submission entry point) + helpers + Dockerfile
scripts/     training, joblib refits, ablations, and figure generation
configs/     segmentation training configs (YAML)
paper/       the accompanying paper + figures
```

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # or docker/requirements-docker.txt for the container
```

Python 3.9+, PyTorch 2.x (CUDA 12.x), MONAI, SimpleITK, scikit-learn, lifelines, scikit-survival.

## Trained weights

Model checkpoints (30 segmentation + deep staging/survival experts) and the fitted
`task23_radio.joblib` are **not** stored in git. See `MODELS.md` for download / access.

## Inference (challenge container)

```bash
docker build -f docker/Dockerfile -t hermes .
# Grand Challenge mounts CT/PET/EHR at /input and reads /output.
```

`docker/inference.py` is the submission entry point: preprocessing → segmentation
ensemble → predicted-mask-driven staging and survival. Weights mount at `/opt/ml/model`.

## Reproduction

- **Segmentation:** train STU-Net folds from the `configs/*.yaml`; deploy the top-3
  checkpoints per fold with test-time augmentation.
- **Task 2/3 deep experts:** `scripts/train_medai_hybrid.py`
  (`MEDAI_RFS_LOSS=sigmoid` selects the concordance-tracking loss; see
  `scripts/_launch_medai_rfs_sigmoid.sh`).
- **Fitted staging/survival model:** `scripts/refit_radio_v15.py`
  (v13 = nodal-geometry N-stage, v14 = + primary-geometry T-stage, v15 = + concordance
  survival expert).
- **Ablations / statistics:** `scripts/n_nodal_geom_probe.py`,
  `scripts/eval_rfs_sigmoid_oof.py`, `scripts/eval_t_geom_oof.py`,
  `scripts/postproc_meandice_oof.py`, `scripts/paper_stats_ci.py`.
- **Figures:** `scripts/make_paper_figures.py`, `scripts/make_fig2_overlays.py`.

## Citation

> [PLACEHOLDER — paper citation once available.] HERMES: A Hybrid Ensemble for
> Head-and-Neck Tumor Segmentation, TN Staging, and Recurrence-Free Survival on
> PET/CT. HECKTOR 2026 Challenge (MICCAI), team AMC_HNC.


## Notes

- Some research scripts under `scripts/` use absolute paths (e.g. dataset/cache
  locations) from our environment; adjust them to your setup. The submission
  entry point `docker/inference.py` reads inputs/outputs via the Grand Challenge
  contract and does not hard-code local paths.
- The release includes the full training/evaluation code used for the challenge,
  including some exploratory components beyond the final HERMES configuration
  (the deployed pipeline is the STU-Net segmentation ensemble + `DualHeadFusionResNet`
  staging/survival experts + geometry features + the concordance loss).

## License

MIT — see `LICENSE`.
