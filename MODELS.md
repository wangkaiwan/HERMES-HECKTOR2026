# Trained weights

The HERMES container mounts its weights at `/opt/ml/model`. These are distributed
separately from the source (they exceed git limits):

- `seg_fold00.ckpt` … `seg_fold29.ckpt` — 30 STU-Net segmentation checkpoints (10 folds × top-3).
- `medai_p96_fold*.ckpt`, `medai_p112_fold*.ckpt` — multi-scale deep triplehead (Task 2/3).
- `medai_rfs_fold*.ckpt`, `medai_rfsaff_fold*.ckpt`, `medai_rfssig_fold*.ckpt` — deep RFS experts
  (Cox, affine-aug Cox, and the concordance-loss expert).
- `task23_radio.joblib` — fitted staging (N geometry / T geometry) + survival ensemble config.
- `config.yaml` — runtime config.

**Access:** the trained weights are available on request. Please contact Kai Wang (kai.2.wang@cuanschutz.edu).

To reproduce the weights from scratch, follow the "Reproduction" section of `README.md`.
