# Trained weights

The HERMES container mounts its weights at `/opt/ml/model`. They are distributed as release
assets rather than in the repository, because they exceed git's file limits.

**Download:** see the [Releases](../../releases) page, tag `weights-v15`.

| asset | contents | size |
|---|---|---|
| `hermes-v15-segmentation.tar` | `seg_fold00.ckpt` … `seg_fold29.ckpt` (10 folds x top-3 STU-Net) + `config.yaml` | ~1.7 GB |
| `hermes-v15-staging-survival.tar` | `medai_p96_fold*`, `medai_p112_fold*` (multi-scale deep triplehead), `medai_rfs_fold*`, `medai_rfsaff_fold*`, `medai_rfssig_fold*` (deep RFS experts: Cox, affine-aug Cox, concordance-loss), `task23_radio.joblib` (fitted staging + survival ensemble config), `config.yaml` | ~0.8 GB |

Each archive carries a `SHA256SUMS`; each asset has a matching `.sha256` beside it. Verify
before use:

```bash
tar xf hermes-v15-segmentation.tar
cd hermes-v15-segmentation && sha256sum -c SHA256SUMS
```

Unpack both into one directory and point the container at it:

```bash
docker run --rm --gpus all \
  -v /path/to/weights:/opt/ml/model:ro \
  -v /path/to/input:/input:ro -v /path/to/output:/output \
  hermes:v15
```

## What these are

Checkpoints are inference weights: the AdamW optimiser state has been removed, which is why
a `seg_fold*.ckpt` here is 57 MB rather than the 168 MB written during training. Nothing the
inference path reads is affected -- it loads `state_dict` and takes its configuration from
`config.yaml`. `state_dict`, `hyper_parameters`, `epoch` and `global_step` are preserved.
Weights are fp32 and were not quantised, pruned or re-exported.

These are the weights behind the numbers in the paper. To rebuild them from scratch instead,
follow the "Reproduction" section of `README.md`.

## What each read-out actually is

Which weights a task uses is not guessable from the file names, and one of them surprises
people. The weights that decide each output, as shipped:

**Segmentation.** The 30 `seg_fold*.ckpt` are STU-Net Small, 10 folds by the top three
checkpoints of each, softmax-averaged with 6-way test-time augmentation. Post-processing is
a per-class connected-component filter (GTVp: components under 1000 mm3 dropped, top 2 kept;
GTVn: under 500 mm3 dropped, top 8 kept -- the filter runs in 1 mm isotropic space, so
voxels and mm3 are the same number) followed by a PET gate that drops any GTVn component
whose peak SUV is below 2.5. The gate is applied to the written mask only, not to the mask
the staging and survival features are computed from.

**T staging.** An equal-weight blend of the multi-scale deep triplehead and a logistic
regression on clinical variables plus GTVp geometry (`t_w_radio = 0.5`), restricted to
T1-T4.

**N staging.** A logistic regression on 13 clinical variables and 7 nodal-geometry features
measured from the predicted mask -- component count, largest and total volume, largest
diameter and their logs. `n_w_radio` is **1.0**, so the deep head contributes nothing to N
despite being loaded. The N model lives inside `task23_radio.joblib`, not in any checkpoint.

**Recurrence-free survival.** An equal-weight z-average of five experts: the multi-scale deep
triplehead, three further deep experts (`medai_rfs_*` Cox, `medai_rfsaff_*` affine-augmented
Cox, `medai_rfssig_*` trained with the concordance loss), and a LASSO-Cox on clinical plus
radiomic features. The z-normalisation statistics and the ensemble mode (`rfs_mode =
equal5`) are in `task23_radio.joblib`.

Scoring note: these are the full ensembles. The cross-validation figures in the paper are
out-of-fold, so do not reproduce them by running this release over the HECKTOR training set
-- any given training patient was seen by 27 of the 30 segmentation checkpoints. Rebuild the
fold assignment with `data/splits.py`, which is deterministic at `random_state=42`
(`--n_folds 10` for segmentation, `--n_folds 5` for staging and survival), and score each
patient with the folds that excluded it.

## Terms

The weights are released under the same MIT licence as this repository (see `LICENSE`), with
two things stated explicitly because a licence alone does not cover them:

- **Research use only.** These models are not a medical device, have not been validated for
  clinical use, and must not be used to inform patient care.
- **The training data is not ours to pass on.** The models were trained on the HECKTOR 2026
  training release. The weights are derived work and are covered by this licence, but the
  images and labels themselves remain under the challenge's own data agreement; obtain them
  from the organisers directly.

If you use the weights, please cite the paper and this repository (see `README.md`).
