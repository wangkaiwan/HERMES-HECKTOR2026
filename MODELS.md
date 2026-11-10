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
