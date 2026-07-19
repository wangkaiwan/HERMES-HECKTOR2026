#!/bin/bash
# SurvLoss experiment: retrain the rfs_only deep RFS expert with the SIGMOID-CONCORDANCE
# loss (Kai's SurvLoss project, MEDAI_RFS_LOSS=sigmoid) instead of cox_efron_loss.
# Everything else IDENTICAL to the deployed rfs10 (medai_10foldmask_rfs_only_aug):
# rfs_only + augment, 10-fold-seg mask cache, p96, batch 8, default schedule.
# 5 folds, 2 concurrent on GPU 1 (GPU 0 in use). Distinct out_dir/log (naming discipline).
set -u
cd /home/kaiwang/project/HECKTOR_2026
PY=/home/kaiwang/.conda/envs/brainiac/bin/python
export MEDAI_CACHE_DIR=/data/kwang/medai_patch_cache_10fold
export MEDAI_RFS_LOSS=sigmoid                       # <-- the only change vs rfs10
GPU=1
NAME=rfs_sigmoid
i=0
for f in 0 1 2 3 4; do
  echo "[$(date '+%F %T')] launch $NAME fold$f on gpu$GPU"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_medai_hybrid.py \
      --fold "$f" --gpu 0 --rfs_mode rfs_only --augment \
      --out_dir "predictions/medai_10foldmask_${NAME}" \
      > "logs/medai_10foldmask_${NAME}_fold${f}.log" 2>&1 &
  i=$((i+1))
  if [ $((i % 2)) -eq 0 ]; then wait; fi     # 2 concurrent on one GPU
done
wait
echo "[$(date '+%F %T')] ALL rfs_sigmoid 5-fold DONE"
