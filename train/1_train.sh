#!/usr/bin/env bash

# ============================================================
# RRWNet Training Script
# ============================================================
# Supported datasets: RITE-train, LES-AV, HRF-Karlsson-w1024
# Supported --base_criterion: BCE3Loss (default), BCE3wminmaxLoss
# Supported --add_criterion: None (default), SpCLLoss
# ============================================================

DATA=/mnt/nasv3/zs/RRWNet_zs/train/_Data

# ---- RITE-train --------------------------------------------------
# python3 train.py --dataset RITE-train --data_folder "$DATA" \
#   --version RITE_rrwnet_bce3 --gpu_id 0

# python3 train.py --dataset RITE-train --data_folder "$DATA" \
#   --version RITE_rrwnet_minmax --gpu_id 0 \
#   --base_criterion BCE3wminmaxLoss

# python3 train.py --dataset RITE-train --data_folder "$DATA" \
#   --version RITE_rrwnet_minmax_spcl --gpu_id 0 \
#   --base_criterion BCE3wminmaxLoss --add_criterion SpCLLoss

# ---- LES-AV ------------------------------------------------------
# python3 train.py --dataset LES-AV --data_folder "$DATA" \
#   --version LES_rrwnet_bce3 --gpu_id 0

# python3 train.py --dataset LES-AV --data_folder "$DATA" \
#   --version LES_rrwnet_minmax --gpu_id 0 \
#   --base_criterion BCE3wminmaxLoss

python3 train.py --dataset LES-AV --data_folder "$DATA" \
  --version LES_rrwnet_minmax_spcl --gpu_id 0 \
  --base_criterion BCE3wminmaxLoss --add_criterion SpCLLoss

# ---- HRF-Karlsson-w1024 ------------------------------------------
# python3 train.py --dataset HRF-Karlsson-w1024 --data_folder "$DATA" \
#   --version HRF_rrwnet_bce3 --gpu_id 0

# python3 train.py --dataset HRF-Karlsson-w1024 --data_folder "$DATA" \
#   --version HRF_rrwnet_minmax --gpu_id 0 \
#   --base_criterion BCE3wminmaxLoss

# python3 train.py --dataset HRF-Karlsson-w1024 --data_folder "$DATA" \
#   --version HRF_rrwnet_minmax_spcl --gpu_id 0 \
#   --base_criterion BCE3wminmaxLoss --add_criterion SpCLLoss

# ---- Resume from checkpoint (any dataset) ------------------------
# python3 train.py --dataset RITE-train --data_folder "$DATA" \
#   --version RITE_rrwnet_resume --gpu_id 0 \
#   --use_checkpoint 1 \
#   --checkpoint_path /path/to/checkpoint_12015.pth \
#   --scheduler_path /path/to/scheduler_12015.pth
