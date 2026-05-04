#!/usr/bin/env bash
set -euo pipefail

cd "${TSL_DIR:-/path/to/Time-Series-Library}"
mkdir -p logs/causal_r1

log_file="logs/causal_r1/train_TimeMixer_SyntheticMulti_fixed_$(date +%Y%m%d_%H%M%S).log"
echo "===== Training TimeMixer SyntheticMulti; log=${log_file} ====="

CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path synthetic_multivariate.csv \
  --model_id Synthetic_Multi_TimeMixer_96_96 \
  --model TimeMixer \
  --data SyntheticMulti \
  --features M \
  --target Target_Var \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 16 \
  --d_ff 32 \
  --n_heads 4 \
  --batch_size 32 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate 0.01 \
  --num_workers 0 \
  --itr 1 \
  --des Exp_Causal_Check \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  2>&1 | tee "${log_file}"

echo "===== Finished TimeMixer SyntheticMulti training ====="
