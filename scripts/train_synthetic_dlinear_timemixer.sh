#!/usr/bin/env bash
set -euo pipefail

cd "${TSL_DIR:-/path/to/Time-Series-Library}"
mkdir -p logs/causal_r1

COMMON_ARGS=(
  --task_name long_term_forecast
  --is_training 1
  --root_path ./dataset/
  --data_path synthetic_multivariate.csv
  --data SyntheticMulti
  --features M
  --target Target_Var
  --seq_len 96
  --label_len 48
  --pred_len 96
  --e_layers 2
  --d_layers 1
  --factor 3
  --enc_in 21
  --dec_in 21
  --c_out 21
  --d_model 128
  --d_ff 256
  --n_heads 4
  --batch_size 32
  --train_epochs 10
  --patience 3
  --learning_rate 0.0001
  --num_workers 0
  --itr 1
  --des Exp_Causal_Check
)

run_model() {
  local model_name="$1"
  local model_id="$2"
  local log_file="logs/causal_r1/train_${model_name}_SyntheticMulti_$(date +%Y%m%d_%H%M%S).log"
  echo "===== Training ${model_name}; log=${log_file} ====="
  CUDA_VISIBLE_DEVICES=0 python -u run.py \
    "${COMMON_ARGS[@]}" \
    --model_id "${model_id}" \
    --model "${model_name}" \
    2>&1 | tee "${log_file}"
}

run_model DLinear Synthetic_Multi_DLinear_96_96
run_model TimeMixer Synthetic_Multi_TimeMixer_96_96

echo "===== Finished DLinear and TimeMixer SyntheticMulti training ====="
