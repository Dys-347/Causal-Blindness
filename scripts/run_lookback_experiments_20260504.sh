#!/usr/bin/env bash

DUET_DIR="${DUET_DIR:-/path/to/DUET-main}"
RUN_ID="lookback_20260504"
LOG_ROOT="${LOG_ROOT:-./causal_r1_runs/${RUN_ID}}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-time_mamba}"

mkdir -p "${LOG_ROOT}"

echo "[INFO] run_id=${RUN_ID}"
echo "[INFO] gpu=${GPU_ID}"
echo "[INFO] start=$(date '+%F %T')"

if [ -n "${CONDA_SH}" ] && [ -f "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

cd "${DUET_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

run_lookback() {
  local seq_len="$1"
  local batch_size="$2"
  local seed="20260504"
  local out_dir="causal_r1_duet/lookback_seq${seq_len}_seed${seed}_${RUN_ID}"
  local log_file="${LOG_ROOT}/duet_mix_seq${seq_len}.log"

  echo "[RUN] seq_len=${seq_len} batch_size=${batch_size}" | tee "${log_file}"
  python train_eval_duet_baseline_curve.py \
    --ci 0 \
    --seq-len "${seq_len}" \
    --pred-len 96 \
    --epochs 20 \
    --patience 5 \
    --batch-size "${batch_size}" \
    --lr 5e-4 \
    --loss MSE \
    --d-model 512 \
    --d-ff 512 \
    --n-heads 8 \
    --seed "${seed}" \
    --output-dir "${out_dir}" >> "${log_file}" 2>&1
  echo "[DONE] seq_len=${seq_len} status=$?" | tee -a "${log_file}"
}

run_lookback 48 32
run_lookback 96 32
run_lookback 192 16
run_lookback 336 8

echo "[INFO] finish=$(date '+%F %T')"
