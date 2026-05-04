#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUET_DIR="${DUET_DIR:-/path/to/DUET-main}"
RUN_ID="${RUN_ID:-v3_mechanism_family_20260504}"
LOG_ROOT="${LOG_ROOT:-./causal_r1_runs/${RUN_ID}}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-time_mamba}"
SEEDS=(${SEEDS:-20260503 20260504 20260505})
MECHANISMS=(${MECHANISMS:-linear_multi_lag nonlinear_sin})

mkdir -p "${LOG_ROOT}"

echo "[INFO] run_id=${RUN_ID}"
echo "[INFO] duet_dir=${DUET_DIR}"
echo "[INFO] gpu=${GPU_ID}"
echo "[INFO] mechanisms=${MECHANISMS[*]}"
echo "[INFO] seeds=${SEEDS[*]}"
echo "[INFO] start=$(date '+%F %T')"

if [ -n "${CONDA_SH}" ] && [ -f "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

cp "${SCRIPT_DIR}/synthetic_mechanism_utils.py" "${DUET_DIR}/synthetic_mechanism_utils.py"
cp "${SCRIPT_DIR}/generate_synthetic_mechanism_family.py" "${DUET_DIR}/generate_synthetic_mechanism_family.py"
cp "${SCRIPT_DIR}/train_eval_duet_synthetic_causal.py" "${DUET_DIR}/train_eval_duet_synthetic_causal.py"
cp "${SCRIPT_DIR}/train_eval_duet_crr_synthetic.py" "${DUET_DIR}/train_eval_duet_crr_synthetic.py"
cp "${SCRIPT_DIR}/train_eval_duet_baseline_curve.py" "${DUET_DIR}/train_eval_duet_baseline_curve.py"

cd "${DUET_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python generate_synthetic_mechanism_family.py \
  --output-dir "dataset/causal_r1_mechanism_family" \
  --mechanisms linear_one_lag linear_multi_lag nonlinear_sin \
  --n-steps 10000 \
  --n-distractors 19 \
  --noise-std 0.05 \
  --seed 20260504

run_one() {
  local mechanism="$1"
  local variant="$2"
  local seed="$3"
  local data_path="dataset/causal_r1_mechanism_family/synthetic_${mechanism}.csv"
  local out_dir="causal_r1_duet/${RUN_ID}/${mechanism}_${variant}_seed${seed}"
  local log_file="${LOG_ROOT}/${mechanism}_${variant}_seed${seed}.log"

  echo "[RUN] mechanism=${mechanism} variant=${variant} seed=${seed}" | tee "${log_file}"
  if [ "${variant}" = "baseline" ]; then
    python train_eval_duet_baseline_curve.py \
      --data-path "${data_path}" \
      --ci 0 \
      --epochs 20 \
      --patience 5 \
      --batch-size 32 \
      --lr 5e-4 \
      --loss MSE \
      --d-model 512 \
      --d-ff 512 \
      --n-heads 8 \
      --seed "${seed}" \
      --output-dir "${out_dir}" >> "${log_file}" 2>&1
  elif [ "${variant}" = "rir" ]; then
    python train_eval_duet_crr_synthetic.py \
      --data-path "${data_path}" \
      --ci 0 \
      --epochs 20 \
      --patience 5 \
      --batch-size 32 \
      --lr 5e-4 \
      --loss MSE \
      --d-model 512 \
      --d-ff 512 \
      --n-heads 8 \
      --lambda-resp 0.05 \
      --lambda-dist 0.005 \
      --selection-pred-weight 0.1 \
      --selection-dist-weight 0.05 \
      --seed "${seed}" \
      --output-dir "${out_dir}" >> "${log_file}" 2>&1
  else
    echo "[ERROR] Unknown variant: ${variant}" | tee -a "${log_file}"
    return 1
  fi
  echo "[DONE] mechanism=${mechanism} variant=${variant} seed=${seed}" | tee -a "${log_file}"
}

for mechanism in "${MECHANISMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "${mechanism}" baseline "${seed}"
    run_one "${mechanism}" rir "${seed}"
  done
done

echo "[INFO] finish=$(date '+%F %T')"
