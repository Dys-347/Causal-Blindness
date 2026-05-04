#!/usr/bin/env bash

TSL_DIR="${TSL_DIR:-/path/to/Time-Series-Library}"
DUET_DIR="${DUET_DIR:-/path/to/DUET-main}"
RUN_ID="priority_20260504"
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

run_bg() {
  local name="$1"
  shift
  local log_file="${LOG_ROOT}/${name}.log"
  echo "[LAUNCH] ${name}" >&2
  echo "[CMD] $*" > "${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    "$@"
  ) >> "${log_file}" 2>&1 &
  echo $!
}

wait_group() {
  local group_name="$1"
  shift
  local pids=("$@")
  echo "[WAIT] ${group_name}: ${pids[*]}"
  local failed=0
  for pid in "${pids[@]}"; do
    if wait "${pid}"; then
      echo "[DONE] pid=${pid}"
    else
      echo "[FAIL] pid=${pid}"
      failed=1
    fi
  done
  echo "[GROUP_DONE] ${group_name} failed=${failed}"
}

run_etth_seed() {
  local seed="$1"
  local seq_len="96"
  local pred_len="$2"
  local aug_path="dataset/ETT-small/ETTh1_RIR_augmented_seed${seed}_pl${pred_len}.csv"
  local base_out="causal_r1_etth_side_effect/baseline_seed${seed}_pl${pred_len}_${RUN_ID}"
  local rir_out="causal_r1_etth_side_effect/rir_seed${seed}_pl${pred_len}_${RUN_ID}"
  cd "${TSL_DIR}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  echo "[ETTH] seed=${seed} pred_len=${pred_len} baseline"
  python train_itransformer_etth1_rir_side_effect.py \
    --make-data \
    --variant baseline \
    --aug-data-path "${aug_path}" \
    --output-dir "${base_out}" \
    --seq-len "${seq_len}" \
    --label-len 48 \
    --pred-len "${pred_len}" \
    --epochs 10 \
    --patience 3 \
    --batch-size 32 \
    --lr 1e-4 \
    --seed "${seed}"
  echo "[ETTH] seed=${seed} pred_len=${pred_len} rir"
  python train_itransformer_etth1_rir_side_effect.py \
    --variant rir \
    --aug-data-path "${aug_path}" \
    --output-dir "${rir_out}" \
    --seq-len "${seq_len}" \
    --label-len 48 \
    --pred-len "${pred_len}" \
    --epochs 10 \
    --patience 3 \
    --batch-size 32 \
    --lr 1e-4 \
    --lambda-resp 0.05 \
    --lambda-dist 0.005 \
    --selection-pred-weight 0.1 \
    --selection-dist-weight 0.05 \
    --seed "${seed}"
}

run_duet_baseline_mix() {
  local seed="$1"
  cd "${DUET_DIR}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  python train_eval_duet_synthetic_causal.py \
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
    --output-dir "causal_r1_duet/mix_baseline_seed${seed}_${RUN_ID}"
}

run_duet_rir_mix() {
  local seed="$1"
  cd "${DUET_DIR}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  python train_eval_duet_crr_synthetic.py \
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
    --output-dir "causal_r1_duet/mix_rir_seed${seed}_${RUN_ID}"
}

run_duet_rir_ci_negative_control() {
  local seed="$1"
  cd "${DUET_DIR}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  python train_eval_duet_crr_synthetic.py \
    --ci 1 \
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
    --output-dir "causal_r1_duet/ci1_rir_seed${seed}_${RUN_ID}"
}

echo "[GROUP] structural negative control + ETTh1 seed 20260504"
p1=$(run_bg "duet_ci1_rir_seed20260503" run_duet_rir_ci_negative_control 20260503)
p2=$(run_bg "etth1_seed20260504_pl96" run_etth_seed 20260504 96)
wait_group "group1" "${p1}" "${p2}"

echo "[GROUP] DUET-Mix seed 20260504 + ETTh1 seed 20260505"
p3=$(run_bg "duet_mix_baseline_seed20260504" run_duet_baseline_mix 20260504)
p4=$(run_bg "duet_mix_rir_seed20260504" run_duet_rir_mix 20260504)
p5=$(run_bg "etth1_seed20260505_pl96" run_etth_seed 20260505 96)
wait_group "group2" "${p3}" "${p4}" "${p5}"

echo "[GROUP] DUET-Mix seed 20260505 + ETTh1 pred_len 192 probe"
p6=$(run_bg "duet_mix_baseline_seed20260505" run_duet_baseline_mix 20260505)
p7=$(run_bg "duet_mix_rir_seed20260505" run_duet_rir_mix 20260505)
p8=$(run_bg "etth1_seed20260504_pl192" run_etth_seed 20260504 192)
wait_group "group3" "${p6}" "${p7}" "${p8}"

echo "[INFO] finish=$(date '+%F %T')"
