#!/usr/bin/env bash

TSL_DIR="${TSL_DIR:-/path/to/Time-Series-Library}"
RUN_ID="ett_augmented_20260504"
LOG_ROOT="${LOG_ROOT:-./causal_r1_runs/${RUN_ID}}"
DATA_DIR="${TSL_DIR}/dataset/ETT-small"
ALT_DATA_DIR="${ALT_DATA_DIR:-}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-time_mamba}"

DATASET_NAME="${DATASET_NAME:-ETTh2}"
DATA_KIND="${DATA_KIND:-ETTh2}"
FREQ="${FREQ:-h}"
SEED="${SEED:-20260504}"
SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"
PRED_LEN="${PRED_LEN:-96}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
LR="${LR:-1e-4}"

SOURCE_FILE="${DATA_DIR}/${DATASET_NAME}.csv"
AUG_FILE="dataset/ETT-small/${DATASET_NAME}_RIR_augmented_seed${SEED}_pl${PRED_LEN}.csv"
BASE_OUT="causal_r1_etth_side_effect/${DATASET_NAME,,}_baseline_seed${SEED}_pl${PRED_LEN}_${RUN_ID}"
RIR_OUT="causal_r1_etth_side_effect/${DATASET_NAME,,}_rir_seed${SEED}_pl${PRED_LEN}_${RUN_ID}"

mkdir -p "${LOG_ROOT}" "${DATA_DIR}"

echo "[INFO] run_id=${RUN_ID}"
echo "[INFO] dataset=${DATASET_NAME} data=${DATA_KIND} freq=${FREQ}"
echo "[INFO] seed=${SEED} seq_len=${SEQ_LEN} pred_len=${PRED_LEN}"
echo "[INFO] gpu=${GPU_ID}"
echo "[INFO] start=$(date '+%F %T')"

if [ ! -f "${SOURCE_FILE}" ]; then
  if [ -n "${ALT_DATA_DIR}" ] && [ -f "${ALT_DATA_DIR}/${DATASET_NAME}.csv" ]; then
    cp "${ALT_DATA_DIR}/${DATASET_NAME}.csv" "${SOURCE_FILE}"
    echo "[INFO] copied ${DATASET_NAME}.csv from alternate dataset directory"
  else
    echo "[ERROR] missing source file: ${SOURCE_FILE}" >&2
    exit 1
  fi
fi

if [ -n "${CONDA_SH}" ] && [ -f "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

cd "${TSL_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

BASE_LOG="${LOG_ROOT}/${DATASET_NAME,,}_baseline_seed${SEED}_pl${PRED_LEN}.log"
RIR_LOG="${LOG_ROOT}/${DATASET_NAME,,}_rir_seed${SEED}_pl${PRED_LEN}.log"

echo "[RUN] ${DATASET_NAME} augmented baseline" | tee "${BASE_LOG}"
python train_itransformer_etth1_rir_side_effect.py \
  --dataset-name "${DATASET_NAME}" \
  --data "${DATA_KIND}" \
  --freq "${FREQ}" \
  --source-data-path "dataset/ETT-small/${DATASET_NAME}.csv" \
  --make-data \
  --variant baseline \
  --aug-data-path "${AUG_FILE}" \
  --output-dir "${BASE_OUT}" \
  --seq-len "${SEQ_LEN}" \
  --label-len "${LABEL_LEN}" \
  --pred-len "${PRED_LEN}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --seed "${SEED}" >> "${BASE_LOG}" 2>&1
BASE_STATUS=$?
echo "[DONE] baseline status=${BASE_STATUS}" | tee -a "${BASE_LOG}"
if [ "${BASE_STATUS}" -ne 0 ]; then
  exit "${BASE_STATUS}"
fi

echo "[RUN] ${DATASET_NAME} augmented RIR" | tee "${RIR_LOG}"
python train_itransformer_etth1_rir_side_effect.py \
  --dataset-name "${DATASET_NAME}" \
  --data "${DATA_KIND}" \
  --freq "${FREQ}" \
  --source-data-path "dataset/ETT-small/${DATASET_NAME}.csv" \
  --variant rir \
  --aug-data-path "${AUG_FILE}" \
  --output-dir "${RIR_OUT}" \
  --seq-len "${SEQ_LEN}" \
  --label-len "${LABEL_LEN}" \
  --pred-len "${PRED_LEN}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --lambda-resp 0.05 \
  --lambda-dist 0.005 \
  --selection-pred-weight 0.1 \
  --selection-dist-weight 0.05 \
  --seed "${SEED}" >> "${RIR_LOG}" 2>&1
RIR_STATUS=$?
echo "[DONE] rir status=${RIR_STATUS}" | tee -a "${RIR_LOG}"
if [ "${RIR_STATUS}" -ne 0 ]; then
  exit "${RIR_STATUS}"
fi

echo "[INFO] finish=$(date '+%F %T')"
