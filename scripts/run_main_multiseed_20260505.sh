#!/usr/bin/env bash
set -eo pipefail

SCRIPT_ROOT="${SCRIPT_ROOT:-/home/ab/yww/Causal-R1-mainseed-scripts}"
TSL_DIR="${TSL_DIR:-/media/ab/yww/adaptive-wavelet/Time-Series-Library/Time-Series-Library-main}"
RUN_ID="${RUN_ID:-main_multiseed_20260505}"
RESULT_ROOT="${RESULT_ROOT:-${TSL_DIR}/causal_r1_results/${RUN_ID}}"
DATA_PATH="${CAUSAL_R1_SYNTHETIC_CSV:-synthetic_multivariate.csv}"

MODELS=(${MODELS:-DLinear PatchTST iTransformer Crossformer TimeMixer})
SEEDS=(${SEEDS:-20260503 20260504 20260505})

mkdir -p "${RESULT_ROOT}"
cd "${TSL_DIR}"

echo "[INFO] run_id=${RUN_ID}"
echo "[INFO] tsl_dir=${TSL_DIR}"
echo "[INFO] result_root=${RESULT_ROOT}"
echo "[INFO] data_path=${DATA_PATH}"
echo "[INFO] start=$(date '+%F %T')"

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_dir="${RESULT_ROOT}/${model}/seed${seed}"
    mkdir -p "${run_dir}"
    echo "[TRAIN] model=${model} seed=${seed}"
    python "${SCRIPT_ROOT}/train_tsl_seeded_synthetic.py" \
      --model "${model}" \
      --seed "${seed}" \
      --batch-size 32 \
      --learning-rate 0.0001 \
      --epochs 10 \
      --patience 3 \
      --data-path "${DATA_PATH}" \
      --run-dir "${run_dir}"

    ckpt_path=$(python - <<PY
import json
from pathlib import Path
meta = json.loads(Path(r"${run_dir}/train_meta.json").read_text(encoding="utf-8"))
print(meta["checkpoint_path"])
PY
)

    echo "[EVAL] model=${model} seed=${seed}"
    python "${SCRIPT_ROOT}/evaluate_tsl_seeded_synthetic.py" \
      --model "${model}" \
      --checkpoint-path "${ckpt_path}" \
      --output-dir "${run_dir}/eval" \
      --batch-size 64 \
      --delta 5.0
  done
done

echo "[INFO] finish=$(date '+%F %T')"
