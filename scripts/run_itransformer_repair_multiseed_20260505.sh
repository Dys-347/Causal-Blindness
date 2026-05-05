#!/usr/bin/env bash
set -eo pipefail

SCRIPT_ROOT="${SCRIPT_ROOT:-/home/ab/yww/Causal-R1-mainseed-scripts}"
TSL_DIR="${TSL_DIR:-/media/ab/yww/adaptive-wavelet/Time-Series-Library/Time-Series-Library-main}"
BASELINE_ROOT="${BASELINE_ROOT:-${TSL_DIR}/causal_r1_results/main_multiseed_20260505/iTransformer}"
RUN_ID="${RUN_ID:-itransformer_repair_multiseed_20260505}"
RESULT_ROOT="${RESULT_ROOT:-${TSL_DIR}/causal_r1_results/${RUN_ID}}"
DATA_PATH="${CAUSAL_R1_SYNTHETIC_CSV:-synthetic_multivariate.csv}"

SEEDS=(${SEEDS:-20260503 20260504 20260505})
VARIANTS=(${VARIANTS:-ft01})

mkdir -p "${RESULT_ROOT}"
cd "${TSL_DIR}"

echo "[INFO] run_id=${RUN_ID}"
echo "[INFO] tsl_dir=${TSL_DIR}"
echo "[INFO] baseline_root=${BASELINE_ROOT}"
echo "[INFO] result_root=${RESULT_ROOT}"
echo "[INFO] data_path=${DATA_PATH}"
echo "[INFO] variants=${VARIANTS[*]}"
echo "[INFO] seeds=${SEEDS[*]}"
echo "[INFO] start=$(date '+%F %T')"

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    baseline_meta="${BASELINE_ROOT}/seed${seed}/train_meta.json"
    if [ ! -f "${baseline_meta}" ]; then
      echo "[ERROR] missing baseline meta: ${baseline_meta}"
      exit 1
    fi
    init_ckpt=$(python - <<PY
import json
from pathlib import Path
meta = json.loads(Path(r"${baseline_meta}").read_text(encoding="utf-8"))
print(meta["checkpoint_path"])
PY
)

    run_dir="${RESULT_ROOT}/${variant}_seed${seed}"
    mkdir -p "${run_dir}"
    echo "[TRAIN] variant=${variant} seed=${seed}"
    if [ "${variant}" = "ft01" ]; then
      lambda_resp="0.1"
      lambda_dist="0.01"
    elif [ "${variant}" = "ft03" ]; then
      lambda_resp="0.3"
      lambda_dist="0.03"
    else
      echo "[ERROR] unknown variant=${variant}"
      exit 1
    fi

    python "${SCRIPT_ROOT}/train_itransformer_crr_h1.py" \
      --output-dir "${run_dir}" \
      --epochs 10 \
      --batch-size 32 \
      --learning-rate 0.0001 \
      --lambda-resp "${lambda_resp}" \
      --lambda-dist "${lambda_dist}" \
      --selection-pred-weight 0.1 \
      --delta 5.0 \
      --patience 3 \
      --seed "${seed}" \
      --init-checkpoint "${init_ckpt}"

    echo "[EVAL] variant=${variant} seed=${seed}"
    python "${SCRIPT_ROOT}/evaluate_tsl_seeded_synthetic.py" \
      --model iTransformer \
      --checkpoint-path "${run_dir}/checkpoint.pth" \
      --output-dir "${run_dir}/eval" \
      --variant "${variant}" \
      --batch-size 64 \
      --delta 5.0
  done
done

echo "[INFO] finish=$(date '+%F %T')"
