#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  ./test.sh [dataset|all] [epochs] [cpu|cuda|xpu] [algorithm|all]

Examples:
  ./test.sh
  ./test.sh TP1 1 cuda epie
  ./test.sh TP1 1 xpu epie
  ./test.sh all 1 cuda epie
  ./test.sh TP1 1 cuda all
  ./test.sh all 1 cuda all

Datasets:
  TP1 TP2 IC1 IC2 NCM FLY1 LFP W LCLS

Algorithms:
  pie epie rpie mpie dm lsqml bh ad_ptycho
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATASET="${1:-TP1}"
EPOCHS="${2:-1}"
DEVICE="${3:-cuda}"
ALGORITHM="${4:-epie}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CONDA_ENV="${CONDA_ENV:-ptychi_rocm}"
CONDA_ENV="${CONDA_ENV:-ptychi}"


DATASETS=(TP1 TP2 IC1 IC2 NCM FLY1 LFP W LCLS)
ALGORITHMS=(pie epie rpie mpie dm lsqml bh ad_ptycho)

if [[ "${DATASET}" == "all" ]]; then
  SELECTED_DATASETS=("${DATASETS[@]}")
else
  SELECTED_DATASETS=("${DATASET}")
fi

if [[ "${ALGORITHM}" == "all" ]]; then
  SELECTED_ALGORITHMS=("${ALGORITHMS[@]}")
else
  SELECTED_ALGORITHMS=("${ALGORITHM}")
fi 

for dataset in "${SELECTED_DATASETS[@]}"; do
  for algorithm in "${SELECTED_ALGORITHMS[@]}"; do
    echo "============================================================"
    echo "Running dataset=${dataset}, algorithm=${algorithm}, epochs=${EPOCHS}, device=${DEVICE}"
    echo "============================================================"
    conda run -n "${CONDA_ENV}" python scripts/run_ptychi_on_ptychopinn.py \
      --dataset "${dataset}" \
      --algorithm "${algorithm}" \
      --device "${DEVICE}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --output "outputs/${dataset}_${algorithm}_${EPOCHS}epoch_recon.npz"
  done
done

# conda run -n ptychi_rocm python scripts/run_ptychi_on_ptychopinn.py --dataset W --algorithm epie --device cuda --epochs 10 --batch-size 25921 --output outputs/W_epie_10epoch_recon.npz
