#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  ./evaluate.sh [recon_npz|all|dataset] [algorithm|all] [epochs]

Examples:
  ./evaluate.sh outputs/TP1_epie_10epoch_recon.npz
  ./evaluate.sh TP1 epie 10
  ./evaluate.sh TP1 all 1
  ./evaluate.sh all epie 1
  ./evaluate.sh all all 1

Notes:
  - Uses PtychoPINN's FRC/FSC metric implementation.
  - Main metric: ptychopinn_frc_auc_0_to_0.5, higher is better.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TARGET="${1:-all}"
ALGORITHM="${2:-all}"
EPOCHS="${3:-1}"

DATASETS=(TP1 TP2 IC1 IC2 NCM FLY1 LFP W LCLS)
ALGORITHMS=(pie epie rpie mpie dm lsqml bh ad_ptycho)

run_eval() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "Missing output: ${file}" >&2
    return 1
  fi
  echo "============================================================"
  echo "Evaluating ${file}"
  echo "============================================================"
  conda run -n ptychopinn_torch python scripts/evaluate_ptychopinn_recon.py "${file}"
}

if [[ "${TARGET}" == *.npz || "${TARGET}" == outputs/*.npz ]]; then
  run_eval "${TARGET}"
  exit 0
fi

if [[ "${TARGET}" == "all" ]]; then
  SELECTED_DATASETS=("${DATASETS[@]}")
else
  SELECTED_DATASETS=("${TARGET}")
fi

if [[ "${ALGORITHM}" == "all" ]]; then
  SELECTED_ALGORITHMS=("${ALGORITHMS[@]}")
else
  SELECTED_ALGORITHMS=("${ALGORITHM}")
fi

for dataset in "${SELECTED_DATASETS[@]}"; do
  for algorithm in "${SELECTED_ALGORITHMS[@]}"; do
    run_eval "outputs/${dataset}_${algorithm}_${EPOCHS}epoch_recon.npz"
  done
done
