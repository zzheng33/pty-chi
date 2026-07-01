#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_ptychi.sh [dataset|all] [epochs|list] [cpu|cuda|xpu] [algorithm|all]

Examples:
  scripts/run_ptychi.sh R1000 100 cuda epie
  scripts/run_ptychi.sh all 100 cuda epie
  scripts/run_ptychi.sh all list cuda all
  scripts/run_ptychi.sh all 10 cuda all
  TEST=true DRY_RUN_PTYCHI=true scripts/run_ptychi.sh R1000 1 cpu epie

Notes:
  - Use the second argument "list", or omit it, to run EPOCHS_LIST below.
  - Edit EPOCHS_LIST=(1 5 10) in this file to change the sweep.

Outputs:
  modeling_exp/<GPU>/<DATASET>/<ALGORITHM>/e<EPOCHS>_bs<BATCH>.log
  modeling_exp/<GPU>/<DATASET>/<ALGORITHM>/e<EPOCHS>_bs<BATCH>_power.csv

Environment variables:
  PYTHON_BIN              Python executable. Default: /home/zhong.zheng/miniforge3/envs/ptychopinn_torch/bin/python
  DATA_ROOT               Data root directory. Default: data
  BATCH_SIZE              Batch size. Default: 1000
  DEVICE_INDEX            GPU index for run and monitor. Default: 0
  VENDOR                  Power monitor vendor: auto, nvidia, amd, intel. Default: auto
  GPU_LABEL               Manual output folder label, e.g. A100 or H200.
  OUTPUT_ROOT             Output root. Default: <repo>/modeling_exp
  INTERVAL                Power sample interval seconds. Default: 0.2
  TEST                    true skips GPU power monitor.
  CONTINUE_ON_ERROR       true continues after a failed run.
  DRY_RUN_PTYCHI          true only loads/setup pty-chi, no reconstruction.
  FIXED_PROBE             true disables probe optimization.
  NO_CENTER_POSITIONS     true keeps original probe positions.
  NO_PROBE_RESCALE        true disables probe rescaling.

Datasets:
  R1000 R2000 R4000 R8000 R12000 R16000 R20000 R26000

Algorithms:
  pie epie rpie mpie dm lsqml bh ad_ptycho
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Edit this list for epoch sweeps.
EPOCHS_LIST=(1 2 4 6)

DATASET="${1:-all}"
EPOCHS_ARG="${2:-list}"
DEVICE="${3:-cuda}"
ALGORITHM="${4:-all}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhong.zheng/miniforge3/envs/ptychopinn_torch/bin/python}"
DATA_ROOT="${DATA_ROOT:-data}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
VENDOR="${VENDOR:-auto}"
INTERVAL="${INTERVAL:-0.2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/modeling_exp}"
MONITOR_SCRIPT="${MONITOR_SCRIPT:-/home/zhong.zheng/PtychoPINN/scripts/monitor_gpu_power.py}"
EXTRA_OBJECT_PIXELS="${EXTRA_OBJECT_PIXELS:-64}"
OBJECT_STEP_SIZE="${OBJECT_STEP_SIZE:-0.1}"
PROBE_STEP_SIZE="${PROBE_STEP_SIZE:-0.1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if ! "${PYTHON_BIN}" --version >/dev/null 2>&1; then
  echo "Python executable cannot run on this node: ${PYTHON_BIN}" >&2
  echo "Node architecture: $(uname -m)" >&2
  echo "If this is a GH200/Grace node, use an ARM/aarch64 Python environment, not the x86_64 ptychopinn_torch env." >&2
  echo "Override with: PYTHON_BIN=/path/to/arm/env/bin/python scripts/run_ptychi.sh ..." >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${EPOCHS_ARG}" == "list" || "${EPOCHS_ARG}" == "all" ]]; then
  SELECTED_EPOCHS=("${EPOCHS_LIST[@]}")
else
  SELECTED_EPOCHS=("${EPOCHS_ARG}")
fi

CMD=(
  "${PYTHON_BIN}" scripts/run_ptychi.py
  --modeling
  --datasets "${DATASET}"
  --algorithms "${ALGORITHM}"
  --epochs "${SELECTED_EPOCHS[@]}"
  --batch-size "${BATCH_SIZE}"
  --data-root "${DATA_ROOT}"
  --device "${DEVICE}"
  --vendor "${VENDOR}"
  --devices "${DEVICE_INDEX}"
  --interval "${INTERVAL}"
  --output-root "${OUTPUT_ROOT}"
  --monitor-script "${MONITOR_SCRIPT}"
  --extra-object-pixels "${EXTRA_OBJECT_PIXELS}"
  --object-step-size "${OBJECT_STEP_SIZE}"
  --probe-step-size "${PROBE_STEP_SIZE}"
)

if [[ -n "${GPU_LABEL:-}" ]]; then
  CMD+=(--gpu-label "${GPU_LABEL}")
fi
if [[ "${TEST:-false}" == "true" ]]; then
  CMD+=(--test)
fi
if [[ "${CONTINUE_ON_ERROR:-false}" == "true" ]]; then
  CMD+=(--continue-on-error)
fi
if [[ "${DRY_RUN_PTYCHI:-false}" == "true" ]]; then
  CMD+=(--dry-run-ptychi)
fi
if [[ "${FIXED_PROBE:-false}" == "true" ]]; then
  CMD+=(--fixed-probe)
fi
if [[ "${NO_CENTER_POSITIONS:-false}" == "true" ]]; then
  CMD+=(--no-center-positions)
fi
if [[ "${NO_PROBE_RESCALE:-false}" == "true" ]]; then
  CMD+=(--no-probe-rescale)
fi

"${CMD[@]}"
