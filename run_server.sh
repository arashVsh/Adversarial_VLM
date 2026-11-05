#!/usr/bin/env bash
set -Eeuo pipefail

# Run the complete canonical seed order on this server. Exact completed job
# identities remain resumable and are skipped regardless of the active shard.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export REPRODUCTION_SEEDS="${REPRODUCTION_SEEDS:-44,43,42}"
export REPRODUCTION_EXPERIMENT_MODE="${REPRODUCTION_EXPERIMENT_MODE:-full}"
export REPRODUCTION_OUTPUT_DIR="${REPRODUCTION_OUTPUT_DIR:-$PROJECT_ROOT/outputs/reproduction}"
export REPRODUCTION_DATA_DIR="${REPRODUCTION_DATA_DIR:-$PROJECT_ROOT/data}"

# These caches survive process restarts and are reused in later experiments.
# The validated cache is populated only after a model loads successfully.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export REPRODUCTION_MODEL_CACHE_DIR="${REPRODUCTION_MODEL_CACHE_DIR:-$HOME/.cache/newpromptshield/validated_models}"

export CUDA_DEVICE_INDEX="${CUDA_DEVICE_INDEX:-0}"
export PYTHONUNBUFFERED="1"
export MPLBACKEND="Agg"

mkdir -p "$REPRODUCTION_OUTPUT_DIR" "$REPRODUCTION_DATA_DIR" \
  "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" \
  "$REPRODUCTION_MODEL_CACHE_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable '$PYTHON_BIN' was not found. Activate the project virtual environment first." >&2
  exit 127
fi

echo "Server seeds: $REPRODUCTION_SEEDS"
echo "Experiment mode: $REPRODUCTION_EXPERIMENT_MODE"
echo "Results: $REPRODUCTION_OUTPUT_DIR/results.json"
echo "Datasets: $REPRODUCTION_DATA_DIR"
echo "Dataset protocol: complete public datasets"
echo "Validated model cache: $REPRODUCTION_MODEL_CACHE_DIR"
echo "CUDA device index: $CUDA_DEVICE_INDEX"

"$PYTHON_BIN" -u reproduce.py 2>&1 | tee -a "$REPRODUCTION_OUTPUT_DIR/reproduction.log"
