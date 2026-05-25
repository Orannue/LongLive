#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-${CONFIG_PATH:-configs/longlive_interactive_inference.yaml}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  interactive_inference.py \
  --config_path "$CONFIG_PATH"
