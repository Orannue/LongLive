#!/usr/bin/env bash
set -euo pipefail

# Batch-generate videos for VBench/VBench-Long standard evaluation.
# The generated files are named exactly as VBench expects:
#   <original prompt>-0.mp4 ... <original prompt>-4.mp4

CONFIG_PATH="${CONFIG_PATH:-configs/longlive_inference.yaml}"
PROMPT_FILE="${PROMPT_FILE:-vbench_prompts/all_dimension.txt}"
EXTENDED_PROMPT_FILE="${EXTENDED_PROMPT_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-videos/vbench_all_dimension}"

# Local model paths. Override these from the command line or environment.
BASE_MODEL_DIR="${BASE_MODEL_DIR:-../models/Wan2.1-T2V-1.3B}"
GENERATOR_CKPT="${GENERATOR_CKPT:-longlive_models/models/longlive_base.pt}"
LORA_CKPT="${LORA_CKPT:-longlive_models/models/lora.pt}"

NUM_SAMPLES="${NUM_SAMPLES:-5}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-120}"
SEED="${SEED:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

usage() {
  cat <<'USAGE'
Usage:
  bash inference_vbench.sh [options]

Options:
  --config PATH              Config yaml. Default: configs/longlive_inference.yaml
  --prompt-file PATH         Original VBench prompt file used for filenames.
  --extended-prompt-file PATH
                             Optional aligned generation prompts, e.g. longer prompts.
                             Filenames still use --prompt-file.
  --output-dir PATH          Output directory for VBench-named mp4 files.
  --base-model-dir PATH      Local Wan base model directory.
  --generator-ckpt PATH      LongLive generator checkpoint.
  --lora-ckpt PATH           LoRA checkpoint. Pass empty string to disable if config allows it.
  --num-samples N            Samples per prompt. VBench standard expects 5.
  --num-output-frames N      Latent output frames. Default: 120.
  --seed N                   Random seed.
  --nproc-per-node N         torchrun processes.
  --master-port PORT         torchrun master port.

Examples:
  bash inference_vbench.sh

  bash inference_vbench.sh \
    --extended-prompt-file vbench_prompts/all_dimension_longer.txt \
    --output-dir videos/vbench_all_dimension_longer
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --extended-prompt-file) EXTENDED_PROMPT_FILE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --base-model-dir) BASE_MODEL_DIR="$2"; shift 2 ;;
    --generator-ckpt) GENERATOR_CKPT="$2"; shift 2 ;;
    --lora-ckpt) LORA_CKPT="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --num-output-frames) NUM_OUTPUT_FRAMES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "$EXTENDED_PROMPT_FILE" ]]; then
  EXTRA_ARGS+=(--extended_prompt_path "$EXTENDED_PROMPT_FILE")
fi

torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  inference.py \
  --config_path "$CONFIG_PATH" \
  --data_path "$PROMPT_FILE" \
  --output_folder "$OUTPUT_DIR" \
  --base_model_dir "$BASE_MODEL_DIR" \
  --generator_ckpt "$GENERATOR_CKPT" \
  --lora_ckpt "$LORA_CKPT" \
  --num_samples "$NUM_SAMPLES" \
  --num_output_frames "$NUM_OUTPUT_FRAMES" \
  --seed "$SEED" \
  --save_with_vbench_names \
  "${EXTRA_ARGS[@]}"
