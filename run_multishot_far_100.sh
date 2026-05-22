#!/usr/bin/env bash
set -euo pipefail

# Edit these paths/values before launching.
# FAR_PROMPTS_PATH="C:/Users/Administrator/Desktop/FAR-Dev/assets/evaluation/eval_caption_multishot_t2v_100.json"
PROMPTS_PATH="eval_caption_multishot_t2v_100_longlive.json"
CHECKPOINT_PATH="checkpoints/LongLive-2.0-5B/model_bf16.pt"
CONFIG_PATH="configs/inference.yaml"
OUTPUT_DIR="videos/multishot_far_100"
WAN_MODEL_ROOT="../models"

# 6 shots * ~5s/shot at 24fps maps to about 181 latent frames.
# With num_frame_per_block=8, use 192 latent frames so each shot is exactly
# 4 blocks and decodes to about 5.3s.
NUM_OUTPUT_FRAMES=192
NUM_FRAME_PER_BLOCK=8
GPUS="0,1,2,3,4,5,6,7"
FPS=24
SEED=0
UNIFORM_SHOT_BLOCKS=1

# Hugging Face upload settings.
HF_REPO_ID="Orannue/Baseline_results"
HF_UPLOAD_PATH="eval_caption_multishot_t2v_100/longlivev2"
HF_REPO_TYPE="dataset"
HF_TOKEN="${HF_TOKEN:-}"

ADAPT_ARGS=()
RUN_ARGS=()
if [[ "${UNIFORM_SHOT_BLOCKS}" == "1" ]]; then
  ADAPT_ARGS+=(--uniform_shot_blocks)
  RUN_ARGS+=(--uniform_shot_blocks)
fi

# python adapt_far_prompts_to_longlive.py \
#   --source_path "${FAR_PROMPTS_PATH}" \
#   --output_path "${PROMPTS_PATH}" \
#   --num_output_frames "${NUM_OUTPUT_FRAMES}" \
#   --num_frame_per_block "${NUM_FRAME_PER_BLOCK}" \
#   "${ADAPT_ARGS[@]}"

python run_multishot_batch.py \
  --prompts_path "${PROMPTS_PATH}" \
  --config_path "${CONFIG_PATH}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_output_frames "${NUM_OUTPUT_FRAMES}" \
  --num_frame_per_block "${NUM_FRAME_PER_BLOCK}" \
  --wan_model_root "${WAN_MODEL_ROOT}" \
  --gpus "${GPUS}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  "${RUN_ARGS[@]}"

hf upload \
  "${HF_REPO_ID}" \
  "${OUTPUT_DIR}" \
  "${HF_UPLOAD_PATH}" \
  --repo-type "${HF_REPO_TYPE}" \
  --token "${HF_TOKEN}"
