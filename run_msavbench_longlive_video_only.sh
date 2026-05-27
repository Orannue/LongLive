#!/usr/bin/env bash
set -euo pipefail

# Converted prompt groups created by convert_msavbench_to_longlive.py.
PROMPT_GROUP_DIR="msavbench_longlive_video_only_by_frames"

CHECKPOINT_PATH="checkpoints/longlive2_5b/model_bf16.pt"
CONFIG_PATH="configs/inference.yaml"
OUTPUT_DIR="videos/msavbench_video_only"
WAN_MODEL_ROOT="../models"

NUM_FRAME_PER_BLOCK=8
GPUS="0,1,2,3,4,5,6,7"
FPS=24
SEED=0
OVERWRITE=0

RUN_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  RUN_ARGS+=(--overwrite)
fi

for PROMPTS_PATH in "${PROMPT_GROUP_DIR}"/msavbench_num_output_frames_*.json; do
  BASENAME="$(basename "${PROMPTS_PATH}")"
  NUM_OUTPUT_FRAMES="${BASENAME#msavbench_num_output_frames_}"
  NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES%.json}"

  echo "Running ${PROMPTS_PATH} with num_output_frames=${NUM_OUTPUT_FRAMES}"
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
done
