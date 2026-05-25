#!/usr/bin/env bash
#SBATCH --job-name=longlive-vbench
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --array=24-119%2
#SBATCH --output=slurm_out/vbench/%A_%a.out
#SBATCH --error=slurm_out/vbench/%A_%a.err

set -euo pipefail

REPO_DIR="/lustre/fs11/portfolios/nvr/projects/nvr_elm_llm/users/yuchaog/workplace/code/chenlan/LongLive"
cd "$REPO_DIR"

CONDA_ENV="${CONDA_ENV:-self_forcing}"
CONFIG_PATH="${CONFIG_PATH:-configs/longlive_inference.yaml}"
PROMPT_JSON="${PROMPT_JSON:-../FAR-Dev/assets/data/meta/vbench/VBench_aug_full_info.json}"
PROMPT_FILE="${PROMPT_FILE:-vbench_prompts/VBench_aug_full_info_prompt_en.txt}"
EXTENDED_PROMPT_FILE="${EXTENDED_PROMPT_FILE:-vbench_prompts/VBench_aug_full_info_aug_prompt_en.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-videos/vbench_all_dimension}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-../models/Wan2.1-T2V-1.3B}"
GENERATOR_CKPT="${GENERATOR_CKPT:-longlive_models/models/longlive_base.pt}"
LORA_CKPT="${LORA_CKPT:-longlive_models/models/lora.pt}"

NUM_SAMPLES="${NUM_SAMPLES:-5}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-120}"
SEED="${SEED:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PROMPTS_PER_TASK="${PROMPTS_PER_TASK:-8}"

python prepare_vbench_aug_prompts.py \
  --json "$PROMPT_JSON" \
  --prompt-out "$PROMPT_FILE" \
  --aug-out "$EXTENDED_PROMPT_FILE"

mkdir -p "$OUTPUT_DIR" slurm_out/vbench

TOTAL_PROMPTS="$(awk 'END { print NR }' "$PROMPT_FILE")"
START_LINE=$(( (SLURM_ARRAY_TASK_ID - 1) * PROMPTS_PER_TASK + 1 ))
END_LINE=$(( START_LINE + PROMPTS_PER_TASK - 1 ))

if (( START_LINE > TOTAL_PROMPTS )); then
  echo "Task ${SLURM_ARRAY_TASK_ID}: start line ${START_LINE} exceeds ${TOTAL_PROMPTS}; nothing to do."
  exit 0
fi

if (( END_LINE > TOTAL_PROMPTS )); then
  END_LINE="$TOTAL_PROMPTS"
fi

if [[ -n "$EXTENDED_PROMPT_FILE" ]]; then
  EXTENDED_TOTAL="$(awk 'END { print NR }' "$EXTENDED_PROMPT_FILE")"
  if (( EXTENDED_TOTAL != TOTAL_PROMPTS )); then
    echo "Extended prompt line count (${EXTENDED_TOTAL}) does not match prompt line count (${TOTAL_PROMPTS})." >&2
    exit 2
  fi
fi

TASK_TMP_DIR="${SLURM_TMPDIR:-/tmp}/longlive_vbench_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
rm -rf "$TASK_TMP_DIR"
mkdir -p "$TASK_TMP_DIR"
trap 'rm -rf "$TASK_TMP_DIR"' EXIT

TASK_LOG_DIR="slurm_out/vbench/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TASK_LOG_DIR"

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1

echo "Task ${SLURM_ARRAY_TASK_ID}: lines ${START_LINE}-${END_LINE} of ${TOTAL_PROMPTS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Conda env: ${CONDA_ENV}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES from Slurm: ${CUDA_VISIBLE_DEVICES:-unset}"

prompt_complete() {
  local prompt_text="$1"
  local sample_idx
  local output_path

  for (( sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++ )); do
    output_path="${OUTPUT_DIR}/${prompt_text}-${sample_idx}.mp4"
    if [[ -e "$output_path" && ! -s "$output_path" ]]; then
      rm -f "$output_path"
    fi
    if [[ ! -s "$output_path" ]]; then
      return 1
    fi
  done

  return 0
}

run_prompt_on_gpu() {
  local line_no="$1"
  local gpu_id="$2"
  local prompt_text
  local task_prompt_file
  local task_seed
  local task_master_port
  local task_extended_args=()

  prompt_text="$(sed -n "${line_no}p" "$PROMPT_FILE")"
  if prompt_complete "$prompt_text"; then
    echo "Line ${line_no}: already has ${NUM_SAMPLES} samples; skipping."
    return 0
  fi

  task_prompt_file="$TASK_TMP_DIR/prompt_${line_no}.txt"
  sed -n "${line_no}p" "$PROMPT_FILE" > "$task_prompt_file"

  if [[ -n "$EXTENDED_PROMPT_FILE" ]]; then
    local task_extended_prompt_file="$TASK_TMP_DIR/extended_prompt_${line_no}.txt"
    sed -n "${line_no}p" "$EXTENDED_PROMPT_FILE" > "$task_extended_prompt_file"
    task_extended_args=(--extended-prompt-file "$task_extended_prompt_file")
  fi

  # Match the seed schedule used by the full prompt file.
  task_seed=$(( SEED + (line_no - 1) * NUM_SAMPLES ))
  task_master_port=$(( 20000 + (SLURM_ARRAY_TASK_ID * GPUS_PER_NODE + gpu_id) % 20000 ))

  echo "Line ${line_no}: running on GPU ${gpu_id}, seed ${task_seed}, port ${task_master_port}."
  CUDA_VISIBLE_DEVICES="$gpu_id" bash inference_vbench.sh \
    --config "$CONFIG_PATH" \
    --prompt-file "$task_prompt_file" \
    "${task_extended_args[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --base-model-dir "$BASE_MODEL_DIR" \
    --generator-ckpt "$GENERATOR_CKPT" \
    --lora-ckpt "$LORA_CKPT" \
    --num-samples "$NUM_SAMPLES" \
    --num-output-frames "$NUM_OUTPUT_FRAMES" \
    --seed "$task_seed" \
    --nproc-per-node 1 \
    --master-port "$task_master_port"
}

pids=()
lines=()
logs=()
gpu_id=0

for (( line_no = START_LINE; line_no <= END_LINE; line_no++ )); do
  log_path="${TASK_LOG_DIR}/line_${line_no}.log"
  run_prompt_on_gpu "$line_no" "$gpu_id" > "$log_path" 2>&1 &
  pids+=("$!")
  lines+=("$line_no")
  logs+=("$log_path")

  gpu_id=$(( gpu_id + 1 ))
  if (( gpu_id >= GPUS_PER_NODE )); then
    break
  fi
done

status=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "Line ${lines[$idx]} finished. Log: ${logs[$idx]}"
  else
    echo "Line ${lines[$idx]} failed. Log: ${logs[$idx]}" >&2
    tail -80 "${logs[$idx]}" >&2 || true
    status=1
  fi
done

exit "$status"
