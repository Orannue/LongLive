#!/usr/bin/env bash
#SBATCH --job-name=vbench-long-resume
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-63%4
#SBATCH --output=slurm_out/vbench_long_eval/%A_%a.out
#SBATCH --error=slurm_out/vbench_long_eval/%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lustre/fs11/portfolios/nvr/projects/nvr_elm_llm/users/yuchaog/workplace/code/chenlan/LongLive}"
VBENCH_DIR="${VBENCH_DIR:-/lustre/fs11/portfolios/nvr/projects/nvr_elm_llm/users/yuchaog/workplace/code/chenlan/VBench}"
CONDA_ENV="${CONDA_ENV:-self_forcing}"
VIDEOS_PATH="${VIDEOS_PATH:-${REPO_DIR}/videos/vbench_all_dimension}"
OUTPUT_PATH="${OUTPUT_PATH:-${VIDEOS_PATH}}"
CACHE_DIR="${CACHE_DIR:-${VIDEOS_PATH}/.cache}"

DIMENSIONS=(
  subject_consistency
  background_consistency
  aesthetic_quality
  imaging_quality
  object_class
  multiple_objects
  color
  spatial_relationship
  scene
  temporal_style
  overall_consistency
  human_action
  temporal_flickering
  motion_smoothness
  dynamic_degree
  appearance_style
)

DIM_INDEX=$((SLURM_ARRAY_TASK_ID % ${#DIMENSIONS[@]}))
PASS_INDEX=$((SLURM_ARRAY_TASK_ID / ${#DIMENSIONS[@]} + 1))
DIMENSION="${DIMENSIONS[$DIM_INDEX]}"
RESULT_FILE="${OUTPUT_PATH}/long_all_${DIMENSION}_eval_results.json"

cd "$REPO_DIR"
mkdir -p "$OUTPUT_PATH" "$CACHE_DIR" slurm_out/vbench_long_eval

echo "Job: ${SLURM_JOB_ID:-unknown}, task: ${SLURM_ARRAY_TASK_ID}, pass: ${PASS_INDEX}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Dimension: ${DIMENSION}"
echo "Videos: ${VIDEOS_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "Cache: ${CACHE_DIR}"
echo "VBench repo: ${VBENCH_DIR}"

if [[ -s "$RESULT_FILE" ]]; then
  if python - "$RESULT_FILE" "$DIMENSION" <<'PY'
import json
import sys

result_file, dimension = sys.argv[1], sys.argv[2]
with open(result_file, "r", encoding="utf-8") as f:
    data = json.load(f)
value = data[dimension]
if isinstance(value, (list, tuple)):
    value = value[0]
float(value)
PY
  then
    echo "Valid existing result found, skipping: ${RESULT_FILE}"
    exit 0
  fi
  echo "Existing result is not valid JSON for ${DIMENSION}; rerunning."
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${VBENCH_DIR}:${PYTHONPATH:-}"

python "$VBENCH_DIR/run_vbench_long_all.py" \
  --vbench_repo "$VBENCH_DIR" \
  --videos_path "$VIDEOS_PATH" \
  --output_path "$OUTPUT_PATH" \
  --cache_dir "$CACHE_DIR" \
  --dimensions "$DIMENSION" \
  --resume \
  --keep_going
