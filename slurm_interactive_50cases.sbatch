#!/usr/bin/env bash
#SBATCH --job-name=longlive-interactive50
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch_block1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_out/interactive_50cases/%j.out
#SBATCH --error=slurm_out/interactive_50cases/%j.err

set -euo pipefail

REPO_DIR="/lustre/fs11/portfolios/nvr/projects/nvr_elm_llm/users/yuchaog/workplace/code/chenlan/LongLive"
cd "$REPO_DIR"

CONDA_ENV="${CONDA_ENV:-self_forcing}"
CONFIG_PATH="${CONFIG_PATH:-configs/longlive_interactive_50cases_12prompts.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29521}"

mkdir -p slurm_out/interactive_50cases videos/interactive_50cases_12prompts_18latent

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1

echo "Config: ${CONFIG_PATH}"
echo "Conda env: ${CONDA_ENV}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES from Slurm: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"

bash interactive_inference.sh "$CONFIG_PATH"
