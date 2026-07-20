#!/usr/bin/env bash
#SBATCH --job-name=wt-llama-preflight
#SBATCH --output=slurm_logs/wt-llama-preflight_%j.out
#SBATCH --error=slurm_logs/wt-llama-preflight_%j.err
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --gres=gpu:nvidia-l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail

REPO="${REPO:-/home/export/sgallagh/WeightTraits-llama32-20260713}"
PYTHON_BIN="${PYTHON_BIN:-/home/export/sgallagh/.conda/envs/weighttraits/bin/python}"
RANK="${RANK:-64}"

cd "${REPO}"
mkdir -p slurm_logs
env PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
  "${PYTHON_BIN}" scripts/llama32_1b_preflight.py --rank "${RANK}"
