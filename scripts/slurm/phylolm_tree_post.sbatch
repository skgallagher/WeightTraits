#!/usr/bin/env bash
#SBATCH --job-name=wt-phylolm-post
#SBATCH --output=slurm_logs/wt-phylolm-post_%j.out
#SBATCH --error=slurm_logs/wt-phylolm-post_%j.err
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

REPO="${REPO:-/home/export/sgallagh/WeightTraits}"
PYTHON_BIN="${PYTHON_BIN:-/home/export/sgallagh/.conda/envs/weighttraits/bin/python}"
: "${POPULATION_DIR:?set POPULATION_DIR to a complete tree population directory}"
: "${ANALYSIS_DIR:?set ANALYSIS_DIR to the tree PhyloLM analysis directory}"

cd "${REPO}"
mkdir -p slurm_logs
PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli build-phylolm-analysis \
  --populations "${POPULATION_DIR}" \
  --out "${ANALYSIS_DIR}"
