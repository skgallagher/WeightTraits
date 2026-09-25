#!/usr/bin/env bash
#SBATCH --job-name=wt-merged-reanalysis
#SBATCH --output=slurm_logs/wt-merged-reanalysis_%A_%a.out
#SBATCH --error=slurm_logs/wt-merged-reanalysis_%A_%a.err
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=1-50%4

set -euo pipefail

CODE_REPO="${CODE_REPO:-/home/export/sgallagh/WeightTraits-llama32-20260713}"
DATA_REPO="${DATA_REPO:-/home/export/sgallagh/WeightTraits}"
PYTHON_BIN="${PYTHON_BIN:-/home/export/sgallagh/.conda/envs/weighttraits/bin/python}"
SUITE="${SUITE:?set SUITE to a completed merged-weight suite}"
DATE_TAG="${DATE_TAG:-20260804}"
ANALYSIS_FAMILY="${ANALYSIS_FAMILY:-flan}"
TREE_ID="$(printf 'confirm_paper_tree_%03d' "${SLURM_ARRAY_TASK_ID}")"
LEDGER="${DATA_REPO}/outputs/confirm_paper_numbers_merged_weights_${DATE_TAG}/${SUITE}/${TREE_ID}/leaves/merged_leaf_ledger.jsonl"
TRUTH="${DATA_REPO}/examples/training/confirm_paper_numbers/assigned_manifests/${TREE_ID}.manifest.jsonl"
OUT="${DATA_REPO}/outputs/analysis_${ANALYSIS_FAMILY}_merged_weights_${DATE_TAG}/${SUITE}/${TREE_ID}/merged_leaf_analysis"

cd "${CODE_REPO}"
mkdir -p slurm_logs
env PYTHONPATH=src "${PYTHON_BIN}" scripts/analyze_existing_merged_tree.py \
  --repo-root "${DATA_REPO}" \
  --ledger "${LEDGER}" \
  --truth-manifest "${TRUTH}" \
  --out "${OUT}" \
  --metric cosine \
  --metric l2 \
  --metric correlation
