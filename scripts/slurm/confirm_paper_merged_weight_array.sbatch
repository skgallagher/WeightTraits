#!/usr/bin/env bash
#SBATCH --job-name=wt-merged-whitebox
#SBATCH --output=slurm_logs/wt-merged-whitebox_%A_%a.out
#SBATCH --error=slurm_logs/wt-merged-whitebox_%A_%a.err
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=1-50%4

set -euo pipefail

REPO="${REPO:-/home/export/sgallagh/WeightTraits-llama32-20260713}"
PYTHON_BIN="${PYTHON_BIN:-/home/export/sgallagh/.conda/envs/weighttraits/bin/python}"
SUITE="${SUITE:?set SUITE to a confirm-paper LoRA training-runlist prefix}"
DATE_TAG="${DATE_TAG:-20260804}"
ANALYSIS_FAMILY="${ANALYSIS_FAMILY:-llama32}"
BASE_REVISION_OVERRIDE="${BASE_REVISION_OVERRIDE:-}"
TREE_INDEX="${SLURM_ARRAY_TASK_ID}"
TREE_ID="$(printf 'confirm_paper_tree_%03d' "${TREE_INDEX}")"
RUN_LIST="examples/training/confirm_paper_numbers/${SUITE}_training_runlists/run_lists/${TREE_ID}.runs.jsonl"
TRUTH="examples/training/confirm_paper_numbers/assigned_manifests/${TREE_ID}.manifest.jsonl"
CHECKPOINT_ROOT="outputs/confirm_paper_numbers_merged_weights_${DATE_TAG}/${SUITE}/${TREE_ID}/leaves"
OUT="outputs/analysis_${ANALYSIS_FAMILY}_merged_weights_${DATE_TAG}/${SUITE}/${TREE_ID}/merged_leaf_analysis"

cd "${REPO}"
mkdir -p slurm_logs

analysis_cmd=(
  env PYTHONPATH=src "${PYTHON_BIN}" scripts/analyze_rematerialized_lora_tree.py
  --repo-root "${REPO}" \
  --run-list "${RUN_LIST}" \
  --truth-manifest "${TRUTH}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --out "${OUT}" \
  --metric cosine \
  --metric l2 \
  --metric correlation \
  --local-files-only
)
if [[ -n "${BASE_REVISION_OVERRIDE}" ]]; then
  analysis_cmd+=(--base-revision-override "${BASE_REVISION_OVERRIDE}")
fi
"${analysis_cmd[@]}"
