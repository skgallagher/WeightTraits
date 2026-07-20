#!/usr/bin/env bash
#SBATCH --job-name=wt-cp-analysis
#SBATCH --output=slurm_logs/wt-cp-analysis_%j.out
#SBATCH --error=slurm_logs/wt-cp-analysis_%j.err
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

REPO="${REPO:-/home/export/sgallagh/WeightTraits}"
PYTHON_BIN="${PYTHON_BIN:-/home/export/sgallagh/.conda/envs/weighttraits/bin/python}"
SUMMARY="${SUMMARY:-examples/training/confirm_paper_numbers/full_finetune_training_run_list_summary.json}"
PATH_BASE="${PATH_BASE:-.}"
ARTIFACT="${ARTIFACT:-model}"
OUT_ROOT="${OUT_ROOT:-outputs/confirm_paper_numbers/full_finetune/run_set_analysis}"
REPORT_OUT="${REPORT_OUT:-outputs/confirm_paper_numbers/full_finetune/direct_run_set_analysis_plan.json}"
ROLLUP_OUT="${ROLLUP_OUT:-outputs/confirm_paper_numbers/full_finetune/direct_run_set_analysis_summary.json}"
CSV_OUT="${CSV_OUT:-outputs/confirm_paper_numbers/full_finetune/direct_run_set_analysis_rows.csv}"
METRICS="${METRICS:-l2 cosine correlation}"
OPTIONAL_ARTIFACTS="${OPTIONAL_ARTIFACTS:-training_log}"
TREE_IDS="${TREE_IDS:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

cd "${REPO}"
mkdir -p slurm_logs

analysis_cmd=(
  env
  PYTHONPATH=src
  "${PYTHON_BIN}" -m weighttraits.cli analyze-training-run-set
  --summary "${SUMMARY}"
  --path-base "${PATH_BASE}"
  --artifact "${ARTIFACT}"
  --out "${OUT_ROOT}"
  --report-out "${REPORT_OUT}"
)
for metric in ${METRICS}; do
  analysis_cmd+=(--metric "${metric}")
done
for optional_artifact in ${OPTIONAL_ARTIFACTS}; do
  analysis_cmd+=(--optional-artifact "${optional_artifact}")
done
for tree_id in ${TREE_IDS}; do
  analysis_cmd+=(--tree-id "${tree_id}")
done
if [[ "${DRY_RUN}" == "1" ]]; then
  analysis_cmd+=(--dry-run)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  analysis_cmd+=(--skip-existing)
fi

printf 'Launching direct run-set analysis:'
printf ' %q' "${analysis_cmd[@]}"
printf '\n'
"${analysis_cmd[@]}"

rollup_cmd=(
  env
  PYTHONPATH=src
  "${PYTHON_BIN}" -m weighttraits.cli summarize-training-run-set-analysis
  --analysis-root "${OUT_ROOT}"
  --artifact "${ARTIFACT}"
  --out "${ROLLUP_OUT}"
  --csv-out "${CSV_OUT}"
)
for tree_id in ${TREE_IDS}; do
  rollup_cmd+=(--tree-id "${tree_id}" --allow-empty)
done

printf 'Launching direct run-set rollup:'
printf ' %q' "${rollup_cmd[@]}"
printf '\n'
"${rollup_cmd[@]}"
