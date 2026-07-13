#!/usr/bin/env bash
#SBATCH --job-name=wt-cp-smoke
#SBATCH --output=slurm_logs/wt-cp-smoke_%j.out
#SBATCH --error=slurm_logs/wt-cp-smoke_%j.err
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --gres=gpu:nvidia-l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail

REPO="${REPO:-/home/export/sgallagh/WeightTraits}"
PYTHON_BIN="${PYTHON_BIN:-/home/export/sgallagh/.conda/envs/weighttraits/bin/python}"
RUN_LIST="${RUN_LIST:-examples/training/confirm_paper_numbers/full_finetune_training_runlists/run_lists/confirm_paper_tree_001.runs.jsonl}"
ROW_INDEX="${ROW_INDEX:-0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1}"
OVERRIDE_MAX_STEPS="${OVERRIDE_MAX_STEPS:-2}"
REPORT_TO="${REPORT_TO:-none}"
RUN_NAME="${RUN_NAME:-confirm-paper-tree-001-n0-smoke}"
WANDB_PROJECT="${WANDB_PROJECT:-weighttraits-confirm-paper}"

cd "${REPO}"
mkdir -p slurm_logs

cmd=(
  env
  PYTHONPATH=src
  "HF_DATASETS_CACHE=${HOME}/.cache/WeightTraits/hf_datasets"
  TOKENIZERS_PARALLELISM=false
  "${PYTHON_BIN}" -m weighttraits.cli run-training-row
  --run-list "${RUN_LIST}"
  --index "${ROW_INDEX}"
  --max-train-samples "${MAX_TRAIN_SAMPLES}"
  --max-eval-samples "${MAX_EVAL_SAMPLES}"
  --override-max-steps "${OVERRIDE_MAX_STEPS}"
)

if [[ "${REPORT_TO}" != "none" ]]; then
  export WANDB_PROJECT
  if [[ "${REPORT_TO}" == *wandb* ]]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
  fi
  cmd+=(--report-to "${REPORT_TO}" --run-name "${RUN_NAME}")
fi

printf 'Launching row smoke:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
