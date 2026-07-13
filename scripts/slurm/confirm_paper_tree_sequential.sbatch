#!/usr/bin/env bash
#SBATCH --job-name=wt-cp-tree
#SBATCH --output=slurm_logs/wt-cp-tree_%A_%a.out
#SBATCH --error=slurm_logs/wt-cp-tree_%A_%a.err
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --gres=gpu:nvidia-l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

REPO="${REPO:-/home/export/sgallagh/WeightTraits}"
PYTHON_BIN="${PYTHON_BIN:-/home/export/sgallagh/.conda/envs/weighttraits/bin/python}"
TREE_INDEX="${TREE_INDEX:-${SLURM_ARRAY_TASK_ID:-1}}"
TREE_ID="$(printf '%03d' "${TREE_INDEX}")"
RUN_LIST_TEMPLATE="${RUN_LIST_TEMPLATE:-examples/training/confirm_paper_numbers/full_finetune_training_runlists/run_lists/confirm_paper_tree_TREEID.runs.jsonl}"
expanded_run_list="${RUN_LIST_TEMPLATE//TREEID/${TREE_ID}}"
expanded_run_list="${expanded_run_list//__TREE_ID__/${TREE_ID}}"
expanded_run_list="${expanded_run_list//\{TREE_ID\}/${TREE_ID}}"
RUN_LIST="${RUN_LIST:-${expanded_run_list}}"
ROW_START="${ROW_START:-0}"
ROW_END="${ROW_END:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
OVERRIDE_MAX_STEPS="${OVERRIDE_MAX_STEPS:-}"
REPORT_TO="${REPORT_TO:-none}"
RUN_NAME_PREFIX_TEMPLATE="${RUN_NAME_PREFIX_TEMPLATE:-confirm-paper-tree-TREEID}"
expanded_run_name_prefix="${RUN_NAME_PREFIX_TEMPLATE//TREEID/${TREE_ID}}"
expanded_run_name_prefix="${expanded_run_name_prefix//__TREE_ID__/${TREE_ID}}"
expanded_run_name_prefix="${expanded_run_name_prefix//\{TREE_ID\}/${TREE_ID}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-${expanded_run_name_prefix}}"
WANDB_PROJECT="${WANDB_PROJECT:-weighttraits-confirm-paper}"

cd "${REPO}"
mkdir -p slurm_logs

row_count="$(
  PYTHONPATH=src "${PYTHON_BIN}" \
    -c 'import sys; print(sum(1 for line in open(sys.argv[1]) if line.strip()))' "${RUN_LIST}"
)"
if [[ -z "${ROW_END}" ]]; then
  ROW_END="$((row_count - 1))"
fi

if [[ "${REPORT_TO}" != "none" ]]; then
  export WANDB_PROJECT
  export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_NAME_PREFIX}}"
  if [[ "${REPORT_TO}" == *wandb* ]]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
  fi
fi

for row_index in $(seq "${ROW_START}" "${ROW_END}"); do
  node_id="$(
    PYTHONPATH=src "${PYTHON_BIN}" \
      -c 'import json, sys; rows=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]; print(rows[int(sys.argv[2])]["node_id"])' \
      "${RUN_LIST}" "${row_index}"
  )"
  cmd=(
    env
    PYTHONPATH=src
    "HF_DATASETS_CACHE=${HOME}/.cache/WeightTraits/hf_datasets"
    TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}" -m weighttraits.cli run-training-row
    --run-list "${RUN_LIST}"
    --index "${row_index}"
  )
  if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
    cmd+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
  fi
  if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
    cmd+=(--max-eval-samples "${MAX_EVAL_SAMPLES}")
  fi
  if [[ -n "${OVERRIDE_MAX_STEPS}" ]]; then
    cmd+=(--override-max-steps "${OVERRIDE_MAX_STEPS}")
  fi
  if [[ "${REPORT_TO}" != "none" ]]; then
    cmd+=(--report-to "${REPORT_TO}" --run-name "${RUN_NAME_PREFIX}-${node_id}")
  fi

  printf 'Launching tree %s row %s (%s):' "${TREE_ID}" "${row_index}" "${node_id}"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
done
