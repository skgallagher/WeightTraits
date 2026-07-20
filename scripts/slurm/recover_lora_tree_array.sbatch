#!/usr/bin/env bash
#SBATCH --job-name=recover_lora_tree
#SBATCH --partition=all
#SBATCH --gres=gpu:nvidia-l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out

set -euo pipefail

: "${REPO:?Set REPO to the WeightTraits checkout}"
: "${RUN_LIST_DIR:?Set RUN_LIST_DIR to the generated run_lists directory}"
: "${TREE_IDS:?Set TREE_IDS to a comma-separated list of tree IDs}"

PYTHON_BIN="${PYTHON_BIN:-/home/export/sgallagh/.conda/envs/weighttraits/bin/python}"
cd "${REPO}"
tree_id="$(${PYTHON_BIN} -c 'import sys; print(sys.argv[1].split(",")[int(sys.argv[2])])' "${TREE_IDS}" "${SLURM_ARRAY_TASK_ID}")"
tree_id="$(${PYTHON_BIN} -c 'import sys; print(f"{int(sys.argv[1]):03d}")' "${tree_id}")"
run_list="${RUN_LIST_DIR}/confirm_paper_tree_${tree_id}.runs.jsonl"
ledger="$(${PYTHON_BIN} -c 'import json,sys; print(json.loads(next(line for line in open(sys.argv[1]) if line.strip()))["ledger_path"])' "${run_list}")"
start_index="$(${PYTHON_BIN} -c '
import json, sys
success = {"completed", "skipped", "stopped_early"}
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
latest = {}
for line in open(sys.argv[2]):
    if line.strip():
        event = json.loads(line)
        latest[event["node_id"]] = event["status"]
for row in rows:
    if latest.get(row["node_id"]) not in success:
        print(row["array_index"])
        break
else:
    print(len(rows))
' "${run_list}" "${ledger}")"
row_count="$(wc -l < "${run_list}")"

if (( start_index >= row_count )); then
  echo "Tree ${tree_id} already has successful terminal statuses for every row"
  exit 0
fi

echo "Recovering tree ${tree_id} from row ${start_index} of ${row_count}"
"${PYTHON_BIN}" scripts/materialize_lora_recovery_parents.py \
  --run-list "${run_list}" \
  --start-index "${start_index}" \
  --repo-root "${REPO}"

export ROW_START="${start_index}"
export ROW_END="$((row_count - 1))"
export PRUNE_INTERNAL_PARENTS=false
export PRUNE_LORA_MERGED_AFTER_TREE=false
export TREE_INDEX="$((10#${tree_id}))"
export RUN_LIST="${run_list}"
bash scripts/slurm/confirm_paper_tree_sequential.sbatch
