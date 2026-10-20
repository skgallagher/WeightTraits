#!/bin/bash
# Run several fixed-2,000-step PhyloLM trees concurrently on one GPU.
# usage: run_fixed2000_pack.sh SUITE PARALLEL TREE_INDEX [TREE_INDEX ...]
set -uo pipefail
SUITE=$1; PARALLEL=$2; shift 2
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY=/home/export/sgallagh/.conda/envs/ellmtrees/bin/python
WT_ROOT=/home/export/sgallagh/WeightTraits-all-results-20260817-clean
CONFIG=scripts/phylolm/configs/weighttraits_fresh_phylolm.yaml
LOGDIR=$REPO/blackbox/data/weighttraits_fixed2000_phylolm/logs
cd "$REPO"
fail=0; pids=()
for idx in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  "$PY" scripts/phylolm/run_weighttraits_fixed2000_tree.py --suite "$SUITE" --tree-index "$idx" \
    --weighttraits-root "$WT_ROOT" --config "$CONFIG" --batch-size 64 \
    > "$LOGDIR/pack_${SUITE}_${SLURM_JOB_ID}_${idx}.out" 2> "$LOGDIR/pack_${SUITE}_${SLURM_JOB_ID}_${idx}.err" &
  pids+=("$!:$idx")
done
for entry in "${pids[@]}"; do
  pid=${entry%%:*}; idx=${entry##*:}
  if ! wait "$pid"; then echo "tree $idx failed"; fail=1; fi
done
exit $fail
