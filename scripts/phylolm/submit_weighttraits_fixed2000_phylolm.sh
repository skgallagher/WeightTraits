#!/bin/bash
# PhyloLM on the fixed-2,000-step Llama-3.2-1B cohorts (the checkpoints behind the paper weight
# rows). Same runner logic, core, config, and seeds as submit_weighttraits_fresh_phylolm.sh;
# only the cohort names, output groups, and weighttraits root differ. Reads the sealed stage only.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
WT_ROOT="${WT_ROOT:-/home/export/sgallagh/WeightTraits-all-results-20260817-clean}"
PY="${PY:-/home/export/sgallagh/.conda/envs/ellmtrees/bin/python}"
CONFIG="${CONFIG:-scripts/phylolm/configs/weighttraits_fresh_phylolm.yaml}"
RUNNER=scripts/phylolm/run_weighttraits_fixed2000_tree.py
THROTTLE="${THROTTLE:-4}"; BATCH_SIZE="${BATCH_SIZE:-64}"; TIME="${TIME:-04:00:00}"; MEM="${MEM:-48G}"
cd "$REPO"
LOGDIR="$REPO/blackbox/data/weighttraits_fixed2000_phylolm/logs"; mkdir -p "$LOGDIR"
for suite in r8 r64 full_ft; do
  "$PY" "$RUNNER" --suite "$suite" --preflight-all --weighttraits-root "$WT_ROOT" --config "$CONFIG" > /dev/null
done
declare -a jobs=()
for suite in r8 r64 full_ft; do
  job=$(sbatch --parsable --partition=all --account=statds --gres=gpu:nvidia-l40:1 \
    --array="1-50%${THROTTLE}" --cpus-per-task=4 --mem="$MEM" --time="$TIME" \
    --job-name="wt-phylolm2k-${suite}" --chdir="$REPO" \
    --output="$LOGDIR/${suite}_%A_%a.out" --error="$LOGDIR/${suite}_%A_%a.err" \
    --wrap="$PY $RUNNER --suite $suite --tree-index \$SLURM_ARRAY_TASK_ID --weighttraits-root $WT_ROOT --config $CONFIG --batch-size $BATCH_SIZE")
  jobs+=("$job")
done
post=$(sbatch --parsable --partition=all --account=statds --cpus-per-task=2 --mem=12G --time=01:00:00 \
  --job-name=wt-phylolm2k-post --chdir="$REPO" --dependency="afterok:${jobs[0]}:${jobs[1]}:${jobs[2]}" \
  --output="$LOGDIR/post_%j.out" --error="$LOGDIR/post_%j.err" \
  --wrap="$PY scripts/phylolm/score_phylolm_table.py --groups runs_weighttraits_llama32_1b_lora_qkv_r8_fixed2000_phylolm runs_weighttraits_llama32_1b_lora_qkv_r64_fixed2000_phylolm runs_weighttraits_llama32_1b_full_finetune_fixed2000_phylolm --out results/aggregate/weighttraits_fixed2000_phylolm")
printf "r8=%s r64=%s full_ft=%s post=%s\n" "${jobs[0]}" "${jobs[1]}" "${jobs[2]}" "$post"
