#!/bin/bash
# Submit the faithful PhyloLM baseline on the 10-model Llama-2-7B zoo (App. D family).
#
#   1. array 0-9%1  -> collect_phylolm.py per model (raw completions, GPU)
#   2. post job     -> phylolm_distance.py (NJ + cube) + score_hf_zoo_recovery.py
#
# Same node/partition defaults as the Mistral run; override NODELIST if the Llama-2
# checkpoints live elsewhere.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-/home/sgallagh/.conda/envs/ellmtrees/bin/python}"
PARTITION="${PARTITION:-Ghidorahs}"
NODELIST="${NODELIST:-ghidorah1}"
COLLECT_TIME="${COLLECT_TIME:-08:00:00}"
POST_TIME="${POST_TIME:-01:00:00}"

TASK="llama2_zoo_phylolm"
CONFIG="scripts/phylolm/configs/${TASK}.yaml"
REGISTRY="scripts/phylolm/configs/models_llama2_zoo.yaml"
TRUTH="outputs/hf_zoo/c_llama2_7b_chat_small_nodilip_resolved/tree.newick"
POP_DIR="blackbox/data/${TASK}/populations"
OUT_DIR="results/hf_zoo/c_llama2_7b_chat_small_nodilip/phylolm"

cd "$REPO"
mkdir -p "blackbox/data/${TASK}/logs" "$OUT_DIR"

collect=$(sbatch --parsable \
    --partition="$PARTITION" \
    --nodelist="$NODELIST" \
    --array=0-9%1 \
    --cpus-per-task=4 \
    --mem=48G \
    --time="$COLLECT_TIME" \
    --job-name="${TASK}_collect" \
    --chdir="$REPO" \
    --output="blackbox/data/${TASK}/logs/slurm_%A_%a.out" \
    --error="blackbox/data/${TASK}/logs/slurm_%A_%a.err" \
    --wrap="$PY scripts/phylolm/collect_phylolm.py --config $CONFIG --registry $REGISTRY --model-index \$SLURM_ARRAY_TASK_ID")

post=$(sbatch --parsable \
    --partition="$PARTITION" \
    --nodelist="$NODELIST" \
    --cpus-per-task=4 \
    --mem=16G \
    --time="$POST_TIME" \
    --job-name="${TASK}_post" \
    --chdir="$REPO" \
    --output="blackbox/data/${TASK}/logs/post_%j.out" \
    --error="blackbox/data/${TASK}/logs/post_%j.err" \
    --dependency="afterok:${collect}" \
    --wrap="$PY scripts/phylolm/phylolm_distance.py --populations $POP_DIR --out $OUT_DIR && \
            $PY scripts/phylolm/score_hf_zoo_recovery.py clades --reference $TRUTH --recon $OUT_DIR/phylolm_nj.newick --out $OUT_DIR/recovery_rescore && \
            $PY scripts/phylolm/score_hf_zoo_recovery.py quartets --reference $TRUTH --distances $OUT_DIR/avg_distance_cosine.npy --metadata $OUT_DIR/distance_metadata.json --out $OUT_DIR/recovery_rescore")

cat <<EOF
Submitted PhyloLM baseline:
  collect (array 0-9%1): ${collect}
  post (dist+NJ+score):  ${post}
Watch:  squeue -u sgallagh
Logs:   blackbox/data/${TASK}/logs/
Result: ${OUT_DIR}/  (phylolm_nj.newick, recovery_rescore/clade_recovery.csv, fourpoint_additivity.json)
EOF
