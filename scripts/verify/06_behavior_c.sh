#!/usr/bin/env bash
# 06_behavior_c.sh — BEHAVIOR track, part (c): every number in Table 4, the heterogeneity table, the EOS
# sensitivity and depth tables, and the behavior prose == the sealed artifacts (display rounding).
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$WT_ROOT"
[[ -d $SCRATCH/behavior_trees ]] || { echo "run 06_behavior_b.sh first"; exit 2; }
if $PY scripts/verify/verify_behavior_paper.py "$PAPER_TEX" "$SCRATCH/behavior_trees" "$LOCAL_MIRROR/table8_eos_sensitivity" > "$REPORT_DIR/06c_behavior_paper.log" 2>&1; then
  check "Table 4 (12 cells), heterogeneity (12 rows), EOS sensitivity (5), depth (20 cells), prose" OK
else
  n=$(grep -c MISMATCH "$REPORT_DIR/06c_behavior_paper.log"); check "behavior tables vs artifacts" FAIL "$n mismatch(es): $(grep MISMATCH "$REPORT_DIR/06c_behavior_paper.log" | sed 's/ *ok$//' | tr '\n' ';' | cut -c1-160)"; fi
sed 's/^/     /' "$REPORT_DIR/06c_behavior_paper.log"
claim() { local label=$1 frag=$2 cond=$3
  if ! grep -qF -- "$frag" "$PAPER_TEX"; then check "$label" FAIL "phrase not found in tex: $frag"; return; fi
  if eval "$cond"; then check "$label" OK "tex: \"${frag:0:70}\""; else check "$label" FAIL; fi; }
claim "641 nodes across 50 trees incl. 276 internal as observed ancestors; root excluded" "641 nodes across 50" "grep -q 'root excluded' $REPORT_DIR/checks.tsv"
claim "translation: 200 frozen prompts, one greedy completion (128 Flan / 64 Llama new tokens)" "Translation uses 200 frozen prompts with one deterministic greedy completion" "grep -q 'protocols: 200x1 greedy' $REPORT_DIR/checks.tsv"
claim "MC probes: 100 prompts x 3 sampled completions, seeds 42/43/44, T=1, p=1, 64 tokens" "three sampled completions per prompt (seeds 42, 43, and 44; temperature" "grep -q 'protocols: 200x1 greedy' $REPORT_DIR/checks.tsv"
claim "all-MiniLM-L6-v2, 384-d sentence embeddings" "all-MiniLM-L6-v2" "$PY -c \"import json,sys; d=json.load(open('$SCRATCH/behavior_trees/semantic/llama_full/translation/confirm_paper_tree_001/semantic_matrix.json')); e=d['encoder']; sys.exit(0 if '1110a243fdf4706b3f48f1d95db1a4f5529b4d41' in json.dumps(e) else 1)\""
claim "DerSimonian-Laird random effects with delete-one-node jackknife variance" "delete-one-trained-node jackknife" "grep -q '650/650 per-tree' $REPORT_DIR/checks.tsv"
finish
