#!/usr/bin/env bash
# 06_behavior_a.sh — BEHAVIOR-FROM-DISTANCE track, part (a): the code works as intended.
#
# This proves the whole bridge chain independently, from raw generated text up to Table 4:
#   1. the behavior unit tests pass;
#   2. raw text -> pinned all-MiniLM-L6-v2 -> paired cosine distance -> mean over (prompt, sample) identities
#      reproduces the sealed semantic matrix of a tree (Llama full FT, translation; 2,800 responses whose
#      stored sha256 match their text) to ~1e-7, and three models are shown answering the same prompt;
#   3. from the sealed semantic + weight-distance matrices, an independent numpy/scipy implementation
#      reproduces, for all 650 trees, the within-tree Pearson r and its delete-one-node jackknife variance,
#      and for all 13 runsets x 2 strata the DerSimonian-Laird r_DL, CI, I^2, tau^2, Q and p to machine precision;
#   4. the encoder snapshot used locally is the pinned revision, and the probe protocols in the repo match the
#      paper's appendix (200 x 1 greedy translation; 100 x 3 sampled MC probes, seeds 42/43/44, T=1, p=1).
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$WT_ROOT"
echo "== 1. unit tests"
if PYTHONPATH=src $PY -m pytest -o addopts="" -q tests/test_behavior_*.py tests/test_native_behavior_pipeline_cli.py tests/test_prompt_pairing_sensitivity.py tests/test_paper_paired.py > "$REPORT_DIR/06a_pytest.log" 2>&1; then
  check "pytest behavior suites" OK "$(tail -1 "$REPORT_DIR/06a_pytest.log")"; else check "pytest behavior suites" FAIL "see $REPORT_DIR/06a_pytest.log"; fi
echo "== 2. raw text -> encoder -> semantic matrix (Llama full FT, tree 001, translation)"
SNAP=$HOME/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
if [[ ! -d $SCRATCH/resp_t1/n0 ]]; then require_socket; mkdir -p "$SCRATCH/resp_t1"
  wr 'cd /home/export/sgallagh/WeightTraits-downstream-20260831-native/outputs/allnode_v1/inference/llama_full/confirm_paper_tree_001 && tar cf ~/verify_20260917/resp_t1.tar */responses.jsonl'
  scp -q -o ControlPath="$SOCK" wright:~/verify_20260917/resp_t1.tar "$SCRATCH/resp_t1/" && (cd "$SCRATCH/resp_t1" && tar xf resp_t1.tar); fi
[[ -d $SCRATCH/behavior_trees ]] || bash "$(dirname "$0")/06_behavior_b.sh" > /dev/null 2>&1 || true
if HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false $PY scripts/verify/verify_semantic_matrix.py "$SCRATCH/resp_t1" "$SCRATCH/behavior_trees/semantic/llama_full/translation/confirm_paper_tree_001/semantic_matrix.json" translation "$SNAP" > "$REPORT_DIR/06a_semantic_matrix.log" 2>&1; then
  check "semantic matrix recomputed from raw text == sealed (2800 responses, sha256-verified)" OK "$(grep 'max |recomputed' "$REPORT_DIR/06a_semantic_matrix.log")"; else check "semantic matrix from raw text" FAIL "see $REPORT_DIR/06a_semantic_matrix.log"; fi
grep -A3 "three models" "$REPORT_DIR/06a_semantic_matrix.log" | sed 's/^/     /'
echo "== 3. independent r / jackknife / DerSimonian-Laird for every tree and runset"
if $PY scripts/verify/verify_behavior_meta.py "$SCRATCH/behavior_trees" "$SCRATCH/behavior_meta_rescore.tsv" > "$REPORT_DIR/06a_behavior_meta.log" 2>&1; then
  check "650/650 per-tree r + jackknife var; 26/26 pooled r_DL, CI, I2, tau2, Q, p reproduced" OK; else check "independent meta-analysis" FAIL "see $REPORT_DIR/06a_behavior_meta.log"; fi
sed 's/^/     /' "$REPORT_DIR/06a_behavior_meta.log" | head -15; cp "$SCRATCH/behavior_meta_rescore.tsv" "$REPORT_DIR/06a_behavior_meta_rescore.tsv"
echo "== 4. encoder pin + probe protocols"
$PY - "$WT_ROOT/examples/behavior/corrected_behavior_protocols_v1.json" "$SNAP" <<'PYEOF' > "$REPORT_DIR/06a_protocols.log" 2>&1 && check "protocols: 200x1 greedy translation; 100x3 MC seeds 42/43/44 T=1 p=1 64 tok; Dolly 30x3; MiniLM rev 1110a243 pinned & cached" OK || check "protocols / encoder pin" FAIL "see $REPORT_DIR/06a_protocols.log"
import json, sys, os
d = json.load(open(sys.argv[1])); p = d["protocols"]; ok = True
t = p["legacy_translation_200x1_greedy"]; ok &= t["prompt_counts"] == {"translation": 200} and t["samples_per_prompt"] == 1 and t["do_sample"] is False and t["architectures"]["causal_lm"]["max_new_tokens"] == 64 and t["architectures"]["seq2seq"]["max_new_tokens"] == 128
m = p["modern_mc_100x3_sampled"]; ok &= m["prompt_counts"] == {k: 100 for k in ("hellaswag", "arc_challenge", "mmlu", "truthfulqa")} and m["draw_seeds"] == [42, 43, 44] and m["temperature"] == 1.0 and m["top_p"] == 1.0 and m["architectures"]["causal_lm"]["max_new_tokens"] == 64
dl = p["dolly_open_ended_30x3_sampled"]; ok &= dl["prompt_counts"] == {"dolly_open_ended": 30} and dl["draw_seeds"] == [42, 43, 44]
ok &= d["embedding_pin"] == {"model_id": "sentence-transformers/all-MiniLM-L6-v2", "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"} and os.path.isdir(sys.argv[2])
ok &= d["model_pins"]["causal_lm"]["revision"] == "4e20de362430cd3b72f300e6b0f18e50e7166e08" and d["model_pins"]["seq2seq"]["revision"] == "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
print({k: {kk: v[kk] for kk in ("prompt_counts", "samples_per_prompt", "draw_seeds", "do_sample", "temperature", "top_p")} for k, v in p.items()}); print("embedding_pin", d["embedding_pin"]); sys.exit(0 if ok else 1)
PYEOF
finish
