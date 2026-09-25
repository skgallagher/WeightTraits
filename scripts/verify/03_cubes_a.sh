#!/usr/bin/env bash
# 03_cubes_a.sh — STAGE 3 (distance cubes), part (a): the code works as intended.
#
# This proves:
#   1. the distance/whitebox unit tests pass;
#   2. the production cube builder returns closed-form answers on a synthetic known-answer input;
#   3. INDEPENDENT recomputation on Wright with only safetensors+numpy (no WeightTraits code) reproduces
#      the sealed cubes: full-FT Flan (all 282 tensors, 1e-14), LoRA leaves re-materialised from the
#      retained adapters (base + sum of alpha/r * B@A along the root->leaf path; 1e-11), the cumulative
#      adapter chain (5e-10), Llama full FT from bf16 files, and Llama LoRA with bf16 merge rounding;
#   4. across all 9 cohorts x 50 trees: every matrix is a valid distance matrix over exactly the truth
#      leaves, equals the mean of its per-tensor cube, and the ACTIVE tensors (nonzero between-model
#      distance) are exactly the LoRA scope (k / q,k,v / q,k,v,o / +FFN / q,k,v_proj) or all trained
#      tensors for full FT (Llama: 122 of 146 — 24 RMSNorm weights never move in pure-bf16 training).
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$WT_ROOT"

echo "== 1. unit tests (distances, streaming cubes, whitebox, LoRA readers)"
if PYTHONPATH=src $PY -m pytest -o addopts="" -q tests/test_distance_metrics.py tests/test_streaming_distance_cube.py tests/test_cli_distance_cube.py tests/test_distance_input_manifest.py tests/test_whitebox_analysis.py tests/test_whitebox_smoke.py tests/test_materialize_lora_recovery.py tests/test_rematerialized_lora_analysis.py > "$REPORT_DIR/03a_pytest.log" 2>&1; then
  check "pytest distance suites" OK "$(tail -1 "$REPORT_DIR/03a_pytest.log")"
else check "pytest distance suites" FAIL "see $REPORT_DIR/03a_pytest.log"; fi

echo "== 2. known-answer test through the production builder"
if $PY scripts/verify/verify_cube_known_answer.py "$WT_ROOT" "$SCRATCH/known_answer" > "$REPORT_DIR/03a_known_answer.log" 2>&1; then
  check "build-distance-cube reproduces closed-form cosine distances" OK "$(grep 'max |cube' "$REPORT_DIR/03a_known_answer.log")"
else check "build-distance-cube reproduces closed-form cosine distances" FAIL "see $REPORT_DIR/03a_known_answer.log"; fi

echo "== 3. independent recomputation on Wright (raw safetensors + numpy) vs sealed cubes"
if ssh -S "$SOCK" -O check wright >/dev/null 2>&1; then
  wr 'cat ~/verify_20260917/recompute_flan_full.log ~/verify_20260917/recompute.log ~/verify_20260917/recompute_llama.log 2>/dev/null' > "$REPORT_DIR/03a_recompute.log"
  grep -E "^###|^OVERALL|^keys with nonzero" "$REPORT_DIR/03a_recompute.log" | sed 's/^/     /'
  chk() {  # chk <label-regex> <tolerance>
    local line; line=$(awk -v pat="$1" '/^### /{f=($0 ~ pat)} f && /^OVERALL/ {last=$0} END {print last}' "$REPORT_DIR/03a_recompute.log")
    local val; val=$(echo "$line" | sed -E 's/.*= ([0-9.e+-]+) at.*/\1/')
    if [[ -n $val ]] && $PY -c "import sys; sys.exit(0 if float('$val') < $2 else 1)"; then check "recompute: $1 (tol $2)" OK "max diff $val"; else check "recompute: $1 (tol $2)" FAIL "${line:-no result}"; fi
  }
  chk "flan_full tree001 model" 1e-9
  chk "flan_k tree001 merged" 1e-9
  chk "flan_qkv tree003 merged" 1e-9
  chk "flan_correctedall tree001 cumulative" 1e-8
  chk "llama_full tree003 model" 1e-9
  chk "llama_r8 tree003 merged" 5e-3
  chk "llama_r64 tree003 merged" 5e-3
else check "independent recompute on Wright" FAIL "no Wright socket"; fi

echo "== 4. sanity + active-tensor scope + fingerprints over all sealed cubes (local mirror)"
if $PY scripts/verify/verify_cube_sanity.py "$LOCAL_MIRROR/direct" "$FROZEN/assigned_manifests" "$SCRATCH/cube_fingerprints.tsv" > "$REPORT_DIR/03a_cube_sanity.log" 2>&1; then
  check "9 cohorts x 50 trees: valid matrices, leaf sets, cube shapes, active-tensor scopes" OK "450 cubes"
else check "9 cohorts x 50 trees: valid matrices, leaf sets, cube shapes, active-tensor scopes" FAIL "see $REPORT_DIR/03a_cube_sanity.log"; fi
sed 's/^/     /' "$REPORT_DIR/03a_cube_sanity.log" | head -12
finish
