#!/usr/bin/env bash
# 03_cubes_c.sh — STAGE 3 (distance cubes), part (c): the paper's distance statements match the artifacts.
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$WT_ROOT"
SAN="$REPORT_DIR/03a_cube_sanity.log"; [[ -s $SAN ]] || $PY scripts/verify/verify_cube_sanity.py "$LOCAL_MIRROR/direct" "$FROZEN/assigned_manifests" "$SCRATCH/cube_fingerprints.tsv" > "$SAN" 2>&1
claim() { local label=$1 frag=$2 cond=$3
  if ! grep -qF -- "$frag" "$PAPER_TEX"; then check "$label" FAIL "phrase not found in tex: $frag"; return; fi
  if eval "$cond"; then check "$label" OK "tex: \"${frag:0:70}\""; else check "$label" FAIL "tex says \"${frag:0:70}\" but artifacts disagree"; fi; }
A="$LOCAL_MIRROR/direct/flan_full/analysis/confirm_paper_tree_001/model_leaf_analysis"
claim "282 trained tensors in the Flan full-FT set" "282-tensor trained set" "[[ \$($PY -c \"import json;print(len(json.load(open('$A/layers.json'))))\") == 282 ]]"
claim "enc_sak = 12 tensors (4.3%) = 7,077,888 of 247,577,856 scalars (2.86%)" "7,077,888/247,577,856" "$PY - <<'PYEOF'
import json, sys
aud = json.load(open('$A/distance_audit.json'))
enc_k = [L for L in aud['layers'] if L['name'].startswith('encoder.') and L['name'].endswith('SelfAttention.k.weight')]
tot = sum(L['numel'] for L in aud['layers'])
ok = len(enc_k) == 12 and sum(L['numel'] for L in enc_k) == 7077888 and tot == 247577856 and round(100*12/282,1) == 4.3 and round(100*7077888/tot,2) == 2.86
print('enc_sak tensors', len(enc_k), 'params', sum(L['numel'] for L in enc_k), 'total', tot); sys.exit(0 if ok else 1)
PYEOF"
claim "cosine distance between flattened tensors" "cosine" "grep -q 'PASS verify_cube_sanity' $SAN"
claim "LoRA rows use only tensors with nonzero between-model cosine distance (active = exact LoRA scope)" "only tensors with nonzero" "grep -qE '^flan_k .*problems=none' $SAN && grep -qE '^flan_qkv .*problems=none' $SAN && grep -qE '^flan_qkvo .*problems=none' $SAN && grep -qE '^llama_r8 .*problems=none' $SAN"
claim "LoRA rows compare sequentially merged full weights (re-materialised merge matches sealed cubes)" "sequentially merged full weights" "grep -q 'recompute: flan_k tree001 merged.*OK' $REPORT_DIR/checks.tsv || grep -qE 'flan_k tree001 merged.*\tOK' $REPORT_DIR/checks.tsv"
claim "dagger row = cumulative update-space (adapter-chain) ablation" "cumulative update-space all-projections ablation" "grep -qE 'flan_correctedall tree001 cumulative.*\tOK' $REPORT_DIR/checks.tsv && grep -qE '^flan_correctedall .*problems=none' $SAN"
claim "Llama-3.2-1B width d=2048; Flan d=768" "(\$d=2048\$)" "$PY -c \"
import json; a=json.load(open('$LOCAL_MIRROR/direct/llama_full/analysis/confirm_paper_tree_001/model_leaf_analysis/distance_audit.json'))
q=[L for L in a['layers'] if L['name'].endswith('layers.0.self_attn.q_proj.weight')][0]; b=json.load(open('$A/distance_audit.json'))
k=[L for L in b['layers'] if L['name']=='encoder.block.0.layer.0.SelfAttention.k.weight'][0]
import sys; sys.exit(0 if q['shape']==[2048,2048] and k['shape']==[768,768] else 1)\""
finish
