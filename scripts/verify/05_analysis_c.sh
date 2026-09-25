#!/usr/bin/env bash
# 05_analysis_c.sh — STAGE 5, part (c): paper statements about ordering, layers, Figure 2, PhyloLM == artifacts.
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$WT_ROOT"
claim() { local label=$1 frag=$2 cond=$3
  if ! grep -qF -- "$frag" "$PAPER_TEX"; then check "$label" FAIL "phrase not found in tex: $frag"; return; fi
  if eval "$cond"; then check "$label" OK "tex: \"${frag:0:70}\""; else check "$label" FAIL "tex says \"${frag:0:70}\" but artifacts disagree"; fi; }
ORD="$REPORT_DIR/05a_ordering.log"; [[ -s $ORD ]] || $PY scripts/verify/verify_ordering.py "$LOCAL_MIRROR/direct" "$FROZEN/assigned_manifests" "$SCRATCH/ordering_rescore.tsv" > "$ORD" 2>&1
ordrow() { awk -v c="$1" '$1==c {gsub(/[()]/,"",$5); gsub(/[()]/,"",$7); printf "%.2f %.2f %.2f %.2f", $4, $5, $6, $7}' "$ORD"; }   # rb rbse rz rse
claim "Table 2 ordering: Flan key 0.52 (0.09) / -0.53 (0.07)" "0.52 (0.09) & \$-0.53\$ (0.07)" "[[ \$(ordrow flan_k) == '0.52 0.09 -0.53 0.07' ]]"
claim "Table 2 ordering: Flan q/k/v 0.56 (0.09) / -0.62 (0.07)" "0.56 (0.09) & \$-0.62\$ (0.07)" "[[ \$(ordrow flan_qkv) == '0.56 0.09 -0.62 0.07' && \$(ordrow flan_qkvo) == '0.56 0.09 -0.62 0.07' ]]"
claim "Table 2 ordering: Flan all-proj 1.00 (0.00) / -0.91 (0.02)" "1.00 (0.00) & \$-0.91\$ (0.02)" "[[ \$(ordrow flan_correctedall) == '1.00 0.00 -0.91 0.02' ]]"
claim "Table 2 ordering: Flan full 0.69 (0.08) / -0.71 (0.06)" "0.69 (0.08) & \$-0.71\$ (0.06)" "[[ \$(ordrow flan_full) == '0.69 0.08 -0.71 0.06' ]]"
claim "Table 2 ordering: Llama r8 0.63 (0.08) / -0.64 (0.07)" "0.63 (0.08) & \$-0.64\$ (0.07)" "[[ \$(ordrow llama_r8) == '0.63 0.08 -0.64 0.07' ]]"
claim "Table 2 ordering: Llama r64 0.65 (0.08) / -0.67 (0.07)" "0.65 (0.08) & \$-0.67\$ (0.07)" "[[ \$(ordrow llama_r64) == '0.65 0.08 -0.67 0.07' ]]"
claim "Table 2 ordering: Llama full 0.64 (0.08) / -0.66 (0.07)" "0.64 (0.08) & \$-0.66\$ (0.07)" "[[ \$(ordrow llama_full) == '0.64 0.08 -0.66 0.07' ]]"
claim "every rank-biserial positive, every r_branch negative" "Every mean rank-biserial effect is positive and every" "$PY -c \"import sys; rows=[l.split() for l in open('$ORD') if l.startswith(('flan','llama'))]; sys.exit(0 if rows and all(float(r[3])>0 and float(r[5])<0 for r in rows) else 1)\""
claim "PhyloLM clade 35 / 24 / 52 (5)" "35\\% (5\\%)" "grep -q '35% (5%)' $REPORT_DIR/05a_phylolm.log && grep -q '24% (5%)' $REPORT_DIR/05a_phylolm.log && grep -q '52% (5%)' $REPORT_DIR/05a_phylolm.log && grep -qF '24\\% (5\\%)' $PAPER_TEX && grep -qF '52\\% (5\\%)' $PAPER_TEX"
LT=reports/paper/final_fixed2000_layertrace.json
claim "tab:layer_subsets: 6 subsets, counts 282/48/24/24/12/36, all 100.0 (0.0) clade and PAER" "\\texttt{enc\\_sak}     & encoder SA.k only                                     & 12  & 4.3           & 100.0 (0.0) & 100.0 (0.0)" "$PY -c \"
import json,sys; d=json.load(open('$LT')); r={x['subset_id']:x for x in d['subset_rows']}
sys.exit(0 if d['n_trees']==46 and [r[k]['n_tensors'] for k in ('full','high_signal','sak','encoder_h','enc_sak','low_signal')]==[282,48,24,24,12,36] and all(x['clade_recovery']==1.0 and x['paer']==1.0 and x['clade_recovery_se']==0.0 for x in d['subset_rows']) else 1)\""
claim "Flan per-tensor: key matrices and other tensors each average about 4% missed clades" "Flan key matrices and other tensors each average about 4\\% missed clades" "$PY -c \"
import json,sys; s=json.load(open('$LT'))['individual_layer_summary']; sys.exit(0 if round(s['mean_fn_key_matrices'])==4 and round(s['mean_fn_other_layers'])==4 else 1)\""
LL=reports/paper/fixed2000_llama_layertrace_summary.json
claim "Llama per-tensor: key 1.1% vs other 24.0% (NOTE: 'other' includes 24 frozen norm tensors at 100%; active non-key mean is 6.9% — decision D6)" "the Llama profile is more key-concentrated (1.1\\% vs.\\ 24.0\\%)" "$PY -c \"
import json,sys; s=json.load(open('$LL'))['individual_tensor_summary']; sys.exit(0 if round(s['mean_fn_key_matrices'],1)==1.1 and round(s['mean_fn_other_tensors'],1)==24.0 else 1)\""
F2=reports/paper/merged_weight_figure2.csv
claim "Figure 2 prose: simulations 92-100%, tensor subsets 90-100%, PhyloLM 24-52% clade recovery" "Tensor-subset estimates recover 90--100\\% of true clades" "$PY - <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open('$F2')))
def rng(level): v=[round(float(r['clade_recovery_pct'])) for r in rows if r['level']==level]; return min(v), max(v)
sim, sub, ph = rng('simulation'), rng('subset'), rng('phylolm')
print('simulation', sim, 'subset', sub, 'phylolm', ph); sys.exit(0 if sim==(92,100) and sub==(90,100) and ph==(24,52) else 1)
PYEOF"
claim "Figure 2 caption: 231 estimators, 46 paired trees, 2,000 replicates, seed 20260805" "paired-bootstrap odds-ratio estimates accompany" "grep -q '231,46,paired-tree percentile,2000,20260805' reports/paper/merged_weight_figure2_bootstrap.csv"
finish
