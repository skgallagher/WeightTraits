#!/usr/bin/env bash
# 02_training_b.sh — STAGE 2 (training), part (b): the trained checkpoints exist on Wright and differ.
#
# This proves, for all 12 cohorts (8 Table-2 rows, 2 diagnostics, 2 matched no-translation):
#   1. the sealed completion bindings say 641/641 nodes completed in 50/50 trees, valid, 0 errors/warnings;
#   2. walking the sealed checkpoint tree on Wright (wright_audit_training.py): every one of the 641 nodes has
#      a completion record at step 2000 with no stop reason, was initialised from its manifest parent, used the
#      pinned base revision, has 9 evaluation points and <=10,000/1,000 rows, and has its weight file on disk;
#   3. the 641 weight files in a cohort have 641 DIFFERENT fingerprints (sha256 of head/middle/tail 4 MiB
#      slices), and the same node never shares a fingerprint across cohorts -> 7,692 distinct trained models;
#   4. the LoRA cohorts resolved exactly the requested modules (k / q,k,v / q,k,v,o / +FFN) with the
#      parameter counts the paper reports.
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
require_socket
cd "$WT_ROOT"
AUD="$SCRATCH/audit"; mkdir -p "$AUD"

echo "== 1. completion bindings (sealed receipts)"
scp -q -o ControlPath="$SOCK" scripts/verify/wright_bindings_table.py wright:~/verify_20260917/
wr 'python3 ~/verify_20260917/wright_bindings_table.py' > "$REPORT_DIR/02b_bindings.log" 2>&1
sed 's/^/     /' "$REPORT_DIR/02b_bindings.log" | cut -c1-200
nb=$(grep -c "" "$REPORT_DIR/02b_bindings.log"); ng=$(grep -c "ok=641/641 trees=50 valid=True err=0 warn=0 rows_valid=50/50" "$REPORT_DIR/02b_bindings.log")
[[ $nb -ge 11 && $ng -eq $nb ]] && check "completion bindings: all say 641/641 nodes, 50/50 trees, valid, 0 errors" OK "$ng/$nb bindings" || check "completion bindings: all say 641/641 nodes, 50/50 trees, valid, 0 errors" FAIL "$ng/$nb"

echo "== 2-3. per-node audit of the sealed checkpoints (produced on Wright by wright_audit_training.py)"
if ! wr 'grep -q ALL_DONE ~/verify_20260917/run_audits.log'; then
  echo "     audit still running on Wright:"; wr 'tail -2 ~/verify_20260917/run_audits.log' | sed 's/^/     /'
  check "per-node audit complete on Wright" FAIL "run_audits.sh not finished; re-run this script later"; finish
fi
rm -f "$AUD"/audit_*.tsv; scp -q -o ControlPath="$SOCK" 'wright:~/verify_20260917/audit_flan_*.tsv' 'wright:~/verify_20260917/audit_llama_*.tsv' "$AUD/"
if $PY scripts/verify/summarize_training_audit.py "$AUD" flan_full flan_k flan_qkv flan_qkvo flan_correctedall flan_legacyall flan_qv llama_r8 llama_r64 llama_full flan_notrans llama_notrans > "$REPORT_DIR/02b_audit_summary.log" 2>&1; then
  check "12 cohorts x 641 nodes: completed@2000, parent-initialised, 641 distinct weights each" OK "table in $REPORT_DIR/02b_audit_summary.log"
else check "12 cohorts x 641 nodes: completed@2000, parent-initialised, 641 distinct weights each" FAIL "see $REPORT_DIR/02b_audit_summary.log"; fi
sed 's/^/     /' "$REPORT_DIR/02b_audit_summary.log"

echo "== 4. LoRA target audits on Wright (resolved modules + trainable parameter counts)"
scp -q -o ControlPath="$SOCK" scripts/verify/lora_audit_table.py wright:~/verify_20260917/ 2>/dev/null || true
wr 'python3 ~/verify_20260917/lora_audit_table.py' > "$REPORT_DIR/02b_lora_audits.log" 2>&1
sed 's/^/     /' "$REPORT_DIR/02b_lora_audits.log" | cut -c1-170
ok=1
grep -q "flan_t5_lora_k_r8.*requested=k .*trainable=   442,368" "$REPORT_DIR/02b_lora_audits.log" || ok=0
grep -q "flan_t5_lora_qkv_r8.*requested=q,k,v .*trainable= 1,327,104" "$REPORT_DIR/02b_lora_audits.log" || ok=0
grep -q "flan_t5_lora_qkvo_r8.*requested=q,k,v,o .*trainable= 1,769,472" "$REPORT_DIR/02b_lora_audits.log" || ok=0
grep -q "all_projections_corrected.*requested=q,k,v,o,wi_0,wi_1,wo .*trainable= 3,391,488" "$REPORT_DIR/02b_lora_audits.log" || ok=0
grep -q "llama32_1b_lora_qkv_r8.*requested=q_proj,k_proj,v_proj" "$REPORT_DIR/02b_lora_audits.log" || ok=0
grep -q "unmatched=\[\]" "$REPORT_DIR/02b_lora_audits.log" || ok=0
[[ $ok -eq 1 ]] && check "LoRA scopes resolved exactly as requested; param counts 442,368 / 1,327,104 / 1,769,472 / 3,391,488" OK || check "LoRA scopes resolved exactly as requested; param counts" FAIL
finish
