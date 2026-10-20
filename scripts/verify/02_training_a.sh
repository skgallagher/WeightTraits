#!/usr/bin/env bash
# 02_training_a.sh — STAGE 2 (task/data assignment + training), part (a): the code works as intended.
#
# This proves:
#   1. the training/protocol unit tests pass (protocol guards: exactly 2,000 steps, no early stopping, seed 42);
#   2. every frozen cohort config resolves to exactly the hyperparameters the paper's appendix states
#      (201 fields over 10 cohorts: revisions, steps, lr, batch, GA, warmup, wd, lengths, LoRA r/alpha/dropout/scope);
#   3. on Wright, regenerating every cohort's run lists from the sealed code + configs + data cache reproduces
#      the sealed run lists BYTE FOR BYTE (12 cohorts x 50 trees; only the run-list directory's own name differs);
#   4. a real 2-node lineage training (3 steps, cached flan-t5-base, CPU) shows the child is initialised from the
#      parent's saved model, every tensor moves, and the child's displacement is aligned with the parent's.
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$WT_ROOT"

echo "== 1. unit tests (training planner / executor / ledger / protocol guards / task assignment)"
if PYTHONPATH=src $PY -m pytest -o addopts="" -q tests/test_training_*.py tests/test_legacy_*_rebuild_protocol.py tests/test_task_assignment.py > "$REPORT_DIR/02a_pytest.log" 2>&1; then
  check "pytest training suites" OK "$(tail -1 "$REPORT_DIR/02a_pytest.log")"
else check "pytest training suites" FAIL "see $REPORT_DIR/02a_pytest.log"; fi

echo "== 2. cohort configs == paper appendix hyperparameters"
if $PY scripts/verify/verify_training_configs.py "$FROZEN" examples/training/translation_holdout_20260803 > "$REPORT_DIR/02a_configs.log" 2>&1; then
  check "10 cohort configs match paper hyperparameters" OK "$(grep -c '  ok$' "$REPORT_DIR/02a_configs.log") fields, 0 mismatches"
else check "10 cohort configs match paper hyperparameters" FAIL "see $REPORT_DIR/02a_configs.log"; fi

echo "== 3. regenerate all run lists on Wright from the sealed stage and byte-compare"
if ssh -S "$SOCK" -O check wright >/dev/null 2>&1; then
  scp -q -o ControlPath="$SOCK" scripts/verify/wright_regen_runlists.sh wright:~/verify_20260917/
  wr 'chmod +x ~/verify_20260917/wright_regen_runlists.sh && ~/verify_20260917/wright_regen_runlists.sh' > "$REPORT_DIR/02a_regen_runlists.log" 2>&1
  n_id=$(grep -c " IDENTICAL " "$REPORT_DIR/02a_regen_runlists.log"); n_all=$(grep -c "" "$REPORT_DIR/02a_regen_runlists.log")
  [[ $n_id -eq 12 && $n_all -eq 12 ]] && check "12 cohorts: regenerated run lists == sealed (byte-identical)" OK "12/12 IDENTICAL, each valid 50 trees / 641 runs / 0 errors" \
                                     || check "12 cohorts: regenerated run lists == sealed (byte-identical)" FAIL "$n_id/12 identical; see $REPORT_DIR/02a_regen_runlists.log"
  sed 's/^/     /' "$REPORT_DIR/02a_regen_runlists.log" | cut -c1-120
else check "12 cohorts: regenerated run lists == sealed (byte-identical)" FAIL "no Wright socket"; fi

echo "== 4. real lineage training smoke (2 nodes, 3 steps, cached flan-t5-base)"
if $PY scripts/verify/verify_lineage_smoke.py "$WT_ROOT" "$SCRATCH/lineage_smoke" > "$REPORT_DIR/02a_lineage_smoke.log" 2>&1; then
  check "lineage smoke: child init_from parent, all tensors move, displacement aligned" OK "$(grep -E '^\|\|n0' "$REPORT_DIR/02a_lineage_smoke.log")"
else check "lineage smoke: child init_from parent, all tensors move, displacement aligned" FAIL "see $REPORT_DIR/02a_lineage_smoke.log"; fi
grep -E "^(n0|n1|tensors)" "$REPORT_DIR/02a_lineage_smoke.log" | sed 's/^/     /'
finish
