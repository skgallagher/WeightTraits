#!/usr/bin/env bash
# 00_env.sh — shared settings for every verification script in this directory.
#
# Source this file; do not run it.  Every path below is READ-ONLY for the
# verification scripts: nothing here ever writes into a sealed directory.
#
#   local repos / mirror
WT_ROOT=/Users/shannon/Desktop/phylo/WeightTraits
ELLM_ROOT=/Users/shannon/Desktop/phylo/ELLMTrees
PAPER_ROOT=/Users/shannon/Desktop/phylo/ELLMTrees-paper
PAPER_TEX=${PAPER_TEX:-$PAPER_ROOT/iclr_draft_v4_numbers_20260911.tex}   # override: PAPER_TEX=.../iclr_draft_v5.tex bash scripts/verify/run_all.sh
LOCAL_MIRROR=/Users/shannon/Desktop/phylo/outputs/01a0813c-87a9-7572-b27f-19208b163c60/sources/wright
CLAIM_INDEX=/Users/shannon/Desktop/phylo/outputs/01a0813c-87a9-7572-b27f-19208b163c60/available_final_paper_results_20260911.csv
BHV_LOCAL=$ELLM_ROOT/results/aggregate/controlled_bhv_fixed2000_20260915_v2
FROZEN=$WT_ROOT/examples/training/confirm_paper_numbers       # frozen trees / assignments / run lists
PY=/opt/homebrew/Caskroom/miniforge/base/envs/ellmtrees/bin/python
# This Mac env ships two OpenMP runtimes (torch's libomp + OpenBLAS's); importing scipy.linalg after
# torch aborts the process depending on import order.  The standard workaround; not a code issue.
export KMP_DUPLICATE_LIB_OK=TRUE
#   Wright (sealed)
WRIGHT_STAGE=/home/export/sgallagh/WeightTraits-all-results-20260817-clean
WRIGHT_DS=/home/export/sgallagh/WeightTraits-downstream-20260831-native
WRIGHT_PY=/home/export/sgallagh/.conda/envs/ellmtrees/bin/python
STAGE_SUMS_SHA=642912687e67d0a48162a8d5cc0a95164ad92350ea629cd396fcfdd7917cd5d1
DEPLOYED_CODE_SHA=11a95522b1e046731666d4f678a39d39103961bc0b83ae9fe75cfbd4f1a433cb
SOCK=/tmp/wright-codex.sock
#   where this run's logs go
VERIFY_DATE=${VERIFY_DATE:-$(date +%Y%m%d)}
REPORT_DIR=$WT_ROOT/reports/verify/$VERIFY_DATE
mkdir -p "$REPORT_DIR"
SCRATCH=${SCRATCH:-/private/tmp/claude-502/-Users-shannon-ELLMTrees/e106527a-288e-4130-95e1-270444828fe6/scratchpad/verify}
mkdir -p "$SCRATCH"

# wr 'command'  — run a read-only command on Wright through the user-opened control socket
wr() { ssh -S "$SOCK" -o BatchMode=yes wright "$@"; }

# sha256 on either side, printed as "<hash>  <path>"
lsha() { shasum -a 256 "$@"; }

# Result bookkeeping: each script calls `check NAME OK|FAIL "detail"` and ends with `finish`.
_FAILS=0; _NAME=${_NAME:-$(basename "${BASH_SOURCE[1]:-$0}" .sh)}
check() {            # check <label> <OK|FAIL> [detail]
  local label=$1 status=$2 detail=${3:-}
  if [[ $status == OK ]]; then printf '  [ok]   %-58s %s\n' "$label" "$detail"
  else printf '  [FAIL] %-58s %s\n' "$label" "$detail"; _FAILS=$((_FAILS+1)); fi
  printf '%s\t%s\t%s\t%s\n' "$_NAME" "$label" "$status" "$detail" >> "$REPORT_DIR/checks.tsv"
}
finish() {
  if [[ $_FAILS -eq 0 ]]; then echo "PASS $_NAME"; exit 0
  else echo "FAIL $_NAME ($_FAILS failing checks)"; exit 1; fi
}
require_socket() {
  if ! ssh -S "$SOCK" -O check wright >/dev/null 2>&1; then
    echo "No Wright control socket. Open one in a terminal first:"
    echo "    ssh -M -S $SOCK -fN wright"; exit 2
  fi
}
