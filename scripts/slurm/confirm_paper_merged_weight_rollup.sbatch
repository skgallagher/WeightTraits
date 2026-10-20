#!/usr/bin/env bash
#SBATCH --job-name=wt-merged-rollup
#SBATCH --output=slurm_logs/wt-merged-rollup_%j.out
#SBATCH --error=slurm_logs/wt-merged-rollup_%j.err
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

CODE_REPO="${CODE_REPO:-${REPO:-/home/export/sgallagh/WeightTraits}}"
DATA_REPO="${DATA_REPO:-${CODE_REPO}}"
PYTHON_BIN="${PYTHON_BIN:-/home/export/sgallagh/.conda/envs/weighttraits/bin/python}"
SUITE="${SUITE:?set SUITE to a completed merged-weight suite}"
DATE_TAG="${DATE_TAG:-20260804}"
ANALYSIS_FAMILY="${ANALYSIS_FAMILY:-flan}"
ROOT="${DATA_REPO}/outputs/analysis_${ANALYSIS_FAMILY}_merged_weights_${DATE_TAG}"
AGGREGATE="${ROOT}/aggregate"

cd "${CODE_REPO}"
mkdir -p "${AGGREGATE}" slurm_logs
env PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli summarize-training-run-set-analysis \
  --analysis-root "${ROOT}/${SUITE}" \
  --artifact merged \
  --out "${AGGREGATE}/${SUITE}_rollup.json" \
  --csv-out "${AGGREGATE}/${SUITE}_rows.csv"
