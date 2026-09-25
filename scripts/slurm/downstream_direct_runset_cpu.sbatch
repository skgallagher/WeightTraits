#!/usr/bin/env bash
#SBATCH --job-name=wt-downstream-direct
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --qos=normal
#SBATCH --nodelist=n03
#SBATCH --exclude=n01
#SBATCH --no-requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

: "${WT_CODE_ROOT:?missing WT_CODE_ROOT}"
: "${WT_CODE_MANIFEST:?missing WT_CODE_MANIFEST}"
: "${WT_CODE_MANIFEST_SHA256:?missing WT_CODE_MANIFEST_SHA256}"
: "${WT_JOB_CONFIG:?missing WT_JOB_CONFIG}"
: "${WT_JOB_CONFIG_SHA256:?missing WT_JOB_CONFIG_SHA256}"

PYTHON_BIN=/home/export/sgallagh/.conda/envs/ellmtrees/bin/python
export PYTHONPATH="${WT_CODE_ROOT}/src"

exec "${PYTHON_BIN}" "${WT_CODE_ROOT}/scripts/run_downstream_job.py" \
  --kind direct_runset_cpu \
  --config "${WT_JOB_CONFIG}" \
  --config-sha256 "${WT_JOB_CONFIG_SHA256}" \
  --code-root "${WT_CODE_ROOT}" \
  --code-manifest "${WT_CODE_MANIFEST}" \
  --code-manifest-sha256 "${WT_CODE_MANIFEST_SHA256}"
