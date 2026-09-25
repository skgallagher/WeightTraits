#!/usr/bin/env bash
#SBATCH --job-name=wt-downstream-phylolm-smoke
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --qos=normal
#SBATCH --nodelist=n03
#SBATCH --exclude=n01
#SBATCH --no-requeue
#SBATCH --gres=gpu:nvidia-l40:1
#SBATCH --array=0-0%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

: "${WT_CODE_ROOT:?missing WT_CODE_ROOT}"
: "${WT_CODE_MANIFEST:?missing WT_CODE_MANIFEST}"
: "${WT_CODE_MANIFEST_SHA256:?missing WT_CODE_MANIFEST_SHA256}"
: "${WT_JOB_CONFIG:?missing WT_JOB_CONFIG}"
: "${WT_JOB_CONFIG_SHA256:?missing WT_JOB_CONFIG_SHA256}"
: "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
[[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]

PYTHON_BIN=/home/export/sgallagh/.conda/envs/ellmtrees/bin/python
export PYTHONPATH="${WT_CODE_ROOT}/src"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

exec "${PYTHON_BIN}" "${WT_CODE_ROOT}/scripts/run_downstream_job.py" \
  --kind phylolm_smoke \
  --config "${WT_JOB_CONFIG}" \
  --config-sha256 "${WT_JOB_CONFIG_SHA256}" \
  --code-root "${WT_CODE_ROOT}" \
  --code-manifest "${WT_CODE_MANIFEST}" \
  --code-manifest-sha256 "${WT_CODE_MANIFEST_SHA256}"
