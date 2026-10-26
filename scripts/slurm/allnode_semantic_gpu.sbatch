#!/usr/bin/env bash
# Production template only. Submission is exclusively controlled by launch.py.
#SBATCH --job-name=wt-allnode-semantic
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --qos=normal
#SBATCH --nodelist=n03
#SBATCH --exclude=n01
#SBATCH --no-requeue
#SBATCH --gres=gpu:nvidia-l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

STAGE_ROOT=/home/export/sgallagh/WeightTraits-downstream-20260831-native
DEPLOYMENT_ROOT="${STAGE_ROOT}/deployments/allnode_v1"
PYTHON_BIN=/home/export/sgallagh/.conda/envs/ellmtrees/bin/python

: "${WT_ALLNODE_COHORT_KEY:?missing WT_ALLNODE_COHORT_KEY}"
: "${WT_ALLNODE_JOB_CONFIG:?missing WT_ALLNODE_JOB_CONFIG}"
: "${WT_ALLNODE_JOB_CONFIG_SHA256:?missing WT_ALLNODE_JOB_CONFIG_SHA256}"
: "${WT_ALLNODE_KIND:?missing WT_ALLNODE_KIND}"
: "${WT_ALLNODE_LOGICAL_ID:?missing WT_ALLNODE_LOGICAL_ID}"
: "${WT_ALLNODE_OUTPUT_DIR:?missing WT_ALLNODE_OUTPUT_DIR}"
: "${WT_ALLNODE_STAGE_ROOT:?missing WT_ALLNODE_STAGE_ROOT}"
: "${WT_ALLNODE_DEPLOYMENT_ROOT:?missing WT_ALLNODE_DEPLOYMENT_ROOT}"
: "${WT_ALLNODE_SOURCE_RESOLUTION:?missing WT_ALLNODE_SOURCE_RESOLUTION}"
: "${WT_ALLNODE_SOURCE_RESOLUTION_SHA256:?missing WT_ALLNODE_SOURCE_RESOLUTION_SHA256}"
: "${WT_ALLNODE_SOURCE_SCOPE_SHA256:?missing WT_ALLNODE_SOURCE_SCOPE_SHA256}"
: "${WT_ALLNODE_CODE_MANIFEST:?missing WT_ALLNODE_CODE_MANIFEST}"
: "${WT_ALLNODE_CODE_MANIFEST_SHA256:?missing WT_ALLNODE_CODE_MANIFEST_SHA256}"
: "${WT_DOWNSTREAM_CODE_MANIFEST:?missing WT_DOWNSTREAM_CODE_MANIFEST}"
: "${WT_DOWNSTREAM_CODE_MANIFEST_SHA256:?missing WT_DOWNSTREAM_CODE_MANIFEST_SHA256}"
: "${WT_ALLNODE_PRODUCTION_INTERPRETER:?missing WT_ALLNODE_PRODUCTION_INTERPRETER}"
: "${WT_ALLNODE_PYTHON_VERSION:?missing WT_ALLNODE_PYTHON_VERSION}"

case "${WT_ALLNODE_COHORT_KEY}" in
  llama_full|llama_notrans|flan_full|flan_notrans) ;;
  *) echo "invalid WT_ALLNODE_COHORT_KEY" >&2; exit 2 ;;
esac
[[ "${WT_ALLNODE_KIND}" == "semantic_allnode_gpu" ]]
[[ "${WT_ALLNODE_LOGICAL_ID}" == "semantic_${WT_ALLNODE_COHORT_KEY}" ]]
[[ "${WT_ALLNODE_JOB_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${WT_ALLNODE_JOB_CONFIG}" == "${DEPLOYMENT_ROOT}/configs/jobs/semantic_${WT_ALLNODE_COHORT_KEY}.json" ]]
[[ "${WT_ALLNODE_OUTPUT_DIR}" == "${STAGE_ROOT}/outputs/allnode_v1/semantic/${WT_ALLNODE_COHORT_KEY}" ]]
[[ "${WT_ALLNODE_STAGE_ROOT}" == "${STAGE_ROOT}" ]]
[[ "${WT_ALLNODE_DEPLOYMENT_ROOT}" == "${DEPLOYMENT_ROOT}" ]]
[[ "${WT_ALLNODE_PRODUCTION_INTERPRETER}" == "${PYTHON_BIN}" ]]

OBSERVED_CONFIG_SHA256="$(sha256sum "${WT_ALLNODE_JOB_CONFIG}" | cut -d' ' -f1)"
[[ "${OBSERVED_CONFIG_SHA256}" == "${WT_ALLNODE_JOB_CONFIG_SHA256}" ]]

export PYTHONPATH="${DEPLOYMENT_ROOT}:${STAGE_ROOT}/src"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" -m weighttraits_allnode.provenance --phase before-wrapper
set +e
"${PYTHON_BIN}" -m weighttraits_allnode.semantic_runner --config "${WT_ALLNODE_JOB_CONFIG}"
RUNNER_STATUS=$?
set -e
"${PYTHON_BIN}" -m weighttraits_allnode.provenance --phase after-wrapper
exit "${RUNNER_STATUS}"
