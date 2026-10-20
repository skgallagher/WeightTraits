#!/usr/bin/env bash
# Thin immutable binding around the original, pinned inference/semantic wrappers.
set -euo pipefail

STAGE_ROOT=/home/export/sgallagh/WeightTraits-downstream-20260831-native
OLD_DEPLOYMENT_ROOT="${STAGE_ROOT}/deployments/allnode_v1"
CONTINUATION_ROOT="${STAGE_ROOT}/deployments/allnode_v1_continuation_20260908_v1c5"
PYTHON_BIN=/home/export/sgallagh/.conda/envs/ellmtrees/bin/python

: "${WT_V1C5_CODE_MANIFEST:?missing WT_V1C5_CODE_MANIFEST}"
: "${WT_V1C5_CODE_MANIFEST_SHA256:?missing WT_V1C5_CODE_MANIFEST_SHA256}"
: "${WT_V1C5_ACCEPTANCE_RECEIPT:?missing WT_V1C5_ACCEPTANCE_RECEIPT}"
: "${WT_V1C5_ACCEPTANCE_RECEIPT_SHA256:?missing WT_V1C5_ACCEPTANCE_RECEIPT_SHA256}"
: "${WT_ALLNODE_CONTINUATION_LOGICAL_ID:?missing WT_ALLNODE_CONTINUATION_LOGICAL_ID}"
: "${WT_ALLNODE_JOB_CONFIG:?missing WT_ALLNODE_JOB_CONFIG}"
: "${WT_ALLNODE_KIND:?missing WT_ALLNODE_KIND}"
: "${WT_ALLNODE_DEPLOYMENT_ROOT:?missing WT_ALLNODE_DEPLOYMENT_ROOT}"

[[ "${WT_V1C5_CODE_MANIFEST}" == "${CONTINUATION_ROOT}/runtime_receipts/ALLNODE_V1C5_CODE_MANIFEST.json" ]]
[[ "${WT_V1C5_ACCEPTANCE_RECEIPT}" == "${CONTINUATION_ROOT}/runtime_receipts/ALLNODE_V1C5_LLAMA_FULL_ARTIFACT_ACCEPTANCE.json" ]]
[[ "${WT_V1C5_CODE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${WT_V1C5_ACCEPTANCE_RECEIPT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${WT_ALLNODE_DEPLOYMENT_ROOT}" == "${OLD_DEPLOYMENT_ROOT}" ]]
case "${WT_ALLNODE_KIND}" in
  inference_allnode_gpu|semantic_allnode_gpu) ;;
  *) echo "unapproved continuation runner kind" >&2; exit 2 ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${CONTINUATION_ROOT}:${OLD_DEPLOYMENT_ROOT}:${STAGE_ROOT}/src"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" -m weighttraits_allnode_continuation attest \
  --code-manifest "${WT_V1C5_CODE_MANIFEST}" \
  --code-manifest-sha256 "${WT_V1C5_CODE_MANIFEST_SHA256}" \
  --acceptance "${WT_V1C5_ACCEPTANCE_RECEIPT}" \
  --acceptance-sha256 "${WT_V1C5_ACCEPTANCE_RECEIPT_SHA256}"

RUN_ARGS=(
  run
  --code-manifest "${WT_V1C5_CODE_MANIFEST}"
  --code-manifest-sha256 "${WT_V1C5_CODE_MANIFEST_SHA256}"
  --acceptance "${WT_V1C5_ACCEPTANCE_RECEIPT}"
  --acceptance-sha256 "${WT_V1C5_ACCEPTANCE_RECEIPT_SHA256}"
  --config "${WT_ALLNODE_JOB_CONFIG}"
)
if [[ "${WT_ALLNODE_KIND}" == "inference_allnode_gpu" ]]; then
  : "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
  RUN_ARGS+=(--array-index "${SLURM_ARRAY_TASK_ID}")
fi

"${PYTHON_BIN}" -m weighttraits_allnode_continuation "${RUN_ARGS[@]}"
