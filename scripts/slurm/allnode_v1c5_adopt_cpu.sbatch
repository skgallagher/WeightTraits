#!/usr/bin/env bash
# Submitted only by the allnode-v1c5 one-shot continuation boundary.
#SBATCH --job-name=wt-an-v1c5-adopt-llama-full
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

STAGE_ROOT=/home/export/sgallagh/WeightTraits-downstream-20260831-native
CONTINUATION_ROOT="${STAGE_ROOT}/deployments/allnode_v1_continuation_20260908_v1c5"
PYTHON_BIN=/home/export/sgallagh/.conda/envs/ellmtrees/bin/python

: "${WT_V1C5_CODE_MANIFEST:?missing WT_V1C5_CODE_MANIFEST}"
: "${WT_V1C5_CODE_MANIFEST_SHA256:?missing WT_V1C5_CODE_MANIFEST_SHA256}"
: "${WT_V1C5_EXPECTED_ACCEPTANCE:?missing WT_V1C5_EXPECTED_ACCEPTANCE}"
: "${WT_V1C5_EXPECTED_ACCEPTANCE_SHA256:?missing WT_V1C5_EXPECTED_ACCEPTANCE_SHA256}"
: "${WT_V1C5_SCHEDULER_EVIDENCE:?missing WT_V1C5_SCHEDULER_EVIDENCE}"
: "${WT_V1C5_SCHEDULER_EVIDENCE_SHA256:?missing WT_V1C5_SCHEDULER_EVIDENCE_SHA256}"
: "${WT_V1C5_ACCEPTANCE_RECEIPT:?missing WT_V1C5_ACCEPTANCE_RECEIPT}"

[[ "${WT_V1C5_CODE_MANIFEST}" == "${CONTINUATION_ROOT}/runtime_receipts/ALLNODE_V1C5_CODE_MANIFEST.json" ]]
[[ "${WT_V1C5_EXPECTED_ACCEPTANCE}" == "${CONTINUATION_ROOT}/runtime_receipts/ALLNODE_V1C5_EXPECTED_LLAMA_FULL_ARTIFACT_ACCEPTANCE.json" ]]
[[ "${WT_V1C5_SCHEDULER_EVIDENCE}" == "${CONTINUATION_ROOT}/runtime_receipts/ALLNODE_V1C5_SCHEDULER_EVIDENCE.json" ]]
[[ "${WT_V1C5_ACCEPTANCE_RECEIPT}" == "${CONTINUATION_ROOT}/runtime_receipts/ALLNODE_V1C5_LLAMA_FULL_ARTIFACT_ACCEPTANCE.json" ]]
[[ "${WT_V1C5_CODE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${WT_V1C5_EXPECTED_ACCEPTANCE_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${WT_V1C5_SCHEDULER_EVIDENCE_SHA256}" =~ ^[0-9a-f]{64}$ ]]

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${CONTINUATION_ROOT}:${STAGE_ROOT}/deployments/allnode_v1:${STAGE_ROOT}/src"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" -m weighttraits_allnode_continuation attest \
  --code-manifest "${WT_V1C5_CODE_MANIFEST}" \
  --code-manifest-sha256 "${WT_V1C5_CODE_MANIFEST_SHA256}"

"${PYTHON_BIN}" -m weighttraits_allnode_continuation adopt \
  --code-manifest "${WT_V1C5_CODE_MANIFEST}" \
  --code-manifest-sha256 "${WT_V1C5_CODE_MANIFEST_SHA256}" \
  --expected-acceptance "${WT_V1C5_EXPECTED_ACCEPTANCE}" \
  --expected-acceptance-sha256 "${WT_V1C5_EXPECTED_ACCEPTANCE_SHA256}" \
  --scheduler-evidence "${WT_V1C5_SCHEDULER_EVIDENCE}" \
  --scheduler-evidence-sha256 "${WT_V1C5_SCHEDULER_EVIDENCE_SHA256}" \
  --acceptance-receipt "${WT_V1C5_ACCEPTANCE_RECEIPT}"

"${PYTHON_BIN}" -m weighttraits_allnode_continuation attest \
  --code-manifest "${WT_V1C5_CODE_MANIFEST}" \
  --code-manifest-sha256 "${WT_V1C5_CODE_MANIFEST_SHA256}" \
  --acceptance "${WT_V1C5_ACCEPTANCE_RECEIPT}" \
  --acceptance-sha256 "${WT_V1C5_EXPECTED_ACCEPTANCE_SHA256}"
