#!/usr/bin/env bash
#SBATCH --job-name=wt-behavior-post
#SBATCH --output=slurm_logs/wt-behavior-post_%j.out
#SBATCH --error=slurm_logs/wt-behavior-post_%j.err
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

REPO="${REPO:-/home/export/sgallagh/WeightTraits}"
PYTHON_BIN="${PYTHON_BIN:-/home/export/sgallagh/.conda/envs/weighttraits/bin/python}"
: "${RESPONSES_DIR:?set RESPONSES_DIR to a complete tree response directory}"

cd "${REPO}"
mkdir -p slurm_logs
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

AUDIT_ARGS=(
  --responses "${RESPONSES_DIR}"
  --expected-samples-per-prompt "${BEHAVIOR_SAMPLES_PER_PROMPT:-3}"
  --out "${RESPONSES_DIR}/audit.json"
)
if [[ "${BEHAVIOR_EMPTY_POLICY:-drop}" == "preserve" ]]; then
  AUDIT_ARGS+=(--allow-empty-completed)
fi
PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli audit-behavior-responses "${AUDIT_ARGS[@]}"

if [[ -n "${BEHAVIOR_SURFACE_DISTANCE_DIR:-}" ]]; then
  PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli build-behavior-surface-distances \
    --responses-dir "${RESPONSES_DIR}" \
    --out "${BEHAVIOR_SURFACE_DISTANCE_DIR}"
fi

if [[ "${BEHAVIOR_BUILD_SEMANTIC:-true}" == "true" ]]; then
  : "${BEHAVIOR_EMBEDDINGS:?set BEHAVIOR_EMBEDDINGS when semantic analysis is enabled}"
  : "${BEHAVIOR_DISTANCE_DIR:?set BEHAVIOR_DISTANCE_DIR when semantic analysis is enabled}"
  EMBED_ARGS=(
    --responses-dir "${RESPONSES_DIR}"
    --embedding-model "${BEHAVIOR_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
    --batch-size "${BEHAVIOR_EMBEDDING_BATCH_SIZE:-64}"
    --local-files-only
    --out "${BEHAVIOR_EMBEDDINGS}"
  )
  if [[ "${BEHAVIOR_NATURAL_LANGUAGE_ONLY:-false}" == "true" ]]; then
    EMBED_ARGS+=(--natural-language-only)
  fi
  PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli embed-behavior-responses "${EMBED_ARGS[@]}"

  PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli build-behavior-embedding-distances \
    --embeddings "${BEHAVIOR_EMBEDDINGS}" \
    --out "${BEHAVIOR_DISTANCE_DIR}"
fi
