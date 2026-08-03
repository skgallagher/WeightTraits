#!/usr/bin/env bash
#SBATCH --job-name=wt-phylolm-leaf
#SBATCH --output=slurm_logs/wt-phylolm-leaf_%A_%a.out
#SBATCH --error=slurm_logs/wt-phylolm-leaf_%A_%a.err
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --gres=gpu:nvidia-l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

REPO="${REPO:-/home/export/sgallagh/WeightTraits}"
PYTHON_BIN="${PYTHON_BIN:-/home/export/sgallagh/.conda/envs/weighttraits/bin/python}"
: "${INPUT_MANIFEST:?set INPUT_MANIFEST to a leaf checkpoint/adapter-chain manifest}"
: "${PHYLOLM_GENOME:?set PHYLOLM_GENOME to the sampled genome JSON}"
: "${POPULATION_DIR:?set POPULATION_DIR to the tree population output directory}"
: "${BASE_MODEL:?set BASE_MODEL to the pinned causal base model}"
: "${BASE_REVISION:?set BASE_REVISION to the pinned causal base revision}"

cd "${REPO}"
mkdir -p slurm_logs "${POPULATION_DIR}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli collect-phylolm-population \
  --input-manifest "${INPUT_MANIFEST}" \
  --model-index "${SLURM_ARRAY_TASK_ID}" \
  --genome "${PHYLOLM_GENOME}" \
  --out-dir "${POPULATION_DIR}" \
  --base-model "${BASE_MODEL}" \
  --base-revision "${BASE_REVISION}" \
  --samples-per-gene "${PHYLOLM_SAMPLES_PER_GENE:-32}" \
  --new-tokens "${PHYLOLM_NEW_TOKENS:-4}" \
  --allele-characters "${PHYLOLM_ALLELE_CHARACTERS:-4}" \
  --temperature "${PHYLOLM_TEMPERATURE:-1.0}" \
  --batch-size "${PHYLOLM_BATCH_SIZE:-64}" \
  --seed "${PHYLOLM_SEED:-0}" \
  --local-files-only
