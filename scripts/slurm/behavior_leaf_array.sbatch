#!/usr/bin/env bash
#SBATCH --job-name=wt-behavior-leaf
#SBATCH --output=slurm_logs/wt-behavior-leaf_%A_%a.out
#SBATCH --error=slurm_logs/wt-behavior-leaf_%A_%a.err
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
: "${BEHAVIOR_PROMPTS:?set BEHAVIOR_PROMPTS to a prepared probe JSONL}"
: "${RESPONSES_DIR:?set RESPONSES_DIR to the tree response output directory}"
: "${RUN_ID:?set RUN_ID to a stable tree/probe run identifier}"
: "${MODEL_TASK:?set MODEL_TASK to seq2seq or causal_lm}"

cd "${REPO}"
mkdir -p slurm_logs "${RESPONSES_DIR}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

ARGS=(
  --input-manifest "${INPUT_MANIFEST}"
  --model-index "${SLURM_ARRAY_TASK_ID}"
  --run-id "${RUN_ID}"
  --prompts "${BEHAVIOR_PROMPTS}"
  --out-dir "${RESPONSES_DIR}"
  --model-task "${MODEL_TASK}"
  --samples-per-prompt "${BEHAVIOR_SAMPLES_PER_PROMPT:-3}"
  --min-new-tokens "${BEHAVIOR_MIN_NEW_TOKENS:-0}"
  --max-new-tokens "${BEHAVIOR_MAX_NEW_TOKENS:-64}"
  --temperature "${BEHAVIOR_TEMPERATURE:-1.0}"
  --top-p "${BEHAVIOR_TOP_P:-1.0}"
  --batch-size "${BEHAVIOR_BATCH_SIZE:-32}"
  --seed "${BEHAVIOR_SEED:-42}"
  --empty-policy "${BEHAVIOR_EMPTY_POLICY:-drop}"
  --local-files-only
)

if [[ "${BEHAVIOR_DO_SAMPLE:-true}" == "true" ]]; then
  ARGS+=(--do-sample)
else
  ARGS+=(--no-do-sample)
fi

if [[ -n "${BASE_MODEL:-}" ]]; then
  ARGS+=(--base-model "${BASE_MODEL}")
fi
if [[ -n "${BASE_REVISION:-}" ]]; then
  ARGS+=(--base-revision "${BASE_REVISION}")
fi
if [[ "${BEHAVIOR_OVERWRITE:-false}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli collect-behavior-responses "${ARGS[@]}"
