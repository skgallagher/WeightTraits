#!/usr/bin/env bash
#SBATCH --job-name=wt-behavior-tree
#SBATCH --output=slurm_logs/wt-behavior-tree_%A_%a.out
#SBATCH --error=slurm_logs/wt-behavior-tree_%A_%a.err
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --gres=gpu:nvidia-l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

# Run one complete behavioral panel for one production tree.  The array index is
# the one-based tree number.  A prompt file may contain several probes; the
# model is loaded only once per leaf, then output-surface cubes and pair tables
# are split by PROBE_IDS during post-processing.  MODEL_TASK and
# MANIFEST_ARTIFACT make the same audited path usable for causal adapter chains
# and merged seq2seq checkpoints.

REPO="${REPO:-/home/export/sgallagh/WeightTraits}"
ARTIFACT_REPO="${ARTIFACT_REPO:-${REPO}}"
PYTHON_BIN="${PYTHON_BIN:-/home/export/sgallagh/.conda/envs/weighttraits/bin/python}"
: "${BEHAVIOR_PROMPTS:?set BEHAVIOR_PROMPTS to one prepared JSONL, optionally multi-probe}"
: "${PROBE_IDS:?set PROBE_IDS to a comma- or colon-separated list of probe IDs}"
: "${PANEL_ID:?set PANEL_ID to the stable condition/prompt-panel identifier}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to the condition output directory}"

MODEL_TASK="${MODEL_TASK:-causal_lm}"
MANIFEST_ARTIFACT="${MANIFEST_ARTIFACT:-adapter_chain}"
case "${MODEL_TASK}" in
  causal_lm|seq2seq) ;;
  *) echo "MODEL_TASK must be causal_lm or seq2seq" >&2; exit 2 ;;
esac
case "${MANIFEST_ARTIFACT}" in
  model|merged|adapter_chain) ;;
  *) echo "MANIFEST_ARTIFACT must be model, merged, or adapter_chain" >&2; exit 2 ;;
esac
if [[ "${MANIFEST_ARTIFACT}" == "adapter_chain" ]]; then
  WEIGHT_REPRESENTATION="${WEIGHT_REPRESENTATION:-lora_cumulative_delta}"
else
  WEIGHT_REPRESENTATION="${WEIGHT_REPRESENTATION:-full_weight}"
fi

TREE_NUMBER="$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")"
TREE_ID="confirm_paper_tree_${TREE_NUMBER}"
RUN_ID="${PANEL_ID}-tree${TREE_NUMBER}"
LEDGER_DIR="${LEDGER_DIR:-${ARTIFACT_REPO}/examples/training/confirm_paper_numbers/llama32_1b_lora_qkv_r8_training_runlists/ledgers}"
TRUTH_DIR="${TRUTH_DIR:-${ARTIFACT_REPO}/examples/training/confirm_paper_numbers/assigned_manifests}"
TREE_OUT="${OUTPUT_ROOT}/tree${TREE_NUMBER}"
RESPONSES_DIR="${TREE_OUT}/responses"
INPUT_MANIFEST="${TREE_OUT}/leaf_models.yaml"

cd "${REPO}"
mkdir -p slurm_logs "${TREE_OUT}" "${RESPONSES_DIR}" "${TREE_OUT}/pairs"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli make-distance-input-manifest \
  --ledger "${LEDGER_DIR}/${TREE_ID}.training_ledger.jsonl" \
  --truth-manifest "${TRUTH_DIR}/${TREE_ID}.manifest.jsonl" \
  --artifact "${MANIFEST_ARTIFACT}" \
  --path-base "${ARTIFACT_REPO}" \
  --out "${INPUT_MANIFEST}"

N_MODELS="$(PYTHONPATH=src "${PYTHON_BIN}" -c \
  'import sys; from weighttraits.distances.manifest import load_distance_input_manifest; print(len(load_distance_input_manifest(sys.argv[1])))' \
  "${INPUT_MANIFEST}")"
if [[ "${N_MODELS}" -lt 2 ]]; then
  echo "Tree ${TREE_ID} has fewer than two retained leaves" >&2
  exit 1
fi

for ((model_index = 0; model_index < N_MODELS; model_index++)); do
  ARGS=(
    --input-manifest "${INPUT_MANIFEST}"
    --model-index "${model_index}"
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
    --empty-policy preserve
    --do-sample
    --local-files-only
  )
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
done

PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli audit-behavior-responses \
  --responses "${RESPONSES_DIR}" \
  --expected-samples-per-prompt "${BEHAVIOR_SAMPLES_PER_PROMPT:-3}" \
  --allow-empty-completed \
  --out "${TREE_OUT}/audit.json"

PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli build-distance-cube \
  --checkpoint-manifest "${INPUT_MANIFEST}" \
  --metric cosine \
  --out "${TREE_OUT}/whitebox_cumulative" \
  --representation "${WEIGHT_REPRESENTATION}"

IFS=',:' read -r -a PROBES <<< "${PROBE_IDS}"
for probe_id in "${PROBES[@]}"; do
  surface_dir="${TREE_OUT}/${probe_id}_surface_distances"
  embeddings="${TREE_OUT}/${probe_id}_embeddings.npz"
  semantic_dir="${TREE_OUT}/${probe_id}_semantic_distances"
  PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli build-behavior-surface-distances \
    --responses-dir "${RESPONSES_DIR}" \
    --probe-id "${probe_id}" \
    --out "${surface_dir}"
  PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli make-behavior-regression-pairs \
    --run-id "tree${TREE_NUMBER}" \
    --weight-cube "${TREE_OUT}/whitebox_cumulative" \
    --behavior-cube "${surface_dir}" \
    --weight-metric cosine \
    --behavior-metric output_surface_paired \
    --out "${TREE_OUT}/pairs/${probe_id}.csv"
  PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli embed-behavior-responses \
    --responses-dir "${RESPONSES_DIR}" \
    --probe-id "${probe_id}" \
    --embedding-model "${BEHAVIOR_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}" \
    --batch-size "${BEHAVIOR_EMBEDDING_BATCH_SIZE:-64}" \
    --local-files-only \
    --out "${embeddings}"
  PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli build-behavior-embedding-distances \
    --embeddings "${embeddings}" \
    --out "${semantic_dir}"
  PYTHONPATH=src "${PYTHON_BIN}" -m weighttraits.cli make-behavior-regression-pairs \
    --run-id "tree${TREE_NUMBER}" \
    --weight-cube "${TREE_OUT}/whitebox_cumulative" \
    --behavior-cube "${semantic_dir}" \
    --weight-metric cosine \
    --behavior-metric semantic_paired \
    --out "${TREE_OUT}/pairs/${probe_id}_semantic.csv"
done
