#!/usr/bin/env bash
#SBATCH --job-name=wt-legacy-cache
#SBATCH --output=slurm_logs/wt-legacy-cache_%j.out
#SBATCH --error=slurm_logs/wt-legacy-cache_%j.err
#SBATCH --account=statds
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-}}"
# This environment preserves the numerical package versions used by the legacy
# Llama runs (torch 2.5.1, transformers 5.8.0, datasets 4.8.5,
# accelerate 1.13.0, tokenizers 0.22.2, and peft 0.19.1).
PYTHON_BIN="${PYTHON_BIN:-/home/export/sgallagh/.conda/envs/ellmtrees/bin/python}"
REGISTRY="${REGISTRY:-examples/training/confirm_paper_numbers/dataset_registry_legacy_causal.yaml}"
FORMATS="${FORMATS:-examples/training/confirm_paper_numbers/dataset_formats_legacy_causal.yaml}"
CACHE_ROOT="${CACHE_ROOT:-data/confirm_paper_numbers/legacy_causal_seed42_cache}"
SUMMARY_OUT="${SUMMARY_OUT:-examples/training/confirm_paper_numbers/legacy_causal_seed42_data_cache_summary.json}"
SOURCE_RECEIPT="${SOURCE_RECEIPT:-examples/training/confirm_paper_numbers/legacy_causal_seed42_cache_source_receipt.json}"
LEGACY_SUMMARY_OUT="${LEGACY_SUMMARY_OUT:-${SUMMARY_OUT%.json}_legacy_sources.json}"
TREC_SUMMARY_OUT="${TREC_SUMMARY_OUT:-${SUMMARY_OUT%.json}_trec_extension.json}"
# Reuse the source cache from the historical ELLMTrees runs. Pointing this job
# at the newer WeightTraits cache would redownload mutable dataset repositories
# and could change selected rows even with the same seed.
LEGACY_HF_DATASETS_CACHE="${LEGACY_HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
LEGACY_DATASET_IDS=(
  cnn_dailymail xsum samsum billsum pubmed govreport arxiv_summ bigpatent dialogsum
  sst2 ag_news mnli imdb rte cola qnli yelp_polarity dbpedia_14
  squad squad_v2 trivia_qa boolq quartz arc_easy arc_challenge sciq
  wmt14_en_fr wmt14_en_de opus_en_de opus_en_fr opus_en_es opus_en_it
  opus_en_ru opus_en_zh opus_en_ja
)

if [[ -z "${REPO}" ]]; then
  printf 'REPO is required (or submit from the intended repository directory)\n' >&2
  exit 2
fi
for required_path in \
  pyproject.toml \
  src/weighttraits/cli.py \
  "${REGISTRY}" \
  "${FORMATS}" \
  "${SOURCE_RECEIPT}"
do
  if [[ "${required_path}" = /* ]]; then
    required_file="${required_path}"
  else
    required_file="${REPO}/${required_path}"
  fi
  if [[ ! -f "${required_file}" ]]; then
    printf 'repository contract path is missing: %s\n' "${required_file}" >&2
    exit 2
  fi
done

legacy_dataset_args=()
for dataset_id in "${LEGACY_DATASET_IDS[@]}"; do
  legacy_dataset_args+=(--dataset-id "${dataset_id}")
done

cd "${REPO}"
mkdir -p slurm_logs

env PYTHONPATH=src "${PYTHON_BIN}" -c \
  'import sys, torch, transformers, datasets, accelerate, tokenizers, peft; print({"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__, "accelerate": accelerate.__version__, "tokenizers": tokenizers.__version__, "peft": peft.__version__})'

env \
  PYTHONPATH=src \
  "HF_DATASETS_CACHE=${LEGACY_HF_DATASETS_CACHE}" \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  "${PYTHON_BIN}" -m weighttraits.cli cache-training-datasets \
  --registry "${REGISTRY}" \
  --formats "${FORMATS}" \
  "${legacy_dataset_args[@]}" \
  --out-dir "${CACHE_ROOT}" \
  --summary-out "${LEGACY_SUMMARY_OUT}" \
  --train-limit 10000 \
  --eval-limit 1000 \
  --min-train-rows 1 \
  --min-eval-rows 1 \
  --sample-strategy legacy_subsample \
  --sample-seed 42

# TREC-QC was not part of the historical source pool. Fetch that one explicit
# extension online at the revision pinned in the registry.
env \
  PYTHONPATH=src \
  "HF_DATASETS_CACHE=${LEGACY_HF_DATASETS_CACHE}" \
  TOKENIZERS_PARALLELISM=false \
  "${PYTHON_BIN}" -m weighttraits.cli cache-training-datasets \
  --registry "${REGISTRY}" \
  --formats "${FORMATS}" \
  --dataset-id trec \
  --out-dir "${CACHE_ROOT}" \
  --summary-out "${TREC_SUMMARY_OUT}" \
  --train-limit 10000 \
  --eval-limit 1000 \
  --min-train-rows 1 \
  --min-eval-rows 1 \
  --sample-strategy legacy_subsample \
  --sample-seed 42

env PYTHONPATH=src "${PYTHON_BIN}" -c '
import hashlib
import json
from pathlib import Path
import sys

source_paths = [Path(value) for value in sys.argv[1:3]]
reports = [json.loads(path.read_text()) for path in source_paths]
splits = [split for report in reports for split in report["splits"]]
dataset_ids = {split["dataset_id"] for split in splits}
source_receipt_path = Path(sys.argv[4])
source_receipt = json.loads(source_receipt_path.read_text())
actual_metadata = sorted(
    [json.loads(Path(split["path"]).with_suffix(".metadata.json").read_text()) for split in splits],
    key=lambda row: (row["dataset_id"], row["split"]),
)
expected_metadata = sorted(
    source_receipt["splits"],
    key=lambda row: (row["dataset_id"], row["split"]),
)
source_receipt_valid = actual_metadata == expected_metadata
combined = {
    "valid": (
        all(report["valid"] for report in reports)
        and len(dataset_ids) == 36
        and source_receipt_valid
    ),
    "out_dir": reports[0]["out_dir"],
    "n_datasets": len(dataset_ids),
    "n_ok": sum(report["n_ok"] for report in reports),
    "sample_strategy": "legacy_subsample",
    "sample_seed": 42,
    "shuffle_buffer_size": reports[0]["shuffle_buffer_size"],
    "source_reports": [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "valid": report["valid"],
            "n_datasets": report["n_datasets"],
        }
        for path, report in zip(source_paths, reports)
    ],
    "source_receipt": {
        "path": str(source_receipt_path),
        "sha256": hashlib.sha256(source_receipt_path.read_bytes()).hexdigest(),
        "valid": source_receipt_valid,
        "n_splits": len(expected_metadata),
    },
    "splits": splits,
}
Path(sys.argv[3]).write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
assert combined["valid"], combined
' "${LEGACY_SUMMARY_OUT}" "${TREC_SUMMARY_OUT}" "${SUMMARY_OUT}" "${SOURCE_RECEIPT}"
