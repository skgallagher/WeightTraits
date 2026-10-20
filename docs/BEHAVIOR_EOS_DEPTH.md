# Corrected behavioral EOS sensitivity and depth summaries

Decoded blank responses are behavioral outcomes in the corrected analysis. Because decoding strips
special tokens, an immediate EOS normally appears as `text == ""`; the current artifact is therefore
an EOS/blank proxy rather than token-level proof of an EOS stop. The primary behavioral endpoint
retains every decoded response, including `text.strip() == ""`; the paired-nonblank result is reported
only as a sensitivity analysis and never replaces the primary endpoint.

`scripts/build_behavior_eos_depth.py` builds the workbook-ready empty-response diagnostics from
native behavioral inference artifacts. It is read-only with respect to inference outputs: every
response JSONL is reached through a completed inference receipt, and every input path is
SHA-256-pinned in an explicit inventory.

## Input contract

The input has schema `weighttraits.behavior.eos_depth_input.v1`:

```json
{
  "schema": "weighttraits.behavior.eos_depth_input.v1",
  "panel_id": "llama_full_ft_modern_mc",
  "registry": {"path": "/sealed/corrected_behavior_protocols_v1.json", "sha256": "..."},
  "prompt_artifacts": {
    "modern_mc_100x3_sampled": {
      "path": "/results/prompts/modern_mc_causal.jsonl",
      "sha256": "..."
    }
  },
  "tree_manifests": [
    {
      "tree_id": "confirm_paper_tree_001",
      "path": "/sealed/trees/confirm_paper_tree_001.manifest.jsonl",
      "sha256": "..."
    }
  ],
  "response_receipts": [
    {"path": "/results/tree_001/n3/receipt.json", "sha256": "..."}
  ]
}
```

Lists are exhaustive, not discovery hints. The analyzer rejects a missing or extra tree, a missing
or extra leaf, duplicate receipts, duplicate prompt/draw identities, incomplete generation grids,
hash drift, receipt/request/provenance drift, and any generation option that differs from the
behavior protocol. A panel must have one cohort, base-model revision, and explicit model task.
When multiple protocols are declared, their probe IDs must not overlap.

The truth manifest is also the depth map. Every trained leaf must have an integer `depth`, a
root-to-node `path` consistent with that depth, and a non-empty training `task_family`. The exact
receipt model IDs must equal the manifest's trained-leaf set.

## Empty-response estimands

The primary endpoint preserves every generated response. The corrected empty definition is
exactly:

```python
text.strip() == ""
```

This counts an empty string and a whitespace-only decoded continuation as empty. It does not add a
natural-language filter and does not remove labels or fragments.

For each probe, tree, and unordered leaf pair, the paired-nonempty sensitivity endpoint starts
from the exact protocol-declared `(prompt_id, sample_id)` grid and excludes an identity if either
leaf is empty under the definition above. Pairing is by identity, never list position, and no
`min(lengths)` truncation is permitted.

The output contains:

- one per-leaf row with panel, probe, tree, node, depth, training task family, output count, empty
  count, and empty percentage;
- one per-depth row with pooled counts and the mean of the depth-specific percentages computed
  independently within each tree;
- a two-sided 95% Student *t* interval over those per-tree percentages (null bounds when only one
  tree contributes, because the interval then has zero degrees of freedom);
- one exact eligibility-count row per tree/probe/leaf pair and one pooled eligibility summary per
  panel/probe.

## CLI

```bash
PYTHONPATH=src python scripts/build_behavior_eos_depth.py \
  --config /results/eos_depth_input.json \
  --out-json /results/eos_depth.json \
  --per-leaf-csv /results/eos_depth_per_leaf.csv \
  --per-depth-csv /results/eos_depth_per_depth.csv \
  --sensitivity-csv /results/eos_sensitivity_pairs.csv \
  --sensitivity-summary-csv /results/eos_sensitivity_summary.csv \
  --receipt /results/eos_depth_receipt.json
```

JSON and CSV files are written through same-directory temporary files and atomically renamed. The
completion receipt is written last and pins every output hash. Existing outputs are rejected unless
`--overwrite` is explicit, and no output may collide with any pinned input.

The implementation is protocol-agnostic. Translation `200 x 1`, multiple-choice `100 x 3`, and
Dolly `30 x 3` are obtained from their registered contracts; the analyzer never guesses or creates
a legacy HellaSwag diagnostic.
