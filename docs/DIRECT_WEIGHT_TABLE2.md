# Corrected direct-weight Table 2 aggregation

`scripts/build_merged_weight_paper_stats.py` is the fail-closed aggregator for
the corrected direct-weight Table 2 rows. It supersedes the old hard-coded
historical-source script. The aggregator recomputes every number from per-tree
rows and never consumes `aggregate_by_metric` values.

## Input contract

The YAML config has `version: 1`, one `metric`, a `truth_manifests` map, and a
non-empty `conditions` list. Relative paths resolve against the config file's
directory unless `--base-dir` is supplied.

```yaml
version: 1
metric: cosine
required_cohort_ids:
  - llama32_1b_full_finetune_legacy_causal_2000
  # ...every reviewed live and diagnostic Table-2 condition...

# Canonical structural eligibility set, declared in advance (26 exact IDs).
ordering_tree_ids:
  - confirm_paper_tree_003
  - confirm_paper_tree_004
  # ...the remaining 24 reviewed ordering-valid IDs...

truth_manifests:
  confirm_paper_tree_001:
    path: ../../examples/training/confirm_paper_numbers/assigned_manifests/confirm_paper_tree_001.manifest.jsonl
    sha256: <reviewed lowercase SHA-256>
  # Declare every topology-eligible tree through 050, except 015/020/037/047.

conditions:
  - cohort_id: llama_full_finetune
    architecture: Llama-3.2-1B
    training: Full fine-tuning
    artifact: model
    representation: full_weight
    cohort_contract:
      path: ../../examples/training/confirm_paper_numbers/llama32_1b_full_finetune_legacy_causal_2000_training_run_list_summary.json
      sha256: <reviewed lowercase SHA-256>
    completion_receipt:
      path: ../../outputs/receipts/llama32_1b_full_finetune_legacy_causal_2000_completion.json
      sha256: <reviewed lowercase SHA-256>
    rollup:
      path: ../../outputs/corrected_direct_weight/llama_full_finetune_rollup.json
      sha256: <reviewed lowercase SHA-256>
```

`required_cohort_ids` is the reviewed completeness gate: the condition list
must contain each ID exactly once and may not add unreviewed conditions. The
cohort contract is the clean 50-tree/641-run declaration and pins every
topology tree's truth-manifest hash. The completion receipt is a separate strict
completed-run-set audit, not merely the run-list declaration. It must prove
50/50 trees ready, 641/641 nodes completed,
1,282/1,282 required artifacts present, and zero failed, missing, in-progress,
not-started, error, or warning counts; every per-tree row is revalidated. Its
hash binds the human-facing cohort ID to completed training. The rollup hash
binds the numeric input artifact. The declared
`artifact` and `representation` must match both the rollup and every selected
per-tree row. Every selected row's truth-manifest file must exist, contain only
the expected `tree_id`, and match the corresponding declared truth hash. Hashes
are intentionally supplied in advance; the aggregation command does not offer a
flag that silently accepts whatever files happen to be present.

## Eligibility and estimators

For the requested metric, each condition must contain exactly one row for every
tree `confirm_paper_tree_001` through `confirm_paper_tree_050`. Duplicate,
missing, or unexpected tree IDs are fatal. The topology sample is exactly 46
trees: 001–050 excluding 015, 020, 037, and 047. Those four excluded trees must
have zero truth splits, while all 46 included trees must have at least one.

Each condition must expose exactly the 26 tree IDs declared by the top-level
`ordering_tree_ids` contract with
`branch_ordering_valid: true` and `branch_ordering_status: ok`. The exact set of
26 tree IDs must match the declaration and be identical across every condition.
The common set is recorded in the JSON output.

The implementation uses these estimators:

- clade recovery, RF, and false negatives: arithmetic mean and sample standard
  error (`stdev / sqrt(n)`) over the 46 topology trees;
- polytomy-aware exact recovery (PAER): binary mean and binomial standard error
  `sqrt(p * (1 - p) / n)` over the same 46 trees;
- rank-biserial: arithmetic mean and sample standard error over the common 26
  ordering-valid trees;
- branch correlation: `tanh(mean(atanh(r)))`; its reported standard error is the
  sample standard error of the 26 raw per-tree `r` values, not z-space values.

All required values must be finite and in their valid ranges. PAER and ordering
validity must be JSON booleans. A branch correlation equal to -1 or 1 is rejected
because its Fisher transform is undefined.

## Command and outputs

Run from the WeightTraits repository root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python scripts/build_merged_weight_paper_stats.py \
  --config examples/paper/corrected_direct_weight_table2.yaml \
  --out outputs/corrected_direct_weight/table2.json \
  --csv-out outputs/corrected_direct_weight/table2.csv
```

The JSON is the authoritative artifact: it records the config hash, all input
hashes, the exact topology and ordering sets, the estimator contract, and the
rows. The CSV is a flat rendering for workbook import. Neither output edits a
workbook or manuscript.
