# Corrected all-projection LoRA results

This bundle contains the paper-facing rollup for the corrected all-projection
LoRA experiment. The adapters target `q`, `k`, `v`, `o`, `wi_0`, `wi_1`, and
`wo`, and analysis uses cumulative `adapter_chain` artifacts.

## Provenance

- Training array: Slurm `154446`, with scoped preemption recoveries `154679`,
  `154856`, `154901`, `155043`, and `155884`.
- Completion audit: Slurm `155044`, exit `0:0`.
- Direct analysis and rollup: Slurm `155045`, exit `0:0`.
- Isolated checkout: `/home/export/sgallagh/WeightTraits-pr1-20260713`.
- Analysis root: `outputs/analysis_v20260713/lora_all_projections_corrected`.

The completion audit reports all 50 trees valid and ready. The rollup is valid,
contains 150 rows, covers all 50 trees for each of correlation, cosine, and L2,
and has no missing tree IDs. Audit and analysis error logs were empty.

## Headline results

| Metric | Theorem certified | Exact recovery | Polytomy-aware exact | Mean clade recovery |
| --- | ---: | ---: | ---: | ---: |
| Correlation | 0.34 | 0.12 | 0.76 | 0.9117 |
| Cosine | 0.34 | 0.12 | 0.76 | 0.9117 |
| L2 | 0.28 | 0.10 | 0.64 | 0.8217 |

## Files

- `lora_all_projections_corrected_summary.json`: validated aggregate rollup and
  per-tree rows.
- `lora_all_projections_corrected_rows.csv`: flat 50-tree by 3-metric table.
- `lora_all_projections_corrected_run.json`: direct-analysis execution report.
