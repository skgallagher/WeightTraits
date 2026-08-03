# Reproducibility Workbook Verification — 2026-08-03

This record summarizes an independent audit of the paper reproducibility workbook associated with
WeightTraits. It documents what was checked, what the audit found, and the state after corrections.
It is a verification record, not a standalone paper-reproduction guide; see
[Paper Reproduction](PAPER_REPRODUCTION.md) for the repository workflow.

## Final status

**Verified after correction.** The statistical inputs regenerate the workbook data, all headline
results independently recompute, the paper-only numeric revision is guarded by an exact allow-list,
and every source-manifest checksum verifies.

| Check | Final result |
|---|---|
| Data preparation from source CSV/JSON files | Reproduced; differences limited to floating-point noise around `1e-15` |
| Source manifest | 173/173 SHA-256 hashes and byte sizes verified |
| Table 2 summary formulas | Verified after correcting the full-FT correlation estimator |
| Weight-to-behavior random-effects pooling | All 12 analyses reproduced |
| Empty-output sensitivity analysis | All 9 rows reproduced |
| Within-task recovery | All 4 rows reproduced |
| PhyloLM comparison | Values reproduced; weight and behavior sample sizes are now shown separately |
| Workbook formula scan | No `#REF!`, `#DIV/0!`, `#VALUE!`, `#NAME?`, or `#N/A` errors |
| Visual workbook review | Passed for all sheets |

Confirmed headline values include:

- cosine clade recovery: **96.83%**;
- polytomy-aware exact recovery: **46/50 (92%)**;
- HellaSwag DerSimonian–Laird pooled correlation: **−0.2036** with 95% CI
  **[−0.3032, −0.0995]**;
- empty-excluding HellaSwag sensitivity: **−0.0813** with 95% CI
  **[−0.1815, 0.0206]**;
- within-tree empty-rate depth slope: **+5.30 percentage points per depth level**, `p=0.00011`.

## Corrections prompted by the audit

### Consistent Fisher-z aggregation in Table 2

The Llama-3.2-1B full-fine-tuning row originally averaged per-run correlations arithmetically while
the other rows used the project convention:

```text
tanh(mean(atanh(r_i)))
```

The formula now uses the same Fisher-z aggregation as every other row. The corrected estimate is
`−0.6829`, reported in the paper as **−0.68**. The earlier `−0.59` value reflected the estimator
switch, not an improvement produced by the new training run.

### Exact paper revision guard

The dated paper copy is now generated from the current live source by nine exact, context-anchored
numeric substitutions. The check fails if prose changes, a substitution is missing or duplicated,
or an unlisted numeric change appears. The figure width remains `0.9\linewidth`; it is no longer
mistaken for a scientific result update.

### Explicit sample sizes

Recovery and ordering sample counts are now distinguished in Table 2. The full-FT recovery summary
uses 50 trees, while its ordering statistics use the 26 trees with valid branch-ordering records.

The PhyloLM workbook table now reports separate weight and behavior sample sizes. The first three
groups use 47/47 runs; full fine-tuning uses 44 weight and 47 behavior runs, and held-out full
fine-tuning uses 46 weight and 47 behavior runs. The paper caption discloses the same range.

## Rebuild boundary

The statistical data-preparation layer uses ordinary Python scientific packages and resolves paths
relative to the workspace (with environment-variable overrides). The formatted `.xlsx` and rendered
sheet previews are built with `@oai/artifact-tool`, which is supplied by the Codex document runtime
and is not an npm package. Consequently, the numbers are portable, while rebuilding the identical
formatting outside that runtime requires a separate workbook-rendering port.

## Audit method

The audit regenerated the structured workbook data from raw sources, independently recomputed each
summary family, inspected key formulas and precedents, scanned the workbook for formula errors,
compared the paper revision against its live baseline, verified source hashes and file sizes, and
reviewed rendered previews of every sheet.
