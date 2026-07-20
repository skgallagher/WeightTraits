# R Regression Cross-Checks

ELLMTrees used Python regression tooling in several places. WeightTraits will use R as an independent check before reporting regression results.

## Minimum Checks

For each paper-facing regression:
- Fit the same fixed-effect model in R with `lm`.
- Fit the mixed/repeated-measure model in R with `lme4::lmer` when grouping is required.
- Export coefficient, standard error, confidence interval, p-value where applicable, formula, n rows, n groups, and package versions.
- Compare sign and magnitude to the Python aggregation.

## Default Inputs

Expected columns for pairwise behavioral regressions:
- `run_id`
- `pair_id`
- `weight_distance`
- `behavior_distance`
- optional task/model/group columns

`scripts/behavior_meta_analysis.R` implements the paper-facing behavioral model
without requiring an R statistics package: within-run Pearson correlations,
Fisher-z transformation, inverse-variance fixed effects, and the headline
DerSimonian--Laird random-effects estimate. It also exports raw,
semi-standardized, and within-run standardized run-fixed-effect regressions,
with CR1 standard errors clustered by run when at least two runs are present.

```bash
Rscript scripts/behavior_meta_analysis.R \
  --input results/behavior_pairs.csv \
  --out-dir results/behavior_statistics \
  --analysis-id llama32_r8_dolly_surface
```

Repeat `--input` to combine independently produced tree tables. The script
requires at least three usable trees before emitting pooled inference, matching
the released paper analysis. A single-tree invocation is still useful as an
end-to-end data and sign check, but its fixed-effect standard error is explicitly
marked `classical_single_run_smoke_only`.

The retained `scripts/regression_r_check.R` is a generic formula-level check;
it is not the paper's Fisher-z/DL pooling estimator.
