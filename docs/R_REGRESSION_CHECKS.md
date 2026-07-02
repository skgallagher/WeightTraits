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

The initial script is `scripts/regression_r_check.R`.

