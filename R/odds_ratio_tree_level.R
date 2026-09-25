#!/usr/bin/env Rscript
# Fractional-logit odds ratios at the estimator-mean and tree level, for several estimator sets.
# The paper's weights-only check (Section 5: odds ratio 5.7 per +0.10 in A, 95% CI 4.0-9.3) is the
# "weights only (no PhyloLM)", "tree level", "A (+0.10)" row.
#   logit(E[p]) = b0 + b1 * x,  p = clade estimation, x = A or log10(max(m~, 0.03)).
# Uncertainty: 2,000-replicate bootstrap resampling the 46 tree IDs jointly (seed 20260923).
#
# usage: Rscript R/odds_ratio_tree_level.R RECORDS_CSV OUT_CSV
#   RECORDS_CSV  merged_weight_figure2_records.csv from scripts/build_final_weight_figure2.py
suppressMessages({library(readr); library(dplyr)})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: odds_ratio_tree_level.R RECORDS_CSV OUT_CSV")
r <- read_csv(args[[1]], show_col_types = FALSE)
r <- r %>% mutate(logm = log10(pmax(atteson_bottleneck_margin, 0.03)))
trees <- sort(unique(r$tree_id)); stopifnot(length(trees) == 46)
fl <- function(x, y) suppressWarnings(glm.fit(cbind(1, x), y, family = quasibinomial())$coefficients[2])

fit_set <- function(d, level_name, B = 2000, seed = 20260923) {
  est_means <- function(dd) dd %>% group_by(estimator_id) %>%
    summarise(p = mean(clade_recovery), A = mean(four_point_mean_additivity),
              lm = log10(pmax(mean(atteson_bottleneck_margin), 0.03)), .groups = "drop")
  e <- est_means(d)
  obs <- c(estA = fl(e$A, e$p), estM = fl(e$lm, e$p),
           treeA = fl(d$four_point_mean_additivity, d$clade_recovery), treeM = fl(d$logm, d$clade_recovery))
  by_tree <- split(d, d$tree_id)
  set.seed(seed)
  bs <- t(replicate(B, {
    s <- sample(trees, 46, replace = TRUE)
    db <- bind_rows(by_tree[s])
    eb <- est_means(db)
    c(fl(eb$A, eb$p), fl(eb$lm, eb$p), fl(db$four_point_mean_additivity, db$clade_recovery), fl(db$logm, db$clade_recovery))
  }))
  scale <- c(0.1, 1, 0.1, 1)
  out <- data.frame(set = level_name,
    fit = c("estimator means", "estimator means", "tree level", "tree level"),
    diagnostic = c("A (+0.10)", "m~ (x10)", "A (+0.10)", "m~ (x10)"),
    n_estimators = nrow(e), n_rows = c(nrow(e), nrow(e), nrow(d), nrow(d)),
    OR = exp(scale * obs),
    lo = exp(scale * apply(bs, 2, quantile, 0.025)),
    hi = exp(scale * apply(bs, 2, quantile, 0.975)))
  out
}
res <- bind_rows(
  fit_set(r, "all estimators (as in Fig 4)"),
  fit_set(r %>% filter(level != "phylolm"), "weights only (no PhyloLM)"),
  fit_set(r %>% filter(level %in% c("simulation", "phylolm")), "10 condition points (sim + PhyloLM)"),
  fit_set(r %>% filter(level == "simulation"), "7 whole-weight conditions only")
)
res[, c("OR", "lo", "hi")] <- round(res[, c("OR", "lo", "hi")], 2)
print(res, row.names = FALSE)
write_csv(res, args[[2]])
