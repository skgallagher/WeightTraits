#!/usr/bin/env Rscript
# Figure 2: clade estimation against the fitted Atteson margin and the tree-likeness score A,
# with fractional-logit fits and paired-tree bootstrap odds ratios (paper Section 5).
#
# usage: Rscript R/figure2.R SUMMARY_CSV RECORDS_CSV BOOTSTRAP_OUT_CSV FIGURE_STEM [REPS]
#   SUMMARY_CSV / RECORDS_CSV  from scripts/build_final_weight_figure2.py
#                              (merged_weight_figure2.csv, merged_weight_figure2_records.csv)
#   BOOTSTRAP_OUT_CSV          odds ratios with 95% paired-bootstrap intervals
#   FIGURE_STEM                output path without extension (.pdf and .png are written)
#   REPS                       bootstrap replicates (default 2000; seed 20260805)
# Ported from ELLMTrees scripts/ggplot/make_merged_weight_figure2.R; numerics unchanged.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("usage: figure2.R SUMMARY_CSV RECORDS_CSV BOOTSTRAP_OUT_CSV FIGURE_STEM [REPS]")
}
summary_csv <- args[[1]]
records_csv <- args[[2]]
bootstrap_out <- args[[3]]
figure_stem <- args[[4]]
bootstrap_reps <- if (length(args) >= 5) as.integer(args[[5]]) else 2000L
bootstrap_seed <- 20260805L

script_dir <- dirname(normalizePath(sub(
  "^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[[1]]
)))
source(file.path(script_dir, "theme_weighttraits.R"))

condition_levels <- c(
  "Full fine-tuning (Flan)",
  "LoRA key only",
  "LoRA q/k/v",
  "LoRA full attention",
  "Full fine-tuning (Llama)",
  "LoRA q/k/v (r=8)",
  "LoRA q/k/v (r=64)"
)
condition_labels <- c(
  "Full fine-tuning (Flan)" = "Flan full FT",
  "LoRA key only" = "Flan key LoRA",
  "LoRA q/k/v" = "Flan q/k/v LoRA",
  "LoRA full attention" = "Flan full-attn LoRA",
  "Full fine-tuning (Llama)" = "Llama full FT",
  "LoRA q/k/v (r=8)" = "Llama q/k/v LoRA r=8",
  "LoRA q/k/v (r=64)" = "Llama q/k/v LoRA r=64"
)
condition_colors <- c(
  "Full fine-tuning (Flan)" = wt_palette[["navy"]],
  "LoRA key only" = wt_palette[["teal"]],
  "LoRA q/k/v" = wt_palette[["purple"]],
  "LoRA full attention" = wt_palette[["gray"]],
  "Full fine-tuning (Llama)" = wt_palette[["charcoal"]],
  "LoRA q/k/v (r=8)" = wt_palette[["orange"]],
  "LoRA q/k/v (r=64)" = wt_palette[["vermillion"]]
)

mark_levels <- c(
  "All-weight estimate",
  "PhyloLM (behavior)",
  "Layer subset",
  "Single matrix (q/k/v)"
)
mark_shapes <- c(
  "All-weight estimate" = 21,
  "PhyloLM (behavior)" = 23,
  "Layer subset" = 22,
  "Single matrix (q/k/v)" = 16
)

diagnostic_levels <- c(
  "A  Fitted Atteson margin (log scale)",
  "B  Tree-likeness score A"
)

points <- read_csv(
  summary_csv,
  show_col_types = FALSE
) %>%
  mutate(
    condition = factor(condition, levels = condition_levels),
    mark = case_when(
      level == "simulation" ~ "All-weight estimate",
      level == "phylolm" ~ "PhyloLM (behavior)",
      level == "subset" ~ "Layer subset",
      level == "layer" ~ "Single matrix (q/k/v)",
      TRUE ~ NA_character_
    ),
    mark = factor(mark, levels = mark_levels)
  )

tree_records <- read_csv(
  records_csv,
  show_col_types = FALSE
)

n_estimators <- nrow(points)
if (n_estimators < 100L || nrow(tree_records) != n_estimators * 46L) {
  stop("Figure 2 bootstrap expects at least 100 estimators x 46 shared tree IDs")
}
cell_counts <- table(tree_records$estimator_id, tree_records$tree_id)
if (!all(cell_counts == 1L)) {
  stop("Every Figure 2 estimator must have exactly one record per shared tree ID")
}
estimator_ids <- points$estimator_id
tree_ids <- sort(unique(tree_records$tree_id))
if (
  length(unique(estimator_ids)) != n_estimators ||
  length(tree_ids) != 46L ||
  !setequal(estimator_ids, unique(tree_records$estimator_id))
) {
  stop("Figure 2 summary and paired-record estimator panels do not match")
}

record_matrix <- function(field) {
  formula <- as.formula(paste(field, "~ estimator_id + tree_id"))
  matrix <- xtabs(formula, data = tree_records)
  unclass(matrix[estimator_ids, tree_ids, drop = FALSE])
}

recovery_matrix <- record_matrix("clade_recovery")
atteson_matrix <- record_matrix("atteson_bottleneck_margin")
additivity_matrix <- record_matrix("four_point_mean_additivity")

observed_recovery <- rowMeans(recovery_matrix)
observed_atteson <- rowMeans(atteson_matrix)
observed_additivity <- rowMeans(additivity_matrix)
if (
  max(abs(100 * observed_recovery - points$clade_recovery_pct)) > 1e-8 ||
  max(abs(observed_atteson - points$atteson_bottleneck_margin)) > 1e-8 ||
  max(abs(observed_additivity - points$four_point_mean_additivity)) > 1e-8
) {
  stop("Paired tree records do not reproduce the plotted Figure 2 summaries")
}

figure_points <- bind_rows(
  points %>% mutate(
    diagnostic = diagnostic_levels[[1]],
    diagnostic_value = log10(pmax(atteson_bottleneck_margin, 0.03))
  ),
  points %>% mutate(
    diagnostic = diagnostic_levels[[2]],
    diagnostic_value = four_point_mean_additivity
  )
) %>%
  mutate(diagnostic = factor(diagnostic, levels = diagnostic_levels))

# Resample the 46 shared topology IDs jointly across all displayed estimators. Each
# replicate recomputes every x/y estimator mean before refitting the fractional
# logit, preserving the paired simulation design.
fit_fractional_logit <- function(x, y) {
  fit <- glm.fit(
    x = cbind("(Intercept)" = 1, diagnostic_value = x),
    y = y,
    weights = rep(46, length(y)),
    family = quasibinomial(link = "logit")
  )
  if (length(fit$coefficients) != 2L || any(!is.finite(fit$coefficients))) {
    stop("Fractional-logit bootstrap fit failed")
  }
  list(coefficients = unname(fit$coefficients), fit = fit)
}

observed_x <- list(
  atteson = log10(pmax(observed_atteson, 0.03)),
  additivity = observed_additivity
)
observed_fits <- list(
  atteson = fit_fractional_logit(observed_x$atteson, observed_recovery),
  additivity = fit_fractional_logit(observed_x$additivity, observed_recovery)
)

set.seed(bootstrap_seed)
bootstrap_coefficients <- list(
  atteson = matrix(NA_real_, nrow = bootstrap_reps, ncol = 2L),
  additivity = matrix(NA_real_, nrow = bootstrap_reps, ncol = 2L)
)
for (replicate_index in seq_len(bootstrap_reps)) {
  sampled_trees <- sample.int(length(tree_ids), length(tree_ids), replace = TRUE)
  replicate_recovery <- rowMeans(recovery_matrix[, sampled_trees, drop = FALSE])
  replicate_atteson <- log10(
    pmax(rowMeans(atteson_matrix[, sampled_trees, drop = FALSE]), 0.03)
  )
  replicate_additivity <- rowMeans(
    additivity_matrix[, sampled_trees, drop = FALSE]
  )
  bootstrap_coefficients$atteson[replicate_index, ] <- fit_fractional_logit(
    replicate_atteson, replicate_recovery
  )$coefficients
  bootstrap_coefficients$additivity[replicate_index, ] <- fit_fractional_logit(
    replicate_additivity, replicate_recovery
  )$coefficients
}

curve_with_bootstrap <- function(diagnostic, x, fit, coefficients) {
  grid <- seq(min(x), max(x), length.out = 240L)
  predictions <- plogis(
    outer(coefficients[, 2], grid) + coefficients[, 1]
  ) * 100
  tibble(
    diagnostic = diagnostic,
    diagnostic_value = grid,
    clade_recovery_pct = 100 * plogis(fit$coefficients[[1]] + fit$coefficients[[2]] * grid),
    bootstrap_lower = apply(predictions, 2, quantile, probs = 0.025),
    bootstrap_upper = apply(predictions, 2, quantile, probs = 0.975)
  )
}

logit_curves <- bind_rows(
  curve_with_bootstrap(
    diagnostic_levels[[1]], observed_x$atteson,
    observed_fits$atteson, bootstrap_coefficients$atteson
  ),
  curve_with_bootstrap(
    diagnostic_levels[[2]], observed_x$additivity,
    observed_fits$additivity, bootstrap_coefficients$additivity
  )
) %>%
  mutate(diagnostic = factor(diagnostic, levels = diagnostic_levels))

odds_ratio_summary <- bind_rows(
  tibble(
    diagnostic = "Fitted Atteson diagnostic",
    contrast = "10-fold increase in mean Atteson diagnostic",
    slope = observed_fits$atteson$coefficients[[2]],
    odds_ratio = exp(observed_fits$atteson$coefficients[[2]]),
    odds_ratio_lower = quantile(exp(bootstrap_coefficients$atteson[, 2]), 0.025),
    odds_ratio_upper = quantile(exp(bootstrap_coefficients$atteson[, 2]), 0.975),
    null_deviance_explained = 1 - observed_fits$atteson$fit$deviance /
      observed_fits$atteson$fit$null.deviance
  ),
  tibble(
    diagnostic = "Four-point additivity A",
    contrast = "0.10 increase in mean additivity",
    slope = observed_fits$additivity$coefficients[[2]],
    odds_ratio = exp(0.10 * observed_fits$additivity$coefficients[[2]]),
    odds_ratio_lower = quantile(
      exp(0.10 * bootstrap_coefficients$additivity[, 2]), 0.025
    ),
    odds_ratio_upper = quantile(
      exp(0.10 * bootstrap_coefficients$additivity[, 2]), 0.975
    ),
    null_deviance_explained = 1 - observed_fits$additivity$fit$deviance /
      observed_fits$additivity$fit$null.deviance
  )
) %>%
  mutate(
    n_estimators = nrow(points),
    n_trees = length(tree_ids),
    interval = "paired-tree percentile",
    bootstrap_reps = bootstrap_reps,
    bootstrap_seed = bootstrap_seed
  )

dir.create(dirname(bootstrap_out), recursive = TRUE, showWarnings = FALSE)
write_csv(odds_ratio_summary, bootstrap_out)

# Fractional-logit odds ratios (paired-tree bootstrap CI), printed in each panel.
odds_ratio_labels <- data.frame(
  diagnostic = factor(diagnostic_levels, levels = diagnostic_levels),
  label = c(
    sprintf(
      "Odds ratio %.1f per 10-fold increase\n(95%% CI %.1f to %.1f)",
      odds_ratio_summary$odds_ratio[[1]],
      odds_ratio_summary$odds_ratio_lower[[1]],
      odds_ratio_summary$odds_ratio_upper[[1]]
    ),
    sprintf(
      "Odds ratio %.1f per +0.10 in A\n(95%% CI %.1f to %.1f)",
      odds_ratio_summary$odds_ratio[[2]],
      odds_ratio_summary$odds_ratio_lower[[2]],
      odds_ratio_summary$odds_ratio_upper[[2]]
    )
  )
)

layer_points <- filter(figure_points, level == "layer")
subset_points <- filter(figure_points, level == "subset")
simulation_points <- filter(figure_points, level == "simulation")
phylolm_points <- filter(figure_points, level == "phylolm")

diagnostic_breaks <- function(limits) {
  if (limits[[1]] < 0) {
    log10(c(0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300))
  } else {
    seq(0.55, 0.95, by = 0.05)
  }
}

diagnostic_labels <- function(values) {
  if (any(values < 0, na.rm = TRUE)) {
    reference <- log10(c(0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300))
    labels <- c(".03", ".1", ".3", "1", "3", "10", "30", "100", "300")
    vapply(values, function(value) {
      if (is.na(value)) return(NA_character_)
      labels[[which.min(abs(reference - value))]]
    }, character(1))
  } else {
    sprintf("%.2f", values)
  }
}

figure <- ggplot(
  figure_points,
  aes(
    x = diagnostic_value,
    y = clade_recovery_pct,
    color = condition,
    shape = mark
  )
) +
  geom_line(
    data = logit_curves,
    aes(
      x = diagnostic_value,
      y = clade_recovery_pct,
      group = diagnostic
    ),
    inherit.aes = FALSE,
    color = wt_palette[["charcoal"]],
    linewidth = 1.0,
    linetype = "dashed",
    alpha = 0.8,
    show.legend = FALSE
  ) +
  geom_point(
    data = layer_points,
    color = wt_palette[["gray"]],
    size = 1.45,
    alpha = 0.24,
    stroke = 0,
    show.legend = TRUE
  ) +
  geom_errorbar(
    data = subset_points,
    aes(
      ymin = clade_recovery_pct - clade_recovery_pct_se,
      ymax = clade_recovery_pct + clade_recovery_pct_se
    ),
    width = 0,
    linewidth = 0.55,
    alpha = 0.85,
    show.legend = FALSE
  ) +
  geom_point(
    data = subset_points,
    color = wt_palette[["gray"]],
    fill = "white",
    size = 2.65,
    stroke = 0.85,
    show.legend = TRUE
  ) +
  geom_errorbar(
    data = simulation_points,
    aes(
      ymin = clade_recovery_pct - clade_recovery_pct_se,
      ymax = clade_recovery_pct + clade_recovery_pct_se
    ),
    width = 0,
    linewidth = 0.75,
    show.legend = FALSE
  ) +
  geom_point(
    data = simulation_points,
    aes(fill = condition),
    size = 3.55,
    stroke = 0.7,
    show.legend = TRUE
  ) +
  geom_errorbar(
    data = phylolm_points,
    aes(
      ymin = clade_recovery_pct - clade_recovery_pct_se,
      ymax = clade_recovery_pct + clade_recovery_pct_se
    ),
    width = 0,
    linewidth = 0.75,
    show.legend = FALSE
  ) +
  geom_point(
    data = phylolm_points,
    aes(fill = condition),
    size = 3.75,
    stroke = 0.75,
    show.legend = TRUE
  ) +
  geom_text(
    data = odds_ratio_labels,
    aes(x = Inf, y = -Inf, label = label),
    inherit.aes = FALSE,
    hjust = 1.04,
    vjust = -0.45,
    size = 3.25,
    lineheight = 0.95,
    color = wt_palette[["charcoal"]]
  ) +
  facet_wrap(vars(diagnostic), nrow = 1, scales = "free_x") +
  scale_color_manual(
    values = condition_colors,
    breaks = condition_levels,
    labels = condition_labels,
    drop = FALSE
  ) +
  scale_fill_manual(values = condition_colors, guide = "none") +
  scale_shape_manual(
    values = mark_shapes,
    breaks = mark_levels,
    drop = FALSE
  ) +
  scale_x_continuous(
    breaks = diagnostic_breaks,
    labels = diagnostic_labels,
    expand = expansion(mult = c(0.045, 0.065))
  ) +
  scale_y_continuous(
    limits = c(0, 102),
    breaks = seq(0, 100, by = 20),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    x = "Mean diagnostic statistic across 46 trees (panel A: log scale)",
    y = "Mean true clades estimated (%)"
  ) +
  guides(
    color = guide_legend(
      title = "Primary condition (color)",
      title.position = "top",
      title.hjust = 0,
      order = 1,
      nrow = 2,
      byrow = TRUE,
      override.aes = list(
        shape = 21,
        fill = unname(condition_colors[condition_levels]),
        size = 3.2,
        alpha = 1,
        stroke = 0.65
      )
    ),
    shape = guide_legend(
      title = "Estimate",
      title.position = "top",
      title.hjust = 0,
      order = 2,
      nrow = 1,
      byrow = TRUE,
      override.aes = list(
        color = wt_palette[["gray"]],
        fill = c(wt_palette[["gray"]], wt_palette[["gray"]], "white", NA),
        size = c(3.0, 3.1, 2.7, 2.1),
        alpha = 1,
        stroke = 0.7
      )
    )
  ) +
  theme_weighttraits(base_size = 10.2) +
  theme(
    strip.text = element_text(size = 10.7, face = "bold", hjust = 0),
    panel.spacing = grid::unit(0.18, "cm"),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.box.just = "left",
    legend.title = element_text(face = "bold", size = 9.8),
    legend.text = element_text(size = 9.1),
    legend.key.height = grid::unit(0.38, "cm"),
    legend.key.width = grid::unit(0.50, "cm"),
    legend.spacing.x = grid::unit(0.28, "cm"),
    legend.spacing.y = grid::unit(0.02, "cm"),
    legend.margin = margin(1, 0, 0, 0),
    plot.margin = margin(3, 5, 8, 3)
  )

dir.create(dirname(figure_stem), recursive = TRUE, showWarnings = FALSE)
save_weighttraits(
  figure,
  figure_stem,
  width = 7.15,
  height = 5.25
)

message("Wrote Figure 2 to ", figure_stem, ".{pdf,png}")
message("Wrote paired-bootstrap odds ratios to ", bootstrap_out)
