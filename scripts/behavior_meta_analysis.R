#!/usr/bin/env Rscript

# Paper-faithful behavioral bridge analysis.
#
# Each training tree is treated as one study.  We estimate the within-tree
# Pearson correlation between weight distance and behavioral similarity,
# transform it with Fisher's z, and pool studies with a DerSimonian--Laird
# random-effects model.  Pairwise sampling variances follow the paper exactly:
# 1 / (n_pairs - 3).  Because leaf pairs are not independent, these variances
# are mildly anticonservative; the script records that limitation explicitly.

parse_args <- function(args) {
  options <- list(
    inputs = character(), out_dir = NULL, analysis_id = NULL,
    iv = "weight_distance", dv = "behavior_similarity", run_col = "run_id"
  )
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--input") {
      i <- i + 1L
      options$inputs <- c(options$inputs, args[[i]])
    } else if (key == "--out-dir") {
      i <- i + 1L
      options$out_dir <- args[[i]]
    } else if (key == "--analysis-id") {
      i <- i + 1L
      options$analysis_id <- args[[i]]
    } else if (key == "--iv") {
      i <- i + 1L
      options$iv <- args[[i]]
    } else if (key == "--dv") {
      i <- i + 1L
      options$dv <- args[[i]]
    } else if (key == "--run-col") {
      i <- i + 1L
      options$run_col <- args[[i]]
    } else {
      stop(sprintf("Unknown or incomplete argument: %s", key), call. = FALSE)
    }
    i <- i + 1L
  }
  if (length(options$inputs) == 0L || is.null(options$out_dir) ||
      is.null(options$analysis_id)) {
    stop(paste(
      "Usage: behavior_meta_analysis.R --input pairs.csv [--input more.csv]",
      "--out-dir DIR --analysis-id ID [--iv weight_distance]",
      "[--dv behavior_similarity] [--run-col run_id]"
    ), call. = FALSE)
  }
  options
}

safe_z <- function(x) {
  if (length(x) < 2L || !is.finite(sd(x)) || sd(x) == 0) {
    return(rep(NA_real_, length(x)))
  }
  as.numeric(scale(x))
}

per_run_statistics <- function(data, run_col, iv, dv) {
  runs <- unique(as.character(data[[run_col]]))
  rows <- lapply(runs, function(run_id) {
    group <- data[as.character(data[[run_col]]) == run_id, , drop = FALSE]
    x <- group[[iv]]
    y <- group[[dv]]
    n <- length(x)
    x_sd <- if (n > 1L) sd(x) else NA_real_
    y_sd <- if (n > 1L) sd(y) else NA_real_
    reason <- ""
    if (n < 4L) reason <- "fewer_than_four_pairs"
    if (!is.finite(x_sd) || x_sd == 0) reason <- "zero_weight_variance"
    if (!is.finite(y_sd) || y_sd == 0) reason <- "zero_behavior_variance"
    usable <- identical(reason, "")
    r <- if (usable) cor(x, y, method = "pearson") else NA_real_
    r_for_z <- if (usable) max(-0.999999, min(0.999999, r)) else NA_real_
    raw_beta <- if (usable) cov(x, y) / var(x) else NA_real_
    data.frame(
      run_id = run_id,
      n_pairs = n,
      weight_mean = mean(x),
      weight_sd = x_sd,
      behavior_mean = mean(y),
      behavior_sd = y_sd,
      raw_beta = raw_beta,
      semi_standardized_beta = if (usable) raw_beta * x_sd else NA_real_,
      standardized_beta = r,
      pearson_r = r,
      fisher_z = if (usable) atanh(r_for_z) else NA_real_,
      sampling_variance = if (usable) 1 / (n - 3) else NA_real_,
      usable = usable,
      exclusion_reason = reason,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

pool_correlations <- function(per_run) {
  studies <- per_run[per_run$usable, , drop = FALSE]
  k <- nrow(studies)
  empty <- data.frame(
    estimator = c("equal_weight", "fixed_effect", "dersimonian_laird"),
    k_runs = k, estimate_r = NA_real_, ci_low = NA_real_, ci_high = NA_real_,
    standard_error_z = NA_real_, statistic_z = NA_real_, p_value = NA_real_,
    Q = NA_real_, Q_df = NA_integer_, Q_p_value = NA_real_,
    I2 = NA_real_, tau2 = NA_real_, status = "fewer_than_three_usable_runs",
    stringsAsFactors = FALSE
  )
  # The source analysis deliberately did not pool fewer than three runs.
  if (k < 3L) return(empty)

  z <- studies$fisher_z
  v <- studies$sampling_variance
  w_fe <- 1 / v
  z_equal <- mean(z)
  z_fe <- sum(w_fe * z) / sum(w_fe)
  se_fe <- sqrt(1 / sum(w_fe))
  Q <- sum(w_fe * (z - z_fe)^2)
  Q_df <- k - 1L
  C <- sum(w_fe) - sum(w_fe^2) / sum(w_fe)
  tau2 <- if (C > 0) max(0, (Q - Q_df) / C) else 0
  I2 <- if (Q > 0) max(0, (Q - Q_df) / Q) else 0
  w_re <- 1 / (v + tau2)
  z_re <- sum(w_re * z) / sum(w_re)
  se_re <- sqrt(1 / sum(w_re))
  q_p <- pchisq(Q, df = Q_df, lower.tail = FALSE)

  make_row <- function(name, estimate_z, se_z = NA_real_) {
    inferential <- is.finite(se_z)
    stat <- if (inferential) estimate_z / se_z else NA_real_
    data.frame(
      estimator = name,
      k_runs = k,
      estimate_r = tanh(estimate_z),
      ci_low = if (inferential) tanh(estimate_z - 1.96 * se_z) else NA_real_,
      ci_high = if (inferential) tanh(estimate_z + 1.96 * se_z) else NA_real_,
      standard_error_z = se_z,
      statistic_z = stat,
      p_value = if (inferential) 2 * pnorm(abs(stat), lower.tail = FALSE) else NA_real_,
      Q = Q, Q_df = Q_df, Q_p_value = q_p, I2 = I2, tau2 = tau2,
      status = "ok",
      stringsAsFactors = FALSE
    )
  }
  rbind(
    make_row("equal_weight", z_equal),
    make_row("fixed_effect", z_fe, se_fe),
    make_row("dersimonian_laird", z_re, se_re)
  )
}

cluster_vcov <- function(fit, clusters) {
  X <- model.matrix(fit)
  residual <- residuals(fit)
  groups <- split(seq_along(clusters), clusters)
  n <- nrow(X)
  p <- ncol(X)
  g <- length(groups)
  bread <- solve(crossprod(X))
  meat <- matrix(0, nrow = p, ncol = p)
  for (idx in groups) {
    score <- crossprod(X[idx, , drop = FALSE], residual[idx])
    meat <- meat + score %*% t(score)
  }
  correction <- (g / (g - 1)) * ((n - 1) / (n - p))
  correction * bread %*% meat %*% bread
}

fit_fixed_effect_variant <- function(data, run_col, iv, dv, variant) {
  x <- data[[iv]]
  y <- data[[dv]]
  run <- as.character(data[[run_col]])
  if (variant %in% c("semi_standardized", "standardized")) {
    x <- ave(x, run, FUN = safe_z)
  }
  if (variant == "standardized") {
    y <- ave(y, run, FUN = safe_z)
  }
  frame <- data.frame(y = y, x = x, run = factor(run))
  frame <- frame[complete.cases(frame), , drop = FALSE]
  g <- nlevels(droplevels(frame$run))
  fit <- if (g >= 2L) lm(y ~ x + run, data = frame) else lm(y ~ x, data = frame)
  estimate <- unname(coef(fit)["x"])
  if (g >= 2L) {
    vc <- cluster_vcov(fit, frame$run)
    se <- sqrt(vc["x", "x"])
    df <- g - 1L
    statistic <- estimate / se
    p_value <- 2 * pt(abs(statistic), df = df, lower.tail = FALSE)
    critical <- qt(0.975, df = df)
    se_type <- "run_cluster_CR1"
  } else {
    se <- unname(summary(fit)$coefficients["x", "Std. Error"])
    df <- df.residual(fit)
    statistic <- estimate / se
    p_value <- 2 * pt(abs(statistic), df = df, lower.tail = FALSE)
    critical <- qt(0.975, df = df)
    se_type <- "classical_single_run_smoke_only"
  }
  data.frame(
    coefficient = variant,
    estimate = estimate,
    std_error = se,
    statistic = statistic,
    degrees_of_freedom = df,
    p_value = p_value,
    ci_low = estimate - critical * se,
    ci_high = estimate + critical * se,
    n_rows = nrow(frame),
    n_runs = g,
    se_type = se_type,
    formula = "outcome ~ predictor + run fixed effects",
    stringsAsFactors = FALSE
  )
}

options <- parse_args(commandArgs(trailingOnly = TRUE))
missing_inputs <- options$inputs[!file.exists(options$inputs)]
if (length(missing_inputs) > 0L) {
  stop(sprintf("Missing input(s): %s", paste(missing_inputs, collapse = ", ")), call. = FALSE)
}

parts <- lapply(options$inputs, read.csv, stringsAsFactors = FALSE)
data <- do.call(rbind, parts)
required <- c(options$run_col, options$iv, options$dv)
missing_columns <- setdiff(required, names(data))
if (length(missing_columns) > 0L) {
  stop(sprintf("Missing columns: %s", paste(missing_columns, collapse = ", ")), call. = FALSE)
}

before <- nrow(data)
finite <- is.finite(data[[options$iv]]) & is.finite(data[[options$dv]]) &
  !is.na(data[[options$run_col]]) & nzchar(as.character(data[[options$run_col]]))
data <- data[finite, , drop = FALSE]
if (nrow(data) == 0L) stop("No complete finite rows remain", call. = FALSE)

per_run <- per_run_statistics(data, options$run_col, options$iv, options$dv)
usable_runs <- per_run$run_id[per_run$usable]
model_data <- data[as.character(data[[options$run_col]]) %in% usable_runs, , drop = FALSE]
pooling <- pool_correlations(per_run)
coefficients <- if (length(usable_runs) > 0L) {
  do.call(rbind, lapply(
    c("raw", "semi_standardized", "standardized"),
    function(variant) fit_fixed_effect_variant(
      model_data, options$run_col, options$iv, options$dv, variant
    )
  ))
} else {
  data.frame()
}

dir.create(options$out_dir, recursive = TRUE, showWarnings = FALSE)
stem <- file.path(options$out_dir, options$analysis_id)
write.csv(per_run, paste0(stem, ".per_run.csv"), row.names = FALSE)
write.csv(pooling, paste0(stem, ".pooling.csv"), row.names = FALSE)
write.csv(coefficients, paste0(stem, ".coefficients.csv"), row.names = FALSE)

audit <- data.frame(
  analysis_id = options$analysis_id,
  input_files = paste(normalizePath(options$inputs), collapse = ";"),
  predictor = options$iv,
  outcome = options$dv,
  run_column = options$run_col,
  rows_input = before,
  rows_analyzed = nrow(data),
  rows_dropped = before - nrow(data),
  runs_observed = length(unique(as.character(data[[options$run_col]]))),
  runs_usable = sum(per_run$usable),
  headline_estimator = "dersimonian_laird_fisher_z",
  pair_variance = "1/(n_pairs-3); shared-leaf pseudo-replication not corrected",
  r_version = R.version.string,
  stringsAsFactors = FALSE
)
write.csv(audit, paste0(stem, ".audit.csv"), row.names = FALSE)

cat(sprintf(
  "Wrote %s.*: %d finite pairs, %d/%d usable runs\n",
  stem, nrow(data), sum(per_run$usable), nrow(per_run)
))
if (sum(per_run$usable) < 3L) {
  cat("DL pooling withheld until at least three usable trees are present.\n")
}
