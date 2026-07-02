#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: regression_r_check.R input.csv output.csv [formula] [group_column]", call. = FALSE)
}

input_path <- args[[1]]
output_path <- args[[2]]
formula_text <- ifelse(length(args) >= 3, args[[3]], "behavior_distance ~ weight_distance")
group_column <- ifelse(length(args) >= 4, args[[4]], "")

data <- read.csv(input_path, stringsAsFactors = FALSE)
fit <- lm(as.formula(formula_text), data = data)
coef_table <- summary(fit)$coefficients
ci <- confint(fit)

rows <- data.frame(
  term = rownames(coef_table),
  estimate = coef_table[, "Estimate"],
  std_error = coef_table[, "Std. Error"],
  statistic = coef_table[, "t value"],
  p_value = coef_table[, "Pr(>|t|)"],
  conf_low = ci[, 1],
  conf_high = ci[, 2],
  n = nrow(data),
  formula = formula_text,
  model = "lm",
  stringsAsFactors = FALSE
)

if (group_column != "" && requireNamespace("lme4", quietly = TRUE)) {
  mixed_formula <- as.formula(paste(formula_text, "+ (1 |", group_column, ")"))
  mixed <- lme4::lmer(mixed_formula, data = data)
  mixed_coef <- summary(mixed)$coefficients
  mixed_ci <- suppressMessages(confint(mixed, parm = rownames(mixed_coef), method = "Wald"))
  mixed_rows <- data.frame(
    term = rownames(mixed_coef),
    estimate = mixed_coef[, "Estimate"],
    std_error = mixed_coef[, "Std. Error"],
    statistic = mixed_coef[, "t value"],
    p_value = NA_real_,
    conf_low = mixed_ci[, 1],
    conf_high = mixed_ci[, 2],
    n = nrow(data),
    formula = deparse(mixed_formula),
    model = "lmer",
    stringsAsFactors = FALSE
  )
  rows <- rbind(rows, mixed_rows)
}

write.csv(rows, output_path, row.names = FALSE)

