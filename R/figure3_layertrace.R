#!/usr/bin/env Rscript
# Figure 3 (App. B): per-layer clades missed for the Flan-T5 encoder and decoder and the
# Llama-3.2 decoder, full fine-tuning, 46 trees.
#
# usage: Rscript R/figure3_layertrace.R FLAN_PROFILE_CSV LLAMA_PROFILE_CSV FIGURE_STEM
#   FLAN_PROFILE_CSV   scripts/summarize_final_layertrace.py --csv-out
#   LLAMA_PROFILE_CSV  scripts/summarize_final_llama_layertrace.py --csv-out
#   FIGURE_STEM        output path without extension (.pdf and .png are written)
# Ported from ELLMTrees scripts/ggplot/make_paper_figures.R (LayerTrace section); unchanged.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("usage: figure3_layertrace.R FLAN_PROFILE_CSV LLAMA_PROFILE_CSV FIGURE_STEM")
}
flan_profile_csv <- args[[1]]
llama_profile_csv <- args[[2]]
figure_stem <- args[[3]]

script_dir <- dirname(normalizePath(sub(
  "^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[[1]]
)))
source(file.path(script_dir, "theme_weighttraits.R"))

layer_flan_fixed2000 <- read_csv(
  flan_profile_csv,
  show_col_types = FALSE
)
layer_llama_fixed2000 <- read_csv(
  llama_profile_csv,
  show_col_types = FALSE
)
layer <- bind_rows(
  layer_flan_fixed2000,
  layer_llama_fixed2000
)

extract_block_index <- function(layer_name) {
  llama <- grepl("model\\.layers\\.[0-9]+", layer_name)
  flan <- grepl("(?:encoder|decoder)\\.block\\.[0-9]+", layer_name)
  value <- rep(NA_character_, length(layer_name))
  value[llama] <- sub(".*model\\.layers\\.([0-9]+).*", "\\1", layer_name[llama])
  value[flan] <- sub(".*(?:encoder|decoder)\\.block\\.([0-9]+).*", "\\1", layer_name[flan])
  as.integer(value)
}

panel_levels <- c(
  "A  Flan-T5 encoder\n12 blocks; n = 46",
  "B  Flan-T5 decoder\n12 blocks; n = 46",
  "C  Llama-3.2 decoder\n16 blocks; n = 46"
)

layer <- layer %>%
  mutate(
    block_index = extract_block_index(layer_name),
    facet_label = case_when(
      grepl("Flan", model) & block == "enc" ~ panel_levels[[1]],
      grepl("Flan", model) & block == "dec" ~ panel_levels[[2]],
      grepl("Llama", model) & block == "dec" ~ panel_levels[[3]],
      TRUE ~ NA_character_
    ),
    facet_label = factor(facet_label, levels = panel_levels),
    ribbon_low = pmax(0, mean_fn - 1.96 * se_fn),
    ribbon_high = mean_fn + 1.96 * se_fn,
    display_family = case_when(
      family == "SA.k" ~ "Self-attention key",
      family == "CA.k" ~ "Cross-attention key",
      family == "LN" ~ "LayerNorm",
      TRUE ~ "Other layer"
    ),
    display_family = factor(
      display_family,
      levels = c("Self-attention key", "Cross-attention key", "LayerNorm", "Other layer")
    )
  ) %>%
  filter(!is.na(facet_label), !is.na(block_index))

highlight_colors <- c(
  "Self-attention key" = wt_palette[["navy"]],
  "Cross-attention key" = wt_palette[["purple"]],
  "LayerNorm" = wt_palette[["vermillion"]],
  "Other layer" = wt_palette[["gray"]]
)
highlight_shapes <- c(
  "Self-attention key" = 16,
  "Cross-attention key" = 18,
  "LayerNorm" = 17,
  "Other layer" = 16
)

other_jitter <- position_jitter(width = 0.13, height = 0, seed = 20260915)

build_layer_plot <- function(panel_data, x_title, y_max, y_breaks) {
  key_layers <- filter(
    panel_data,
    display_family %in% c("Self-attention key", "Cross-attention key")
  )
  highlight_layers <- filter(panel_data, display_family != "Other layer")
  ggplot(panel_data, aes(x = block_index, y = mean_fn)) +
    geom_linerange(
      data = filter(panel_data, display_family == "Other layer"),
      aes(ymin = ribbon_low, ymax = ribbon_high),
      color = wt_palette[["light_gray"]], linewidth = 0.32, alpha = 0.42,
      position = other_jitter
    ) +
    geom_point(
      data = filter(panel_data, display_family == "Other layer"),
      aes(color = display_family, shape = display_family),
      size = 0.85, alpha = 0.45, position = other_jitter
    ) +
    geom_line(
      data = key_layers,
      aes(color = display_family, group = interaction(facet_label, display_family)),
      linewidth = 0.78, alpha = 0.9
    ) +
    geom_linerange(
      data = highlight_layers,
      aes(ymin = ribbon_low, ymax = ribbon_high, color = display_family),
      linewidth = 0.52, alpha = 0.76
    ) +
    geom_point(
      data = highlight_layers,
      aes(color = display_family, shape = display_family),
      size = 2.25, stroke = 0.4
    ) +
    facet_wrap(vars(facet_label), nrow = 1, scales = "free_x") +
    scale_color_manual(
      values = highlight_colors, breaks = names(highlight_colors), drop = FALSE
    ) +
    scale_shape_manual(
      values = highlight_shapes, breaks = names(highlight_shapes), drop = FALSE
    ) +
    scale_x_continuous(
      breaks = function(limits) pretty(limits, n = 6),
      expand = expansion(mult = c(0.045, 0.045))
    ) +
    scale_y_continuous(
      breaks = y_breaks,
      expand = expansion(mult = c(0, 0.02))
    ) +
    coord_cartesian(ylim = c(-5, y_max), clip = "on") +
    labs(
      x = x_title,
      y = "True clades missed (%)",
      color = "Layer family",
      shape = "Layer family"
    ) +
    guides(
      color = guide_legend(
        nrow = 1,
        override.aes = list(
          size = 2.7,
          alpha = 1,
          linewidth = 0
        )
      ),
      shape = "none"
    ) +
    theme_weighttraits(base_size = 11) +
    theme(
      strip.text = element_text(size = 10.0, lineheight = 1.05),
      panel.spacing = grid::unit(0.14, "cm"),
      legend.position = "bottom",
      legend.text = element_text(size = 9.3),
      legend.title = element_text(face = "bold", size = 9.5),
      legend.key.width = grid::unit(0.48, "cm"),
      plot.margin = margin(3, 4, 3, 4)
    )
}

fig_layer_flan_encoder <- build_layer_plot(
  filter(layer, facet_label == panel_levels[[1]], family != "relbias"), NULL,
  y_max = 5, y_breaks = seq(0, 5, 1)
) + theme(legend.position = "none")
fig_layer_flan_decoder <- build_layer_plot(
  filter(layer, facet_label == panel_levels[[2]], family != "relbias"), NULL,
  y_max = 30, y_breaks = seq(0, 30, 5)
) + theme(
  legend.position = "none",
  axis.title.y = element_blank()
)
fig_layer_flan <- fig_layer_flan_encoder + fig_layer_flan_decoder +
  plot_layout(widths = c(1, 1))
fig_layer_llama <- build_layer_plot(
  filter(layer, facet_label == panel_levels[[3]]), "Transformer block",
  y_max = 102, y_breaks = seq(0, 100, 20)
) + theme(legend.position = "none")
legend_data <- tibble(
  x = seq_along(highlight_colors),
  y = 1,
  display_family = factor(names(highlight_colors), levels = names(highlight_colors))
)
fig_layer_legend <- ggplot(
  legend_data,
  aes(x = x, y = y, color = display_family, shape = display_family)
) +
  geom_point(alpha = 0) +
  scale_color_manual(values = highlight_colors, breaks = names(highlight_colors)) +
  scale_shape_manual(values = highlight_shapes, breaks = names(highlight_shapes)) +
  labs(color = "Layer family", shape = "Layer family") +
  guides(
    color = guide_legend(
      nrow = 1,
      override.aes = list(size = 2.7, alpha = 1, linewidth = 0)
    ),
    shape = "none"
  ) +
  theme_void(base_size = 11) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 9.3, color = wt_palette[["charcoal"]]),
    legend.title = element_text(face = "bold", size = 9.5, color = wt_palette[["charcoal"]]),
    legend.key.width = grid::unit(0.48, "cm"),
    plot.margin = margin(0, 0, 0, 0)
  )
fig_layer <- (fig_layer_flan / fig_layer_llama / fig_layer_legend) +
  plot_layout(heights = c(1, 1.08, 0.10))

dir.create(dirname(figure_stem), recursive = TRUE, showWarnings = FALSE)
save_weighttraits(
  fig_layer,
  figure_stem,
  width = 7.15, height = 6.35
)

message("Wrote Figure 3 to ", figure_stem, ".{pdf,png}")
