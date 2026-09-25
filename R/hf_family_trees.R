#!/usr/bin/env Rscript
# Figs 5 and 6 (App. D): true tree | weight-based estimate | PhyloLM estimate for the two
# documented Hugging Face families.
#
# usage: Rscript R/hf_family_trees.R DATA_DIR FIGURE_DIR
#   DATA_DIR    holds {mistral,llama2}_tree_{segments,tips}.csv from
#               weighttraits.paper.hf_family_figures.write_family_figure_data
#   FIGURE_DIR  receives hf_mistral_hero_3panel.{pdf,png} and hf_llama2_hero_3panel.{pdf,png}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: hf_family_trees.R DATA_DIR FIGURE_DIR")
data_dir <- args[[1]]
figure_dir <- args[[2]]
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

script_dir <- dirname(normalizePath(sub(
  "^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[[1]]
)))
source(file.path(script_dir, "theme_weighttraits.R"))

build_tree_figure <- function(stem, lineage_colors, x_max, text_size) {
  tree_segments <- read_csv(
    file.path(data_dir, paste0(stem, "_tree_segments.csv")), show_col_types = FALSE
  )
  tree_tips <- read_csv(
    file.path(data_dir, paste0(stem, "_tree_tips.csv")), show_col_types = FALSE
  )
  panel_levels <- tree_tips %>% arrange(panel_order) %>% distinct(panel) %>% pull(panel)
  tree_segments$panel <- factor(tree_segments$panel, levels = panel_levels)
  tree_tips$panel <- factor(tree_tips$panel, levels = panel_levels)

  ggplot() +
    geom_segment(
      data = filter(tree_segments, kind == "tree"),
      aes(x = x, xend = xend, y = y, yend = yend),
      color = wt_palette[["charcoal"]], linewidth = 1.25,
      lineend = "round"
    ) +
    geom_segment(
      data = filter(tree_segments, kind == "connector"),
      aes(x = x, xend = xend, y = y, yend = yend),
      color = "#AEB6C0", linewidth = 0.8, linetype = "dotted"
    ) +
    geom_point(
      data = tree_tips,
      aes(x = x + 0.015, y = y, color = lineage),
      size = 3.2
    ) +
    geom_text(
      data = tree_tips,
      aes(x = x + 0.055, y = y, label = display),
      hjust = 0, size = text_size, family = "Helvetica",
      color = wt_palette[["charcoal"]]
    ) +
    facet_wrap(vars(panel), nrow = 1) +
    scale_color_manual(values = lineage_colors, breaks = names(lineage_colors)) +
    scale_x_continuous(limits = c(-0.02, x_max), expand = c(0, 0)) +
    scale_y_reverse(expand = expansion(add = 0.45)) +
    coord_cartesian(clip = "off") +
    labs(color = NULL) +
    theme_void(base_size = 11, base_family = "Helvetica") +
    theme(
      strip.text = element_text(
        face = "bold", size = 11.5, lineheight = 1.05,
        color = wt_palette[["charcoal"]], margin = margin(b = 5)
      ),
      strip.background = element_rect(fill = "#F3F4F6", color = NA),
      panel.spacing.x = grid::unit(0.12, "in"),
      legend.position = "bottom",
      legend.justification = "left",
      legend.box.just = "left",
      legend.text = element_text(size = 10.3),
      legend.key.width = grid::unit(0.35, "cm"),
      plot.margin = margin(5, 4, 4, 4)
    ) +
    guides(color = guide_legend(nrow = 1, override.aes = list(size = 3.5)))
}

mistral_colors <- c(
  "Zephyr fork" = wt_palette[["navy"]],
  "OpenChat-Starling alpha" = wt_palette[["vermillion"]],
  "OpenChat-Starling beta" = wt_palette[["teal"]],
  "Other checkpoints" = wt_palette[["gray"]]
)
save_weighttraits(
  build_tree_figure("mistral", mistral_colors, x_max = 2.18, text_size = 3.75),
  file.path(figure_dir, "hf_mistral_hero_3panel"),
  width = 7.15, height = 3.85
)

llama2_colors <- c(
  "Chat clade (chat + 6 descendants)" = wt_palette[["navy"]],
  "Base / Nous-Hermes / Vicuna" = wt_palette[["vermillion"]]
)
save_weighttraits(
  build_tree_figure("llama2", llama2_colors, x_max = 2.55, text_size = 3.3),
  file.path(figure_dir, "hf_llama2_hero_3panel"),
  width = 7.15, height = 3.85
)

message("Wrote documented-family tree figures to ", figure_dir)
