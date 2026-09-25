# Shared publication styling for the WeightTraits / ELLMTrees paper figures.

wt_palette <- c(
  navy = "#0072B2",
  sky = "#56B4E9",
  teal = "#009E73",
  orange = "#E69F00",
  vermillion = "#D55E00",
  purple = "#CC79A7",
  charcoal = "#273142",
  gray = "#7A8491",
  light_gray = "#D6DAE0"
)

theme_weighttraits <- function(base_size = 9, base_family = "Helvetica") {
  ggplot2::theme_minimal(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      text = ggplot2::element_text(color = wt_palette[["charcoal"]]),
      plot.title = ggplot2::element_text(
        size = ggplot2::rel(1.15), face = "bold", hjust = 0,
        margin = ggplot2::margin(b = 7)
      ),
      plot.subtitle = ggplot2::element_text(
        size = ggplot2::rel(0.95), color = "#4B5563",
        margin = ggplot2::margin(b = 8)
      ),
      axis.title = ggplot2::element_text(size = ggplot2::rel(1.0)),
      axis.title.x = ggplot2::element_text(margin = ggplot2::margin(t = 6)),
      axis.title.y = ggplot2::element_text(margin = ggplot2::margin(r = 6)),
      axis.text = ggplot2::element_text(color = "#374151"),
      panel.grid.major = ggplot2::element_line(color = "#E5E7EB", linewidth = 0.5),
      panel.grid.minor = ggplot2::element_blank(),
      panel.border = ggplot2::element_rect(
        color = "#C7CDD4", fill = NA, linewidth = 0.65
      ),
      strip.text = ggplot2::element_text(face = "bold", hjust = 0),
      strip.background = ggplot2::element_rect(fill = "#F3F4F6", color = NA),
      legend.position = "bottom",
      legend.justification = "left",
      legend.box.just = "left",
      legend.title = ggplot2::element_blank(),
      legend.text = ggplot2::element_text(size = ggplot2::rel(0.88)),
      legend.key.height = grid::unit(0.35, "cm"),
      legend.key.width = grid::unit(0.55, "cm"),
      plot.margin = ggplot2::margin(7, 9, 7, 7)
    )
}

save_weighttraits <- function(plot, stem, width, height, dpi = 320) {
  dir.create(dirname(stem), recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(
    paste0(stem, ".pdf"), plot = plot, width = width, height = height,
    units = "in", device = "pdf", bg = "white"
  )
  ggplot2::ggsave(
    paste0(stem, ".png"), plot = plot, width = width, height = height,
    units = "in", dpi = dpi, bg = "white"
  )
}
