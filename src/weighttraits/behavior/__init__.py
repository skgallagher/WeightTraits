"""Behavioral probing, distance, meta-analysis, and PhyloLM helpers."""

from weighttraits.behavior.distances import (
    BehaviorDistanceResult,
    paired_cosine_distances,
)
from weighttraits.behavior.meta import (
    CorrelationEffect,
    RandomEffectsCorrelation,
    dersimonian_laird_correlations,
)
from weighttraits.behavior.phylolm import (
    PHYLOLM_UPSTREAM_COMMIT,
    compute_population,
    nei_distance_matrix,
    nei_similarity,
    sample_genes,
)
from weighttraits.behavior.probes import (
    BehaviorPrompt,
    hellaswag_prompts_from_rows,
    load_behavior_prompts,
    write_behavior_prompts,
)
from weighttraits.behavior.responses import (
    BehaviorResponse,
    audit_behavior_responses,
    load_behavior_responses,
    write_behavior_responses,
)
from weighttraits.behavior.regression import paired_distance_rows, write_paired_distance_rows

__all__ = [
    "BehaviorDistanceResult",
    "BehaviorPrompt",
    "BehaviorResponse",
    "CorrelationEffect",
    "PHYLOLM_UPSTREAM_COMMIT",
    "RandomEffectsCorrelation",
    "audit_behavior_responses",
    "compute_population",
    "dersimonian_laird_correlations",
    "hellaswag_prompts_from_rows",
    "load_behavior_prompts",
    "load_behavior_responses",
    "nei_distance_matrix",
    "nei_similarity",
    "paired_cosine_distances",
    "paired_distance_rows",
    "sample_genes",
    "write_behavior_prompts",
    "write_behavior_responses",
    "write_paired_distance_rows",
]
