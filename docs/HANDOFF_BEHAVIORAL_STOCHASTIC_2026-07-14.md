# Behavioral Stochastic Protocol Handoff

Date: 2026-07-14

## Decision

The primary WeightTraits behavioral bridge will use three stochastic generations for every
prompt/model pair. The frozen defaults are temperature 1.0, top-p 1.0, and base seed 42. Draw IDs
0, 1, and 2 use seeds 42, 43, and 44 across all models. A one-generation greedy run is retained only
as a sensitivity analysis.

Each generation is embedded independently with `all-MiniLM-L6-v2`. Model-pair cosine distances are
computed for aligned `(prompt_id, sample_id)` observations and averaged over the complete audited
grid. The regression CSV exports behavioral distance and `behavior_similarity = 1 - distance`.
Paper coefficients use similarity, so inherited behavior corresponds to a negative association with
weight distance.

## Implemented

- `collect-behavior-responses` defaults to three draws with sampling enabled.
- Each draw resets and records its own seed; generation metadata records temperature, top-p, batch
  size, maximum new tokens, and base seed.
- Wright wrappers default to the same three-draw contract, including post-job grid validation.
- Response audits report per-model mean unique draws per prompt and the fraction of prompts whose
  draws vary, in addition to completeness, empty-output, uniqueness, dominance, and ROUGE-L signals.
- Empty prompt sets and all-empty embedding exports fail explicitly.
- The live ICLR methods paragraph specifies the protocol.

## Interpretation and provenance warning

Archived Wright behavioral jobs `154614`, `154621`, `154624`, `154633`, `154635`, and `154636` used
the earlier one-greedy-generation path. They remain useful probe-health diagnostics, but they are not
the final three-draw estimator. Existing ICLR Table 3 values also predate this WeightTraits
confirmation. Regenerate them before claiming the stochastic protocol produced those numbers.

## Next execution gate

Wait for a production Llama r8 or r64 tree to complete. Run one 100-prompt, three-draw HellaSwag tree
first, then inspect the response audit and finite semantic cube before submitting the full array.
Required health review includes empty rate, unique responses by model, dominant response fraction,
mean unique draws per prompt, varying-prompt fraction, reference ROUGE-L, and raw examples. Do not
promote a collapsed probe merely because its artifacts are complete.

After the gate passes, run the remaining held-out probes, export paired regression rows, compute one
correlation per tree with behavioral similarity, pool tree effects with DerSimonian-Laird, and
cross-check the final coefficients independently in R.

## Open statistical audit item

`behavior/meta.py` currently uses the conventional Fisher-z sampling variance `1 / (n_pairs - 3)` for
each tree correlation. Model pairs from the same tree share leaves and are not independent. Before
final confidence intervals are treated as inferential, add a leaf-aware bootstrap or another
clustered uncertainty calculation and compare it with the current DerSimonian-Laird result. The
point estimates are still valid descriptive within-tree correlations; this caveat concerns their
weights and confidence intervals.
