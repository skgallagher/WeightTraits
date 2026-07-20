# ELLMTrees Reference Notes

Initial inspection on 2026-07-02 found:

- `ELLMTrees` is a Git worktree under `/Users/shannon/Desktop/phylo/ELLMTrees`.
- The workspace is dirty and contains many untracked outputs/results.
- Approximate size: 12G.
- The old repository has script-heavy analysis and cluster workflows.
- `CLAUDE.md` and `codex/HANDOFF.md` contain fresh July 2026 state and cluster guardrails that must be treated as provenance.

Important current guardrails from the handoff:
- Do not submit broad catch-up arrays.
- Use explicit run lists and dry runs.
- Keep cloned-topology comparisons on constant run sets.
- Do not pursue `runs_llama8b_full_ft_approx` for the current straggler batch unless this is deliberately revisited.
- Wright straggler work should use partition `all`, account `statds`, and throttled concurrency.

## ICLR v2 Reference Surface

As of 2026-07-07, the live draft target is `../ELLMTrees-paper/iclr_draft_v2.tex`.
Shannon is roughly two-thirds through the draft and revising the results section.
Treat the latest paper state as ground truth: when old ELLMTrees notes, scripts, or generated outputs
conflict with the current paper, preserve the latest paper-grounded artifact in the WeightTraits
registry rather than carrying forward stale old-reference wording.

The current paper-facing reference snapshot is tracked in `paper/reference_registry.yaml` and can be
checked with:

```bash
PYTHONPATH=src python -m weighttraits.cli validate-reference-registry \
  --registry paper/reference_registry.yaml
```

The initial registry records:

- the active draft path, checked for existence while the draft is still changing;
- old reproducibility/runbook provenance docs;
- the five figure assets referenced by `iclr_draft_v2.tex`;
- main results targets for `tab:variants`, `fig:coherence_recovery`, `tab:behavior_holdout`, and
  `fig:real-trees`;
- static/supporting tables for related work and TaskHoldout translation-probe examples;
- appendix support targets for LayerTrace, regression diagnostics, heterogeneity, within-task
  recovery, and LoRA/layer definition tables.

One provenance detail is worth preserving: the paper copy
`../ELLMTrees-paper/figures/fig4_coherence_atteson.png` currently matches
`../ELLMTrees/results/aggregate/recovery_rescore/fig4_atteson_layermeans.png`, not the older
same-name `fig4_coherence_atteson.png` in the old recovery-rescore directory.

The first generated old-reference table is now `reports/paper/ellmtrees_variants_reference.{json,csv}`,
built from `paper/ellmtrees_variants_registry.yaml`. It reconstructs the active draft's
`tab:variants` rows from old recovery and branch-structure CSVs, including per-run SEs where the old
per-run source is available.
`reports/paper/ellmtrees_variants_reference_compare.json` compares the generated JSON and CSV forms by
`variant_id` and currently validates all 8 rows with no mismatches.
