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

