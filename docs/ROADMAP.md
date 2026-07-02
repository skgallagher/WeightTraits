# WeightTraits Rebuild Roadmap

This is the working plan for the multi-week ELLMTrees rebuild. The intent is not to polish the old repository; it is to reconstruct the scientific pipeline in a cleaner repository with stronger tests and explicit provenance.

## Operating Rules

1. Treat ELLMTrees as a reference archive. Do not mutate it during the rebuild.
2. Every nontrivial result gets a command log, config snapshot, input manifest digest, output digest, and comparison target.
3. No cluster array runs without a dry run, an explicit run list, and a concurrency limit.
4. Prefer small deterministic checks first, then scale up.
5. Regression claims must be checked in R before they enter the paper.
6. The paper is rebuilt from registries, not hand-copied figures and tables.

## Phase 0: Repository Foundation

Deliverables:
- Private `WeightTraits` repo under `/Users/shannon/Desktop/phylo`.
- Package skeleton in `src/weighttraits`.
- Fast unit tests for manifest parsing, distance metrics, and audit inventory.
- Local and cluster config files with explicit guardrails.
- Initial ELLMTrees inventory report.

Exit criteria:
- `PYTHONPATH=src python -m pytest -q` passes.
- `weighttraits.cli audit-ellmtrees --source ../ELLMTrees` produces a JSON inventory.
- Git repository exists and is ready to push privately.

## Phase 1: Freeze the Reference Surface

Deliverables:
- Complete inventory of ELLMTrees code, configs, generated outputs, paper figures, and result tables.
- Reference manifest digests for key experiment groups.
- Reference output digests for current paper figures and tables.
- A migration map from old scripts to new package modules.
- A list of results that are paper-critical, provisional, stale, or excluded.

Checks:
- Compare local Git status and untracked result state in ELLMTrees.
- Record July 2026 handoff constraints before running any cluster work.
- Verify which outputs can be regenerated locally versus only on Wright/Ghidorah.

## Phase 2: Rebuild Core Phylogenetic Logic

Deliverables:
- Flexible topology generation with clean separation from task/dataset assignment.
- Experiment-flow schemas for: make trees, pick tasks/datasets, train, analyze weights, or analyze behaviors.
- Manifest and topology code.
- Weight-distance readers and distance cubes.
- Neighbor-joining reconstruction and RF/clade recovery.
- Tree-space and Frechet mean pieces, if retained for the paper.

Checks:
- Generate the same topology under multiple task policies and confirm training manifests are comparable.
- Generate multiple topologies under one task policy and confirm downstream code does not assume one tree shape.
- Unit tests from known bug cases: multifurcating roots, identity nodes, star-tree filters, cumulative LoRA measurement.
- Numerical comparisons against ELLMTrees toy/cpu-smoke outputs.
- Independent small synthetic examples with known tree topology and known distance matrix.

## Phase 3: Smoke/Toy Rebuild

This is the first scientific target.

Deliverables:
- Rebuild toy/cpu-smoke manifests using `ellmtrees_balanced`, Poisson branching, and pruned-binary-backbone topology.
- Run no-download checks for topology, leaf sets, reference splits, and small synthetic distance matrices.
- Reproduce the old smoke/toy recovery outputs where local artifacts exist.

Checks:
- Minimum depth and minimum leaves are enforced.
- Polytomies are present in at least one smoke topology.
- RF scoring handles shallow, deep, chainy, and polytomous reference trees.

## Phase 4: Rebuild RF Tables

This is the second scientific target.

Deliverables:
- Whitebox recovery result schema.
- RF/clade-recovery tables for paper-critical groups.
- Constant-run-set comparison ledgers for cloned topologies.
- Old-vs-new RF comparison report.

Checks:
- Run counts and leaf counts are audited before aggregation.
- Star-tree and `<4` common-leaf exclusions are reported explicitly.
- New RF tables match ELLMTrees within declared tolerances or explain deviations.

## Phase 5: Rebuild Whitebox Experiments

Deliverables:
- Clean experiment specification model.
- Local smoke workflow.
- Cluster workflow with explicit run lists and resumable job ledgers.
- Result schemas for per-run distance cubes, reconstructed trees, RF summaries, and aggregate recovery.

Checks:
- Reproduce smoke and toy runs first.
- Reproduce selected ELLMTrees run outputs where checkpoints are already local.
- Confirm cumulative-vs-increment behavior for LoRA-style checkpoints.
- Keep run-set comparisons fixed across cloned topologies.

## Phase 6: Rebuild Behavioral and Regression Analyses

Deliverables:
- Behavioral result ingestion.
- Pairwise semantic/ROUGE/BLEU distance builders.
- Python aggregation for convenience.
- R regression scripts as the authoritative cross-check for reported coefficients.
- Regression diagnostic plots regenerated from the new outputs.

Checks:
- R `lm` and `lme4::lmer` outputs match expected signs, magnitudes, and confidence intervals.
- Compare paired versus centroid semantic DVs where both are available.
- Preserve caveats for known task-label vocabulary collapse in TaskHoldout multiple-choice probes.

## Phase 7: Rebuild HF-Zoo and Blackbox Validation

Deliverables:
- Organic model-family manifests.
- Whitebox recovery scorer.
- Blackbox response/embedding distance bridge.
- Side-by-side whitebox versus behavior-only recovery table for the same truth tree.

Checks:
- Re-score Mistral `d_mistral_7b` from frozen references.
- Compare whitebox 3/3 split recovery against blackbox task-specific recovery.
- Audit prompt counts, response counts, embedding shapes, and dropped-empty records.

## Phase 8: Rebuild the ICLR Draft

Deliverables:
- Figure registry with source command and digest per figure.
- Table registry with source command and digest per table.
- Paper build target.
- Appendix/rebuttal material separated from main claims.

Checks:
- Rebuild the PDF from a clean checkout.
- Verify every cited numeric claim has a generated source file.
- Run a final stale-claim audit against the old draft and notes.

## Recurring Cadence

Daily:
- Run fast tests.
- Commit small, reviewable changes.
- Update the audit ledger when outputs change.

Weekly:
- Run smoke tests.
- Regenerate the reference comparison dashboard.
- Review cluster queue state and job failures.
- Update this roadmap with completed/blocked items.

Before any paper-facing change:
- Re-run the relevant figure/table command.
- Re-run the relevant R regression check if the claim is statistical.
- Record tolerances and differences from ELLMTrees.

## Definition of Done

The rebuild is done when a clean clone of `WeightTraits` can recreate the full ICLR draft and all paper-critical results from declared inputs, with tests passing, R regression checks recorded, and any deviations from ELLMTrees documented rather than hidden.
