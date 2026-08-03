# WeightTraits Documentation

Start with the main [README](../README.md) for installation and a five-minute topology workflow.
The guides below are grouped by what a user is trying to accomplish. Some older files are retained
as dated project records; they are useful for provenance but are not the best onboarding path.

## Understand the workflow

- [Pipeline Flow](PIPELINE_FLOW.md) — separation between topology, task/data assignment, training,
  white-box analysis, behavior analysis, and black-box studies.
- [Audit Protocol](AUDIT_PROTOCOL.md) — evidence required before a result is treated as reproduced.
- [Tree Generation](TREE_GENERATION.md) and [Tree Generator Math](TREE_GENERATOR_MATH.md) — supported
  topology families, constraints, and sampling definitions.

## Run local examples

- [Run Tree Examples](RUN_TREE_EXAMPLES.md) — generate and verify the bundled topology examples.
- [Training Examples](../examples/training/README.md) — plan jobs without model downloads, then move
  to tiny real-training and lineage smokes.
- [Distance Input Examples](../examples/distance_inputs/README.md) — manifests for full-weight,
  cumulative-LoRA, and merged-LoRA analyses.

## Analyze weights and trees

- [Distance Metrics](DISTANCE_METRICS.md) — supported metrics and metric-registry design.
- [Streaming Distance Cubes](STREAMING_DISTANCE_CUBES.md) — chunked distance computation for large
  checkpoints.
- [LoRA Distance Model](LORA_DISTANCE_MODEL.md) — cumulative adapter displacement and merged-weight
  semantics.
- [Recovery Scoring](RECOVERY_SCORING.md) — clade matching, RF, FP/FN, PAER, and uncertainty.

## Train model families

- [Trainer Design](TRAINER_DESIGN.md) — planning, prompt resolution, ledgers, retention, and resume
  behavior.
- [Task and Dataset Candidates](TASK_DATA_CANDIDATES.md) — dataset contracts and audit status.
- [Experiment Checklist](EXPERIMENT_CHECKLIST.md) — current queue-facing research checklist; this
  includes environment-specific paths and is primarily for maintainers.

## Analyze behavior

- [Behavioral Rebuild](BEHAVIORAL_REBUILD.md) — response collection, embeddings, paired distances,
  and regression inputs.
- [PhyloLM Provenance](PHYLOLM_PROVENANCE.md) — clean implementation boundary and external gene-pool
  requirements.
- [R Regression Checks](R_REGRESSION_CHECKS.md) — independent statistical cross-checks.

## Reproduce paper artifacts

- [Paper Reproduction](PAPER_REPRODUCTION.md) — registries, commands, comparison gates, and known
  external-input requirements.
- [Paper Workspace](../paper/README.md) — paper registry files and artifact builders.
- [Workbook Verification Record](WORKBOOK_REPRO_CHECK_20260803.md) — independent audit of the paper
  reproducibility workbook and the corrections made after that audit.

## Scientific reference

- [Flexible Tree Mathematics](TREE_GENERATOR_MATH.md) — exact generator definitions.
- [Recovery Scoring](RECOVERY_SCORING.md) — estimands and polytomy behavior.

## Maintainer and historical records

The following files capture project state at particular dates. They may contain local paths, cluster
job IDs, or superseded next steps and should not be treated as general setup guides:

- [Current Handoff](HANDOFF.md)
- [Roadmap](ROADMAP.md)
- [Behavioral Stochastic Handoff, 2026-07-14](HANDOFF_BEHAVIORAL_STOCHASTIC_2026-07-14.md)
- [Weekend Handoff, 2026-07-08](WEEKEND_HANDOFF_2026-07-08.md)
- [Llama-3.2-1B Experiment Plan](LLAMA32_1B_EXPERIMENT_PLAN.md)
