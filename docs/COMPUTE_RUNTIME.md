# Compute and runtime

This page records how long the main WeightTraits runs took and where they ran. The numbers below
come from Slurm accounting (`ElapsedRaw`, allocation, start, and end) queried on 2026-08-03. They
are observed runtimes, not estimates.

## Completed Flan-T5 training sets

Each row contains 50 independent training trees. A tree ran sequentially on one NVIDIA L40 GPU so
that parent artifacts existed before their descendants. Different trees ran concurrently through
Slurm.

| Condition | Trees | Mean per tree | Longest tree | Total GPU-hours | Submission-to-completion window | Slurm jobs |
|---|---:|---:|---:|---:|---:|---|
| Full fine-tuning | 50 | 3.07 h | 3.76 h | 153.26 | 3 d 7 h 11 m | `153745`, `153752` |
| LoRA q/v | 50 | 3.63 h | 4.46 h | 181.47 | 4 d 5 h 59 m | `153785` |
| LoRA k | 50 | 3.28 h | 3.89 h | 164.17 | 2 d 2 h 3 m | `153909` |
| LoRA q/k/v | 50 | 4.15 h | 5.02 h | 207.49 | 2 d 13 h 13 m | `153910` |
| LoRA q/k/v/o | 50 | 4.62 h | 5.53 h | 230.88 | 3 d 9 h 6 m | `153911` |
| LoRA legacy full-FT approximation | 50 | 4.92 h | 5.89 h | 246.24 | 3 d 20 h 18 m | `153912` |
| LoRA all projections, corrected | 50 | 4.89 h | 6.43 h | 244.31 | 3 d 3 h 43 m | `154446` plus recoveries |
| **Total** | **350** | — | — | **1,427.82** | — | — |

These jobs ran on the Wright Slurm cluster's `all` partition. Each tree requested one NVIDIA L40,
4 CPU cores, and 64 GB RAM. The completed training sets consumed 1,427.82 GPU-hours, or 59.49
serial GPU-days. Their calendar windows overlap, so those windows must not be added together.

The legacy full-FT-approximation declaration requested `q`, `k`, `v`, `o`, `wi`, and `wo`. On
Flan-T5 the actual PEFT scope omitted `wi` because the model exposes `wi_0` and `wi_1`. The corrected
all-projection run explicitly targeted both split input projections. Its accounting includes the
successful tasks from the original array and scoped recovery jobs `154679`, `154856`, `154901`,
and `155884`.

## CPU analysis and rollup

The weight-distance, reconstruction, scoring, and paper-rollup jobs reused the persisted training
artifacts and did not request GPUs.

| Analysis | Elapsed | Allocation | Slurm job |
|---|---:|---:|---|
| Full fine-tuning | 46 m 27 s | 8 CPU, 96 GB | `154440` |
| LoRA q/v | 5 m 20 s | 8 CPU, 96 GB | `154441` |
| LoRA k | 3 m 24 s | 8 CPU, 96 GB | `154442` |
| LoRA q/k/v | 7 m 16 s | 8 CPU, 96 GB | `154443` |
| LoRA q/k/v/o | 9 m 5 s | 8 CPU, 96 GB | `154444` |
| LoRA legacy full-FT approximation | 13 m 4 s | 8 CPU, 96 GB | `154445` |
| Corrected all-projection completion audit | 23 s | 2 CPU, 8 GB | `155044` |
| Corrected all-projection analysis and rollup | 12 m 19 s | 8 CPU, 96 GB | `155045` |
| **Total job time** | **1 h 37 m 18 s** | **12.94 allocated CPU-core-hours** | — |

Several analysis jobs ran at the same time. Total job time is the sum of their individual elapsed
times; it is not the end-to-end calendar time.

## Llama 3.2 1B benchmarks and partial arrays

Before broad execution, one complete 14-node tree was run for each Llama condition with bounded
training settings:

| Condition | One-tree runtime | Environment | Slurm job |
|---|---:|---|---|
| Full fine-tuning | 14 m 28 s | Wright, 1 L40, 4 CPU, 64 GB | `154555` |
| LoRA q/k/v rank 8 | 9 m 40 s | Wright, 1 L40, 4 CPU, 64 GB | `154556` |
| LoRA q/k/v rank 64 | 9 m 58 s | Wright, 1 L40, 4 CPU, 64 GB | `154557` |

The subsequent 50-tree Llama arrays were not complete and are not presented as reproduced result
sets. Their compute is disclosed here so failed work is not hidden:

| Condition | Completed | Failed | Total GPU-hours, including failures | Mean completed tree | Run window | Slurm job |
|---|---:|---:|---:|---:|---:|---|
| LoRA q/k/v rank 8 | 37/50 | 13 | 466.37 | 11.64 h | 3 d 17 h 12 m | `154594` |
| LoRA q/k/v rank 64 | 32/50 | 18 | 392.26 | 10.93 h | 2 d 8 h 6 m | `154595` |

## Behavioral and PhyloLM validation runs

These were contract and complete-tree smoke tests, not publication-scale production collections.

| Validation | GPU job time | CPU post-processing | Environment | Slurm jobs |
|---|---:|---:|---|---|
| Generic complete-tree behavioral probe | 1 m 54 s across 8 leaves | 13 s | Wright, L40 leaves | `154614`, `154621` |
| 100-prompt behavioral diagnostic | 2 m 6 s across 8 leaves | — | Wright, L40 leaves | `154624` |
| One-leaf causal probe preflight | 23 s | — | Wright, 1 L40 | `154633` |
| Complete-tree causal alignment | 2 m 22 s across 7 leaves | 21 s | Wright, L40 leaves | `154635`, `154636` |
| One-leaf PhyloLM collection smoke | 15 s | — | Wright, 1 L40 | `154601` |

## Local validation baseline

On 2026-08-03, the full Python test suite plus Ruff completed in **3.04 seconds** on a 10-core Apple
M5 MacBook Pro with 16 GB RAM. This is a developer baseline rather than a cross-platform benchmark;
test runtime varies with filesystem cache, Python environment, and optional dependencies.

## How the figures were calculated

The source query for each run set was equivalent to:

```bash
sacct -j JOB_IDS -X -n \
  --format=JobIDRaw,JobName,State,ElapsedRaw,AllocTRES,Start,End -P
```

- **Mean per tree** is the sum of `ElapsedRaw` for completed GPU tasks divided by the number of
  completed trees.
- **Total GPU-hours** is the sum of task elapsed time multiplied by the one-GPU allocation.
- **Submission-to-completion window** spans the earliest task start to the latest successful task
  end. It includes concurrency and gaps between tasks, but not time before the first task started.
- **Allocated CPU-core-hours** multiplies each CPU job's elapsed time by its requested core count.

This is an accounting of the named reproducibility runs, not a claim about the project's entire
development footprint. It excludes unrecorded local exploration, queue time before allocation,
download and cache warming, and historical source-artifact generation performed outside
WeightTraits. When new production runs are added, their job IDs and accounting snapshot should be
added here rather than replacing these observed values.
