# Compute and runtime

Observed compute for the paper's fixed-2,000-step cohorts and downstream analyses. The source is
Slurm accounting on Wright (`sacct`, `ElapsedRaw` × allocated GPUs), queried on 20 Oct 2026.
These are not estimates. Times are Wright local time (EDT). GPU-hours include attempts that were
preempted or retried.

## Training

The hardware was NVIDIA L40 (48 GB). Each tree trained sequentially on one GPU, so that every
parent checkpoint existed before its children. Different trees ran concurrently. Every cohort has
50 trees and 641 trained nodes, and every node ran exactly 2,000 optimizer steps.

| Cohort | GPU-hours | Main array (start → end) | Notes |
| --- | ---: | --- | --- |
| Llama-3.2-1B full fine-tuning + Llama no-translation | 769.5 | 10 Aug 13:23 → 13 Aug 15:43 | trained together (`wt-cp-tree`, 78 tasks, plus 24 single-tree recoveries) |
| Flan-T5-base full fine-tuning | 315.2 | 17 Aug 16:52 → 20 Aug 00:07 | tree 001 canary first (17 Aug 10:05) |
| Flan LoRA q/k/v/o + FFN (summed adapters) | 302.3 | 17 Aug 16:52 → 19 Aug 01:58 | 12 trees preempted and rerun 19–20 Aug |
| Flan LoRA q/v | 202.8 | 19 Aug 17:19 → 21 Aug 08:33 | |
| Flan LoRA key only | 179.1 | 21 Aug 04:27 → 22 Aug 13:51 | |
| Flan LoRA q/k/v | 233.4 | 22 Aug 09:57 → 24 Aug 04:18 | |
| Flan LoRA q/k/v/o | 259.0 | 23 Aug 23:35 → 25 Aug 22:59 | |
| Flan LoRA legacy all-projection declaration | 273.8 | 25 Aug 17:35 → 27 Aug 19:37 | |
| Llama LoRA q/k/v r=8 | 290.1 | 21 Aug 02:46 → 24 Aug 05:25 | |
| Llama LoRA q/k/v r=64 | 295.8 | 24 Aug 05:25 → 27 Aug 11:33 | |
| Flan no-translation | 308.5 | 27 Aug 11:34 → 30 Aug 18:44 | |
| **Total** | **3,429.6** | 10 Aug → 30 Aug | |

From the August ledgers, median training time per node is 27.5 min for Flan-T5-base full
fine-tuning and 19.5 min for Llama-3.2-1B full fine-tuning. The Flan full fine-tuning cohort ran on
about five GPUs at once, so its 315 GPU-hours took 62 hours of wall time.

## Downstream analysis (31 Aug – 30 Sep)

| Stage | GPU-hours | Allocated CPU core-hours |
| --- | ---: | ---: |
| Direct-weight distance cubes and run sets (all conditions) | 0 | ≈ 209 |
| Behavior inference, all 641 nodes × 4 cohorts (`wt-an-*-infer`) | 66.8 | ≈ 535 |
| Behavior semantic matrices | 10.8 | ≈ 87 |
| MoTHer baseline (8 conditions) | 0 | ≈ 1,067 |
| PhyloLM rerun on the fixed-2,000-step Llama cohorts (24 Sep) | 5.1 | ≈ 61 |
| **All jobs, including small ones not listed** | **83.3** | **≈ 1,984** |

## Other runs

* The July early-stopped cohorts are a training-depth sensitivity analysis, not the primary
  result. Seven Flan-T5 conditions used 1,427.8 GPU-hours, recorded on branch
  `docs/compute-runtime`.
* For the October task-replacement sensitivity experiment, see
  [TASK_REPLACEMENT_SENSITIVITY.md](TASK_REPLACEMENT_SENSITIVITY.md) on its branch.
