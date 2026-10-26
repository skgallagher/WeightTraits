# All-node behavior study (`weighttraits.behavior.allnode`)

This package computes the paper's behavior results: Table 3 (`tab:behavior_holdout`), the App. E
heterogeneity table, and App. F (`tab:eos-sensitivity`, `tab:eos-depth`). It compares weight
distance with behavioral distance over every trained model in a tree, with internal nodes treated
as observed ancestors. It then pools the per-tree associations across trees with a
DerSimonian–Laird meta-analysis and jackknife intervals.

## Provenance

The study ran on Wright from deployments under
`/home/export/sgallagh/WeightTraits-downstream-20260831-native/deployments/`:

* `allnode_v1`: base deployment, cohort configs, runtime inputs;
* continuations `20260902`, `v1c2` … `v1c5`.

The package code was never in git before this port. The modules here are the final deployment's
`weighttraits_allnode` package (the v1c3–v1c5 core) plus its `continuation` module. They are
byte-identical apart from the package rename. Strings that name the deployment's own directory
layout (`weighttraits_allnode/…`) are unchanged, because the deployment-attestation modules check
that layout. The five `scripts/slurm/allnode_*.sbatch` wrappers are the deployed wrappers, kept
verbatim as the record of what ran. The deployment's 28 test files are ported as
`tests/test_allnode_*.py`.

## What each module does

**Statistics and data contracts**

| Module | Role |
| --- | --- |
| `semantic_analysis` | Per-tree regressions of semantic similarity on weight distance (all trained-node pairs, plus ancestry strata and the leaf-only sensitivity); DerSimonian–Laird pooling with jackknife intervals; clustered and legacy regressions |
| `analysis` | Prespecified per-tree Pearson correlations and equal-weight aggregation |
| `covariates` | Node design: depth, role, and the training-task labels on each path |
| `matrices`, `manifest` | Validated all-node distance matrices and the topology contracts that order them |
| `inference`, `runtime` | Exact request and response grids for behavior inference (prompts × draws × nodes) |
| `encoder_snapshot` | The pinned MiniLM sentence encoder |
| `production` | Scope of the four cohorts: 50 trees and 641 trained nodes each |

**Runners** (filesystem boundaries that execute and cache each stage)

| Module | Role |
| --- | --- |
| `inference_runner` | Generates responses for every trained node (GPU) |
| `semantic_runner` | Embeds responses, builds per-tree semantic matrices, runs `semantic_analysis`, and writes run sets |
| `direct_execution`, `direct_runner` | Direct weight cosine distances over all trained nodes |

**Paper post-processors and verification**

| Module | Role |
| --- | --- |
| `replay` | Read-only replay of a cohort's semantic study from its saved artifacts (see below) |
| `eos_excluded` | App. F sensitivity: the bridge after excluding pairs where either response is empty |
| `empty_rates` | App. F empty-output rates, pooled and by leaf depth |

**Deployment orchestration** (kept as the execution record)

`audit`, `checkpoint_identity`, `checkpoints`, `compatibility_preflight`,
`direct_compatibility`, `gate`, `launch`, `loader`, `oneshot`, `preflight`, `provenance`,
`resolver`, `source_specs`, `symlink_inventory`, `continuation`, `jsonl`. These attest
checkpoints, pin inputs, gate submission, and drive the Slurm one-shot launches on Wright.

## Reproducing the paper's numbers

Every command below is read-only against the downstream root and runs on Wright from a checkout
of this repository, with `PYTHONPATH=src`. Each cohort's job config is
`deployments/allnode_v1/configs/jobs/semantic_<cohort>.json`. Cohorts are `llama_full`,
`llama_notrans`, `flan_full`, and `flan_notrans`.

```bash
# Table 3 and App. E: replay every endpoint of a cohort and compare with the saved receipts
python -m weighttraits.behavior.allnode.replay --config <job config>

# App. F, empty-excluded column (Llama full fine-tuning)
python -m weighttraits.behavior.allnode.eos_excluded --config <llama_full job config> \
  --endpoint hellaswag --output <path>

# App. F, empty rates pooled and by depth
python -m weighttraits.behavior.allnode.empty_rates --cohort llama_full
```

## Verification (20 Oct 2026)

Everything was run on Wright from this package against the saved study.

* **Table 3 and App. E.** `replay` recomputed all 13 cohort–endpoint run sets: Llama and Flan full
  fine-tuning on every endpoint, plus held-out translation for both no-translation cohorts. It
  covered 650 tree analyses (13 × 50 trees).
  * Every tree analysis equals the saved one.
  * Every run-set analysis has the same hash.
  * Each run set took about 40 minutes on 2 CPUs.

  | Cohort | Endpoint | r_DL [95% CI] |
  | --- | --- | --- |
  | Llama full | translation (trained) | −0.20 [−0.26, −0.13] |
  | Llama no-translation | translation (held out) | −0.23 [−0.29, −0.17] |
  | Llama full | HellaSwag | −0.17 [−0.24, −0.10] |
  | Llama full | ARC-Challenge | −0.20 [−0.26, −0.13] |
  | Llama full | MMLU | −0.19 [−0.26, −0.12] |
  | Llama full | TruthfulQA | −0.22 [−0.27, −0.16] |
  | Llama full | Dolly open-ended | −0.08 [−0.13, −0.03] |
  | Flan full | translation (trained) | −0.10 [−0.16, −0.04] |
  | Flan no-translation | translation (held out) | +0.04 [−0.01, +0.09] |
  | Flan full | HellaSwag | +0.09 [+0.05, +0.14] |
  | Flan full | ARC-Challenge | −0.14 [−0.19, −0.09] |
  | Flan full | MMLU | −0.16 [−0.20, −0.11] |
  | Flan full | TruthfulQA | −0.17 [−0.22, −0.12] |

* **App. F, empty excluded.** `eos_excluded` reproduces all five Llama endpoints' saved artifacts
  byte for byte, including the MiniLM re-embedding of blank coordinates; HellaSwag gives −0.26
  [−0.34, −0.19].
* **App. F, depth table.** `empty_rates` reproduces the saved summary exactly: HellaSwag 16.72,
  30.86, 41.13, 40.25% by depth, 38.64% overall.
* **Not rerun.** Behavior inference (GPU response generation) and the MiniLM embedding of the full
  response sets. The replay starts from the saved responses and semantic matrices, and checks them
  against their pinned hashes.
