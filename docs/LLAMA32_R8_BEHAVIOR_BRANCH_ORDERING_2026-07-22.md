# Llama 3.2 1B q/k/v LoRA Rank-8 Behavioral Branch Ordering

Date: 2026-07-22

## Question and estimands

The primary behavioral analysis asks whether continuous cumulative-adapter weight distance predicts
behavioral similarity within each tree. Its paper-facing estimate is the DerSimonian--Laird
random-effects mean of within-tree correlations. All ten Llama q/k/v rank-8 estimates were null.

The branch-ordering analysis asks a coarser question: are behavioral distances between leaves on
different root branches larger than distances between leaves on the same root branch? A pair is
`same_branch` when the two leaf paths share their first ancestor below the manifest root. The
directional alternative is `cross_branch > same_branch`.

Two inferential summaries are required:

1. **Legacy pooled Mann--Whitney.** Pool all eligible leaf pairs and run the old paper's one-sided
   Mann--Whitney U test. Report its rank-biserial effect size, oriented positive when cross-branch
   distances are larger. This is retained for continuity with the old paper.
2. **Tree-level robustness.** Compute the rank-biserial effect independently within every tree that
   contains both branch classes, then report its mean and standard error, a one-sided Wilcoxon
   signed-rank test against zero, and the Fisher-z mean of the within-tree point-biserial
   correlations. This treats trees, rather than dependent leaf pairs, as the replication unit.

Twenty-six of the 50 production trees contain both same- and cross-root-branch pairs. The other 24
have a one-child root and are valid recovery/behavior trees but cannot contribute a within-tree
branch-ordering effect. Across the full panel there are 1,228 leaf pairs: 864 same-branch and 364
cross-branch pairs.

## Results

| Probe | Endpoint | Pooled MW p | Pooled rank-biserial | Tree mean rank-biserial (SE) | Tree Wilcoxon p | Within-tree Fisher r |
|---|---|---:|---:|---:|---:|---:|
| ARC-Challenge | Surface | .646 | -.013 | -.031 (.067) | .913 | .032 |
| ARC-Challenge | Semantic | .809 | -.032 | -.058 (.065) | .951 | .027 |
| Dolly | Surface | .019 | .075 | .029 (.068) | .742 | -.091 |
| Dolly | Semantic | .337 | .015 | -.003 (.058) | .767 | .027 |
| HellaSwag | Surface | 8.91e-4 | .113 | .083 (.067) | .251 | -.082 |
| HellaSwag | Semantic | .076 | .052 | .055 (.062) | .363 | -.071 |
| MMLU | Surface | .729 | -.022 | -.068 (.064) | .961 | .050 |
| MMLU | Semantic | .817 | -.033 | -.041 (.066) | .957 | .046 |
| TruthfulQA | Surface | .064 | .055 | .036 (.068) | .615 | -.030 |
| TruthfulQA | Semantic | .470 | .003 | .039 (.054) | .446 | -.002 |

The pooled legacy test detects HellaSwag surface separation and nominal Dolly surface separation.
HellaSwag remains below .05 after a simple Bonferroni correction over the ten endpoint/probe tests,
but multiplicity correction does not fix the shared-leaf dependence in the pooled test. No endpoint
passes the tree-level robustness test, and none of the semantic endpoints passes even the pooled
test.

For context, the old Llama-3.1-8B q/k/v translation-semantic analysis reported pooled
Mann--Whitney p=1.57e-3, rank-biserial=.074, and within-tree Fisher r=-.162 over 2,205 pairs. The new
HellaSwag and Dolly surface effects reach a similar pooled rank-biserial magnitude, but the new
semantic and tree-level evidence is weaker.

## Interpretation for the appendix

The new q/k/v rank-8 condition preserves lineage extremely strongly in parameter space: for
correlation/cosine distance, mean within-tree rank-biserial is .996 (SE .003) and the Fisher-z
within-tree branch correlation is -.924 across the 26 ordering-valid trees. In contrast,
behavioral branch separation is small, endpoint-specific, and not consistent across trees.

The defensible claim is therefore not that behavior contains no trace whatsoever. The output-form
endpoint contains a weak pooled branch signal for HellaSwag and Dolly, while the semantic endpoint
does not. Because those pooled findings do not survive tree-level inference, they should be framed
as exploratory coarse branch separation rather than confirmation of inherited behavioral
geometry. This complements the null continuous weight--behavior correlations: a low-rank lineage
can retain highly recoverable parameter geometry while transmitting, at most, a fragile
surface-form behavioral trace.

## Provenance

- Model condition: Llama 3.2 1B, cumulative q/k/v LoRA, rank 8.
- Behavioral production jobs: Wright arrays `156351` and `156352`.
- Behavioral pair roots:
  `outputs/behavior/llama32_r8_mc_four_s3_all50` and
  `outputs/behavior/llama32_r8_dolly_open_s3_n30_all50` in the Wright behavioral worktree.
- Truth manifests:
  `examples/training/confirm_paper_numbers/assigned_manifests/confirm_paper_tree_*.manifest.jsonl`
  in the Wright Llama worktree.
- Compact machine-readable values:
  `results/behavior/llama32_r8_branch_ordering_20260722/paper_summary.csv`.
- The pooled calculation matches the legacy `ELLMTrees/scripts/analyze_branch_structure.py`
  direction and rank-biserial definition.

## Required fields for future behavioral tables

Every future model/probe/endpoint row should include or link to all of the following:

- model family, tuning regime, targeted modules, rank, and weight representation;
- probe and endpoint (`surface` or `semantic`);
- total trees, DL-usable trees, and branch-ordering-valid trees;
- total, same-branch, and cross-branch pair counts;
- semantic valid observations and coverage, plus preserved empty-output count;
- DerSimonian--Laird correlation, 95% CI, p-value, and heterogeneity (`I2`/`tau2`);
- pooled one-sided Mann--Whitney p and pooled rank-biserial for legacy continuity;
- mean per-tree rank-biserial with SE and a tree-level Wilcoxon or sign-test p-value;
- within-tree Fisher-z branch correlation; and
- a dependency note stating that pooled leaf-pair p-values are descriptive because pairs share
  leaves, while the tree-level result is the inferential robustness check.

Do not combine surface and semantic endpoints, silently omit unary-root trees, infer a global pair
count such as 36, or present a pooled Mann--Whitney p-value without its effect size and tree-level
check.
