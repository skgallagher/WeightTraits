# Paper results: code, commands, and inputs

This is a map from each result in the ICLR 2027 submission (`iclr_draft_v6.tex`, ELLMTrees-paper
`main` efec10b) to the code that computes it, how to run it, and what it reads. Wright paths refer
to the cluster that holds the trained artifacts. Two read-only roots hold them:

* `/home/export/sgallagh/WeightTraits-all-results-20260817-clean` ("sealed root"): the trained
  cohorts, run lists, and ledgers;
* `/home/export/sgallagh/WeightTraits-downstream-20260831-native` ("downstream root"): direct-weight
  run sets, behavior outputs, and receipts.

Local mirrors carry a `receipt.json` with each Wright path and sha256. Execution details are in
[FINAL_RESULTS_REPRODUCTION_2026-09-14.md](FINAL_RESULTS_REPRODUCTION_2026-09-14.md).

Population: 50 trees and 641 trained nodes per cohort. Topology statistics use the 46 trees that
have a clade to estimate (all but 015, 020, 037, 047). Ordering statistics use the 26 trees whose
root has more than one child.

## Main text

| Result | Code | Command | Inputs |
| --- | --- | --- | --- |
| Table 2 weight clade, full estimation, FN; App. A RF table (BHV) | `weighttraits.paper.bhv_table`, `weighttraits.phylo.bhv` | `wt bhv-table --suite ID=ROOT ... --truth-root examples/training/confirm_paper_numbers/assigned_manifests --out-dir DIR` (raw scale, seed 0, 300 epochs) | per-layer cosine cubes, `direct/<condition>/analysis` in the downstream root |
| Table 2 rank-biserial, branch-length r | `weighttraits.paper.direct_weight_table` | `scripts/build_merged_weight_paper_stats.py --config examples/paper/corrected_direct_weight_table2.yaml` (on Wright) | direct-weight run sets, completion receipts, cohort contracts; see [DIRECT_WEIGHT_TABLE2.md](DIRECT_WEIGHT_TABLE2.md) |
| Table 2 PhyloLM column (Llama) | `scripts/phylolm/` on `weighttraits.behavior.phylolm` | see [PhyloLM](#phylolm) | populations generated on the fixed-2,000-step checkpoints |
| Table 2 chance level (caption) | `weighttraits.phylo.chance` | `wt chance-baseline --manifest-dir examples/training/confirm_paper_numbers/assigned_manifests` | truth manifests |
| Figure 2 points | `scripts/build_merged_weight_figure2.py` | `--weighttraits-root MIRROR --phylolm-metrics per_run_metrics.csv --phylolm-cohort fixed2000 --out ... --csv-out ... --records-out ...` | run-set rollups, per-layer cubes, PhyloLM per-run metrics |
| Figure 2 fits, odds ratios, panels | `R/figure2.R`; Python port `weighttraits.stats.fractional_logit` | `Rscript R/figure2.R SUMMARY_CSV RECORDS_CSV BOOTSTRAP_OUT FIGURE_STEM` (2,000 paired-tree replicates, seed 20260805) | Figure 2 summary and records CSVs |
| Figure 2 tree-level odds ratios | `R/odds_ratio_tree_level.R` | `Rscript R/odds_ratio_tree_level.R RECORDS_CSV OUT_CSV` | Figure 2 records CSV |
| Table 3 (behavior, held-out tasks) | `weighttraits.behavior.allnode` (`semantic_analysis`, `replay`) | `python -m weighttraits.behavior.allnode.replay --config <semantic job config>` (on Wright); see [behavior](#behavior) | all-node responses, semantic matrices, direct cosine |

Submitted values:

* **Table 2.** Values are rank-biserial, r_branch, weight clade %, weight full estimation %,
  weight FN; App. A RF in brackets.
  * Flan LoRA key only: .52, −.53, 92, 85, .17 [2.39].
  * Flan LoRA q/k/v and q/k/v/o: .56, −.62, 99, 98, .02 [2.09].
  * Flan LoRA q/k/v/o + FFN (summed adapter updates): 1.00, −.91, 96, 89, .11 [2.26].
  * Flan full fine-tuning: .69, −.71, 100, 100, .00 [2.04].
  * Llama LoRA r8: .63, −.64, 100, 100, .00 [2.04].
  * Llama LoRA r64: .65, −.67, 100, 100, .00 [2.04].
  * Llama full fine-tuning: .64, −.66, 100, 100, .00 [1.83].
* **Table 2 (PhyloLM, fixed-2,000-step rerun of 24 Sep).** Clade % (SE): r8 35.3 (5.0), r64
  41.3 (4.6), full fine-tuning 56.3 (5.2). App. A PhyloLM RF: 5.52 / 5.09 / 4.26; FN: 1.74 /
  1.52 / 1.11.
* **Chance.** 8.7% (SE 1.1) clade estimation and 3.2% full estimation.
* **Figure 2.**
  * Odds ratio per +0.10 in A: 4.54 [3.87, 5.79].
  * Odds ratio per tenfold m̃: 15.19 [8.48, 68.00].
  * Both over 231 estimators and 46 paired trees.

Recorded outputs (`reports/paper/fixed2000_phylolm/`, sha256):

| File | sha256 |
| --- | --- |
| `merged_weight_figure2.json` | `99c60e5c…` |
| `merged_weight_figure2.csv` | `a9b887ac…` |
| `merged_weight_figure2_records.csv` | `c8114c1b…` |
| `merged_weight_figure2_bootstrap.csv` | `7e0d6119…` |
| `or_tree_level.csv` | `9075c9d3…` |

Table 2 ordering output: `table2.json` `55a9c2cb…`.

## Appendices

| Result | Code | Command / notes |
| --- | --- | --- |
| App. A RF table | `wt bhv-table` | `rf_mean`/`rf_se` columns of the same run |
| App. A training configuration table | `examples/training/confirm_paper_numbers/*_2000.yaml` | configs as trained: tag `training-code-august-20260817` |
| App. A q/k/v vs q/k/v/o trees | `weighttraits.paper.condition_compare` | `wt compare-conditions --left ROOT --right ROOT --pattern '{tree_id}/...newick'`; 8/46 BHV and 5/50 mean-matrix NJ trees differ, every difference resolves a polytomy of the truth |
| App. B layer subsets | `scripts/summarize_final_layertrace.py`, `scripts/summarize_final_llama_layertrace.py` | all six Flan full-FT subsets estimate every clade; smallest is 12 encoder key matrices (2.86% of parameters) |
| App. B LayerTrace grid | `R/figure3_layertrace.R` | `Rscript R/figure3_layertrace.R FLAN_PROFILE_CSV LLAMA_PROFILE_CSV FIGURE_STEM` |
| App. C PhyloLM details | see [PhyloLM](#phylolm) | locked configuration: G 128, N 32, T 4, sampling seed 20260803, `genes_math.json` |
| App. D Hugging Face families: scores | `weighttraits.paper.hf_families` (`score_family`, `per_layer_clade_support`) | Mistral 3/3 clades, A 0.86 (weights) vs 0.68 (PhyloLM); Llama-2 1/1, A 0.82 vs 0.64; chat clade in 275/289 per-layer trees with positive length (95.2%) |
| App. D tree figures | `weighttraits.paper.hf_family_figures`, `R/hf_family_trees.R` | `Rscript R/hf_family_trees.R DATA_DIR FIGURE_DIR`; cubes live on Ghidorah (`/home/sgallagh/ELLMTrees/results/hf_zoo/`) |
| App. E heterogeneity table | `weighttraits.behavior.allnode.semantic_analysis` | the same run sets as Table 3; see [behavior](#behavior) |
| App. E Flan output audit | `weighttraits.behavior.output_audit.audit_cohort(inference_root)` | unrelated training labels: ARC-C 20.4%, MMLU 22.9%, HellaSwag 21.3%, TruthfulQA 30.3%, translation 14.7% trained / 22.6% held out |
| App. F empty-response sensitivity and depth | `weighttraits.behavior.allnode.eos_excluded` (empty excluded) and `.empty_rates` (rates by depth) | see [BEHAVIOR_ALLNODE.md](BEHAVIOR_ALLNODE.md) |

## PhyloLM

The Table 2 column comes from a 24 Sep rerun on the fixed-2,000-step Llama checkpoints. Before that
rerun, the column used an earlier early-stopped training of the same trees.

* Code: [`scripts/phylolm/`](../scripts/phylolm/README.md). `run_weighttraits_fixed2000_tree.py`
  generates and `score_phylolm_table.py` scores (originally in ELLMTrees `codex/phylolm`).
  Rescoring the 150 saved runs reproduces the column byte for byte (20 Oct 2026).
* Output: `results/aggregate/weighttraits_fixed2000_phylolm/`, mirrored with a receipt under
  `sources/wright/phylolm_fixed2000`.
* Agreement: WeightTraits' `weighttraits.behavior.phylolm.nei_distance_matrix` with
  `neighbor_joining_newick` gives the same distances (to 1e-16) and the same trees (50/50).
* Licensing: see [PHYLOLM_PROVENANCE.md](PHYLOLM_PROVENANCE.md) for the gene pool.

## Behavior

Table 3, the App. E heterogeneity table, and App. F come from the all-node behavior study.

* **Origin.** It first ran on Wright as the `weighttraits_allnode` deployment
  (`deployments/allnode_v1`, plus continuations through `v1c5`, under the downstream root).
* **Now.** That package is `weighttraits.behavior.allnode`, byte-identical apart from the package
  name; see [BEHAVIOR_ALLNODE.md](BEHAVIOR_ALLNODE.md).
* **Verified on Wright (20 Oct 2026).**
  * Replaying all 13 run sets from the saved responses and matrices reproduces every tree analysis
    and run-set hash.
  * Both App. F post-processors reproduce their saved outputs byte for byte.
* **Not rerun.** GPU response generation.

## Training

The cohorts were trained by the code at tag `training-code-august-20260817`. See
[TRAINER_DESIGN.md](TRAINER_DESIGN.md#training-code-of-record) and the tag's
`TRAINING_CODE_AUGUST_2026.md`. The training ledgers (706 MB) stay in the sealed root.

Retraining checks on 20 Oct 2026:

* **Flan-T5, tree 001 `n0`**: all 282 tensors are bit-identical to the paper's checkpoint.
* **Llama, tree 025 `n0`**: GPU training is nondeterministic. The retrained weights differ by up to
  1.04e-4 in cosine distance per tensor, about a quarter of the smallest leaf-to-leaf distance in
  that tree.

## Known differences from the submitted text

These are found after submission and kept for the rebuttal. The submitted values above are not
changed.

* BHV per-layer trees were read from five-decimal Newick, so NJ edges shorter than 5e-6 became
  zero. With `wt bhv-table --branch-length-format %.17g`, two rows change:
  * Flan LoRA key only: clade 92 → 91%, full estimation 85 → 83%, RF 2.39 → 2.43, FN 0.17 → 0.20;
  * Llama full fine-tuning: App. A RF 1.83 → 2.04.
* `phylo.bhv.geodesic` now visits shared splits in a fixed order. Python's string hashing
  previously made branch lengths vary between runs. Table 2 is unchanged.
