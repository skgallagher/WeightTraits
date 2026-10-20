# PhyloLM scripts

These scripts compute the behavior-only PhyloLM baseline used in the paper:

* the Table 2 PhyloLM column and the PhyloLM points in Figure 2;
* App. C;
* the PhyloLM panels of App. D.

The method is PhyloLM (Yax, Oudeyer et al., arXiv:2404.04671; https://github.com/Nicolas-Yax/PhyloLM
at commit `8c70edf062a0adce2a3e6c8c79cd23a645fd0905`). The method isn't ours, so it stays as
scripts rather than library code.

## Licensing

The upstream repository is GPL-3.0, while WeightTraits is MIT. No upstream source is included here.

* `phylolm_core.py` is our code: gene sampling, batched allele generation from local
  checkpoints, and population I/O.
* Allele frequencies and Nei similarity come from WeightTraits' own implementation
  (`weighttraits.behavior.phylolm`).
* The upstream gene pool is not redistributed. Extract the `math` genome from
  `data/kl_pop/math.zip` at the pinned commit as a JSON list of its 4,096 genes. Place it at
  `scripts/phylolm/genes_math.json` (git-ignored), or point `PHYLOLM_GENE_POOL` at it. Its sha256
  must be `98d553eb4042cb084a7a17ce3a932ee147629698315e8cc2910dde6b05f84445`; the scripts check it.
  See [docs/PHYLOLM_PROVENANCE.md](../../docs/PHYLOLM_PROVENANCE.md).

## Protocol (locked)

`configs/weighttraits_fresh_phylolm.yaml`:

* G = 128 genes, drawn with `gene_seed` 0;
* N = 32 samples per gene;
* T = 4 new tokens; an allele is the first 4 characters of the generated text;
* temperature 1.0, with `sampling_seed` 20260803 reset before each model;
* the prompt is the raw gene text, with no chat template.

Trees use distance −log(max(Nei similarity, 1e-3)) and neighbor joining. The two family configs
for App. D (`configs/mistral_zoo_phylolm.yaml`, `configs/llama2_zoo_phylolm.yaml`) use the same G,
N, T, and seeds.

## Files

| File | Role |
| --- | --- |
| `phylolm_core.py` | Gene pool and sampling, allele generation, populations, similarity matrix |
| `run_weighttraits_fixed2000_tree.py` | One controlled tree: populations for every leaf, then distances and an NJ tree |
| `submit_weighttraits_fixed2000_phylolm.sh`, `run_fixed2000_pack.sh` | Wright submission for the three Llama cohorts (several trees per GPU) |
| `score_phylolm_table.py` (+ `score_hf_zoo_recovery.py`, `atteson_margin.py`) | Table 2 metrics: clade estimation, full estimation, RF, FN, A, fitted margin |
| `collect_phylolm.py`, `phylolm_distance.py` | One model per job from a registry, then distances and NJ (App. D families) |
| `submit_mistral_phylolm.sh`, `submit_llama2_phylolm.sh` | The original App. D submissions on Ghidorah (partition and paths are Ghidorah's) |

Outputs follow the layout the scorer reads, under `PHYLOLM_ROOT` (default: the repository root):

* `blackbox/data/<group>/populations/<run>/`;
* `results/<group>/<run>/weight/`;
* `outputs/<group>/<run>/tree.newick`.

## Paper column

```bash
# on Wright (normal priority, one L40 per array task)
bash scripts/phylolm/submit_weighttraits_fixed2000_phylolm.sh
python scripts/phylolm/score_phylolm_table.py --groups \
  runs_weighttraits_llama32_1b_lora_qkv_r8_fixed2000_phylolm \
  runs_weighttraits_llama32_1b_lora_qkv_r64_fixed2000_phylolm \
  runs_weighttraits_llama32_1b_full_finetune_fixed2000_phylolm --out <dir>
```

## Verification (20 Oct 2026)

On Wright, the paper's 150 PhyloLM runs (3 cohorts × 50 trees) were recomputed from their saved
populations with these scripts and rescored:

* **Table 2 column:** `table_metrics.csv` is byte-identical to the submitted one.
* **Distances:** they agree to at most 2.2e-16.
* **Trees:** 149 of 150 are identical. The other (full fine-tuning, run 033) is the same unrooted
  tree; NJ placed the root on the other side of the final join.
* **Other per-run metrics:** A and the margin differ by at most 3.3e-16.

One script was also fixed. The label-permutation p-value (`p_FN_vs_label`, not reported in the
paper) used to draw permutations from a Python set. Its order follows per-process string hashing,
so the p-values changed between runs. Permutations now draw from sorted names, so per-run metrics
are identical across hash seeds.
