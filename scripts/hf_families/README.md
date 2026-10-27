# Documented Hugging Face families (App. D)

These scripts redo App. D on Wright from public checkpoints. The original run was on Ghidorah,
with checkpoints downloaded in September and no revisions recorded. App. D covers two families,
listed in `tab:hf_checkpoints`:

* a ten-checkpoint Mistral-7B family;
* a ten-checkpoint Llama-2-7B family.

Their documented trees and model maps are in `examples/hf_families/<family>/`.

## Steps

| Step | Script | Where |
| --- | --- | --- |
| 1. Download every checkpoint at a pinned revision and verify each file against the Hub's sha256 | `download_families.py ROOT` | CPU, local scratch (about 334 GB) |
| 2. Convert sharded `.bin` checkpoints to safetensors, tensor for tensor, and verify | `convert_bin_to_safetensors.py DIR...` | CPU |
| 3. Build the per-tensor distance cube over the shared tensors | `build_family_cube.py --manifest ROOT/download_manifest.json --family F --out DIR` | CPU |
| 4. Build BHV and NJ trees and score them against the documented tree | `family_trees.py --cube-dir DIR --truth examples/hf_families/F/tree.newick [--phylolm DIR]` | CPU |
| 5. Run PhyloLM, streaming one model at a time through one GPU | `phylolm_stream.py --family F --config ../phylolm/configs/F_zoo_phylolm_wright.yaml ...` | GPU |
| 6. Repeat PhyloLM over sampling seeds from the local checkpoints, and score each seed | `phylolm_seeds.py --family F --models-dir ROOT/models --seeds 1,2,... ...` | GPU |

Revisions are pinned in `download_families.FAMILIES`, as commit prefixes; the first 40 hex
characters are recorded in `download_manifest.json`. The two `meta-llama` repos are gated and
need a Hugging Face token with access.

## Settings that reproduce the paper

* **Tensors.** All tensors shared by the family with equal shapes, up to a size limit set per
  family in the original run (`PAPER_MAX_ELEMENTS`):
  * Llama-2: 50,000,000 elements, which drops the token embedding and output head (289 layers);
  * Mistral: 60,000,000 elements, which keeps the 58.7M-element MLP matrices (289 layers).
    OpenChat's larger vocabulary makes the embedding shapes differ, so those are dropped anyway.
* **Cube storage.** Each per-tensor cosine matrix is stored in float32 before averaging, as the
  original streaming script did. `--float64` keeps full precision.
* **Weight tree.** The BHV Fréchet mean of per-layer NJ trees (five-decimal Newick, negative
  pendant lengths clipped), with 25 epochs, seed 0, and tolerance 1e-9.
* **Scores.** `weighttraits.paper.hf_families` gives the documented clades estimated and the
  all-quartet A. `per_layer_clade_support` gives the 95.2% figure.
* **PhyloLM.** The locked protocol: G 128, N 32, T 4, `gene_seed` 0. The four fp32 Llama-2
  checkpoints are loaded as float16, as originally done. The original App. D PhyloLM run set no
  sampling seed. The Wright redo uses seed 20260803 (`*_zoo_phylolm_wright.yaml`), so its
  populations are reproducible but not identical to the originals.

## Checks

`family_trees.py`, run on the original Ghidorah Llama-2 cube, reproduces App. D:

* 1 of 1 documented clades for both methods;
* A = 0.817 for the weights and 0.643 for PhyloLM;
* the chat clade in 275 of 289 per-layer trees (95.2%);
* a BHV tree with the same splits as the original.

Full redo on Wright (20 Oct 2026), from fresh downloads at the pinned revisions:

| Family | Layers kept | Agreement with the original cube | Weight clades | A (paper) | Weight BHV tree vs paper |
| --- | --- | --- | --- | --- | --- |
| Llama-2-7B | 289 of 291 shared | per layer within 4.7e-10 (float32 storage); mean matrix within 1.6e-12 | 1/1 | 0.817 (0.82) | identical splits |
| Mistral-7B | 289 of 291 shared | mean matrix within 4.4e-11 | 3/3 | 0.858 (0.86) | identical splits |

The Llama-2 chat clade appears in 275 of 289 per-layer trees (95.2%), as in the paper. The new
Mistral cube orders its layers differently from the original; scored in either order, it gives the
same tree and scores.

PhyloLM redo (Slurm job 237508, one L40, 21 Oct 2026). `phylolm_stream.py` was run with the
`*_zoo_phylolm_wright.yaml` configs (sampling seed 20260803), streaming each model from a fresh
pinned download. Every allele is nonempty.

| Family | PhyloLM clades | PhyloLM A, redo | PhyloLM A, paper | Weight A |
| --- | --- | --- | --- | --- |
| Llama-2-7B | 1/1 | 0.651 | 0.643 (0.64) | 0.817 |
| Mistral-7B | 3/3 | 0.806 | 0.681 (0.68) | 0.858 |

* **Clades.** PhyloLM again estimates every documented clade in both families, as the paper says.
* **Scorer check.** The same scorer gives 0.643 and 0.681 on the original PhyloLM matrices
  (`ELLMTrees/results/hf_zoo/{c_llama2_7b_chat_small_nodilip,d_mistral_7b}/phylolm`). So the
  difference is in the sampled populations, not in the scoring.
* **Mistral.** The original run set no sampling seed. Its redo Nei similarities correlate 0.97 with
  the original's. The largest pair difference is 0.06, and the redo averages 0.016 higher. Yet A
  rises from 0.68 to 0.81.

Seed repeats (`phylolm_seeds.py`, Slurm jobs 250434–250435 on n03, 26 Oct 2026, commit
`6929dbe`). These are 11 seeds per family: 20260803 and 1–10, all from the downloaded checkpoints.
Seed 20260803 reproduced job 237508's allele populations exactly for all 20 models.

| Family | Seeds with every clade | PhyloLM A, mean (sd) | Range | Paper | Weight A |
| --- | --- | --- | --- | --- | --- |
| Llama-2-7B | 11/11 | 0.655 (0.015) | 0.625–0.683 | 0.643 | 0.817 |
| Mistral-7B | 11/11 | 0.827 (0.018) | 0.806–0.855 | 0.681 | 0.858 |

* **Llama-2.** The paper's value lies inside the seed range. Weights are more tree-like than
  PhyloLM by about 0.16.
* **Mistral.** The paper's 0.68 lies about eight standard deviations below the seed mean, so it
  does not reproduce. Its population files predate the dtype field. They come from the first
  version of the collection script (2 July). Its sampling code is the same as now apart from the
  optional seed, so the cause is probably the July environment or checkpoint snapshot, which were
  not recorded. Under the documented protocol, weights are more tree-like than PhyloLM in all 11
  seeds, but only by 0.00–0.05, not the paper's 0.18.
* **Clades.** Every seed estimates every documented clade in both families. The paper's clade
  claims stand.
