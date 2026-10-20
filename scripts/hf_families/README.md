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

Results of the full Wright redo are recorded in `docs/PAPER_RESULTS.md`.
