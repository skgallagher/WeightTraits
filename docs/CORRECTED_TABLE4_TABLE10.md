# Corrected Table 4 and Table 10 contracts

These builders are deliberately fail-closed. They consume reviewed SHA-256
pins, recompute every estimator from per-tree rows, and never edit a workbook
or manuscript.

## Table 4: Flan LayerTrace subsets

`scripts/build_layer_subset_table.py` consumes a version-1 YAML contract:

```yaml
version: 1
metric: cosine
cohort_id: flan_full_finetune
table2_full_ft_cohort_id: flan_full_finetune
truth_manifests:
  confirm_paper_tree_001: {path: path/to/tree_001.manifest.jsonl, sha256: <sha256>}
  # Exactly the canonical 46 IDs: 001..050 except 015, 020, 037, 047.
analysis_inventory: {path: path/to/table4_input_inventory.json, sha256: <sha256>}
table2_receipt: {path: path/to/corrected_table2.json, sha256: <sha256>}
```

The pinned inventory uses schema
`weighttraits.layer_subset_input_inventory.v1`, declares
`valid: true`, `artifact: model`, `representation: full_weight`, and
`metric: cosine`. It pins the strict 50-tree training-completion receipt and
the valid 50-tree/150-row direct-analysis rollup, then maps each exact
canonical tree ID to pinned `summary`, `layers`, `models`, `distance_layers`,
and `truth_manifest` files. Build this inventory directly from the corrected
rollup; do not assemble it by hand:

```bash
PYTHONPATH=src python scripts/build_layer_subset_inventory.py \
  --rollup outputs/analysis_corrected/flan_full_ft_rollup.json \
  --completion-receipt outputs/receipts/flan_full_ft_completion.json \
  --cohort-id flan_full_finetune \
  --base-dir . \
  --out outputs/corrected_paper/table4_input_inventory.json
```

The inventory builder requires the completion receipt to prove 50/50 ready
trees, 641/641 completed nodes, 1282/1282 artifacts, and zero
failures/missing/errors/warnings. It requires exactly one cosine row for each
of all 50 controlled trees before selecting the canonical topology-eligible
46.
The scorer requires the same ordered 282 Flan tensor names on every tree and
resolves these exact subsets:

| Subset | Exact count |
|---|---:|
| `full` | 282 |
| `high_signal` | 48 |
| `sak` | 24 |
| `encoder_h` | 24 |
| `enc_sak` | 12 |
| `low_signal` | 36 |

For each subset it averages the selected tensor distance matrices before
neighbor joining. The JSON receipt records all 276 tree/subset observations,
selected tensor names and hashes, inputs and hashes, and aggregate estimates.
The build aborts unless the `full` row matches the pinned corrected Table 2
full-fine-tuning row for clade recovery, PAER, RF, FN, and every SE.

```bash
PYTHONPATH=src python scripts/build_layer_subset_table.py \
  --config examples/paper/corrected_layer_subset_table4.yaml \
  --out outputs/corrected_paper/table4_layers.json \
  --csv-out outputs/corrected_paper/table4_layers.csv
```

## Table 10: weights versus PhyloLM

`scripts/build_phylolm_table.py` accepts exactly these suite IDs—no historical
aliases or fuzzy resolution:

- `llama32_1b_lora_qkv_r8_legacy_causal_2000`
- `llama32_1b_lora_qkv_r64_legacy_causal_2000`
- `llama32_1b_full_finetune_legacy_causal_2000`

Its version-1 YAML contract pins the same exact 46 truth manifests and, for
each suite, the 50-tree cohort declaration, strict completion receipt, weight
rollup, and PhyloLM receipt:

```yaml
version: 1
metric: cosine
truth_manifests: { ...exact canonical 46 pinned files... }
suites:
  - suite_id: llama32_1b_lora_qkv_r8_legacy_causal_2000
    cohort_contract: {path: path/to/runlist_summary.json, sha256: <sha256>}
    completion_receipt: {path: path/to/strict_completion_audit.json, sha256: <sha256>}
    weight_rollup: {path: path/to/weight_rollup.json, sha256: <sha256>}
    phylolm_receipt: {path: path/to/phylolm_receipt.json, sha256: <sha256>}
  # r64 and full_finetune are both required.
```

`cohort_contract` proves the declared 50-tree/641-run protocol and truth
hashes. `completion_receipt` separately proves that all 641 nodes actually
completed and all 1282 required training artifacts exist. Both are mandatory.

Each PhyloLM input receipt must use schema
`weighttraits.phylolm_controlled_runset.v1`, declare the exact suite and group
name, include the exact 46 per-tree rows, and carry both an exact
tree-ID-to-truth-SHA map and its canonical digest. Each row repeats its truth
SHA. The combiner independently verifies the weight row's readable truth file.
It rejects two 46-row inputs when the IDs differ: equal set size or a 46-row
intersection is not sufficient.

Native producers live in `weighttraits.behavior.phylolm`. They lock upstream
commit `8c70edf062a0adce2a3e6c8c79cd23a645fd0905`, 128 genes, 32 samples per
gene, four new tokens/four allele characters, temperature 1, gene seed 0,
sampling seed 20260803, raw continuations, Nei similarity with a 1e-3 floor,
and WeightTraits neighbor joining. The sampled genome is built once:

```bash
PYTHONPATH=src python scripts/build_phylolm_genome.py \
  --gene-pool path/to/pinned/genes_math.json \
  --out outputs/corrected_phylolm/genome.json
```

After GPU collection writes one pinned population receipt per leaf,
`scripts/analyze_phylolm_tree.py` creates a tree matrix/Newick/score receipt.
`scripts/build_phylolm_runset_receipt.py` then requires the exact canonical 46
tree receipts, a strict 50/50-tree 641/641-node training-completion receipt,
and the frozen-stage manifest before it emits the Table 10 PhyloLM input. It
cannot accept a merely valid run-list declaration as proof that training
completed.

```bash
PYTHONPATH=src python scripts/build_phylolm_table.py \
  --config examples/paper/corrected_phylolm_table10.yaml \
  --out outputs/corrected_paper/table10_phylolm.json \
  --csv-out outputs/corrected_paper/table10_phylolm.csv
```

The Table 10 JSON is the authoritative receipt. It includes all 138 paired
suite/tree observations, exact common IDs, truth and source hashes, the suite
resolver, estimator definitions, and three flat summary rows.
