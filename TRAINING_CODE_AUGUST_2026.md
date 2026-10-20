# Training code of record, August 2026

This commit (tag `training-code-august-20260817`) archives the code and run lists that trained
the paper's controlled cohorts. It is not a development branch.

* **Code**: `src/weighttraits` is the 50-file package from the sealed results checkout
  `/home/export/sgallagh/WeightTraits-all-results-20260817-clean` on Wright. Those files pass that
  checkout's freeze checksums (`.stage_provenance/STAGE_SHA256SUMS`, 17 Aug 2026). The package
  receipt (`weighttraits.training.executor._weighttraits_source_receipt`,
  `sha256-relative-path-and-content-v1`) is
  `64494e920bd7d18e459738d3df7d210cc5f9055346505bbbceed69c374e3df01`.
* **Run lists and configs**: `examples/training/confirm_paper_numbers/` holds the August versions
  of every run list (1,050 files, byte-identical to the sealed checkout), the run-list summaries,
  the Llama and Flan base configs, and the rebuild notes.

Cohorts trained from this checkout:

| Cohort | Trained |
| --- | --- |
| Llama-3.2-1B full fine-tuning | 10–13 Aug |
| Flan-T5-base full fine-tuning | 17–20 Aug |
| Flan-T5 LoRA variants, Llama LoRA r8/r64, Flan no-translation | 20–30 Aug |

The Llama cohort ran in an earlier directory. Its `training/executor.py` was last edited at 17:47
UTC on 10 Aug, after the first wave of Llama nodes had started, so those nodes may have run an
earlier executor that no longer exists.

## Relation to `training-code-frozen-20260810`

The base of this commit is the later tag `training-code-frozen-20260810` (d22bf52). That tag's
60-file package (receipt `1445e45d…`) adds provenance guards: runtime, source, and cache pins. It
also regenerated the run lists with those pins inside `trainer`. Its argument builder, unchanged
from August, forwards unknown trainer keys to Hugging Face, which rejects the pins. As a result,
neither the tag nor this archive can train the tag's run lists.

The August run lists here have no pins. All 6,410 rows of the ten `*_2000` conditions equal the
tag's rows once the three pin keys and the config hash are removed. The training code paths
(arguments, tokenization, collation, trainer) are identical between the two packages; only
provenance code differs. Current `main` strips the pins before calling Hugging Face.

## Reproducing a node

From this checkout, with the seed-42 data cache at
`data/confirm_paper_numbers/legacy_causal_seed42_cache` and the pinned base-model revisions in the
Hugging Face cache, run:

```bash
PYTHONPATH=src python -m weighttraits.cli run-training-row \
  --run-list examples/training/confirm_paper_numbers/<condition>_training_runlists/run_lists/<tree>.runs.jsonl \
  --node-id n0
```

Use the numerical stack recorded in the ledgers:

* Python 3.11.15
* torch 2.5.1+cu121
* transformers 5.8.0
* datasets 4.8.5
* accelerate 1.13.0
* tokenizers 0.22.2

Results of retraining on 20 Oct 2026 (Wright jobs 237440 and 237441, one L40 each):

* **Flan-T5, tree 001 `n0`**: all 282 tensors are bit-identical to the paper's checkpoint.
* **Llama, tree 025 `n0`**: training on GPU is not bit-deterministic.
  * 24 of 146 tensors match exactly.
  * The rest differ by at most 1.04e-4 in cosine distance (mean 1.7e-5 over trained tensors).
  * That is about a quarter of the smallest leaf-to-leaf distance in that tree.
