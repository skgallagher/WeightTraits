# Behavioral and PhyloLM Rebuild

WeightTraits now has the behavioral response, inference, embedding, and paired-distance path for
both seq2seq checkpoints and causal full-checkpoint/cumulative-LoRA leaves. The first useful held-out
probe for the existing confirm-paper trees is HellaSwag. Translation is present in every
confirm-paper training tree, so a translation-held-out claim requires a separately generated and
trained assignment set.

## Free-text response, surface, and semantic-distance contract

Behavioral responses use JSONL records keyed by
`(run_id, model_id, probe_id, prompt_id, sample_id)`. The audit rejects duplicate keys, completed
records with empty text, prompt-ID/text conflicts, and incomplete repeat grids. Sentence embeddings
are stored in an aligned NPZ with arrays `embeddings` (model x observation x dimension), `model_ids`,
`observation_ids`, and optional `valid_mask`. Paired cosine distances average only aligned observations
valid for both models and persist the exact per-pair denominator. The response audit also reports
length, uniqueness, and reference ROUGE-L health signals without turning them into arbitrary validity
thresholds.

When the scientific endpoint is the emitted text itself, use `--empty-policy preserve`. Immediate
EOS is then retained as a completed empty string rather than discarded. The
`build-behavior-surface-distances` command classifies every aligned output as empty, label-only,
repetition-loop, natural-language, or fragment and combines those indicators with bounded length,
character-type, and lexical-diversity features. Its paired per-prompt Gower distance measures output
form without pretending that `0`, repeated labels, or EOS have ordinary sentence semantics.
`embed-behavior-responses --natural-language-only` remains a separate semantic endpoint and reports
coverage per leaf; semantic results must always be interpreted alongside that coverage.

Commands:

```bash
PYTHONPATH=src python -m weighttraits.cli audit-behavior-responses \
  --responses responses_dir --expected-samples-per-prompt 3 --out response_audit.json

PYTHONPATH=src python -m weighttraits.cli build-behavior-embedding-distances \
  --embeddings embeddings.npz --out behavior_distance_cube

PYTHONPATH=src python -m weighttraits.cli build-behavior-surface-distances \
  --responses-dir responses_dir --out output_surface_distance_cube

PYTHONPATH=src python -m weighttraits.cli make-behavior-regression-pairs \
  --run-id confirm_paper_tree_001 \
  --weight-cube whitebox_distance_cube \
  --behavior-cube behavior_distance_cube \
  --weight-metric cosine \
  --out regression_pairs.csv
```

Prepare deterministic task-specific prompts with `prepare-hellaswag-probe`. Collect one JSONL per
leaf with `collect-behavior-responses`, then embed the complete response directory with
`embed-behavior-responses`. The wrappers `scripts/slurm/behavior_leaf_array.sbatch` and
`scripts/slurm/behavior_tree_post.sbatch` make that a GPU leaf array followed by a CPU post job. The
post job first audits the complete response directory, then embeds and builds distances; invalid or
incomplete grids stop the dependency chain.

The versioned open-ended gate is `examples/behavior/dolly_open_ended.yaml`. It samples only Dolly
open QA, general QA, brainstorming, and creative-writing instructions, none of which expose answer
choices or labels. Dolly is not one of the 36 WeightTraits training datasets. Run this one-tree gate
with empty preservation and surface analysis before considering a broad panel; compute semantic
distances only for outputs classified as natural language and report the excluded coverage.
Before calling `sbatch`, create `${REPO}/slurm_logs`; Slurm opens the declared log path before the
wrapper itself can create directories. Array indices are zero-based: for a manifest with N leaves,
submit `--array=0-$((N-1))%20`.

The first probe declaration is `examples/behavior/hellaswag_paired.yaml`. Wright jobs `154614` and
`154621` passed the complete eight-leaf tree-001 path with two prompts: 16/16 nonempty aligned
responses, a 384-dimensional embedding tensor, and a finite non-degenerate 8x8 semantic distance
cube. The tiny health audit found 0.8125 unique-response fraction and mean reference ROUGE-L 0.065;
raw outputs were content-bearing but often retained serialized QA-style wrappers learned during
fine-tuning. Preserve those raw outputs and report the health caveat rather than silently cleaning
them. The next gate is a 100-prompt diagnostic on this one tree before a 50-tree launch.

That gate failed scientifically on Flan despite complete artifacts. Wright array `154624` generated
800/800 valid responses over 100 prompts and 8 leaves, but two leaves produced only 2 unique responses
each, a third produced 11, and the overall mean reference ROUGE-L was 0.037. This is task-format
collapse, so do not launch HellaSwag broadly on the Flan confirm-paper trees. Retain the artifacts as
a documented negative diagnostic and repeat the probe on the first completed causal Llama trees.
The one-leaf causal preflight is encouraging: Wright job `154633` rematerialized the archived r8
`n0 -> n2 -> n7` adapter chain and produced 100/100 unique outputs, dominant-response fraction 0.01,
and mean reference ROUGE-L 0.122. Repeat this across a complete production tree before broad launch.
The archived complete-tree follow-up also passed: jobs `154635` and `154636` generated 700/700 valid
responses across 7 leaves, with 100 unique outputs and 0.01 dominant-response fraction within every
leaf. Mean reference ROUGE-L was 0.124, and the aligned 100-observation semantic cube was finite and
non-degenerate (maximum pair distance 0.1031). Cross-model output agreement is high, as expected for
the deliberately low-divergence 10-step benchmark. The remaining gate is repetition on a completed
production r8/r64 tree.

The confirmation protocol uses three stochastic completions per prompt. Defaults are sampling on,
temperature 1.0, top-p 1.0, and base seed 42; draw IDs 0, 1, and 2 use seeds 42, 43, and 44 for every
model. Each output is embedded separately. Paired cosine distance is averaged over the complete
prompt-by-draw grid, which gives every prompt equal weight when the audited grid is rectangular.
The regression export includes both `behavior_distance` and `behavior_similarity = 1 - distance`;
the paper's negative coefficients use similarity. One greedy completion remains a sensitivity
analysis, not a repeated-draw estimator. The archived jobs above predate this decision and are
health diagnostics rather than final paper estimates.
The regression-pair command aligns cubes by model ID rather than matrix position, records excluded
models and layer selections in an audit, and emits the stable columns expected by the independent R
cross-check.

## Faithful PhyloLM path

The PhyloLM implementation pins upstream commit
`8c70edf062a0adce2a3e6c8c79cd23a645fd0905` and preserves the paper contract: 128 shared math genes,
32 raw completions per gene, four new tokens, four-character alleles, temperature 1, Nei similarity,
`-log(max(similarity, 1e-3))`, and neighbor joining. It intentionally applies no chat template.

For adapter-only Llama trees, each leaf is rematerialized from the pinned base by merging the complete
root-to-leaf adapter chain in order. Only the compact allele population is retained; the rematerialized
model exists in GPU memory for one leaf collection job and is then released.

Prepare one shared genome:

```bash
PYTHONPATH=src python -m weighttraits.cli make-phylolm-genome \
  --gene-pool ../ELLMTrees/codex/phylolm/genes_math.json \
  --n-genes 128 --seed 0 --out outputs/phylolm/genome_math_g128_seed0.json
```

For a completed tree, derive a leaf adapter-chain manifest with the existing
`make-distance-input-manifest --artifact adapter_chain` command. Submit one GPU array task per leaf
using `scripts/slurm/phylolm_leaf_array.sbatch`, then run
`scripts/slurm/phylolm_tree_post.sbatch` after the array succeeds. The post job writes a native
WeightTraits distance cube and `phylolm_nj.newick`, ready for the existing recovery, additivity, and
Atteson scoring commands. These leaf-array indices are also zero-based.

## Remaining gates

1. Keep the released math gene pool external and checksum-pinned per `docs/PHYLOLM_PROVENANCE.md`;
   do not vendor the GPLv3 upstream artifact into the MIT repository without review.
2. Run the full PhyloLM contract on the first completed r8 and r64 production trees. Wright job
   `154601` already passed a two-gene/two-sample cumulative-LoRA smoke with zero empty alleles.
3. Compare the reconstructed trees to the same truth manifests and whitebox scorer.
4. Run the three-draw, 100-prompt protocol on the first completed production Llama tree, inspect
   per-model output and within-prompt draw diversity, then cross-check behavioral regression
   coefficients with `scripts/regression_r_check.R`.

After a one-tree output-surface gate passes, `scripts/slurm/behavior_condition_tree.sbatch`
provides the full-condition path. Its Slurm array index is the one-based production tree number.
It rematerializes each leaf once for a prompt JSONL that may contain multiple probes, preserves
empty generations as observed behavior, then emits a separate surface cube and regression pair
table for every comma-separated `PROBE_IDS` entry. This avoids reloading every leaf once per probe.
The resulting per-tree pair CSVs are the direct inputs to
`scripts/behavior_meta_analysis.R`; do not report pooled inference until at least three tree jobs
have passed their audits.
