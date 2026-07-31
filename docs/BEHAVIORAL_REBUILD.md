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

Collection artifacts carry a SHA-256 provenance fingerprint over the prompt records, model source,
base revision, seeds, sampling controls, token limits, and batch size. An existing response JSONL is
skipped only when both that fingerprint and the complete prompt-by-draw grid match; otherwise resume
fails and requires an explicit `--overwrite`. The five held-out datasets and the sentence encoder are
pinned to the exact Hugging Face commits cached for the production Flan panel.

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
The regression-pair command aligns cubes by model ID rather than matrix position and requires exact
model-set equality. `--allow-model-subset` is an audit-only escape hatch. Pair rows carry
`behavior_observations`, the exact shared-output denominator, alongside the stable columns expected
by the independent R cross-check.

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
It rematerializes each leaf once for a prompt JSONL that may contain multiple probes and preserves
empty generations as observed behavior. For every comma- or colon-separated `PROBE_IDS` entry it
emits two parallel endpoints: a surface cube/pair table for output form and a paired
sentence-embedding cube/pair table for semantic similarity. Semantic metadata records coverage;
empty strings remain part of the surface endpoint but, lacking linguistic content, are excluded
from the semantic endpoint. This avoids reloading every leaf once per probe. The colon form is
convenient inside Slurm's comma-delimited `--export` argument.
The resulting per-tree pair CSVs are the direct inputs to
`scripts/behavior_meta_analysis.R`; do not report pooled inference until at least three tree jobs
have passed their audits.

The wrapper also supports the paper-matched Flan-T5 LoRA condition without a parallel analysis
implementation. Set `MODEL_TASK=seq2seq`, `MANIFEST_ARTIFACT=merged`, and point `LEDGER_DIR` and
`ARTIFACT_REPO` at the completed Flan q/v rank-8 run. The wrapper then loads the retained merged
leaf checkpoints and uses `full_weight` for the white-box cube. Causal cumulative-LoRA runs retain
the defaults `MODEL_TASK=causal_lm`, `MANIFEST_ARTIFACT=adapter_chain`, and
`WEIGHT_REPRESENTATION=lora_cumulative_delta`.

## Final Flan q/v rank-8 behavioral panel

The 50-tree Flan panel completed in Wright jobs `156398`, `156399`, `156411`, and `156412`.
All trees passed complete-grid, finite-cube, aligned-leaf, and dynamic `choose(L, 2)` pair-count
audits. The multiple-choice panel retained 4,154 empty outputs among 438,000 records; semantic
coverage ranged from 97.97% (TruthfulQA) to 99.60% (MMLU). Dolly retained 818 empty outputs among
32,850 records and had 97.51% semantic coverage. These are observations, not silently discarded
generation failures.

The DerSimonian--Laird random-effects estimates relate weight distance to behavioral similarity,
so the expected phylogenetic signal is negative. Across 50/50 usable trees, surface/semantic
estimates were ARC-Challenge -0.230/-0.249, Dolly -0.156/-0.219, MMLU -0.230/-0.252, and
TruthfulQA -0.128/-0.212. HellaSwag surface form was null (-0.008), while semantic similarity was
weakly negative (-0.112). Compact audit and R outputs are versioned under
`results/behavior/flan_qv_r8_final_20260720/`.

### Flan output-collapse appendix diagnostic

`scripts/flan_degeneracy_diagnostics.py` joins each retained leaf to its terminal training task,
classifies response form, and reports exact-response concentration plus nonempty semantic-embedding
coverage. The audited appendix run covers 50 trees, 365 unique leaves, five probes, and 1,825
leaf-probe rows with no alignment or completeness issues. Compact tables and PDF/SVG figures are
under `results/behavior/flan_qv_r8_degeneracy_appendix/`.

Across probes, terminal classification leaves produced 78.9% label-only and 10.8%
natural-language outputs. Natural-language fractions were 41.2% for QA, 48.8% for summarization,
and 31.4% for translation terminal leaves. Nonempty semantic coverage remained 96.8-99.7% by
terminal family, showing that embedding coverage alone is not evidence of healthy natural-language
generation. Terminal family is descriptive rather than causal because every leaf inherits its
entire root-to-leaf task history.

## Final Llama 3.2 1B q/k/v rank-8 behavioral panel

The 50-tree Llama panel completed in Wright jobs `156351` and `156352` after the rank-8 recovery
dependency succeeded. All trees passed complete three-draw grids, finite and aligned cumulative
LoRA white-box/surface/semantic cubes, and dynamic `choose(L, 2)` pair-count audits. The tree set
contains 4-10 leaves and contributes 1,228 pairs per endpoint.

The multiple-choice probes each contain 109,500 outputs. Empty-output counts and semantic coverage
are: ARC-Challenge 271 and 99.75%, HellaSwag 41,430 and 62.16%, MMLU 378 and 99.65%, and
TruthfulQA 652 and 99.40%. Dolly contains 32,850 outputs, 556 empties, and 98.31% semantic
coverage. The unusually low HellaSwag coverage is reported directly; empty outputs remain valid
surface observations and are excluded only from sentence embedding.

Across 50/50 usable trees, the DerSimonian--Laird surface/semantic correlations were
ARC-Challenge 0.034/-0.008, Dolly 0.019/0.015, HellaSwag -0.062/-0.023, MMLU 0.035/-0.007, and
TruthfulQA 0.026/-0.027. None was distinguishable from zero. Compact audits, within-tree
correlations, pooled estimates, and raw/semi-standardized/standardized run-fixed-effect
coefficients are versioned under `results/behavior/llama32_r8_final_20260722/`.

### Llama behavioral branch-ordering appendix

The paper-matched one-sided pooled Mann--Whitney test (`cross_branch > same_branch`) finds a small
HellaSwag surface effect (p=8.91e-4, rank-biserial=.113) and a nominal Dolly surface effect
(p=.019, rank-biserial=.075). All semantic tests are null; HellaSwag is closest at p=.076.
Only 26/50 trees contain both branch classes, because 24 trees have a one-child root. When each
ordering-valid tree is treated as the replication unit, no endpoint passes the one-sided Wilcoxon
robustness test (HellaSwag surface p=.251; Dolly surface p=.742). The pooled results are therefore
exploratory continuity statistics, not independent-pair inference.

The full appendix writeup and exact ten-row summary are in
`docs/LLAMA32_R8_BEHAVIOR_BRANCH_ORDERING_2026-07-22.md` and
`results/behavior/llama32_r8_branch_ordering_20260722/paper_summary.csv`.
Rebuild an endpoint directly from the per-tree pair tables with:

```bash
python scripts/behavior_branch_ordering.py \
  --condition llama32_1b_qkv_lora_r8 \
  --probe hellaswag \
  --endpoint surface \
  --pairs-glob 'outputs/behavior/llama32_r8_mc_four_s3_all50/tree*/pairs/hellaswag.csv' \
  --manifest-dir examples/training/confirm_paper_numbers/assigned_manifests \
  --expected-trees 50 \
  --out-dir outputs/behavior/llama32_r8_branch_ordering/hellaswag_surface
```

The script fails on duplicate/incomplete pair grids, non-finite distances, unknown/nonleaf models,
or a missing expected tree. It writes the pooled legacy statistics, per-tree effects, labeled pair
audit, and tree-level robustness tests separately.

For every future behavioral table, retain separate surface and semantic rows and report: model
condition and weight representation; probe/endpoint; total, DL-usable, and ordering-valid trees;
all/same/cross pair counts; empty-output and semantic-coverage counts; DL r, CI, p, I2, and tau2;
pooled one-sided Mann--Whitney p plus rank-biserial; mean per-tree rank-biserial plus SE and a
tree-level Wilcoxon/sign test; and the within-tree Fisher-z branch correlation. Pooled leaf-pair
p-values must be labeled descriptive because pairs share leaves. Never assume 36 pairs globally or
hide unary-root trees; derive `choose(L, 2)` and branch-class availability from each truth manifest.
