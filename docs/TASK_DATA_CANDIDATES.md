# Task And Dataset Candidates

This is a draft pool for expanding beyond the original ELLMTrees tasks. The project should keep the old task families for comparability, then add harder pools only after loader, license, split, and prompt-format audits.

The machine-readable draft is [configs/task_data_candidates.yaml](../configs/task_data_candidates.yaml).

## Existing Families To Preserve

- Summarization: CNN/DailyMail, XSum, SAMSum, BillSum, PubMed, GovReport, arXiv, BigPatent, DialogSum.
- Classification: SST-2, AG News, MNLI, IMDB, RTE, CoLA, QNLI, Yelp Polarity, DBPedia.
- QA/reasoning: SQuAD, SQuAD v2, BoolQ, ARC Easy/Challenge, OpenBookQA, CommonsenseQA, HellaSwag, MMLU auxiliary train.
- Translation: WMT14 and OPUS pairs already used in ELLMTrees.

## Harder Candidates

Math reasoning:
- `openai/gsm8k` has `main` and `socratic` subsets with train/test splits.
- `EleutherAI/hendrycks_math` is preferable to `hendrycks/competition_math`, because the latter is currently disabled on Hugging Face; the EleutherAI mirror exposes seven subject subsets.
- `deepmind/math_dataset` is a possible large synthetic source, but needs loader design before use.

Code generation:
- `codeparrot/apps` has train/test splits and difficulty filters including competition-level problems.
- `google-research-datasets/mbpp` has full and sanitized subsets.
- `bigcode/the-stack-smol` can provide language-filtered code continuation data, but access and license conditions need auditing.
- `openai/openai_humaneval` is evaluation-only, not a training pool.

Instruction following and behavior probes:
- `google/IFEval` is useful as an instruction-following probe, but it is small and should not be treated like a large training pool.
- TruthfulQA is validation-only and best used as a probe unless we deliberately build a training variant.
- MMLU has many subject subsets and a large auxiliary split; subject leakage and train/eval separation need careful handling.

## Audits Before Use

For every candidate:
- verify the dataset still exists and is downloadable from the intended environment;
- record license and access restrictions;
- record split names and row counts;
- implement one loader test with a tiny sample;
- confirm prompt format does not accidentally trigger label-vocabulary collapse;
- mark whether it is training, validation, evaluation-only, or behavior-probe only.

The first offline prompt-field check is `wt validate-training-data`, documented in [Trainer Design](TRAINER_DESIGN.md). It validates planned prompt templates against dataset-format contracts before any dataset download or training launch.

The first registry/split check is `wt audit-datasets`. Use `--no-load` for local dry runs that should
not download anything; omit it on a prepared cluster or local environment to call Hugging Face
`load_dataset` and record available splits plus row counts.

The first real row-level prompt check is `wt audit-training-samples`. It loads a tiny sample for
planned jobs, applies the dataset `field_map`, renders prompts, and reports missing fields or empty
renders without writing raw samples to the report.
