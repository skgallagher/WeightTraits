# Paper Workspace

This directory will hold the rebuilt ICLR draft and artifact registries.

The old paper in `../ELLMTrees/paper` is a reference target only. Legacy analysis code and derived
tables are never inputs to WeightTraits candidate statistics or figures. Candidate artifacts are
computed by WeightTraits from the new experiment outputs; old results may be compared only after
the independent candidate artifact is frozen.

Current registries:

- `reference_registry.yaml`: pinned v2 draft, figure, table, and source-artifact reference surface
  from the old ELLMTrees paper/results repos.
- `ellmtrees_variants_registry.yaml`: row mapping for the old ELLMTrees `tab:variants` reference
  table.
- `weighttraits_variants_registry.yaml`: row mapping from the versioned WeightTraits run-set
  rollups to the five rebuilt Flan-T5 rows in `tab:variants`.
- `weighttraits_completed_conditions_registry.yaml`: the five currently complete fresh conditions
  (four LoRA scopes plus full fine-tuning), independent of the unfinished legacy-scope row.
- `weighttraits_runset_diagnostics_registry.yaml`: native three-metric recovery, four-point, and
  Atteson diagnostics for those same five completed conditions.
- `recovery_registry.yaml`: verified whitebox recovery summary artifacts.
- `table_registry.yaml`: paper-facing table definitions and source commands.

Validate the reference surface:

```bash
PYTHONPATH=src python -m weighttraits.cli validate-reference-registry \
  --registry paper/reference_registry.yaml
```

Build the provisional whitebox recovery table in an environment where the registered `outputs/`
paths exist:

```bash
PYTHONPATH=src python -m weighttraits.cli make-recovery-table \
  --registry paper/recovery_registry.yaml \
  --out reports/paper/whitebox_smoke_recovery.json \
  --csv-out reports/paper/whitebox_smoke_recovery.csv
```

Build the legacy ELLMTrees `tab:variants` reference table from pinned old CSVs:

```bash
PYTHONPATH=src python -m weighttraits.cli make-ellmtrees-variants-table \
  --registry paper/ellmtrees_variants_registry.yaml \
  --out reports/paper/ellmtrees_variants_reference.json \
  --csv-out reports/paper/ellmtrees_variants_reference.csv
```

After the versioned direct analyses finish, rerun their lightweight rollups with the current code.
The rollup derives the paper's per-tree same-branch/cross-branch rank-biserial and within-run
correlation from each saved distance matrix; it does not reload checkpoints or recompute weights.
For example:

```bash
PYTHONPATH=src python -m weighttraits.cli summarize-training-run-set-analysis \
  --analysis-root outputs/analysis_v20260713/lora_qkv \
  --artifact adapter_chain \
  --truth-manifest-root examples/training/confirm_paper_numbers/assigned_manifests \
  --out outputs/analysis_v20260713/lora_qkv_summary.json \
  --csv-out outputs/analysis_v20260713/lora_qkv_rows.csv
```

`--truth-manifest-root` is useful after pulling analysis directories from Wright: it relocates an
absolute cluster-authored manifest path by filename without modifying the saved analysis artifact.

Build the five WeightTraits Flan-T5 candidate rows in the same stable schema as the legacy table:

```bash
PYTHONPATH=src python -m weighttraits.cli make-weighttraits-variants-table \
  --registry paper/weighttraits_variants_registry.yaml \
  --out reports/paper/weighttraits_variants_rebuild.json \
  --csv-out reports/paper/weighttraits_variants_rebuild.csv
```

Plot the independent candidate estimands and standard errors:

```bash
PYTHONPATH=src python -m weighttraits.cli plot-weighttraits-variants \
  --table reports/paper/weighttraits_variants_rebuild.json \
  --out reports/paper/weighttraits_variants_rebuild.svg
```

The plotting command requires the provenance-bearing `weighttraits.variants.v1` JSON written by
`make-weighttraits-variants-table`. It rejects legacy-style JSON and CSV inputs so a reference
table cannot accidentally enter the candidate figure path.

Build the completed-condition robustness and additivity artifacts directly from native run-set
summaries:

```bash
PYTHONPATH=src python -m weighttraits.cli make-weighttraits-runset-diagnostics \
  --registry paper/weighttraits_runset_diagnostics_registry.yaml \
  --out reports/paper/weighttraits_runset_diagnostics.json \
  --csv-out reports/paper/weighttraits_runset_diagnostics.csv

PYTHONPATH=src python -m weighttraits.cli plot-weighttraits-metric-robustness \
  --diagnostics reports/paper/weighttraits_runset_diagnostics.json \
  --out reports/paper/weighttraits_metric_robustness.svg

PYTHONPATH=src python -m weighttraits.cli plot-weighttraits-additivity-recovery \
  --diagnostics reports/paper/weighttraits_runset_diagnostics.json \
  --metric cosine \
  --out reports/paper/weighttraits_additivity_recovery.svg
```

The diagnostics builder validates every contributing row as native `direct` analysis and rejects
mixed or legacy analysis engines. The JSON records `producer: weighttraits`; both plotting commands
require that provenance-bearing JSON rather than accepting arbitrary CSV input.

Recovery and ordering sample counts are tracked separately. A Poisson draw may give the root only
one child; such a tree remains valid for recovery but has no cross-root-branch pairs and therefore
does not contribute a rank-biserial or within-run correlation. This matches the legacy analysis
behavior instead of manufacturing an ordering label for an undefined comparison.

Extract the live draft `tab:behavior_holdout` reference table:

```bash
PYTHONPATH=src python -m weighttraits.cli make-behavior-holdout-table \
  --draft ../ELLMTrees-paper/iclr_draft_v2.tex \
  --out reports/paper/behavior_holdout_reference.json \
  --csv-out reports/paper/behavior_holdout_reference.csv
```

Run registered table comparisons against paper-grounded references:

```bash
PYTHONPATH=src python -m weighttraits.cli run-table-comparisons \
  --registry paper/table_registry.yaml \
  --out reports/paper/table_comparison_validation.json
```

Compare one rebuilt table artifact to the paper-grounded reference directly:

```bash
PYTHONPATH=src python -m weighttraits.cli compare-table-artifacts \
  --reference reports/paper/ellmtrees_variants_reference.csv \
  --candidate reports/paper/ellmtrees_variants_reference.json \
  --key-column variant_id \
  --out reports/paper/ellmtrees_variants_reference_compare.json
```

Validate the table registry and generated row counts:

```bash
PYTHONPATH=src python -m weighttraits.cli validate-table-registry \
  --registry paper/table_registry.yaml \
  --require-outputs
```
