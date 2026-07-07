# Paper Workspace

This directory will hold the rebuilt ICLR draft and artifact registries.

The old paper in `../ELLMTrees/paper` is a reference target. WeightTraits should regenerate figures and tables from source commands before they are used here.

Current registries:

- `reference_registry.yaml`: pinned v2 draft, figure, table, and source-artifact reference surface
  from the old ELLMTrees paper/results repos.
- `ellmtrees_variants_registry.yaml`: row mapping for the old ELLMTrees `tab:variants` reference
  table.
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

Compare a rebuilt table artifact to the paper-grounded reference:

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
