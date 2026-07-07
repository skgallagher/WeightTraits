# Paper Workspace

This directory will hold the rebuilt ICLR draft and artifact registries.

The old paper in `../ELLMTrees/paper` is a reference target. WeightTraits should regenerate figures and tables from source commands before they are used here.

Current registries:

- `recovery_registry.yaml`: verified whitebox recovery summary artifacts.
- `table_registry.yaml`: paper-facing table definitions and source commands.

Build the provisional whitebox recovery table in an environment where the registered `outputs/`
paths exist:

```bash
PYTHONPATH=src python -m weighttraits.cli make-recovery-table \
  --registry paper/recovery_registry.yaml \
  --out reports/paper/whitebox_smoke_recovery.json \
  --csv-out reports/paper/whitebox_smoke_recovery.csv
```
