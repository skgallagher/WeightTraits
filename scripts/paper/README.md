# Reproducibility Workbook Builder

`rebuild_reproducibility_workbook.mjs` rebuilds the audit workbook from a
preserved `.xlsx` snapshot plus the checked-in status/source contract. It refuses
to overwrite its input, verifies the input SHA-256, scans formulas, renders every
sheet for visual review, and writes inspection and reconciliation sidecars beside
the output workbook.

## Runtime contract

The checked-in launcher pins the environment that produced the verified rebuild:

| Component | Required version |
| --- | --- |
| Codex primary runtime bundle | `26.905.11957` |
| Node | `v24.19.0` |
| `@oai/artifact-tool` | `2.8.59` |

Artifact-tool is private and bundles private `@oai` dependencies. It is supplied
by `load_workspace_dependencies`; it is not vendored here and cannot be represented
honestly by a public npm lockfile. The launcher discovers the standard Codex cache
location, verifies `runtime.json` and the package manifest, exposes that exact
module directory only for the builder process, and removes its transient symlink
on exit. If the loader returns a different runtime root, pass it without checking
a host-specific path into git:

```bash
export WEIGHTTRAITS_CODEX_RUNTIME=/loader/provided/codex-primary-runtime
```

## Rebuild command

Run from the WeightTraits repository root:

```bash
scripts/paper/run_rebuild_reproducibility_workbook.sh \
  --input ../ELLMTrees-paper/WeightTraits_all_results_reproducibility.xlsx \
  --status examples/paper/workbook_rebuild_status_20260810.json \
  --output ../outputs/weighttraits-workbook-20260810/WeightTraits_all_results_reproducibility_rebuild.xlsx \
  --workspace-root .. \
  --preview-dir ../outputs/weighttraits-workbook-20260810/previews
```

The output must be distinct from the preserved input. A completed rebuild is
acceptable only after the formula-error scan is empty, source hashes reconcile,
and all rendered sheets pass visual review. The manuscript remains read-only; the
workbook is a reconciliation surface, not authority to change manuscript prose.

`workbook_rebuild_status_20260810.json` is the source receipt for the workbook
already rendered from it, not a live scheduler dashboard. Newer protocol-readiness
state is recorded in the rebuild documents under
`examples/training/confirm_paper_numbers/`. If that status JSON changes, rebuild
and reverify the workbook rather than presenting an old render as current.
