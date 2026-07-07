import csv
import hashlib
import json
from pathlib import Path

import pytest

from weighttraits.paper.results import (
    compare_table_artifacts,
    ellmtrees_variants_table_rows,
    recovery_table_rows,
    run_table_registry_comparisons,
    validate_reference_registry,
    validate_table_registry,
    write_ellmtrees_variants_table_csv,
    write_ellmtrees_variants_table_json,
    write_recovery_table_csv,
    write_recovery_table_json,
)


def _write_summary(path: Path, *, artifact: str = "model") -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "artifact": artifact,
                "representation": "full_weight",
                "ledger": "outputs/example/training_ledger.jsonl",
                "truth_manifest": "examples/training/truth.jsonl",
                "n_models": 4,
                "n_layers": 12,
                "results": [
                    {
                        "metric": "l2",
                        "rf": 0,
                        "normalized_rf": 0.0,
                        "exact_tree_recovery": True,
                        "clade_recovery": 1.0,
                        "split_precision": 1.0,
                        "distance_mean": 0.125,
                        "distance_min": 0.01,
                        "distance_max": 0.2,
                    },
                    {
                        "metric": "cosine",
                        "rf": 2,
                        "normalized_rf": 0.5,
                        "exact_tree_recovery": False,
                        "clade_recovery": 0.5,
                        "split_precision": 0.5,
                        "distance_mean": 0.25,
                        "distance_min": 0.02,
                        "distance_max": 0.4,
                    },
                ],
                "aggregate_recovery": {
                    "exact_tree_recovery_rate": 0.5,
                    "n_truth_leaves_mean": 4.0,
                    "n_truth_splits_mean": 1.0,
                },
            }
        )
        + "\n"
    )


def test_recovery_table_rows_load_registered_summary(tmp_path):
    summary = tmp_path / "outputs/example/summary.json"
    registry = tmp_path / "paper/recovery_registry.yaml"
    _write_summary(summary)
    registry.parent.mkdir()
    registry.write_text(
        """
result_sets:
  - id: example_whitebox
    title: Example whitebox result
    status: smoke_verified
    environment: local
    truth_manifest: examples/training/truth.jsonl
    summaries:
      - id: full_model
        artifact: model
        summary: outputs/example/summary.json
"""
    )

    rows = recovery_table_rows(registry, base_dir=tmp_path)

    assert len(rows) == 2
    assert rows[0]["result_set_id"] == "example_whitebox"
    assert rows[0]["summary_id"] == "full_model"
    assert rows[0]["metric"] == "l2"
    assert rows[0]["exact_tree_recovery"] is True
    assert rows[0]["exact_tree_recovery_rate"] == 0.5
    assert rows[0]["environment"] == "local"
    assert rows[1]["metric"] == "cosine"


def test_recovery_table_writers_emit_json_and_csv(tmp_path):
    rows = [
        {
            "result_set_id": "example",
            "result_set_title": "Example",
            "summary_id": "full_model",
            "artifact": "model",
            "representation": "full_weight",
            "metric": "l2",
            "n_models": 4,
            "n_layers": 12,
            "n_truth_leaves": 4.0,
            "n_truth_splits": 1.0,
            "rf": 0,
            "normalized_rf": 0.0,
            "exact_tree_recovery": True,
            "exact_tree_recovery_rate": 1.0,
            "clade_recovery": 1.0,
            "split_precision": 1.0,
            "distance_mean": 0.125,
            "distance_min": 0.01,
            "distance_max": 0.2,
            "environment": "local",
            "status": "smoke_verified",
            "truth_manifest": "truth.jsonl",
            "ledger": "ledger.jsonl",
            "summary": "summary.json",
        }
    ]
    json_out = tmp_path / "table.json"
    csv_out = tmp_path / "table.csv"

    write_recovery_table_json(rows, json_out, registry="paper/recovery_registry.yaml")
    write_recovery_table_csv(rows, csv_out)

    payload = json.loads(json_out.read_text())
    assert payload["n_rows"] == 1
    assert payload["rows"][0]["metric"] == "l2"
    with csv_out.open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["exact_tree_recovery"] == "true"
    assert csv_rows[0]["distance_mean"] == "0.125"


def test_recovery_table_rows_reject_artifact_mismatch(tmp_path):
    summary = tmp_path / "outputs/example/summary.json"
    registry = tmp_path / "recovery_registry.yaml"
    _write_summary(summary, artifact="merged")
    registry.write_text(
        """
result_sets:
  - id: example_whitebox
    summaries:
      - id: full_model
        artifact: model
        summary: outputs/example/summary.json
"""
    )

    with pytest.raises(ValueError, match="summary artifact mismatch"):
        recovery_table_rows(registry, base_dir=tmp_path)


def test_validate_table_registry_checks_inputs_outputs_and_rows(tmp_path):
    source_input = tmp_path / "paper/recovery_registry.yaml"
    json_out = tmp_path / "reports/paper/recovery.json"
    csv_out = tmp_path / "reports/paper/recovery.csv"
    registry = tmp_path / "paper/table_registry.yaml"
    source_input.parent.mkdir(parents=True)
    source_input.write_text("result_sets: []\n")
    rows = [
        {
            "result_set_id": "example",
            "result_set_title": "Example",
            "summary_id": "full_model",
            "artifact": "model",
            "representation": "full_weight",
            "metric": "l2",
            "n_models": 4,
            "n_layers": 12,
            "n_truth_leaves": 4.0,
            "n_truth_splits": 1.0,
            "rf": 0,
            "normalized_rf": 0.0,
            "exact_tree_recovery": True,
            "exact_tree_recovery_rate": 1.0,
            "clade_recovery": 1.0,
            "split_precision": 1.0,
            "distance_mean": 0.125,
            "distance_min": 0.01,
            "distance_max": 0.2,
            "environment": "local",
            "status": "smoke_verified",
            "truth_manifest": "truth.jsonl",
            "ledger": "ledger.jsonl",
            "summary": "summary.json",
        }
    ]
    write_recovery_table_json(rows, json_out)
    write_recovery_table_csv(rows, csv_out)
    registry.write_text(
        """
tables:
  - id: whitebox_smoke_recovery
    title: Whitebox smoke recovery
    status: smoke_verified
    source_command: wt make-recovery-table
    source_inputs:
      - paper/recovery_registry.yaml
    outputs:
      - reports/paper/recovery.json
      - reports/paper/recovery.csv
    expected_rows: 1
    paper_location: provisional
    verification_status: local_verified
"""
    )

    report = validate_table_registry(registry, base_dir=tmp_path, require_outputs=True)

    assert report["valid"]
    assert report["n_tables"] == 1
    assert report["tables"][0]["id"] == "whitebox_smoke_recovery"
    assert report["tables"][0]["expected_rows"] == 1
    assert [item["observed_rows"] for item in report["tables"][0]["outputs"]] == [1, 1]
    assert [item["expected_sha256"] for item in report["tables"][0]["outputs"]] == [None, None]
    assert all(item["observed_sha256"] for item in report["tables"][0]["outputs"])


def test_validate_table_registry_reports_missing_required_output(tmp_path):
    source_input = tmp_path / "paper/recovery_registry.yaml"
    registry = tmp_path / "paper/table_registry.yaml"
    source_input.parent.mkdir(parents=True)
    source_input.write_text("result_sets: []\n")
    registry.write_text(
        """
tables:
  - id: whitebox_smoke_recovery
    title: Whitebox smoke recovery
    source_command: wt make-recovery-table
    source_inputs:
      - paper/recovery_registry.yaml
    outputs:
      - reports/paper/missing.json
    expected_rows: 1
    paper_location: provisional
    verification_status: local_verified
"""
    )

    report = validate_table_registry(registry, base_dir=tmp_path, require_outputs=True)

    assert not report["valid"]
    assert report["issues"][0]["code"] == "missing_output"


def test_validate_table_registry_reports_row_count_mismatch(tmp_path):
    source_input = tmp_path / "paper/recovery_registry.yaml"
    json_out = tmp_path / "reports/paper/recovery.json"
    registry = tmp_path / "paper/table_registry.yaml"
    source_input.parent.mkdir(parents=True)
    source_input.write_text("result_sets: []\n")
    json_out.parent.mkdir(parents=True)
    json_out.write_text(json.dumps({"n_rows": 2, "rows": [{}, {}]}) + "\n")
    registry.write_text(
        """
tables:
  - id: whitebox_smoke_recovery
    title: Whitebox smoke recovery
    source_command: wt make-recovery-table
    source_inputs:
      - paper/recovery_registry.yaml
    outputs:
      - reports/paper/recovery.json
    expected_rows: 1
    paper_location: provisional
    verification_status: local_verified
"""
    )

    report = validate_table_registry(registry, base_dir=tmp_path, require_outputs=True)

    assert not report["valid"]
    assert report["issues"][0]["code"] == "row_count_mismatch"


def test_validate_table_registry_accepts_pinned_output_sha256(tmp_path):
    source_input = tmp_path / "paper/recovery_registry.yaml"
    json_out = tmp_path / "reports/paper/recovery.json"
    registry = tmp_path / "paper/table_registry.yaml"
    source_input.parent.mkdir(parents=True)
    source_input.write_text("result_sets: []\n")
    json_out.parent.mkdir(parents=True)
    json_out.write_text(json.dumps({"n_rows": 1, "rows": [{}]}) + "\n")
    digest = hashlib.sha256(json_out.read_bytes()).hexdigest()
    registry.write_text(
        f"""
tables:
  - id: whitebox_smoke_recovery
    title: Whitebox smoke recovery
    source_command: wt make-recovery-table
    source_inputs:
      - paper/recovery_registry.yaml
    outputs:
      - path: reports/paper/recovery.json
        sha256: {digest}
    expected_rows: 1
    paper_location: provisional
    verification_status: local_verified
"""
    )

    report = validate_table_registry(registry, base_dir=tmp_path, require_outputs=True)

    assert report["valid"]
    assert report["tables"][0]["outputs"][0]["expected_sha256"] == digest
    assert report["tables"][0]["outputs"][0]["observed_sha256"] == digest


def test_validate_table_registry_reports_output_sha256_mismatch(tmp_path):
    source_input = tmp_path / "paper/recovery_registry.yaml"
    json_out = tmp_path / "reports/paper/recovery.json"
    registry = tmp_path / "paper/table_registry.yaml"
    source_input.parent.mkdir(parents=True)
    source_input.write_text("result_sets: []\n")
    json_out.parent.mkdir(parents=True)
    json_out.write_text(json.dumps({"n_rows": 1, "rows": [{}]}) + "\n")
    registry.write_text(
        """
tables:
  - id: whitebox_smoke_recovery
    title: Whitebox smoke recovery
    source_command: wt make-recovery-table
    source_inputs:
      - paper/recovery_registry.yaml
    outputs:
      - path: reports/paper/recovery.json
        sha256: "0000000000000000000000000000000000000000000000000000000000000000"
    expected_rows: 1
    paper_location: provisional
    verification_status: local_verified
"""
    )

    report = validate_table_registry(registry, base_dir=tmp_path, require_outputs=True)

    assert not report["valid"]
    assert report["issues"][0]["code"] == "sha256_mismatch"


def test_compare_table_artifacts_accepts_matching_rows_with_numeric_tolerance(tmp_path):
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.json"
    reference.write_text(
        "\n".join(
            [
                "variant_id,label,score,source",
                "a,Alpha,1.000000001,old.csv",
                "b,Beta,2.5,old.csv",
            ]
        )
        + "\n"
    )
    candidate.write_text(
        json.dumps(
            {
                "rows": [
                    {"variant_id": "b", "label": "Beta", "score": 2.5, "source": "new.csv"},
                    {"variant_id": "a", "label": "Alpha", "score": 1.0, "source": "new.csv"},
                ]
            }
        )
        + "\n"
    )

    report = compare_table_artifacts(
        reference,
        candidate,
        key_columns=["variant_id"],
        numeric_columns=["score"],
        ignore_columns=["source"],
        atol=1e-6,
    )

    assert report["valid"]
    assert report["n_reference_rows"] == 2
    assert report["n_candidate_rows"] == 2
    assert report["n_matched_rows"] == 2
    assert report["n_issues"] == 0


def test_compare_table_artifacts_reports_row_and_value_mismatches(tmp_path):
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    reference.write_text("id,label,score\none,One,1.0\ntwo,Two,2.0\n")
    candidate.write_text("id,label,score\none,Uno,1.25\nthree,Three,3.0\n")

    report = compare_table_artifacts(
        reference,
        candidate,
        key_columns=["id"],
        numeric_columns=["score"],
        atol=0.01,
    )

    assert not report["valid"]
    assert [issue["code"] for issue in report["issues"]] == [
        "missing_row",
        "extra_row",
        "value_mismatch",
        "numeric_mismatch",
    ]


def test_run_table_registry_comparisons_writes_declared_reports(tmp_path):
    reference = tmp_path / "reports/paper/reference.csv"
    candidate = tmp_path / "reports/paper/candidate.json"
    registry = tmp_path / "paper/table_registry.yaml"
    out = tmp_path / "reports/paper/comparison.json"
    reference.parent.mkdir(parents=True)
    registry.parent.mkdir(parents=True)
    reference.write_text("id,label,score,source\none,One,1.00001,old.csv\n")
    candidate.write_text(
        json.dumps(
            {
                "rows": [
                    {"id": "one", "label": "One", "score": 1.0, "source": "new.csv"},
                ]
            }
        )
        + "\n"
    )
    registry.write_text(
        """
tables:
  - id: example_table
    title: Example
    source_command: wt make-example-table
    source_inputs:
      - reports/paper/reference.csv
    outputs:
      - reports/paper/candidate.json
    expected_rows: 1
    paper_location: smoke
    verification_status: local_verified
    comparisons:
      - id: example_json_csv
        reference: reports/paper/reference.csv
        candidate: reports/paper/candidate.json
        key_columns:
          - id
        numeric_columns:
          - score
        ignore_columns:
          - source
        atol: 0.001
        out: reports/paper/comparison.json
"""
    )

    report = run_table_registry_comparisons(registry, base_dir=tmp_path)

    assert report["valid"]
    assert report["n_comparisons"] == 1
    assert report["comparisons"][0]["id"] == "example_json_csv"
    assert report["comparisons"][0]["n_compared_cells"] == 2
    written = json.loads(out.read_text())
    assert written["valid"]
    assert written["table_id"] == "example_table"


def test_run_table_registry_comparisons_reports_mismatches(tmp_path):
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    registry = tmp_path / "table_registry.yaml"
    reference.write_text("id,label\none,One\n")
    candidate.write_text("id,label\none,Uno\n")
    registry.write_text(
        """
tables:
  - id: example_table
    title: Example
    source_command: wt make-example-table
    source_inputs:
      - reference.csv
    outputs:
      - candidate.csv
    expected_rows: 1
    paper_location: smoke
    verification_status: local_verified
    comparisons:
      - id: example_mismatch
        reference: reference.csv
        candidate: candidate.csv
        key_columns:
          - id
"""
    )

    report = run_table_registry_comparisons(registry, base_dir=tmp_path)

    assert not report["valid"]
    assert report["issues"][0]["code"] == "comparison_failed"
    assert report["comparisons"][0]["issues"][0]["code"] == "value_mismatch"


def test_validate_reference_registry_checks_paths_sources_and_digests(tmp_path):
    draft = tmp_path / "ELLMTrees-paper/iclr_draft_v2.tex"
    figure = tmp_path / "ELLMTrees-paper/figures/fig.png"
    source = tmp_path / "ELLMTrees/scripts/make_fig.py"
    registry = tmp_path / "paper/reference_registry.yaml"
    draft.parent.mkdir(parents=True)
    figure.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    registry.parent.mkdir(parents=True)
    draft.write_text("draft\n")
    figure.write_bytes(b"figure")
    source.write_text("print('figure')\n")
    figure_digest = hashlib.sha256(figure.read_bytes()).hexdigest()
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    registry.write_text(
        f"""
version: 1
entries:
  - id: active_draft_v2
    kind: draft
    classification: live_target
    status: in_progress
    path: ELLMTrees-paper/iclr_draft_v2.tex
  - id: fig:example
    kind: figure
    classification: paper_critical
    status: reference_pinned
    path: ELLMTrees-paper/figures/fig.png
    sha256: {figure_digest}
    source_inputs:
      - path: ELLMTrees/scripts/make_fig.py
        role: generator
        sha256: {source_digest}
"""
    )

    report = validate_reference_registry(registry, base_dir=tmp_path)

    assert report["valid"]
    assert report["n_entries"] == 2
    assert report["entries"][1]["path"]["observed_sha256"] == figure_digest
    assert report["entries"][1]["source_inputs"][0]["observed_sha256"] == source_digest


def test_validate_reference_registry_checks_active_draft_label_coverage(tmp_path):
    draft = tmp_path / "iclr_draft_v2.tex"
    registry = tmp_path / "reference_registry.yaml"
    draft.write_text(
        r"""
\begin{figure}
\label{fig:covered}
\end{figure}
\begin{table}
\label{tab:missing}
\end{table}
"""
    )
    registry.write_text(
        """
version: 1
active_draft: iclr_draft_v2.tex
entries:
  - id: fig:covered
    kind: figure
    classification: paper_critical
    status: reference_pinned
    path: iclr_draft_v2.tex
  - id: fig:stale
    kind: figure
    classification: paper_critical
    status: reference_pinned
    path: iclr_draft_v2.tex
"""
    )

    report = validate_reference_registry(registry, base_dir=tmp_path)

    assert not report["valid"]
    assert report["draft_label_coverage"]["labels"] == ["fig:covered", "tab:missing"]
    assert report["draft_label_coverage"]["missing_labels"] == ["tab:missing"]
    assert report["draft_label_coverage"]["stale_registered_labels"] == ["fig:stale"]
    assert [issue["code"] for issue in report["issues"]] == [
        "missing_draft_label_entry",
        "stale_draft_label_entry",
    ]


def test_validate_reference_registry_reports_missing_source(tmp_path):
    figure = tmp_path / "fig.png"
    registry = tmp_path / "reference_registry.yaml"
    figure.write_bytes(b"figure")
    registry.write_text(
        """
version: 1
entries:
  - id: fig:missing_source
    kind: figure
    classification: paper_critical
    status: reference_pinned
    path: fig.png
    source_inputs:
      - path: missing.py
        role: generator
"""
    )

    report = validate_reference_registry(registry, base_dir=tmp_path)

    assert not report["valid"]
    assert report["issues"][0]["code"] == "missing_source_input"


def test_validate_reference_registry_reports_digest_mismatch(tmp_path):
    draft = tmp_path / "iclr_draft_v2.tex"
    registry = tmp_path / "reference_registry.yaml"
    draft.write_text("draft\n")
    registry.write_text(
        """
version: 1
entries:
  - id: active_draft_v2
    kind: draft
    classification: live_target
    status: in_progress
    path: iclr_draft_v2.tex
    sha256: "0000000000000000000000000000000000000000000000000000000000000000"
"""
    )

    report = validate_reference_registry(registry, base_dir=tmp_path)

    assert not report["valid"]
    assert report["issues"][0]["code"] == "sha256_mismatch"


def test_ellmtrees_variants_table_rows_join_recovery_and_branch_stats(tmp_path):
    recovery = tmp_path / "aggregate/by_group_cosine.csv"
    per_run = tmp_path / "aggregate/per_run_cosine.csv"
    branch = tmp_path / "aggregate/branch_structure.csv"
    registry = tmp_path / "paper/ellmtrees_variants_registry.yaml"
    recovery.parent.mkdir(parents=True)
    registry.parent.mkdir(parents=True)
    recovery.write_text(
        "\n".join(
            [
                "group,n_runs,mean_RF,mean_FN,mean_clade_recovery,pct_refinement_FN0",
                "runs_example,2,1.0,0.5,75.0,50.0",
            ]
        )
        + "\n"
    )
    per_run.write_text(
        "\n".join(
            [
                "group,run,clade_recovery,refinement,RF,FN",
                "runs_example,run_001,1.0,1,0,0",
                "runs_example,run_002,0.5,0,2,1",
            ]
        )
        + "\n"
    )
    branch.write_text(
        "\n".join(
            [
                "run_id,node_i,node_j,same_branch,cosine_dist",
                "run_001,a,b,1,0.1",
                "run_001,a,c,0,0.4",
                "run_001,b,c,0,0.3",
                "run_002,a,b,1,0.2",
                "run_002,a,c,0,0.3",
                "run_002,b,c,0,0.4",
            ]
        )
        + "\n"
    )
    registry.write_text(
        """
version: 1
recovery_source: aggregate/by_group_cosine.csv
per_run_recovery_source: aggregate/per_run_cosine.csv
variants:
  - id: example
    section: Example
    model: ExampleModel
    label: Example variant
    recovery_group: runs_example
    branch_source: aggregate/branch_structure.csv
"""
    )

    rows = ellmtrees_variants_table_rows(registry, base_dir=tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["variant_id"] == "example"
    assert row["n_runs"] == 2
    assert row["n_ordering_runs"] == 2
    assert row["n_pairs"] == 6
    assert row["rank_biserial"] == 1.0
    assert row["clade_recovery_pct"] == 75.0
    assert row["exact_recovery_pct"] == 50.0
    assert row["rf_mean"] == 1.0
    assert row["fn_mean"] == 0.5


def test_ellmtrees_variants_table_writers_emit_json_and_csv(tmp_path):
    rows = [
        {
            "variant_id": "example",
            "section": "Example",
            "model": "ExampleModel",
            "label": "Example variant",
            "recovery_group": "runs_example",
            "n_runs": 2,
            "n_ordering_runs": 2,
            "n_pairs": 6,
            "rank_biserial": 1.0,
            "rank_biserial_se": 0.0,
            "within_run_r": -0.9,
            "within_run_r_se": 0.1,
            "clade_recovery_pct": 75.0,
            "clade_recovery_se_pct": 25.0,
            "exact_recovery_pct": 50.0,
            "exact_recovery_se_pct": 35.35,
            "rf_mean": 1.0,
            "rf_se": 1.0,
            "fn_mean": 0.5,
            "fn_se": 0.5,
            "recovery_source": "by_group.csv",
            "per_run_recovery_source": "per_run.csv",
            "branch_source": "branch.csv",
        }
    ]
    json_out = tmp_path / "variants.json"
    csv_out = tmp_path / "variants.csv"

    write_ellmtrees_variants_table_json(rows, json_out, registry="paper/variants.yaml")
    write_ellmtrees_variants_table_csv(rows, csv_out)

    payload = json.loads(json_out.read_text())
    assert payload["n_rows"] == 1
    assert payload["rows"][0]["variant_id"] == "example"
    with csv_out.open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["rank_biserial"] == "1.0"
    assert csv_rows[0]["branch_source"] == "branch.csv"
