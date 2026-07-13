from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from weighttraits.paper.diagnostics import (
    WEIGHTTRAITS_RUNSET_DIAGNOSTICS_SCHEMA,
    weighttraits_runset_diagnostic_rows,
    write_weighttraits_runset_diagnostics_json,
)
from weighttraits.paper.figures import (
    load_weighttraits_runset_diagnostics_artifact,
    plot_weighttraits_additivity_recovery,
    plot_weighttraits_metric_robustness,
)


def _aggregate(offset: float) -> dict[str, float | int]:
    return {
        "n_trees": 2,
        "clade_recovery_mean": 0.8 + offset,
        "clade_recovery_se": 0.02,
        "polytomy_aware_exact_recovery_rate": 0.6 + offset,
        "polytomy_aware_exact_recovery_rate_se": 0.03,
        "rf_mean": 2.0 - offset,
        "rf_se": 0.2,
        "false_negative_mean": 0.3 - offset,
        "false_negative_se": 0.05,
        "four_point_mean_additivity_mean": 0.7 + offset,
        "four_point_mean_additivity_se": 0.01,
        "atteson_bottleneck_margin_mean": 1.1 + offset,
        "atteson_bottleneck_margin_se": 0.1,
        "atteson_theorem_certified_rate": 0.4 + offset,
        "atteson_theorem_certified_rate_se": 0.04,
    }


def _write_summary(path: Path, *, representation: str, engine: str = "direct") -> None:
    path.write_text(
        json.dumps(
            {
                "artifact": "adapter_chain",
                "n_tree_summaries": 2,
                "rows": [
                    {"analysis_engine": engine, "representation": representation},
                    {"analysis_engine": engine, "representation": representation},
                ],
                "aggregate_by_metric": {
                    "l2": _aggregate(0.00),
                    "cosine": _aggregate(0.05),
                    "correlation": _aggregate(0.04),
                },
            }
        )
    )


def _write_registry(path: Path, summaries: list[Path]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "metrics": ["l2", "cosine", "correlation"],
                "conditions": [
                    {
                        "id": f"condition-{index}",
                        "label": f"Condition {index}",
                        "artifact": "adapter_chain",
                        "representation": "lora_cumulative_delta",
                        "summary": summary.name,
                        "expected_trees": 2,
                    }
                    for index, summary in enumerate(summaries, start=1)
                ],
            }
        )
    )


def test_runset_diagnostics_builds_provenance_artifact(tmp_path: Path) -> None:
    summaries = [tmp_path / "one.json", tmp_path / "two.json"]
    for summary in summaries:
        _write_summary(summary, representation="lora_cumulative_delta")
    registry = tmp_path / "registry.yaml"
    _write_registry(registry, summaries)

    rows = weighttraits_runset_diagnostic_rows(registry, base_dir=tmp_path)

    assert len(rows) == 6
    assert {row["metric"] for row in rows} == {"l2", "cosine", "correlation"}
    assert rows[1]["clade_recovery_pct"] == pytest.approx(85.0)
    artifact = tmp_path / "diagnostics.json"
    write_weighttraits_runset_diagnostics_json(rows, artifact, registry=registry)
    payload = json.loads(artifact.read_text())
    assert payload["schema"] == WEIGHTTRAITS_RUNSET_DIAGNOSTICS_SCHEMA
    assert payload["producer"] == "weighttraits"
    assert load_weighttraits_runset_diagnostics_artifact(artifact) == rows


def test_runset_diagnostics_rejects_non_direct_analysis(tmp_path: Path) -> None:
    summary = tmp_path / "legacy.json"
    _write_summary(summary, representation="lora_cumulative_delta", engine="legacy")
    registry = tmp_path / "registry.yaml"
    _write_registry(registry, [summary])

    with pytest.raises(ValueError, match="not native direct analysis"):
        weighttraits_runset_diagnostic_rows(registry, base_dir=tmp_path)


def test_runset_diagnostic_plots(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    summaries = [tmp_path / "one.json", tmp_path / "two.json"]
    for summary in summaries:
        _write_summary(summary, representation="lora_cumulative_delta")
    registry = tmp_path / "registry.yaml"
    _write_registry(registry, summaries)
    rows = weighttraits_runset_diagnostic_rows(registry, base_dir=tmp_path)

    robustness = tmp_path / "robustness.svg"
    additivity = tmp_path / "additivity.svg"
    plot_weighttraits_metric_robustness(rows, robustness)
    plot_weighttraits_additivity_recovery(rows, additivity)

    assert "Clade recovery" in robustness.read_text()
    assert "Spearman" in additivity.read_text()

