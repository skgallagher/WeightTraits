from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from weighttraits.paper.figures import (
    load_weighttraits_paired_comparisons_artifact,
    plot_weighttraits_paired_effects,
)
from weighttraits.paper.paired import (
    WEIGHTTRAITS_PAIRED_COMPARISONS_SCHEMA,
    weighttraits_paired_comparison_rows,
    write_weighttraits_paired_comparisons_json,
)


def _write_summary(
    path: Path,
    values: dict[str, list[float]],
    *,
    engine: str = "direct",
) -> None:
    rows = []
    for metric, metric_values in values.items():
        for index, value in enumerate(metric_values, start=1):
            rows.append(
                {
                    "analysis_engine": engine,
                    "artifact": "adapter_chain",
                    "representation": "lora_cumulative_delta",
                    "tree_id": f"tree-{index}",
                    "metric": metric,
                    "clade_recovery": value,
                    "rf": 5.0 - value,
                }
            )
    path.write_text(
        json.dumps(
            {
                "artifact": "adapter_chain",
                "n_tree_summaries": len(next(iter(values.values()))),
                "rows": rows,
            }
        )
    )


def _write_registry(path: Path, left: Path, right: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "n_bootstrap": 200,
                "seed": 7,
                "conditions": [
                    {
                        "id": "left",
                        "summary": left.name,
                        "artifact": "adapter_chain",
                        "representation": "lora_cumulative_delta",
                        "expected_trees": 3,
                    },
                    {
                        "id": "right",
                        "summary": right.name,
                        "artifact": "adapter_chain",
                        "representation": "lora_cumulative_delta",
                        "expected_trees": 3,
                    },
                ],
                "outcomes": [
                    {
                        "id": "clade",
                        "field": "clade_recovery",
                        "label": "Clade recovery",
                        "units": "percentage points",
                        "direction": "higher",
                        "scale": 100,
                    },
                    {
                        "id": "rf",
                        "field": "rf",
                        "label": "RF improvement",
                        "units": "fewer splits",
                        "direction": "lower",
                    },
                ],
                "comparisons": [
                    {
                        "id": "left_vs_right",
                        "group": "scope",
                        "label": "Left vs right",
                        "left": {"condition": "left", "metric": "cosine"},
                        "right": {"condition": "right", "metric": "cosine"},
                    }
                ],
            }
        )
    )


def test_paired_comparison_effects_and_provenance(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_summary(left, {"cosine": [0.8, 0.9, 1.0]})
    _write_summary(right, {"cosine": [0.7, 0.8, 0.9]})
    registry = tmp_path / "registry.yaml"
    _write_registry(registry, left, right)

    rows = weighttraits_paired_comparison_rows(registry, base_dir=tmp_path)

    assert len(rows) == 2
    assert rows[0]["mean_effect"] == pytest.approx(10.0)
    assert rows[0]["wins"] == 3
    assert rows[0]["sign_test_p"] == pytest.approx(0.25)
    assert rows[1]["mean_effect"] == pytest.approx(0.1)
    assert rows == weighttraits_paired_comparison_rows(registry, base_dir=tmp_path)

    artifact = tmp_path / "paired.json"
    write_weighttraits_paired_comparisons_json(rows, artifact, registry=registry)
    payload = json.loads(artifact.read_text())
    assert payload["schema"] == WEIGHTTRAITS_PAIRED_COMPARISONS_SCHEMA
    assert payload["producer"] == "weighttraits"
    assert load_weighttraits_paired_comparisons_artifact(artifact) == rows


def test_paired_comparison_rejects_non_direct_summary(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_summary(left, {"cosine": [0.8, 0.9, 1.0]}, engine="legacy")
    _write_summary(right, {"cosine": [0.7, 0.8, 0.9]})
    registry = tmp_path / "registry.yaml"
    _write_registry(registry, left, right)

    with pytest.raises(ValueError, match="not native direct analysis"):
        weighttraits_paired_comparison_rows(registry, base_dir=tmp_path)


def test_paired_comparison_rejects_mismatched_topologies(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_summary(left, {"cosine": [0.8, 0.9, 1.0]})
    _write_summary(right, {"cosine": [0.7, 0.8, 0.9]})
    payload = json.loads(right.read_text())
    payload["rows"][-1]["tree_id"] = "other-tree"
    right.write_text(json.dumps(payload))
    registry = tmp_path / "registry.yaml"
    _write_registry(registry, left, right)

    with pytest.raises(ValueError, match="mismatched topology IDs"):
        weighttraits_paired_comparison_rows(registry, base_dir=tmp_path)


def test_paired_effect_plot(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_summary(left, {"cosine": [0.8, 0.9, 1.0]})
    _write_summary(right, {"cosine": [0.7, 0.8, 0.9]})
    registry = tmp_path / "registry.yaml"
    _write_registry(registry, left, right)
    rows = weighttraits_paired_comparison_rows(registry, base_dir=tmp_path)

    out = tmp_path / "paired.svg"
    plot_weighttraits_paired_effects(rows, out, group="scope")

    assert "Clade recovery" in out.read_text()
    assert "Left vs right" in out.read_text()

