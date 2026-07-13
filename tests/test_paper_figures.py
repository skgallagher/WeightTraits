import json

import pytest

from weighttraits.paper.figures import (
    load_weighttraits_variants_artifact,
    plot_weighttraits_variants_diagnostics,
)
from weighttraits.paper.results import write_weighttraits_variants_table_json


def _row(variant_id: str, label: str, offset: float = 0.0) -> dict:
    return {
        "variant_id": variant_id,
        "label": label,
        "rank_biserial": 0.7 + offset,
        "rank_biserial_se": 0.04,
        "within_run_r": -0.7 + offset,
        "within_run_r_se": 0.05,
        "clade_recovery_pct": 80.0 + 10.0 * offset,
        "clade_recovery_se_pct": 3.0,
        "exact_recovery_pct": 70.0 + 10.0 * offset,
        "exact_recovery_se_pct": 4.0,
        "rf_mean": 1.5 + offset,
        "rf_se": 0.2,
        "fn_mean": 0.3 + offset,
        "fn_se": 0.05,
    }


def test_weighttraits_variants_figure_requires_native_provenance(tmp_path):
    legacy_style = tmp_path / "legacy.json"
    legacy_style.write_text(json.dumps({"n_rows": 1, "rows": [_row("a", "A")]}) + "\n")

    with pytest.raises(ValueError, match="schema"):
        load_weighttraits_variants_artifact(legacy_style)


def test_weighttraits_variants_figure_loads_native_artifact(tmp_path):
    artifact = tmp_path / "variants.json"
    rows = [_row("a", "Variant A"), _row("b", "Variant B", 0.1)]
    write_weighttraits_variants_table_json(rows, artifact, registry="paper/rebuild.yaml")

    loaded = load_weighttraits_variants_artifact(artifact)

    assert [row["variant_id"] for row in loaded] == ["a", "b"]
    assert loaded[0]["clade_recovery_pct"] == 80.0


def test_weighttraits_variants_figure_writes_svg(tmp_path):
    pytest.importorskip("matplotlib")
    artifact = tmp_path / "variants.json"
    figure = tmp_path / "variants.svg"
    write_weighttraits_variants_table_json(
        [_row("a", "Variant A"), _row("b", "Variant B", 0.1)],
        artifact,
    )

    summary = plot_weighttraits_variants_diagnostics(
        load_weighttraits_variants_artifact(artifact),
        figure,
    )

    assert summary["n_variants"] == 2
    assert summary["variant_ids"] == ["a", "b"]
    assert figure.exists()
    text = figure.read_text()
    assert "Clade recovery (%)" in text
    assert "Variant A" in text
