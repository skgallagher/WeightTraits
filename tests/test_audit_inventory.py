from pathlib import Path

from weighttraits.audit.ellmtrees import inventory_ellmtrees


def test_inventory_counts_expected_sections(tmp_path: Path):
    root = tmp_path / "ELLMTrees"
    (root / "scripts").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "paper" / "figures").mkdir(parents=True)
    (root / "results" / "run_001").mkdir(parents=True)
    (root / "outputs" / "run_001").mkdir(parents=True)
    (root / "scripts" / "run_analysis.py").write_text("print('ok')\n")
    (root / "scripts" / "submit.sh").write_text("#!/usr/bin/env bash\n")
    (root / "tests" / "test_example.py").write_text("def test_ok(): assert True\n")
    (root / "paper" / "paper.tex").write_text("\\documentclass{article}\n")
    (root / "paper" / "figures" / "fig.png").write_bytes(b"not really png")

    report = inventory_ellmtrees(root)

    assert report["script_count"] == 1
    assert report["shell_script_count"] == 1
    assert report["test_count"] == 1
    assert report["paper_tex_exists"] is True
    assert report["paper_figure_count"] == 1
    assert report["n_result_groups"] == 1
    assert report["n_output_groups"] == 1

