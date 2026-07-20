from __future__ import annotations

import csv
import math
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is not installed")
def test_behavior_meta_analysis_matches_dl_equations(tmp_path: Path) -> None:
    input_path = tmp_path / "pairs.csv"
    rows = []
    outcomes = {
        "tree-a": [0.11, 0.19, 0.28, 0.44, 0.39, 0.62],
        "tree-b": [0.61, 0.49, 0.52, 0.31, 0.27, 0.20],
        "tree-c": [0.18, 0.23, 0.21, 0.51, 0.42, 0.58],
    }
    weights = [0.10, 0.22, 0.31, 0.47, 0.56, 0.71]
    for run_id, behavior in outcomes.items():
        for index, (weight, similarity) in enumerate(zip(weights, behavior, strict=True)):
            rows.append(
                {
                    "run_id": run_id,
                    "pair_id": f"p{index}",
                    "weight_distance": weight,
                    "behavior_similarity": similarity,
                }
            )
    with input_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            "Rscript",
            str(repo / "scripts" / "behavior_meta_analysis.R"),
            "--input",
            str(input_path),
            "--out-dir",
            str(tmp_path),
            "--analysis-id",
            "smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with (tmp_path / "smoke.per_run.csv").open(newline="") as handle:
        studies = list(csv.DictReader(handle))
    with (tmp_path / "smoke.pooling.csv").open(newline="") as handle:
        pooling = {row["estimator"]: row for row in csv.DictReader(handle)}

    z = [float(row["fisher_z"]) for row in studies]
    variance = [float(row["sampling_variance"]) for row in studies]
    fixed_weights = [1 / value for value in variance]
    fixed_z = sum(
        weight * effect for weight, effect in zip(fixed_weights, z, strict=True)
    ) / sum(fixed_weights)
    q = sum(
        weight * (effect - fixed_z) ** 2
        for weight, effect in zip(fixed_weights, z, strict=True)
    )
    c_value = sum(fixed_weights) - sum(weight**2 for weight in fixed_weights) / sum(
        fixed_weights
    )
    tau2 = max(0.0, (q - (len(z) - 1)) / c_value)
    random_weights = [1 / (value + tau2) for value in variance]
    random_z = sum(
        weight * effect for weight, effect in zip(random_weights, z, strict=True)
    ) / sum(random_weights)

    dl = pooling["dersimonian_laird"]
    assert dl["status"] == "ok"
    assert float(dl["estimate_r"]) == pytest.approx(math.tanh(random_z))
    assert float(dl["tau2"]) == pytest.approx(tau2)
    assert float(dl["Q"]) == pytest.approx(q)


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is not installed")
def test_behavior_meta_analysis_withholds_single_tree_pooling(tmp_path: Path) -> None:
    input_path = tmp_path / "pairs.csv"
    input_path.write_text(
        "run_id,weight_distance,behavior_similarity\n"
        "tree-a,0.1,0.3\n"
        "tree-a,0.2,0.1\n"
        "tree-a,0.3,0.4\n"
        "tree-a,0.4,0.2\n"
    )
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            "Rscript",
            str(repo / "scripts" / "behavior_meta_analysis.R"),
            "--input",
            str(input_path),
            "--out-dir",
            str(tmp_path),
            "--analysis-id",
            "one",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with (tmp_path / "one.pooling.csv").open(newline="") as handle:
        pooling = list(csv.DictReader(handle))
    assert {row["status"] for row in pooling} == {"fewer_than_three_usable_runs"}
    assert all(row["estimate_r"] == "NA" for row in pooling)
