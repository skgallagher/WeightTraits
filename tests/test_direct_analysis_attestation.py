from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("Bio")
pytest.importorskip("dendropy")
safetensors_np = pytest.importorskip("safetensors.numpy")

from weighttraits.analysis.direct_attestation import (  # noqa: E402
    DIRECT_ANALYSIS_RECEIPT_NAME,
    produce_attested_direct_analysis,
    validate_pinned_direct_replay_receipt,
    verify_direct_analysis_receipt,
    write_direct_analysis_replay_receipt,
)
from weighttraits.paper.analysis_contracts import ALL_TREE_IDS, canonical_sha256  # noqa: E402
from weighttraits.training.ledger import (  # noqa: E402
    TrainingLedgerEvent,
    append_ledger_event,
)


COHORT_ID = "attested_direct_test_cohort"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_counts() -> dict[str, int]:
    return {tree_id: 12 if index < 9 else 13 for index, tree_id in enumerate(ALL_TREE_IDS)}


def _strict_completion() -> dict[str, Any]:
    rows = []
    for tree_id, count in _run_counts().items():
        rows.append(
            {
                "tree_id": tree_id,
                "valid": True,
                "ready_for_analysis": True,
                "n_runs": count,
                "n_ledger_nodes": count,
                "n_terminal_nodes": count,
                "n_ok_nodes": count,
                "n_failed_nodes": 0,
                "n_missing_nodes": 0,
                "n_errors": 0,
                "n_expected_artifacts": 2 * count,
                "n_existing_artifacts": 2 * count,
                "status_counts": {"completed": count},
            }
        )
    return {
        "cohort_id": COHORT_ID,
        "valid": True,
        "n_trees": 50,
        "n_ready": 50,
        "n_failed": 0,
        "n_in_progress": 0,
        "n_not_started": 0,
        "n_total_runs": 641,
        "n_terminal_nodes": 641,
        "n_ok_nodes": 641,
        "n_failed_nodes": 0,
        "n_missing_nodes": 0,
        "n_errors": 0,
        "n_warnings": 0,
        "rows": rows,
    }


def _case(tmp_path: Path) -> dict[str, Path]:
    tree_id = ALL_TREE_IDS[0]
    truth = tmp_path / "truth" / f"{tree_id}.manifest.jsonl"
    truth.parent.mkdir(parents=True)
    truth_rows = [
        {
            "tree_id": tree_id,
            "node_id": "n0",
            "parent_id": "root",
            "path": ["root", "n0"],
            "grow": "train",
        },
        {
            "tree_id": tree_id,
            "node_id": "n1",
            "parent_id": "root",
            "path": ["root", "n1"],
            "grow": "train",
        },
        {
            "tree_id": tree_id,
            "node_id": "n2",
            "parent_id": "n0",
            "path": ["root", "n0", "n2"],
            "grow": "train",
        },
        {
            "tree_id": tree_id,
            "node_id": "n3",
            "parent_id": "n0",
            "path": ["root", "n0", "n3"],
            "grow": "train",
        },
        {
            "tree_id": tree_id,
            "node_id": "n4",
            "parent_id": "n1",
            "path": ["root", "n1", "n4"],
            "grow": "train",
        },
        {
            "tree_id": tree_id,
            "node_id": "n5",
            "parent_id": "n1",
            "path": ["root", "n1", "n5"],
            "grow": "train",
        },
    ]
    truth.write_text("\n".join(json.dumps(row) for row in truth_rows) + "\n")

    ledger = tmp_path / "ledgers" / f"{tree_id}.training_ledger.jsonl"
    run_list = tmp_path / "run_lists" / f"{tree_id}.runs.jsonl"
    run_list.parent.mkdir(parents=True)
    node_ids = ["n2", "n3", "n4", "n5", *(f"x{index}" for index in range(8))]
    values = {"n2": 0.0, "n3": 1.0, "n4": 10.0, "n5": 11.0}
    rows = []
    for index, node_id in enumerate(node_ids):
        value = values.get(node_id, 20.0 + index)
        model_dir = tmp_path / "checkpoints" / tree_id / node_id / "model"
        model_dir.mkdir(parents=True)
        safetensors_np.save_file(
            {
                "layer0.weight": np.array([value], dtype=np.float32),
                "layer1.weight": np.array([2.0 * value], dtype=np.float32),
            },
            str(model_dir / "model.safetensors"),
        )
        training_log = model_dir.parent / "training_log.jsonl"
        training_log.write_text(json.dumps({"step": 1, "train_loss": value}) + "\n")
        relative_model = model_dir.relative_to(tmp_path).as_posix()
        relative_log = training_log.relative_to(tmp_path).as_posix()
        rows.append(
            {
                "array_index": index,
                "run_id": f"{tree_id}-{node_id}",
                "node_id": node_id,
                "parent_id": "root",
                "depth": 1,
                "method": "full",
                "dataset_id": "squad",
                "task_family": "qa",
                "init_from": "base",
                "output_dir": model_dir.parent.relative_to(tmp_path).as_posix(),
                "expected_artifacts": {
                    "model": relative_model,
                    "training_log": relative_log,
                },
                "ledger_path": ledger.relative_to(tmp_path).as_posix(),
                "runner": {},
                "job": {},
            }
        )
        append_ledger_event(
            ledger,
            TrainingLedgerEvent(
                node_id=node_id,
                status="completed",
                extra={"artifacts": {"model": relative_model}},
            ),
        )
    run_list.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")

    contract_files = {}
    for name in ("assignment_summary", "formats", "registry"):
        path = tmp_path / "contracts" / f"{name}.json"
        _write_json(path, {"name": name})
        contract_files[name] = path
    config = tmp_path / "contracts" / f"{COHORT_ID}.yaml"
    config.write_text("model: synthetic\n")
    contract_files["config"] = config
    report = tmp_path / "reports" / f"{tree_id}.json"
    _write_json(report, {"valid": True})

    selected_row = {
        "tree_id": tree_id,
        "valid": True,
        "n_runs": 12,
        "n_errors": 0,
        "n_warnings": 0,
        "ledger": ledger.relative_to(tmp_path).as_posix(),
        "manifest": truth.relative_to(tmp_path).as_posix(),
        "manifest_sha256": _sha(truth),
        "run_list": run_list.relative_to(tmp_path).as_posix(),
        "run_list_sha256": _sha(run_list),
        "report": report.relative_to(tmp_path).as_posix(),
        "report_sha256": _sha(report),
    }
    summary_rows = []
    for current_id, count in _run_counts().items():
        summary_rows.append(
            selected_row
            if current_id == tree_id
            else {
                **selected_row,
                "tree_id": current_id,
                "n_runs": count,
            }
        )
    summary = tmp_path / "training_summary.json"
    summary_payload = {
        "assignment_summary": contract_files["assignment_summary"].relative_to(
            tmp_path
        ).as_posix(),
        "assignment_summary_sha256": _sha(contract_files["assignment_summary"]),
        "config": config.relative_to(tmp_path).as_posix(),
        "config_sha256": _sha(config),
        "formats": contract_files["formats"].relative_to(tmp_path).as_posix(),
        "formats_sha256": _sha(contract_files["formats"]),
        "registry": contract_files["registry"].relative_to(tmp_path).as_posix(),
        "registry_sha256": _sha(contract_files["registry"]),
        "n_trees": 50,
        "n_runs": 641,
        "n_errors": 0,
        "n_warnings": 0,
        "trees": summary_rows,
    }
    _write_json(summary, summary_payload)
    completion = tmp_path / "completion.json"
    _write_json(completion, _strict_completion())
    stage = tmp_path / "stage_manifest.json"
    _write_json(stage, {"valid": True, "sealed": True})
    return {
        "summary": summary,
        "completion": completion,
        "stage": stage,
        "ledger": ledger,
        "truth": truth,
        "run_list": run_list,
        "checkpoint": tmp_path
        / "checkpoints"
        / tree_id
        / "n2"
        / "model"
        / "model.safetensors",
    }


def _produce(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    case = _case(tmp_path)
    out = tmp_path / "analysis"
    produce_attested_direct_analysis(
        training_summary_path=case["summary"],
        completion_receipt_path=case["completion"],
        stage_manifest_path=case["stage"],
        cohort_id=COHORT_ID,
        tree_id=ALL_TREE_IDS[0],
        out_dir=out,
        artifact="model",
        metrics=["l2"],
        path_base=tmp_path,
        chunk_size=1,
    )
    return case, out / DIRECT_ANALYSIS_RECEIPT_NAME


def test_attested_direct_analysis_deep_replay_and_consumer_contract(tmp_path: Path) -> None:
    _, receipt = _produce(tmp_path)
    receipt_digest = _sha(receipt)
    verified = verify_direct_analysis_receipt(
        receipt,
        expected_sha256=receipt_digest,
        replay=True,
    )
    assert verified["replayed"] is True
    assert verified["exact"] is True
    assert verified["comparisons"] == [
        {
            "metric": "l2",
            "shape": [2, 4, 4],
            "n_layers": 2,
            "n_models": 4,
            "n_values": 32,
            "matches": True,
            "max_abs_error": 0.0,
        }
    ]

    replay = tmp_path / "receipts" / "tree001.direct-replay.json"
    write_direct_analysis_replay_receipt(
        receipt,
        replay,
        expected_direct_sha256=receipt_digest,
    )
    consumed = validate_pinned_direct_replay_receipt(
        {"path": str(replay), "sha256": _sha(replay)},
        expected_cohort_id=COHORT_ID,
        expected_tree_id=ALL_TREE_IDS[0],
        expected_artifact="model",
        expected_metric="l2",
    )
    assert consumed["tree_id"] == ALL_TREE_IDS[0]
    direct_payload = consumed["direct_payload"]
    assert direct_payload["layer_names"] == ["layer0.weight", "layer1.weight"]
    assert {row["relative_path"] for row in direct_payload["outputs"]} == {
        path.relative_to(receipt.parent).as_posix()
        for path in receipt.parent.iterdir()
        if path.is_file() and path != receipt
    }


def test_replay_rejects_selective_cube_forgery_that_preserves_full_mean(
    tmp_path: Path,
) -> None:
    _, receipt = _produce(tmp_path)
    cube_path = receipt.parent / "direct_distance_layers.npz"
    matrix_before = np.load(receipt.parent / "distance_matrix_l2.npy", allow_pickle=False)
    with np.load(cube_path, allow_pickle=False) as archive:
        cube = np.asarray(archive["l2"]).copy()
    cube[0, 0, 1] += 1.0
    cube[0, 1, 0] += 1.0
    cube[1, 0, 1] -= 1.0
    cube[1, 1, 0] -= 1.0
    np.savez_compressed(cube_path, l2=cube)
    assert np.array_equal(cube.mean(axis=0), matrix_before)

    payload = json.loads(receipt.read_text())
    for row in payload["outputs"]:
        if row["relative_path"] == "direct_distance_layers.npz":
            row["size"] = cube_path.stat().st_size
            row["sha256"] = _sha(cube_path)
    payload["outputs_sha256"] = canonical_sha256(payload["outputs"])
    _write_json(receipt, payload)

    with pytest.raises(ValueError, match="direct replay mismatch for l2"):
        verify_direct_analysis_receipt(receipt, replay=True)


def test_pinned_replay_rejects_post_replay_direct_receipt_rewrite(tmp_path: Path) -> None:
    _, receipt = _produce(tmp_path)
    replay = tmp_path / "receipts" / "tree001.direct-replay.json"
    write_direct_analysis_replay_receipt(receipt, replay)
    replay_pin = {"path": str(replay), "sha256": _sha(replay)}

    direct_payload = json.loads(receipt.read_text())
    direct_payload["runtime"]["python"] = "forged"
    _write_json(receipt, direct_payload)
    with pytest.raises(ValueError, match="direct-analysis receipt sha256 mismatch"):
        validate_pinned_direct_replay_receipt(replay_pin)


def test_deep_verifier_rejects_checkpoint_drift(tmp_path: Path) -> None:
    case, receipt = _produce(tmp_path)
    case["checkpoint"].write_bytes(case["checkpoint"].read_bytes() + b"drift")
    with pytest.raises(ValueError, match="checkpoint .* no longer matches"):
        verify_direct_analysis_receipt(receipt)


def test_producer_rejects_non_strict_completion_before_analysis(tmp_path: Path) -> None:
    case = _case(tmp_path)
    completion = json.loads(case["completion"].read_text())
    completion["n_ready"] = 49
    _write_json(case["completion"], completion)
    with pytest.raises(ValueError, match="completion receipt requires n_ready=50"):
        produce_attested_direct_analysis(
            training_summary_path=case["summary"],
            completion_receipt_path=case["completion"],
            stage_manifest_path=case["stage"],
            cohort_id=COHORT_ID,
            tree_id=ALL_TREE_IDS[0],
            out_dir=tmp_path / "analysis",
            artifact="model",
            metrics=["l2"],
            path_base=tmp_path,
        )

