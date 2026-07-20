import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "materialize_lora_recovery_parents.py"
SPEC = importlib.util.spec_from_file_location("materialize_lora_recovery_parents", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(root: Path, index: int, node: str, parent: str) -> dict[str, object]:
    output = Path("outputs") / node
    return {
        "array_index": index,
        "node_id": node,
        "parent_id": parent,
        "init_from": "base/model" if parent == "root" else f"outputs/{parent}/merged",
        "expected_artifacts": {
            "adapter": str(output / "adapter"),
            "merged": str(output / "merged"),
        },
        "job": {"trainer": {"model_task": "causal_lm"}},
    }


def test_materializes_only_missing_ancestors_before_retry(tmp_path: Path, monkeypatch) -> None:
    rows = [
        _row(tmp_path, 0, "n0", "root"),
        _row(tmp_path, 1, "n1", "n0"),
        _row(tmp_path, 2, "n2", "n0"),
        _row(tmp_path, 3, "n3", "n1"),
        _row(tmp_path, 4, "n4", "n2"),
    ]
    run_list = tmp_path / "runs.jsonl"
    run_list.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for node in ("n0", "n1", "n2"):
        (tmp_path / "outputs" / node / "adapter").mkdir(parents=True)
    calls: list[str] = []

    def fake_materialize(row: dict[str, object], root: Path) -> None:
        node = str(row["node_id"])
        calls.append(node)
        (root / "outputs" / node / "merged").mkdir(parents=True)

    monkeypatch.setattr(MODULE, "_materialize", fake_materialize)
    materialized = MODULE.materialize_recovery_parents(run_list, 3, tmp_path)

    assert materialized == ["n0", "n1", "n2"]
    assert calls == materialized


def test_recovery_fails_if_a_required_adapter_is_missing(tmp_path: Path) -> None:
    rows = [_row(tmp_path, 0, "n0", "root"), _row(tmp_path, 1, "n1", "n0")]
    run_list = tmp_path / "runs.jsonl"
    run_list.write_text("".join(json.dumps(row) + "\n" for row in rows))

    try:
        MODULE.materialize_recovery_parents(run_list, 1, tmp_path)
    except FileNotFoundError as exc:
        assert "missing retained adapter for n0" in str(exc)
    else:
        raise AssertionError("expected a missing retained adapter to stop recovery")
