import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_rematerialized_lora_tree.py"
SPEC = importlib.util.spec_from_file_location("merged_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_leaf_chains_are_derived_in_root_to_leaf_order(tmp_path):
    def row(node, parent, path):
        adapter = tmp_path / path
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        return {
            "node_id": node,
            "parent_id": parent,
            "job": {"base_model": "base/model", "base_model_revision": "abc123"},
            "expected_artifacts": {"adapter": path},
        }

    rows = [
        row("n0", "root", "outputs/n0/adapter"),
        row("n1", "n0", "outputs/n1/adapter"),
        row("n2", "n0", "outputs/n2/adapter"),
    ]
    chains, base, revision = MODULE._leaf_chains(rows, tmp_path)

    assert list(chains) == ["n1", "n2"]
    assert chains["n1"] == [
        tmp_path / "outputs/n0/adapter",
        tmp_path / "outputs/n1/adapter",
    ]
    assert base == "base/model"
    assert revision == "abc123"


def test_leaf_chains_accept_revision_override_for_historical_run_list(tmp_path):
    def row(node, parent, path):
        adapter = tmp_path / path
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        return {
            "node_id": node,
            "parent_id": parent,
            "job": {"base_model": "base/model"},
            "expected_artifacts": {"adapter": path},
        }

    rows = [
        row("n0", "root", "outputs/n0/adapter"),
        row("n1", "n0", "outputs/n1/adapter"),
    ]
    chains, base, revision = MODULE._leaf_chains(
        rows,
        tmp_path,
        revision_override="immutable-snapshot",
    )

    assert list(chains) == ["n1"]
    assert base == "base/model"
    assert revision == "immutable-snapshot"
