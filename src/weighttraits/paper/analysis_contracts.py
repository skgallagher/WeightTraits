"""Shared fail-closed helpers for corrected paper-analysis contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


TREE_PREFIX = "confirm_paper_tree_"
ALL_TREE_IDS = tuple(f"{TREE_PREFIX}{index:03d}" for index in range(1, 51))
TOPOLOGY_EXCLUDED_TREE_IDS = frozenset(
    f"{TREE_PREFIX}{index:03d}" for index in (15, 20, 37, 47)
)
TOPOLOGY_TREE_IDS = tuple(
    tree_id for tree_id in ALL_TREE_IDS if tree_id not in TOPOLOGY_EXCLUDED_TREE_IDS
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TREE_ID_PATTERN = re.compile(
    r"^(?:confirm_paper_tree_|tree_|run_)?(?P<index>[0-9]{1,3})$"
)


def sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible object using a stable canonical encoding."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_tree_id(value: object) -> str:
    """Normalize a deliberately small set of controlled-tree ID spellings."""

    text = str(value).strip()
    match = _TREE_ID_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid controlled-tree ID: {value!r}")
    index = int(match.group("index"))
    if index < 1 or index > 50:
        raise ValueError(f"controlled-tree ID is outside 001..050: {value!r}")
    return f"{TREE_PREFIX}{index:03d}"


def required_text(mapping: Mapping[str, Any], key: str, *, context: str) -> str:
    """Read one required nonempty string field."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires nonempty text field {key!r}")
    return value.strip()


def resolve_path(raw: object, *, base: Path) -> Path:
    """Resolve a path relative to a contract base directory."""

    path = Path(str(raw))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def pinned_file(
    raw: object,
    *,
    label: str,
    base: Path,
) -> tuple[Path, str]:
    """Resolve a ``{path, sha256}`` input and verify its digest."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a mapping with path and sha256")
    path = resolve_path(required_text(raw, "path", context=label), base=base)
    digest = required_text(raw, "sha256", context=label)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} sha256 must be 64 lowercase hexadecimal characters")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != digest:
        raise ValueError(
            f"{label} sha256 mismatch: declared {digest}, observed {actual}: {path}"
        )
    return path, digest


def load_truth_specs(
    raw_specs: object,
    *,
    base: Path,
    expected_tree_ids: tuple[str, ...] = TOPOLOGY_TREE_IDS,
) -> dict[str, dict[str, Any]]:
    """Load an exact, pinned truth-manifest map and validate its tree IDs."""

    if not isinstance(raw_specs, Mapping):
        raise ValueError("contract requires a truth_manifests mapping")
    expected = set(expected_tree_ids)
    observed = {canonical_tree_id(tree_id) for tree_id in raw_specs}
    if len(observed) != len(raw_specs):
        raise ValueError("truth_manifests contains duplicate canonical tree IDs")
    if observed != expected:
        raise ValueError(
            "truth_manifests must declare exactly the canonical topology-eligible "
            f"46 tree IDs; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    out: dict[str, dict[str, Any]] = {}
    for raw_tree_id, raw in raw_specs.items():
        tree_id = canonical_tree_id(raw_tree_id)
        path, digest = pinned_file(raw, label=f"truth manifest {tree_id}", base=base)
        _validate_manifest_tree_id(path, expected_tree_id=tree_id)
        out[tree_id] = {"path": path, "sha256": digest}
    return {tree_id: out[tree_id] for tree_id in expected_tree_ids}


def truth_hashes_sha256(specs: Mapping[str, Mapping[str, Any]]) -> str:
    """Return the canonical digest of a truth tree-ID-to-file-hash map."""

    return canonical_sha256(
        {tree_id: str(specs[tree_id]["sha256"]) for tree_id in sorted(specs)}
    )


def validate_strict_training_completion(
    receipt: object,
    *,
    cohort_id: str | None = None,
    artifact_factor: int | None = None,
) -> None:
    """Require the full 50-tree/641-node strict completion contract.

    Full-fine-tuning receipts contain two required artifacts per run.  The
    controlled LoRA receipts contain four (adapter plus merged-model evidence).
    A derived cohort-binding receipt may declare ``artifact_factor`` explicitly;
    callers validating an immutable source receipt can instead pass the expected
    factor.  The default remains the historical full-fine-tuning factor of two.
    """

    if not isinstance(receipt, Mapping):
        raise ValueError("training completion receipt must be a JSON object")
    if receipt.get("valid") is not True:
        raise ValueError("training completion receipt is not explicitly valid")
    declared = receipt.get("cohort_id")
    if cohort_id is not None and declared != cohort_id:
        raise ValueError(
            f"completion receipt cohort_id {declared!r} differs from {cohort_id!r}"
        )
    declared_factor = receipt.get("artifact_factor", 2)
    expected_factor = declared_factor if artifact_factor is None else artifact_factor
    if (
        isinstance(expected_factor, bool)
        or not isinstance(expected_factor, int)
        or expected_factor not in {2, 4}
    ):
        raise ValueError("training completion artifact factor must be exactly 2 or 4")
    if "artifact_factor" in receipt and declared_factor != expected_factor:
        raise ValueError(
            "completion receipt artifact_factor "
            f"{declared_factor!r} differs from {expected_factor!r}"
        )
    expected = {
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
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(
                f"completion receipt requires {field}={value}, got {receipt.get(field)!r}"
            )
    rows = receipt.get("rows")
    if not isinstance(rows, list) or len(rows) != 50:
        raise ValueError("completion receipt must contain exactly 50 per-tree rows")
    tree_ids: set[str] = set()
    n_runs = 0
    n_artifacts = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"completion receipt row {index} is not an object")
        tree_id = canonical_tree_id(row.get("tree_id"))
        if tree_id in tree_ids:
            raise ValueError(f"completion receipt duplicates tree {tree_id}")
        tree_ids.add(tree_id)
        run_count = row.get("n_runs")
        if isinstance(run_count, bool) or not isinstance(run_count, int) or run_count < 1:
            raise ValueError(f"completion receipt {tree_id} has invalid n_runs")
        required_equal = {
            "n_ledger_nodes": run_count,
            "n_terminal_nodes": run_count,
            "n_ok_nodes": run_count,
            "n_failed_nodes": 0,
            "n_missing_nodes": 0,
            "n_errors": 0,
        }
        for field, value in required_equal.items():
            if row.get(field) != value:
                raise ValueError(
                    f"completion receipt {tree_id} requires {field}={value}, "
                    f"got {row.get(field)!r}"
                )
        if row.get("valid") is not True or row.get("ready_for_analysis") is not True:
            raise ValueError(f"completion receipt {tree_id} is not ready for analysis")
        if row.get("status_counts") != {"completed": run_count}:
            raise ValueError(
                f"completion receipt {tree_id} does not contain only completed nodes"
            )
        expected_artifacts = row.get("n_expected_artifacts")
        existing_artifacts = row.get("n_existing_artifacts")
        if (
            expected_artifacts != expected_factor * run_count
            or existing_artifacts != expected_artifacts
        ):
            raise ValueError(
                f"completion receipt {tree_id} artifact counts are not exactly "
                f"{expected_factor}*n_runs"
            )
        n_runs += run_count
        n_artifacts += expected_artifacts
    expected_total_artifacts = expected_factor * 641
    if (
        tree_ids != set(ALL_TREE_IDS)
        or n_runs != 641
        or n_artifacts != expected_total_artifacts
    ):
        raise ValueError(
            "completion receipt per-tree totals must be exact: 50 trees, 641 runs, "
            f"{expected_total_artifacts} artifacts"
        )


def _validate_manifest_tree_id(path: Path, *, expected_tree_id: str) -> None:
    seen = set()
    n_rows = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            n_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in truth manifest {path}:{line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"truth manifest row must be a mapping: {path}:{line_number}"
                )
            tree_id = row.get("tree_id")
            if tree_id is not None:
                seen.add(canonical_tree_id(tree_id))
    if n_rows == 0:
        raise ValueError(f"truth manifest is empty: {path}")
    if seen and seen != {expected_tree_id}:
        raise ValueError(
            f"truth manifest {path} has tree IDs {sorted(seen)}, "
            f"expected only {expected_tree_id}"
        )
