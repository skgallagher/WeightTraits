"""Flan output audit (App. E): how often held-out outputs are just a fine-tuning-task label.

Repeated Flan fine-tuning makes many outputs collapse onto a label from an unrelated
fine-tuning task (for example ``entailment`` on an MMLU prompt). Each decoded output is
lowercased and whitespace-normalised, then classified as

* ``valid_choice_label`` - a bare answer-option label (``a``-``d`` or ``1``-``4``), which is a
  legitimate answer on the multiple-choice probes (ARC-Challenge, MMLU);
* ``unrelated_training_label`` - exactly one of the classification labels in
  :data:`UNRELATED_TRAINING_LABELS`;
* ``other`` - anything else.

The label list is the fixed set used for the paper's audit. It is a lower bound on label
collapse: outputs that are other fine-tuning labels (for example ``yes``/``no`` or TREC
categories) are counted as ``other``.

Ported from the audit Codex ran on Wright on 2026-09-21 (``audit_final_flan_mc_categories.py``).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

VALID_CHOICE_LABELS = frozenset({"a", "b", "c", "d", "1", "2", "3", "4"})
UNRELATED_TRAINING_LABELS = frozenset(
    {
        "entailment",
        "not_entailment",
        "positive",
        "negative",
        "neutral",
        "contradiction",
        "acceptable",
        "unacceptable",
        "sci/tech",
        "business",
        "world",
        "sports",
        "unanswerable",
    }
)
CATEGORIES = ("valid_choice_label", "unrelated_training_label", "other")


def normalize_output(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


def categorize_output(text: str | None) -> str:
    normalized = normalize_output(text)
    if normalized in VALID_CHOICE_LABELS:
        return "valid_choice_label"
    if normalized in UNRELATED_TRAINING_LABELS:
        return "unrelated_training_label"
    return "other"


def audit_response_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Count categories per ``probe:population`` for decoded response rows.

    Each row needs ``text``, ``node_role`` (``leaf`` or ``internal``) and
    ``response_row.probe_id``. Populations are ``all_nodes`` and ``leaves``.
    """

    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        probe = row["response_row"]["probe_id"]
        category = categorize_output(row.get("text"))
        counts[(probe, "all_nodes")][category] += 1
        if row["node_role"] == "leaf":
            counts[(probe, "leaves")][category] += 1
    out = {}
    for (probe, population), values in sorted(counts.items()):
        total = sum(values.values())
        out[f"{probe}:{population}"] = {
            "n": total,
            "counts": {name: values.get(name, 0) for name in CATEGORIES},
            "fractions": {name: values.get(name, 0) / total for name in CATEGORIES},
        }
    return out


def iter_response_rows(inference_root: Path) -> Iterable[dict[str, Any]]:
    """Rows of ``<root>/confirm_paper_tree_*/n*/responses.jsonl`` for one cohort."""

    for path in sorted(Path(inference_root).glob("confirm_paper_tree_*/n*/responses.jsonl")):
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def audit_cohort(inference_root: Path) -> dict[str, Any]:
    return {
        "schema": "weighttraits.flan_output_audit.v1",
        "inference_root": str(inference_root),
        "unrelated_training_labels": sorted(UNRELATED_TRAINING_LABELS),
        "valid_choice_labels": sorted(VALID_CHOICE_LABELS),
        "probes": audit_response_rows(iter_response_rows(inference_root)),
    }
