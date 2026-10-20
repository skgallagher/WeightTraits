"""Strict JSONL decoding with literal-LF framing.

Python ``str.splitlines()`` treats Unicode separators such as U+2028 as
record boundaries.  JSONL is LF framed, so decoding must split the raw bytes
only on ``b"\n"`` and leave every other Unicode code point inside its JSON
string.
"""

from __future__ import annotations

import json
from typing import Any


def decode_lf_json_objects(payload: bytes, label: str) -> list[dict[str, Any]]:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError(f"{label} is empty")
    if not payload.endswith(b"\n"):
        raise ValueError(f"{label} lacks the required terminal LF")
    raw_rows = payload[:-1].split(b"\n")
    if not raw_rows or any(not row for row in raw_rows):
        raise ValueError(f"{label} contains a blank LF-framed row")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(raw_rows, start=1):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{label} has invalid UTF-8 at LF record {line_number}"
            ) from exc
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label} has invalid JSON at LF record {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} LF record {line_number} is not an object")
        rows.append(value)
    return rows


__all__ = ["decode_lf_json_objects"]
