"""Backend-agnostic execution planning for all-node behavior inference.

No model library is imported and no file is read or written.  Fresh execution
returns response and terminal-receipt bytes in explicit publish order so a
caller can perform exclusive atomic writes.  Cache validation consumes bytes
supplied by the caller and returns no write plan.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Callable, Mapping, Sequence

from .checkpoints import ArtifactPin, INTERNAL_NODE_TREATMENT, POPULATION
from .jsonl import decode_lf_json_objects
from .inference import (
    AllNodeInferenceRequest,
    GenerationGroup,
    ResponseCoordinate,
    ResponseRow,
    build_terminal_response_receipt,
    read_inference_request,
    validate_terminal_response_receipt,
)


DECODED_RESPONSE_SCHEMA = "weighttraits.all_node_decoded_response.v1"


GenerateBatchCallback = Callable[
    [Sequence[object], GenerationGroup, AllNodeInferenceRequest], Sequence[str]
]


@dataclass(frozen=True)
class PlannedAtomicWrite:
    """Bytes for one exclusive atomic publish step."""

    path: str
    payload: bytes
    publish_order: int
    terminal: bool

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


@dataclass(frozen=True)
class RuntimeExecutionResult:
    """Either a validated cache hit or a complete two-file write plan."""

    cache_hit: bool
    request: AllNodeInferenceRequest
    response_rows: tuple[ResponseRow, ...]
    receipt: Mapping[str, Any]
    writes: tuple[PlannedAtomicWrite, ...]


def execute_all_node_inference(
    request: AllNodeInferenceRequest,
    *,
    prompt_payloads: Mapping[ResponseCoordinate, object],
    generate_batch: GenerateBatchCallback | None,
    observed_backend_provenance: Mapping[str, Any] | None,
    output_path: str,
    receipt_path: str,
    existing_output: bytes | None = None,
    existing_receipt: bytes | None = None,
) -> RuntimeExecutionResult:
    """Execute one exact frozen grid without performing filesystem writes.

    Existing output is accepted only when both artifacts form an exact valid
    terminal cache for ``request``.  Callback failures propagate before any
    publishable write plan or terminal receipt is returned.
    """

    verified_request = read_inference_request(request.to_dict())
    _reject_leaf_ordinal(verified_request.to_dict())
    output = _absolute_path(output_path, "output_path")
    receipt_file = _absolute_path(receipt_path, "receipt_path")
    if output == receipt_file:
        raise ValueError("output_path and receipt_path must be different")
    if (existing_output is None) != (existing_receipt is None):
        raise FileExistsError(
            "existing output is refused without its exact terminal receipt/cache"
        )
    if existing_output is not None and existing_receipt is not None:
        return _validate_cache(
            verified_request,
            output_path=output,
            output_bytes=_bytes(existing_output, "existing_output"),
            receipt_bytes=_bytes(existing_receipt, "existing_receipt"),
        )
    if not callable(generate_batch):
        raise ValueError("fresh inference requires callable generate_batch")
    if not isinstance(observed_backend_provenance, Mapping):
        raise ValueError("fresh inference requires observed_backend_provenance")

    coordinates = verified_request.coordinates()
    payloads = _validated_prompt_payloads(prompt_payloads, coordinates=coordinates)
    generated: dict[ResponseCoordinate, str] = {}
    for group, group_coordinates in verified_request.grouped_coordinates():
        batch_payloads = [payloads[coordinate] for coordinate in group_coordinates]
        # The callback is invoked exactly once per protocol/probe/draw group.
        # Its bound generation_options carry seed_scope='probe_draw', so the
        # existing HuggingFace backend seeds once before internal minibatches.
        texts = generate_batch(batch_payloads, group, verified_request)
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("generate_batch must return a sequence of text responses")
        if len(texts) != len(group_coordinates):
            raise ValueError(
                f"generate_batch returned {len(texts)} rows for group "
                f"{group.protocol_id}/{group.probe_id}/draw{group.sample_id}, "
                f"expected {len(group_coordinates)}"
            )
        for coordinate, text in zip(group_coordinates, texts):
            if not isinstance(text, str):
                raise TypeError(
                    "generate_batch must return text at every coordinate; "
                    f"got {type(text).__name__} for {coordinate!r}"
                )
            if coordinate in generated:
                raise AssertionError(f"generation groups duplicate coordinate {coordinate!r}")
            generated[coordinate] = text
    if set(generated) != set(coordinates):
        raise AssertionError("generation groups did not populate the exact response grid")

    rows: list[ResponseRow] = []
    records: list[dict[str, Any]] = []
    for coordinate in coordinates:
        # Nothing is serialized or returned until every batch succeeds.
        text = generated[coordinate]
        response_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = ResponseRow(
            request_sha256=verified_request.request_sha256,
            cohort_id=verified_request.cohort_id,
            tree_id=verified_request.tree_id,
            node_id=verified_request.node_id,
            protocol_id=coordinate.protocol_id,
            probe_id=coordinate.probe_id,
            prompt_id=coordinate.prompt_id,
            sample_id=coordinate.sample_id,
            response_sha256=response_sha,
            completed=True,
        )
        rows.append(row)
        records.append(_decoded_record(verified_request, row=row, text=text))

    output_bytes = _jsonl_bytes(records)
    response_pin = ArtifactPin(output, hashlib.sha256(output_bytes).hexdigest())
    receipt = build_terminal_response_receipt(
        verified_request,
        rows,
        responses=response_pin,
        observed_backend_provenance=observed_backend_provenance,
    )
    receipt_bytes = _json_bytes(receipt)
    # Revalidate what will be published, including the independently measured
    # response bytes, before exposing either write step.
    validate_terminal_response_receipt(
        receipt,
        expected_request=verified_request,
        observed_responses=response_pin,
    )
    writes = (
        PlannedAtomicWrite(output, output_bytes, publish_order=0, terminal=False),
        PlannedAtomicWrite(receipt_file, receipt_bytes, publish_order=1, terminal=True),
    )
    return RuntimeExecutionResult(
        cache_hit=False,
        request=verified_request,
        response_rows=tuple(rows),
        receipt=receipt,
        writes=writes,
    )


def _validate_cache(
    request: AllNodeInferenceRequest,
    *,
    output_path: str,
    output_bytes: bytes,
    receipt_bytes: bytes,
) -> RuntimeExecutionResult:
    receipt = _json_object(receipt_bytes, "existing_receipt")
    response_pin = ArtifactPin(
        output_path,
        hashlib.sha256(output_bytes).hexdigest(),
    )
    validated_receipt = validate_terminal_response_receipt(
        receipt,
        expected_request=request,
        observed_responses=response_pin,
    )
    records = _decoded_records(output_bytes)
    raw_rows = validated_receipt.get("response_rows")
    assert isinstance(raw_rows, list)
    if len(records) != len(raw_rows):
        raise ValueError("cached decoded response count differs from terminal receipt")
    rows: list[ResponseRow] = []
    for index, (record, raw_row) in enumerate(zip(records, raw_rows)):
        row = _row_from_receipt(raw_row)
        expected_record = _decoded_record(request, row=row, text=record.get("text"))
        if record != expected_record:
            raise ValueError(f"cached decoded response drift at row {index}")
        rows.append(row)
    return RuntimeExecutionResult(
        cache_hit=True,
        request=request,
        response_rows=tuple(rows),
        receipt=validated_receipt,
        writes=(),
    )


def _decoded_record(
    request: AllNodeInferenceRequest,
    *,
    row: ResponseRow,
    text: object,
) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("decoded response text must be a string")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != row.response_sha256:
        raise ValueError("decoded response text hash differs from its response row")
    return {
        "schema": DECODED_RESPONSE_SCHEMA,
        "population": POPULATION,
        "internal_node_treatment": INTERNAL_NODE_TREATMENT,
        "cohort_id": request.cohort_id,
        "tree_id": request.tree_id,
        "node_id": request.node_id,
        "node_ordinal": request.node_ordinal,
        "parent_id": request.parent_id,
        "depth": request.depth,
        "node_role": request.node_role,
        "request_sha256": request.request_sha256,
        "response_row": row.to_dict(),
        "text": text,
    }


def _row_from_receipt(value: object) -> ResponseRow:
    if not isinstance(value, Mapping):
        raise ValueError("cached response row must be a mapping")
    required = {
        "schema",
        "request_sha256",
        "cohort_id",
        "tree_id",
        "node_id",
        "protocol_id",
        "probe_id",
        "prompt_id",
        "sample_id",
        "response_sha256",
        "completed",
    }
    if set(value) != required:
        raise ValueError("cached response row keys drift")
    if value.get("schema") != "weighttraits.all_node_response_row.v1":
        raise ValueError("cached response row schema drift")
    row = ResponseRow(
        request_sha256=_text(value["request_sha256"], "request_sha256"),
        cohort_id=_text(value["cohort_id"], "cohort_id"),
        tree_id=_text(value["tree_id"], "tree_id"),
        node_id=_text(value["node_id"], "node_id"),
        protocol_id=_text(value["protocol_id"], "protocol_id"),
        probe_id=_text(value["probe_id"], "probe_id"),
        prompt_id=_text(value["prompt_id"], "prompt_id"),
        sample_id=_integer(value["sample_id"], "sample_id", minimum=0),
        response_sha256=_text(value["response_sha256"], "response_sha256"),
        completed=_boolean(value["completed"], "completed"),
    )
    return row


def _validated_prompt_payloads(
    values: Mapping[ResponseCoordinate, object],
    *,
    coordinates: Sequence[ResponseCoordinate],
) -> dict[ResponseCoordinate, object]:
    if not isinstance(values, Mapping):
        raise ValueError("prompt_payloads must be a mapping keyed by ResponseCoordinate")
    expected = set(coordinates)
    if any(not isinstance(key, ResponseCoordinate) for key in values):
        raise ValueError("prompt_payloads keys must be ResponseCoordinate values")
    observed = set(values)
    if observed != expected:
        raise ValueError(
            "prompt payload coordinates differ from the frozen grid: "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )
    return dict(values)


def _decoded_records(payload: bytes) -> list[dict[str, Any]]:
    rows = decode_lf_json_objects(payload, "cached response artifact")
    for value in rows:
        _reject_leaf_ordinal(value)
    return rows


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not one UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    _reject_leaf_ordinal(value)
    return value


def _reject_leaf_ordinal(value: object) -> None:
    if isinstance(value, Mapping):
        if "leaf_ordinal" in value:
            raise ValueError("all-node runtime forbids leaf_ordinal")
        for child in value.values():
            _reject_leaf_ordinal(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_leaf_ordinal(child)


def _absolute_path(value: object, label: str) -> str:
    path = _text(value, label)
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


def _bytes(value: object, label: str) -> bytes:
    if not isinstance(value, bytes):
        raise ValueError(f"{label} must be bytes")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty string without surrounding whitespace")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value
