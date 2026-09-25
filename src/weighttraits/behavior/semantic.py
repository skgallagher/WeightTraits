"""Strict prompt/draw-paired semantic distance construction.

Every leaf must provide the exact same attested response grid. Pairing is by
``(prompt_id, sample_id)`` rather than list position. Production construction
loads the pinned MiniLM backend internally; injectable encoders are explicit,
non-production test artifacts and cannot be consumed accidentally.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from weighttraits.behavior.direct_bridge import array_values_sha256, ordered_strings_sha256
from weighttraits.verification.digests import sha256_file


PAIRED_SEMANTIC_SCHEMA = "weighttraits.behavior.paired_semantic.v2"
SEMANTIC_METADATA = "semantic_metadata.json"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_MODEL_ALIAS = "all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
PINNED_ENCODER_MODE = "pinned_internal_minilm"
TEST_ENCODER_MODE = "explicit_test_override"

SEMANTIC_ARRAY_FILES = {
    "primary": "dist_semantic_paired.npy",
    "eos_excluded": "dist_semantic_paired_eos_excluded.npy",
    "eos_included_counts": "eos_included_counts.npy",
    "per_identity": "per_identity_distances.npy",
    "nonempty_mask": "nonempty_mask.npy",
}


@dataclass(frozen=True, order=True)
class ResponseIdentity:
    """Exact identity of one prompt/draw response."""

    prompt_id: str
    sample_id: str

    def __post_init__(self) -> None:
        _validate_identifier(self.prompt_id, field="prompt_id")
        _validate_identifier(self.sample_id, field="sample_id")


@dataclass(frozen=True)
class SemanticPanelSpec:
    """Ordered expected response identities for one behavioral panel."""

    panel_id: str
    identities: tuple[ResponseIdentity, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.panel_id, field="panel_id")
        identities = tuple(self.identities)
        object.__setattr__(self, "identities", identities)
        if not identities:
            raise ValueError("semantic panel must contain at least one response identity")
        if not all(isinstance(identity, ResponseIdentity) for identity in identities):
            raise ValueError("semantic panel identities must be ResponseIdentity values")
        if len(set(identities)) != len(identities):
            raise ValueError("semantic panel response identities must be unique")

    @classmethod
    def from_prompt_samples(
        cls,
        *,
        panel_id: str,
        prompt_ids: Sequence[str],
        sample_ids: Sequence[str],
    ) -> "SemanticPanelSpec":
        """Build an ordered prompt-major Cartesian panel contract."""

        prompts = _validated_identifier_sequence(prompt_ids, field="prompt_id")
        samples = _validated_identifier_sequence(sample_ids, field="sample_id")
        if len(set(prompts)) != len(prompts):
            raise ValueError("semantic panel prompt IDs must be unique")
        if len(set(samples)) != len(samples):
            raise ValueError("semantic panel sample IDs must be unique")
        return cls(
            panel_id=panel_id,
            identities=tuple(
                ResponseIdentity(prompt_id=prompt_id, sample_id=sample_id)
                for prompt_id in prompts
                for sample_id in samples
            ),
        )

    @property
    def prompt_ids(self) -> tuple[str, ...]:
        """Unique prompt IDs in first-occurrence order."""

        return tuple(dict.fromkeys(identity.prompt_id for identity in self.identities))

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Unique sample IDs in first-occurrence order."""

        return tuple(dict.fromkeys(identity.sample_id for identity in self.identities))


@dataclass(frozen=True)
class SemanticAnalysisAttestation:
    """Strict orchestration hook binding a semantic artifact to frozen inputs."""

    cohort_id: str
    tree_id: str
    protocol_id: str
    probe_id: str
    panel_id: str
    expected_prompt_count: int
    expected_sample_ids: tuple[str, ...]
    registry_sha256: str
    fixture_sha256: str
    prompt_artifact_sha256: str
    response_receipt_sha256: str
    response_rows_sha256: str
    checkpoint_set_sha256: str
    encoder_receipt_sha256: str

    def __post_init__(self) -> None:
        for field in ("cohort_id", "tree_id", "protocol_id", "probe_id", "panel_id"):
            _validate_identifier(getattr(self, field), field=field)
        if (
            isinstance(self.expected_prompt_count, bool)
            or not isinstance(self.expected_prompt_count, int)
            or self.expected_prompt_count < 1
        ):
            raise ValueError("expected_prompt_count must be a positive integer")
        sample_ids = _validated_identifier_sequence(
            self.expected_sample_ids, field="expected_sample_id"
        )
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("expected_sample_ids must be non-empty and unique")
        object.__setattr__(self, "expected_sample_ids", sample_ids)
        for field in (
            "registry_sha256",
            "fixture_sha256",
            "prompt_artifact_sha256",
            "response_receipt_sha256",
            "response_rows_sha256",
            "checkpoint_set_sha256",
            "encoder_receipt_sha256",
        ):
            _validated_sha256(getattr(self, field), context=field)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON provenance contract."""

        return {
            "cohort_id": self.cohort_id,
            "tree_id": self.tree_id,
            "protocol_id": self.protocol_id,
            "probe_id": self.probe_id,
            "panel_id": self.panel_id,
            "expected_prompt_count": self.expected_prompt_count,
            "expected_sample_ids": list(self.expected_sample_ids),
            "registry_sha256": self.registry_sha256,
            "fixture_sha256": self.fixture_sha256,
            "prompt_artifact_sha256": self.prompt_artifact_sha256,
            "response_receipt_sha256": self.response_receipt_sha256,
            "response_rows_sha256": self.response_rows_sha256,
            "checkpoint_set_sha256": self.checkpoint_set_sha256,
            "encoder_receipt_sha256": self.encoder_receipt_sha256,
        }


@dataclass(frozen=True)
class SemanticResponse:
    """One model response, including exact prompt and draw identity."""

    model_id: str
    prompt_id: str
    sample_id: str
    text: str

    def __post_init__(self) -> None:
        _validate_identifier(self.model_id, field="model_id")
        _validate_identifier(self.prompt_id, field="prompt_id")
        _validate_identifier(self.sample_id, field="sample_id")
        if not isinstance(self.text, str):
            raise ValueError("semantic response text must be a string")

    @property
    def identity(self) -> ResponseIdentity:
        return ResponseIdentity(prompt_id=self.prompt_id, sample_id=self.sample_id)


@dataclass(frozen=True)
class SemanticDistancePair:
    """One exact model pair with both distance and the contracted similarity."""

    left_index: int
    right_index: int
    left_model_id: str
    right_model_id: str
    distance: float

    @property
    def similarity(self) -> float:
        """Return the paper endpoint: one minus cosine distance."""

        return 1.0 - self.distance


@dataclass(frozen=True)
class PairedSemanticDistances:
    """Paired semantic matrices and exact identity provenance."""

    panel: SemanticPanelSpec
    attestation: SemanticAnalysisAttestation
    model_ids: tuple[str, ...]
    embedding_model: str
    embedding_revision: str
    encoder_mode: str
    response_text_sha256: str
    per_identity_distances: np.ndarray
    primary: np.ndarray
    eos_excluded: np.ndarray
    eos_included_counts: np.ndarray
    nonempty_mask: np.ndarray

    def upper_triangle_pairs(self) -> tuple[SemanticDistancePair, ...]:
        """Return primary-distance pairs in deterministic matrix order."""

        return tuple(
            SemanticDistancePair(
                left_index=left,
                right_index=right,
                left_model_id=self.model_ids[left],
                right_model_id=self.model_ids[right],
                distance=float(self.primary[left, right]),
            )
            for left in range(len(self.model_ids))
            for right in range(left + 1, len(self.model_ids))
        )


def semantic_response_rows_sha256(
    responses: Sequence[SemanticResponse | Mapping[str, Any]],
    *,
    panel: SemanticPanelSpec,
    model_ids: Sequence[str],
) -> str:
    """Hash exact response rows after strict model/prompt/sample-grid validation."""

    ordered_models = _validated_model_ids(model_ids)
    normalized = tuple(_coerce_response(response) for response in responses)
    indexed = _index_exact_panel_responses(normalized, model_ids=ordered_models, panel=panel)
    return _response_text_hash(model_ids=ordered_models, panel=panel, indexed=indexed)


def build_paired_semantic_distances(
    responses: Sequence[SemanticResponse | Mapping[str, Any]],
    *,
    panel: SemanticPanelSpec,
    model_ids: Sequence[str],
    attestation: SemanticAnalysisAttestation,
    embedding_model: str,
    embedding_revision: str,
    encoder_snapshot: str | Path | None = None,
    encoder: Callable[[Sequence[str]], np.ndarray] | None = None,
    allow_test_encoder: bool = False,
) -> PairedSemanticDistances:
    """Build compare-then-average cosine distances with exact pairing.

    Production callers omit ``encoder`` so this function constructs the pinned
    MiniLM backend itself. A custom encoder requires ``allow_test_encoder=True``
    and produces a visibly non-production artifact that readers reject by default.
    """

    if not isinstance(panel, SemanticPanelSpec):
        raise ValueError("panel must be a SemanticPanelSpec")
    if not isinstance(attestation, SemanticAnalysisAttestation):
        raise ValueError("attestation must be a SemanticAnalysisAttestation")
    _validate_panel_attestation(panel, attestation)
    ordered_models = _validated_model_ids(model_ids)
    canonical_model = _validate_minilm_pin(embedding_model, embedding_revision)
    normalized_responses = tuple(_coerce_response(response) for response in responses)
    indexed = _index_exact_panel_responses(
        normalized_responses,
        model_ids=ordered_models,
        panel=panel,
    )

    response_hash = _response_text_hash(
        model_ids=ordered_models,
        panel=panel,
        indexed=indexed,
    )
    if response_hash != attestation.response_rows_sha256:
        raise ValueError("semantic response rows do not match the attested response hash")
    ordered_texts = [
        indexed[model_id][identity] for model_id in ordered_models for identity in panel.identities
    ]
    if encoder is None:
        if encoder_snapshot is None:
            raise ValueError(
                "production semantic analysis requires the receipt-pinned local MiniLM snapshot"
            )
        active_encoder = pinned_minilm_encoder(encoder_snapshot)
        encoder_mode = PINNED_ENCODER_MODE
    else:
        if not allow_test_encoder:
            raise ValueError(
                "custom semantic encoders are test-only; production must load pinned MiniLM"
            )
        active_encoder = encoder
        encoder_mode = TEST_ENCODER_MODE

    raw_embeddings = np.asarray(active_encoder(ordered_texts))
    expected_rows = len(ordered_models) * len(panel.identities)
    if raw_embeddings.ndim != 2 or raw_embeddings.shape[0] != expected_rows:
        raise ValueError(
            "semantic encoder must return exactly one embedding per response: "
            f"expected {expected_rows} rows, got shape {raw_embeddings.shape}"
        )
    if raw_embeddings.shape[1] < 1:
        raise ValueError("semantic encoder returned zero-dimensional embeddings")
    if not np.issubdtype(raw_embeddings.dtype, np.floating):
        raise ValueError("semantic encoder embeddings must use a real floating-point dtype")
    embeddings = np.asarray(raw_embeddings, dtype=np.float64)
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("semantic encoder returned non-finite embeddings")
    max_abs = np.max(np.abs(embeddings), axis=1)
    if np.any(max_abs <= 0):
        raise ValueError("semantic encoder returned a non-finite or zero-norm embedding")
    scaled_norms = np.linalg.norm(embeddings / max_abs[:, None], axis=1)
    if np.any(max_abs > np.finfo(np.float64).max / scaled_norms):
        raise ValueError("semantic encoder returned a non-finite or zero-norm embedding")
    norms = max_abs * scaled_norms
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("semantic encoder returned a non-finite or zero-norm embedding")
    embeddings = embeddings / norms[:, None]
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("semantic encoder normalization produced non-finite embeddings")
    embeddings = embeddings.reshape(
        len(ordered_models),
        len(panel.identities),
        embeddings.shape[1],
    )

    similarities = np.einsum("mid,nid->imn", embeddings, embeddings)
    per_identity = 1.0 - np.clip(similarities, -1.0, 1.0)
    per_identity = np.clip(per_identity, 0.0, 2.0)
    for identity_index in range(len(panel.identities)):
        np.fill_diagonal(per_identity[identity_index], 0.0)
    primary = np.mean(per_identity, axis=0)
    np.fill_diagonal(primary, 0.0)

    nonempty = np.asarray(
        [
            [bool(indexed[model_id][identity].strip()) for identity in panel.identities]
            for model_id in ordered_models
        ],
        dtype=bool,
    )
    eos_counts = nonempty.astype(np.int64) @ nonempty.astype(np.int64).T
    eos_excluded = _eos_excluded_from_components(
        per_identity,
        nonempty,
        model_ids=ordered_models,
    )

    for array in (per_identity, primary, eos_excluded, eos_counts, nonempty):
        array.setflags(write=False)
    result = PairedSemanticDistances(
        panel=panel,
        attestation=attestation,
        model_ids=ordered_models,
        embedding_model=canonical_model,
        embedding_revision=embedding_revision,
        encoder_mode=encoder_mode,
        response_text_sha256=response_hash,
        per_identity_distances=per_identity,
        primary=primary,
        eos_excluded=eos_excluded,
        eos_included_counts=eos_counts,
        nonempty_mask=nonempty,
    )
    _validate_semantic_result(result, allow_test_artifact=allow_test_encoder)
    return result


def pinned_minilm_encoder(
    snapshot_path: str | Path,
    *,
    batch_size: int = 128,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Load the audited MiniLM bytes from one already-verified local snapshot."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional heavyweight dependency
        raise RuntimeError(
            "sentence-transformers is required to load the pinned MiniLM encoder"
        ) from exc
    snapshot = Path(snapshot_path).resolve(strict=True)
    if not snapshot.is_dir():
        raise ValueError(f"MiniLM snapshot must be a directory: {snapshot}")
    model = SentenceTransformer(str(snapshot), local_files_only=True)

    def encode(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            model.encode(
                list(texts),
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )

    return encode


def write_paired_semantic_artifact(
    result: PairedSemanticDistances,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    allow_test_artifact: bool = False,
) -> Path:
    """Write hash-verified matrices, publishing the metadata receipt last."""

    _validate_semantic_result(result, allow_test_artifact=allow_test_artifact)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "primary": result.primary,
        "eos_excluded": result.eos_excluded,
        "eos_included_counts": result.eos_included_counts,
        "per_identity": result.per_identity_distances,
        "nonempty_mask": result.nonempty_mask,
    }
    metadata_path = out / SEMANTIC_METADATA
    targets = [out / filename for filename in SEMANTIC_ARRAY_FILES.values()] + [metadata_path]
    with _artifact_write_lock(out, name="paired-semantic"):
        if not overwrite and any(path.exists() for path in targets):
            raise FileExistsError(
                f"paired-semantic artifact already exists under {out}; "
                "pass overwrite=True explicitly"
            )
        array_metadata: dict[str, Any] = {}
        for name, values in arrays.items():
            filename = SEMANTIC_ARRAY_FILES[name]
            target = out / filename
            _atomic_save_npy(target, values)
            array_metadata[name] = {
                "file": filename,
                "sha256": sha256_file(target),
                "values_sha256": array_values_sha256(values),
                "dtype": values.dtype.str,
                "shape": list(values.shape),
            }

        identity_rows = _identity_rows(result.panel)
        metadata = {
            "schema": PAIRED_SEMANTIC_SCHEMA,
            "producer": "weighttraits",
            "production_valid": result.encoder_mode == PINNED_ENCODER_MODE,
            "encoder_mode": result.encoder_mode,
            "attestation": result.attestation.to_dict(),
            "panel_id": result.panel.panel_id,
            "n_prompts": len(result.panel.prompt_ids),
            "n_samples_per_prompt": len(result.panel.sample_ids),
            "n_response_identities": len(result.panel.identities),
            "response_identities": identity_rows,
            "response_identities_sha256": hashlib.sha256(
                _canonical_json_bytes(identity_rows)
            ).hexdigest(),
            "n_models": len(result.model_ids),
            "model_ids": list(result.model_ids),
            "model_ids_sha256": ordered_strings_sha256(result.model_ids),
            "embedding": {
                "model": result.embedding_model,
                "revision": result.embedding_revision,
            },
            "response_text_sha256": result.response_text_sha256,
            "primary_endpoint": "mean same-prompt-and-sample cosine distance",
            "meta_outcome": "semantic similarity = 1 - primary cosine distance",
            "empty_string_policy": {
                "primary": "preserved and embedded without substitution",
                "eos_excluded": ("exclude identity when either paired response strips to empty"),
            },
            "arrays": array_metadata,
        }
        _atomic_write_json(metadata_path, metadata)
    return metadata_path


def read_paired_semantic_artifact(
    path: str | Path,
    *,
    expected_attestation: SemanticAnalysisAttestation,
    allow_test_artifact: bool = False,
) -> PairedSemanticDistances:
    """Read a paired-semantic artifact only after full self-validation."""

    if not isinstance(expected_attestation, SemanticAnalysisAttestation):
        raise ValueError("expected_attestation must be a SemanticAnalysisAttestation")
    raw_path = Path(path)
    metadata_path = raw_path / SEMANTIC_METADATA if raw_path.is_dir() else raw_path
    metadata = _load_json_mapping(metadata_path)
    if metadata.get("schema") != PAIRED_SEMANTIC_SCHEMA:
        raise ValueError(f"unsupported paired-semantic schema in {metadata_path}")
    if metadata.get("producer") != "weighttraits":
        raise ValueError("paired-semantic producer mismatch")
    if metadata.get("attestation") != expected_attestation.to_dict():
        raise ValueError("paired-semantic attestation does not match expected provenance")
    encoder_mode = metadata.get("encoder_mode")
    production_valid = metadata.get("production_valid")
    if encoder_mode not in {PINNED_ENCODER_MODE, TEST_ENCODER_MODE}:
        raise ValueError("paired-semantic encoder mode is invalid")
    if production_valid != (encoder_mode == PINNED_ENCODER_MODE):
        raise ValueError("paired-semantic production-valid flag is inconsistent")
    if not production_valid and not allow_test_artifact:
        raise ValueError("test-encoder semantic artifacts are not valid production inputs")

    identity_rows = metadata.get("response_identities")
    if not isinstance(identity_rows, list) or not identity_rows:
        raise ValueError("paired-semantic response identities must be a non-empty list")
    identities: list[ResponseIdentity] = []
    for index, row in enumerate(identity_rows):
        if not isinstance(row, Mapping) or set(row) != {"prompt_id", "sample_id"}:
            raise ValueError(f"invalid paired-semantic identity row {index}")
        identities.append(ResponseIdentity(prompt_id=row["prompt_id"], sample_id=row["sample_id"]))
    panel = SemanticPanelSpec(panel_id=metadata.get("panel_id"), identities=tuple(identities))
    _validate_panel_attestation(panel, expected_attestation)
    if (
        metadata.get("response_identities_sha256")
        != hashlib.sha256(_canonical_json_bytes(_identity_rows(panel))).hexdigest()
    ):
        raise ValueError("paired-semantic response-identity hash mismatch")
    if metadata.get("n_prompts") != len(panel.prompt_ids):
        raise ValueError("paired-semantic prompt count mismatch")
    if metadata.get("n_samples_per_prompt") != len(panel.sample_ids):
        raise ValueError("paired-semantic sample count mismatch")
    if metadata.get("n_response_identities") != len(panel.identities):
        raise ValueError("paired-semantic response-identity count mismatch")

    model_ids = _validated_model_ids(metadata.get("model_ids"))
    if metadata.get("n_models") != len(model_ids):
        raise ValueError("paired-semantic model count mismatch")
    if metadata.get("model_ids_sha256") != ordered_strings_sha256(model_ids):
        raise ValueError("paired-semantic ordered model-ID hash mismatch")
    embedding = metadata.get("embedding")
    if not isinstance(embedding, Mapping):
        raise ValueError("paired-semantic embedding metadata must be a mapping")
    canonical_model = _validate_minilm_pin(embedding.get("model"), embedding.get("revision"))
    response_hash = metadata.get("response_text_sha256")
    if response_hash != expected_attestation.response_rows_sha256:
        raise ValueError("paired-semantic response hash does not match attestation")
    if metadata.get("meta_outcome") != "semantic similarity = 1 - primary cosine distance":
        raise ValueError("paired-semantic meta outcome/sign contract mismatch")

    arrays_metadata = metadata.get("arrays")
    if not isinstance(arrays_metadata, Mapping) or set(arrays_metadata) != set(
        SEMANTIC_ARRAY_FILES
    ):
        raise ValueError("paired-semantic array metadata keys are invalid")
    arrays = {
        name: _read_verified_array(
            metadata_path.parent,
            name=name,
            metadata=arrays_metadata[name],
        )
        for name in SEMANTIC_ARRAY_FILES
    }
    result = PairedSemanticDistances(
        panel=panel,
        attestation=expected_attestation,
        model_ids=model_ids,
        embedding_model=canonical_model,
        embedding_revision=embedding["revision"],
        encoder_mode=encoder_mode,
        response_text_sha256=response_hash,
        per_identity_distances=arrays["per_identity"],
        primary=arrays["primary"],
        eos_excluded=arrays["eos_excluded"],
        eos_included_counts=arrays["eos_included_counts"],
        nonempty_mask=arrays["nonempty_mask"],
    )
    _validate_semantic_result(result, allow_test_artifact=allow_test_artifact)
    for array in arrays.values():
        array.setflags(write=False)
    return result


def _validate_semantic_result(
    result: PairedSemanticDistances,
    *,
    allow_test_artifact: bool,
) -> None:
    if not isinstance(result, PairedSemanticDistances):
        raise ValueError("result must be a PairedSemanticDistances")
    if not isinstance(result.attestation, SemanticAnalysisAttestation):
        raise ValueError("semantic result attestation is invalid")
    _validate_panel_attestation(result.panel, result.attestation)
    model_ids = _validated_model_ids(result.model_ids)
    _validate_minilm_pin(result.embedding_model, result.embedding_revision)
    if result.encoder_mode not in {PINNED_ENCODER_MODE, TEST_ENCODER_MODE}:
        raise ValueError("semantic result encoder mode is invalid")
    if result.encoder_mode == TEST_ENCODER_MODE and not allow_test_artifact:
        raise ValueError("test-encoder semantic results are not valid production artifacts")
    if result.response_text_sha256 != result.attestation.response_rows_sha256:
        raise ValueError("semantic result response hash does not match attestation")
    n_models = len(model_ids)
    n_identities = len(result.panel.identities)
    expected_shapes = {
        "per_identity": (n_identities, n_models, n_models),
        "primary": (n_models, n_models),
        "eos_excluded": (n_models, n_models),
        "eos_included_counts": (n_models, n_models),
        "nonempty_mask": (n_models, n_identities),
    }
    arrays = {
        "per_identity": np.asarray(result.per_identity_distances),
        "primary": np.asarray(result.primary),
        "eos_excluded": np.asarray(result.eos_excluded),
        "eos_included_counts": np.asarray(result.eos_included_counts),
        "nonempty_mask": np.asarray(result.nonempty_mask),
    }
    for name, values in arrays.items():
        if values.shape != expected_shapes[name]:
            raise ValueError(
                f"semantic {name} shape {values.shape} does not match {expected_shapes[name]}"
            )
    for name in ("per_identity", "primary", "eos_excluded"):
        values = arrays[name]
        if not np.issubdtype(values.dtype, np.floating) or not np.all(np.isfinite(values)):
            raise ValueError(f"semantic {name} must contain finite floating-point values")
        if np.any(values < 0) or np.any(values > 2):
            raise ValueError(f"semantic {name} values must lie within [0, 2]")
    counts = arrays["eos_included_counts"]
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("semantic EOS-included counts must use an integer dtype")
    mask = arrays["nonempty_mask"]
    if mask.dtype != np.dtype(bool):
        raise ValueError("semantic nonempty mask must use a boolean dtype")
    per_identity = arrays["per_identity"]
    for index in range(n_identities):
        _validate_distance_matrix(per_identity[index], context=f"per_identity[{index}]")
    _validate_distance_matrix(arrays["primary"], context="primary")
    _validate_distance_matrix(arrays["eos_excluded"], context="eos_excluded")
    expected_primary = np.mean(per_identity, axis=0)
    np.fill_diagonal(expected_primary, 0.0)
    if not np.array_equal(arrays["primary"], expected_primary):
        raise ValueError("semantic primary matrix is not the exact per-identity mean")
    expected_counts = mask.astype(np.int64) @ mask.astype(np.int64).T
    if not np.array_equal(counts, expected_counts):
        raise ValueError("semantic EOS-included counts do not match the nonempty mask")
    expected_eos = _eos_excluded_from_components(
        per_identity,
        mask,
        model_ids=model_ids,
    )
    if not np.array_equal(arrays["eos_excluded"], expected_eos):
        raise ValueError("semantic EOS-excluded matrix does not match its component arrays")


def _validate_distance_matrix(values: np.ndarray, *, context: str) -> None:
    if not np.array_equal(values, values.T):
        raise ValueError(f"semantic {context} matrix must be exactly symmetric")
    if np.any(np.diag(values) != 0):
        raise ValueError(f"semantic {context} matrix diagonal must be exactly zero")


def _eos_excluded_from_components(
    per_identity: np.ndarray,
    nonempty: np.ndarray,
    *,
    model_ids: Sequence[str],
) -> np.ndarray:
    result = np.zeros((len(model_ids), len(model_ids)), dtype=per_identity.dtype)
    for left in range(len(model_ids)):
        for right in range(left + 1, len(model_ids)):
            included = nonempty[left] & nonempty[right]
            if not np.any(included):
                raise ValueError(
                    "EOS-excluded semantic distance has no shared non-empty responses for "
                    f"{model_ids[left]!r} and {model_ids[right]!r}"
                )
            value = float(np.mean(per_identity[included, left, right]))
            result[left, right] = result[right, left] = value
    return result


def _validate_panel_attestation(
    panel: SemanticPanelSpec,
    attestation: SemanticAnalysisAttestation,
) -> None:
    if panel.panel_id != attestation.panel_id:
        raise ValueError("semantic panel ID does not match the attestation")
    if len(panel.prompt_ids) != attestation.expected_prompt_count:
        raise ValueError(
            f"semantic panel has {len(panel.prompt_ids)} prompts, "
            f"expected {attestation.expected_prompt_count}"
        )
    expected = tuple(
        ResponseIdentity(prompt_id=prompt_id, sample_id=sample_id)
        for prompt_id in panel.prompt_ids
        for sample_id in attestation.expected_sample_ids
    )
    if panel.identities != expected:
        raise ValueError(
            "semantic panel must be the exact prompt-major Cartesian product of "
            "the attested prompt and sample IDs"
        )


def _coerce_response(raw: SemanticResponse | Mapping[str, Any]) -> SemanticResponse:
    if isinstance(raw, SemanticResponse):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("semantic responses must be SemanticResponse values or mappings")
    has_sample = "sample_id" in raw
    has_draw = "draw_id" in raw
    if not has_sample and not has_draw:
        raise ValueError("semantic response requires sample_id or draw_id")
    if has_sample and has_draw and raw["sample_id"] != raw["draw_id"]:
        raise ValueError("semantic response sample_id and draw_id disagree")
    sample_id = raw["sample_id"] if has_sample else raw["draw_id"]
    try:
        model_id = raw["model_id"]
        prompt_id = raw["prompt_id"]
        text = raw["text"]
    except KeyError as exc:
        raise ValueError(f"semantic response is missing {exc.args[0]}") from exc
    for field, value in (
        ("model_id", model_id),
        ("prompt_id", prompt_id),
        ("sample_id", sample_id),
    ):
        if not isinstance(value, str):
            raise ValueError(f"semantic response {field} must be a string")
    return SemanticResponse(
        model_id=model_id,
        prompt_id=prompt_id,
        sample_id=sample_id,
        text=text,
    )


def _index_exact_panel_responses(
    responses: Sequence[SemanticResponse],
    *,
    model_ids: tuple[str, ...],
    panel: SemanticPanelSpec,
) -> dict[str, dict[ResponseIdentity, str]]:
    indexed = {model_id: {} for model_id in model_ids}
    expected_models = set(model_ids)
    observed_models = {response.model_id for response in responses}
    if observed_models != expected_models:
        raise ValueError(
            "semantic response model IDs do not match the ordered model contract: "
            f"missing={sorted(expected_models - observed_models)}, "
            f"extra={sorted(observed_models - expected_models)}"
        )
    for response in responses:
        identity = response.identity
        model_rows = indexed[response.model_id]
        if identity in model_rows:
            raise ValueError(
                "duplicate semantic response identity for "
                f"{response.model_id!r}: ({identity.prompt_id!r}, {identity.sample_id!r})"
            )
        model_rows[identity] = response.text

    expected_identities = set(panel.identities)
    for model_id in model_ids:
        observed = set(indexed[model_id])
        if observed != expected_identities:
            missing = sorted(expected_identities - observed)
            extra = sorted(observed - expected_identities)
            raise ValueError(
                f"semantic response identity mismatch for {model_id!r}: "
                f"missing={missing}, extra={extra}"
            )
    return indexed


def _validated_model_ids(raw: object) -> tuple[str, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("paired semantic model IDs must be a sequence of strings")
    model_ids = tuple(raw)
    if len(model_ids) < 2:
        raise ValueError("paired semantic distances require at least two ordered model IDs")
    for model_id in model_ids:
        _validate_identifier(model_id, field="model_id")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("paired semantic model IDs must be unique")
    return model_ids


def _validated_identifier_sequence(raw: object, *, field: str) -> tuple[str, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{field} values must be a sequence of strings")
    values = tuple(raw)
    for value in values:
        _validate_identifier(value, field=field)
    return values


def _validate_minilm_pin(model: object, revision: object) -> str:
    if model not in {MINILM_MODEL, MINILM_MODEL_ALIAS}:
        raise ValueError(f"semantic embedding model must be {MINILM_MODEL!r}, got {model!r}")
    if revision != MINILM_REVISION:
        raise ValueError(
            f"semantic embedding revision must be pinned to {MINILM_REVISION}, got {revision!r}"
        )
    return MINILM_MODEL


def _response_text_hash(
    *,
    model_ids: tuple[str, ...],
    panel: SemanticPanelSpec,
    indexed: Mapping[str, Mapping[ResponseIdentity, str]],
) -> str:
    rows = [
        {
            "model_id": model_id,
            "prompt_id": identity.prompt_id,
            "sample_id": identity.sample_id,
            "text": indexed[model_id][identity],
        }
        for model_id in model_ids
        for identity in panel.identities
    ]
    return hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()


def _identity_rows(panel: SemanticPanelSpec) -> list[dict[str, str]]:
    return [
        {"prompt_id": identity.prompt_id, "sample_id": identity.sample_id}
        for identity in panel.identities
    ]


def _read_verified_array(
    directory: Path,
    *,
    name: str,
    metadata: object,
) -> np.ndarray:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"paired-semantic {name} metadata must be a mapping")
    expected_file = SEMANTIC_ARRAY_FILES[name]
    if metadata.get("file") != expected_file:
        raise ValueError(f"paired-semantic {name} must use canonical filename")
    path = directory / expected_file
    if not path.is_file():
        raise FileNotFoundError(f"paired-semantic array is missing: {path}")
    if metadata.get("sha256") != sha256_file(path):
        raise ValueError(f"paired-semantic {name} file hash mismatch")
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid paired-semantic array: {path}") from exc
    if metadata.get("values_sha256") != array_values_sha256(values):
        raise ValueError(f"paired-semantic {name} numeric hash mismatch")
    if metadata.get("dtype") != values.dtype.str:
        raise ValueError(f"paired-semantic {name} dtype mismatch")
    if metadata.get("shape") != list(values.shape):
        raise ValueError(f"paired-semantic {name} shape metadata mismatch")
    return values


def _load_json_mapping(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            out[key] = value
        return out

    try:
        loaded = json.loads(path.read_text(), object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON artifact must contain a mapping: {path}")
    return loaded


def _validate_identifier(value: object, *, field: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty string without surrounding whitespace")


def _validated_sha256(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase 64-character SHA-256")
    return value


def _canonical_json_bytes(value: object) -> bytes:
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


@contextmanager
def _artifact_write_lock(directory: Path, *, name: str) -> Iterator[None]:
    lock_path = directory / f".{name}.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"another {name} writer holds {lock_path}") from exc
    os.close(descriptor)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
