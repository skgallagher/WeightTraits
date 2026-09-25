from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from weighttraits.behavior.direct_bridge import (
    DIRECT_DISTANCE_MATRIX,
    DirectDistanceSourceAttestation,
    array_values_sha256,
    export_native_direct_distance,
    ordered_strings_sha256,
    read_direct_distance_artifact,
)
from weighttraits.behavior.meta import (
    BEHAVIOR_META_SCHEMA,
    SEMANTIC_SIMILARITY_OUTCOME,
    CorrelationStudy,
    StudyInputProvenance,
    fisher_z_dersimonian_laird,
    paired_semantic_similarity_study,
    pearson_correlation_study,
    write_meta_summaries_csv,
    write_meta_summary_json,
)
from weighttraits.behavior.semantic import (
    MINILM_MODEL,
    MINILM_REVISION,
    ResponseIdentity,
    SemanticAnalysisAttestation,
    SemanticPanelSpec,
    SemanticResponse,
    build_paired_semantic_distances,
    read_paired_semantic_artifact,
    semantic_response_rows_sha256,
    write_paired_semantic_artifact,
)
from weighttraits.verification.digests import sha256_file


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_direct_summary(
    root: Path,
) -> tuple[Path, np.ndarray, DirectDistanceSourceAttestation]:
    analysis = root / "tree-001" / "model_leaf_analysis"
    analysis.mkdir(parents=True)
    sources = root / "sources"
    sources.mkdir()
    ledger = sources / "ledger.jsonl"
    truth = sources / "truth.jsonl"
    receipt = sources / "completion.json"
    ledger.write_text('{"status":"completed"}\n')
    truth.write_text('{"node_id":"n0"}\n')
    receipt.write_text('{"valid":true}\n')
    matrix = np.asarray(
        [
            [0.0, 0.25, 0.75],
            [0.25, 0.0, 0.5],
            [0.75, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    matrix_path = analysis / "distance_matrix_cosine.npy"
    np.save(matrix_path, matrix)
    model_ids = ["leaf-c", "leaf-a", "leaf-b"]
    summary = {
        "analysis_engine": "direct",
        "distance_engine": "direct_streaming_sufficient_stats",
        "artifact": "model",
        "representation": "full_weight",
        "ledger": str(ledger),
        "truth_manifest": str(truth),
        "n_models": 3,
        "model_ids": model_ids,
        "metrics": ["cosine"],
        "results": [{"metric": "cosine"}],
    }
    summary_path = analysis / "summary.json"
    summary_path.write_text(json.dumps(summary))
    attestation = DirectDistanceSourceAttestation(
        cohort_id="ordinary-full-ft",
        tree_id="tree-001",
        metric="cosine",
        artifact="model",
        representation="full_weight",
        summary=str(summary_path),
        summary_sha256=sha256_file(summary_path),
        matrix_sha256=sha256_file(matrix_path),
        model_ids_sha256=ordered_strings_sha256(model_ids),
        ledger=str(ledger),
        ledger_sha256=sha256_file(ledger),
        truth_manifest=str(truth),
        truth_manifest_sha256=sha256_file(truth),
        completion_receipt=str(receipt),
        completion_receipt_sha256=sha256_file(receipt),
    )
    return summary_path, matrix, attestation


def _reattest_direct(summary_path: Path) -> DirectDistanceSourceAttestation:
    summary = json.loads(summary_path.read_text())
    matrix_path = summary_path.parent / "distance_matrix_cosine.npy"
    ledger = Path(summary["ledger"])
    truth = Path(summary["truth_manifest"])
    receipt = summary_path.parent.parent.parent / "sources" / "completion.json"
    return DirectDistanceSourceAttestation(
        cohort_id="ordinary-full-ft",
        tree_id="tree-001",
        metric="cosine",
        artifact="model",
        representation="full_weight",
        summary=str(summary_path),
        summary_sha256=sha256_file(summary_path),
        matrix_sha256=sha256_file(matrix_path),
        model_ids_sha256=ordered_strings_sha256(summary["model_ids"]),
        ledger=str(ledger),
        ledger_sha256=sha256_file(ledger),
        truth_manifest=str(truth),
        truth_manifest_sha256=sha256_file(truth),
        completion_receipt=str(receipt),
        completion_receipt_sha256=sha256_file(receipt),
    )


def test_direct_distance_export_is_byte_order_and_provenance_preserving(
    tmp_path: Path,
) -> None:
    summary_path, source_matrix, attestation = _write_direct_summary(tmp_path / "analysis")
    source_npy = summary_path.parent / "distance_matrix_cosine.npy"

    artifact = export_native_direct_distance(
        summary_path,
        metric="cosine",
        out_dir=tmp_path / "bridge",
        attestation=attestation,
    )

    assert artifact.attestation == attestation
    assert artifact.model_ids == ("leaf-c", "leaf-a", "leaf-b")
    assert artifact.matrix.dtype == source_matrix.dtype
    assert np.array_equal(artifact.matrix, source_matrix)
    assert not artifact.matrix.flags.writeable
    assert artifact.matrix_path.read_bytes() == source_npy.read_bytes()
    assert artifact.metadata["source_attestation"] == attestation.to_dict()
    assert artifact.metadata["matrix"]["values_sha256"] == array_values_sha256(source_matrix)
    assert [
        (pair.left_model_id, pair.right_model_id, pair.distance)
        for pair in artifact.upper_triangle_pairs()
    ] == [
        ("leaf-c", "leaf-a", 0.25),
        ("leaf-c", "leaf-b", 0.75),
        ("leaf-a", "leaf-b", 0.5),
    ]
    assert not list((tmp_path / "bridge").glob("*.tmp"))
    assert not list((tmp_path / "bridge").glob("*.lock"))


def test_direct_reader_rejects_matrix_order_and_provenance_tampering(
    tmp_path: Path,
) -> None:
    summary_path, _, attestation = _write_direct_summary(tmp_path / "analysis")
    artifact_dir = tmp_path / "bridge"
    artifact = export_native_direct_distance(
        summary_path,
        metric="cosine",
        out_dir=artifact_dir,
        attestation=attestation,
    )

    tampered = np.asarray(artifact.matrix).copy()
    tampered[0, 1] = tampered[1, 0] = 0.9
    np.save(artifact_dir / DIRECT_DISTANCE_MATRIX, tampered)
    with pytest.raises(ValueError, match="file hash mismatch"):
        read_direct_distance_artifact(artifact_dir, expected_attestation=attestation)

    export_native_direct_distance(
        summary_path,
        metric="cosine",
        out_dir=artifact_dir,
        attestation=attestation,
        overwrite=True,
    )
    metadata_path = artifact_dir / "direct_distance.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["tree_id"] = "tree-999"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="tree_id does not match expected provenance"):
        read_direct_distance_artifact(artifact_dir, expected_attestation=attestation)


def test_direct_export_rejects_unattested_sources_and_fake_native_labels(
    tmp_path: Path,
) -> None:
    summary_path, _, attestation = _write_direct_summary(tmp_path / "analysis")
    Path(attestation.ledger).write_text("tampered\n")
    with pytest.raises(ValueError, match="ledger SHA-256 mismatch"):
        export_native_direct_distance(
            summary_path,
            metric="cosine",
            out_dir=tmp_path / "bridge",
            attestation=attestation,
        )

    summary_path, _, _ = _write_direct_summary(tmp_path / "analysis-2")
    summary = json.loads(summary_path.read_text())
    summary["distance_engine"] = "made_up"
    summary_path.write_text(json.dumps(summary))
    attestation = _reattest_direct(summary_path)
    with pytest.raises(ValueError, match="wrong direct distance engine"):
        export_native_direct_distance(
            summary_path,
            metric="cosine",
            out_dir=tmp_path / "bridge-2",
            attestation=attestation,
        )


def _semantic_panel() -> SemanticPanelSpec:
    return SemanticPanelSpec.from_prompt_samples(
        panel_id="modern-mc/hellaswag",
        prompt_ids=["prompt-1", "prompt-2"],
        sample_ids=["draw-0"],
    )


def _semantic_responses() -> list[SemanticResponse]:
    return [
        SemanticResponse("leaf-b", "prompt-2", "draw-0", "same"),
        SemanticResponse("leaf-a", "prompt-1", "draw-0", ""),
        SemanticResponse("leaf-b", "prompt-1", "draw-0", "opposite"),
        SemanticResponse("leaf-a", "prompt-2", "draw-0", "same"),
    ]


def _semantic_attestation(
    responses: list[SemanticResponse] | None = None,
    panel: SemanticPanelSpec | None = None,
) -> SemanticAnalysisAttestation:
    active_panel = panel or _semantic_panel()
    active_responses = responses or _semantic_responses()
    response_hash = semantic_response_rows_sha256(
        active_responses,
        panel=active_panel,
        model_ids=["leaf-a", "leaf-b"],
    )
    return SemanticAnalysisAttestation(
        cohort_id="ordinary-full-ft",
        tree_id="tree-001",
        protocol_id="modern-mc",
        probe_id="hellaswag",
        panel_id=active_panel.panel_id,
        expected_prompt_count=2,
        expected_sample_ids=("draw-0",),
        registry_sha256=_digest("registry"),
        fixture_sha256=_digest("fixture"),
        prompt_artifact_sha256=_digest("prompt-artifact"),
        response_receipt_sha256=_digest("response-receipt"),
        response_rows_sha256=response_hash,
        checkpoint_set_sha256=_digest("checkpoint-set"),
        encoder_receipt_sha256=_digest("encoder-receipt"),
    )


def _build_test_semantic(
    *,
    encoder,
    responses: list[SemanticResponse] | None = None,
    panel: SemanticPanelSpec | None = None,
):
    active_panel = panel or _semantic_panel()
    active_responses = responses or _semantic_responses()
    return build_paired_semantic_distances(
        active_responses,
        panel=active_panel,
        model_ids=["leaf-a", "leaf-b"],
        attestation=_semantic_attestation(active_responses, active_panel),
        encoder=encoder,
        allow_test_encoder=True,
        embedding_model="all-MiniLM-L6-v2",
        embedding_revision=MINILM_REVISION,
    )


def test_paired_semantic_exact_pairing_empty_policy_sign_and_verified_reader(
    tmp_path: Path,
) -> None:
    observed_texts: list[str] = []
    vectors = {
        "": [1.0, 0.0],
        "opposite": [-1.0, 0.0],
        "same": [0.0, 1.0],
    }

    def encoder(texts: list[str]) -> np.ndarray:
        observed_texts.extend(texts)
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)

    result = _build_test_semantic(encoder=encoder)

    assert observed_texts == ["", "same", "opposite", "same"]
    assert result.embedding_model == MINILM_MODEL
    assert result.model_ids == ("leaf-a", "leaf-b")
    assert result.per_identity_distances[:, 0, 1] == pytest.approx([2.0, 0.0])
    assert result.primary[0, 1] == pytest.approx(1.0)
    assert result.eos_excluded[0, 1] == pytest.approx(0.0)
    assert result.eos_included_counts[0, 1] == 1
    pair = result.upper_triangle_pairs()[0]
    assert pair.distance == pytest.approx(1.0)
    assert pair.similarity == pytest.approx(0.0)

    with pytest.raises(ValueError, match="not valid production artifacts"):
        write_paired_semantic_artifact(result, tmp_path / "semantic")
    metadata_path = write_paired_semantic_artifact(
        result,
        tmp_path / "semantic",
        allow_test_artifact=True,
    )
    metadata = json.loads(metadata_path.read_text())
    assert metadata["production_valid"] is False
    assert metadata["meta_outcome"] == ("semantic similarity = 1 - primary cosine distance")
    reread = read_paired_semantic_artifact(
        tmp_path / "semantic",
        expected_attestation=result.attestation,
        allow_test_artifact=True,
    )
    assert np.array_equal(reread.primary, result.primary)
    assert not reread.primary.flags.writeable
    assert not list((tmp_path / "semantic").glob("*.tmp"))
    assert not list((tmp_path / "semantic").glob("*.lock"))


def test_paired_semantic_reader_rejects_array_and_attestation_tampering(
    tmp_path: Path,
) -> None:
    result = _build_test_semantic(
        encoder=lambda texts: np.asarray(
            [[1.0, 0.0] if text != "same" else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )
    )
    out = tmp_path / "semantic"
    write_paired_semantic_artifact(result, out, allow_test_artifact=True)
    primary_path = out / "dist_semantic_paired.npy"
    primary = np.load(primary_path)
    primary[0, 1] = primary[1, 0] = 0.4
    np.save(primary_path, primary)
    with pytest.raises(ValueError, match="primary file hash mismatch"):
        read_paired_semantic_artifact(
            out,
            expected_attestation=result.attestation,
            allow_test_artifact=True,
        )

    write_paired_semantic_artifact(
        result,
        out,
        overwrite=True,
        allow_test_artifact=True,
    )
    metadata_path = out / "semantic_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["attestation"]["tree_id"] = "tree-999"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="does not match expected provenance"):
        read_paired_semantic_artifact(
            out,
            expected_attestation=result.attestation,
            allow_test_artifact=True,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda rows: rows[:-1], "identity mismatch"),
        (
            lambda rows: rows + [SemanticResponse("leaf-a", "prompt-1", "draw-0", "duplicate")],
            "duplicate semantic response identity",
        ),
        (
            lambda rows: rows + [SemanticResponse("leaf-a", "prompt-1", "draw-1", "extra")],
            "identity mismatch",
        ),
    ],
)
def test_paired_semantic_rejects_missing_duplicate_or_extra_identity(
    mutator,
    message: str,
) -> None:
    rows = mutator(_semantic_responses())
    attestation = _semantic_attestation()
    with pytest.raises(ValueError, match=message):
        build_paired_semantic_distances(
            rows,
            panel=_semantic_panel(),
            model_ids=["leaf-a", "leaf-b"],
            attestation=attestation,
            encoder=lambda texts: np.ones((len(texts), 2), dtype=np.float64),
            allow_test_encoder=True,
            embedding_model=MINILM_MODEL,
            embedding_revision=MINILM_REVISION,
        )


def test_paired_semantic_requires_pinned_internal_encoder_for_production() -> None:
    with pytest.raises(ValueError, match="custom semantic encoders are test-only"):
        build_paired_semantic_distances(
            _semantic_responses(),
            panel=_semantic_panel(),
            model_ids=["leaf-a", "leaf-b"],
            attestation=_semantic_attestation(),
            encoder=lambda texts: np.ones((len(texts), 2), dtype=np.float64),
            embedding_model=MINILM_MODEL,
            embedding_revision=MINILM_REVISION,
        )
    with pytest.raises(ValueError, match="revision must be pinned"):
        build_paired_semantic_distances(
            _semantic_responses(),
            panel=_semantic_panel(),
            model_ids=["leaf-a", "leaf-b"],
            attestation=_semantic_attestation(),
            encoder=lambda texts: np.ones((len(texts), 2), dtype=np.float64),
            allow_test_encoder=True,
            embedding_model=MINILM_MODEL,
            embedding_revision="main",
        )


def test_paired_semantic_rejects_nonstring_ids_noncartesian_panel_and_bad_norms() -> None:
    rows = [row.__dict__ for row in _semantic_responses()]
    rows[0]["sample_id"] = 0
    with pytest.raises(ValueError, match="sample_id must be a string"):
        semantic_response_rows_sha256(
            rows,
            panel=_semantic_panel(),
            model_ids=["leaf-a", "leaf-b"],
        )

    noncartesian = SemanticPanelSpec(
        panel_id="modern-mc/hellaswag",
        identities=(
            ResponseIdentity("prompt-1", "draw-0"),
            ResponseIdentity("prompt-2", "draw-1"),
        ),
    )
    attestation = SemanticAnalysisAttestation(
        cohort_id="ordinary-full-ft",
        tree_id="tree-001",
        protocol_id="modern-mc",
        probe_id="hellaswag",
        panel_id=noncartesian.panel_id,
        expected_prompt_count=2,
        expected_sample_ids=("draw-0", "draw-1"),
        registry_sha256=_digest("registry"),
        fixture_sha256=_digest("fixture"),
        prompt_artifact_sha256=_digest("prompts"),
        response_receipt_sha256=_digest("responses"),
        response_rows_sha256=_digest("rows"),
        checkpoint_set_sha256=_digest("checkpoints"),
        encoder_receipt_sha256=_digest("encoder"),
    )
    with pytest.raises(ValueError, match="exact prompt-major Cartesian product"):
        build_paired_semantic_distances(
            [],
            panel=noncartesian,
            model_ids=["leaf-a", "leaf-b"],
            attestation=attestation,
            encoder=lambda texts: np.ones((len(texts), 2), dtype=np.float64),
            allow_test_encoder=True,
            embedding_model=MINILM_MODEL,
            embedding_revision=MINILM_REVISION,
        )

    with pytest.raises(ValueError, match="non-finite or zero-norm"):
        _build_test_semantic(
            encoder=lambda texts: np.full((len(texts), 2), 1.5e308, dtype=np.float64)
        )


def _pairs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    keys = [
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    ]
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    direct = [
        {"left_model_id": left, "right_model_id": right, "distance": value}
        for (left, right), value in zip(keys, weights, strict=True)
    ]
    semantic = [
        {
            "left_model_id": left,
            "right_model_id": right,
            "distance": 1.0 - value,
        }
        for (left, right), value in zip(keys, weights, strict=True)
    ]
    return direct, semantic


def test_pair_key_join_is_order_independent_and_similarity_sign_is_explicit() -> None:
    direct, semantic = _pairs()
    study = paired_semantic_similarity_study(
        "tree-001",
        direct_pairs=direct,
        semantic_pairs=list(reversed(semantic)),
        direct_artifact_sha256=_digest("direct-artifact"),
        semantic_artifact_sha256=_digest("semantic-artifact"),
    )
    assert study.correlation == pytest.approx(1.0)
    assert study.input_provenance.outcome_definition == SEMANTIC_SIMILARITY_OUTCOME

    with pytest.raises(ValueError, match="do not match exactly"):
        paired_semantic_similarity_study(
            "tree-001",
            direct_pairs=direct,
            semantic_pairs=semantic[:-1],
            direct_artifact_sha256=_digest("direct-artifact"),
            semantic_artifact_sha256=_digest("semantic-artifact"),
        )


def _study(study_id: str, correlation: float, n_pairs: int) -> CorrelationStudy:
    provenance = StudyInputProvenance(
        pair_ids_sha256=_digest(f"{study_id}-pairs"),
        predictor_values_sha256=_digest(f"{study_id}-predictor"),
        outcome_values_sha256=_digest(f"{study_id}-outcome"),
        direct_artifact_sha256=_digest(f"{study_id}-direct"),
        semantic_artifact_sha256=_digest(f"{study_id}-semantic"),
    )
    return CorrelationStudy(study_id, correlation, n_pairs, provenance)


def test_fisher_z_dl_matches_independent_golden_values_and_binds_outputs(
    tmp_path: Path,
) -> None:
    summary = fisher_z_dersimonian_laird(
        [
            _study("tree-1", 0.10, 20),
            _study("tree-2", 0.30, 25),
            _study("tree-3", 0.50, 30),
            _study("tree-4", 0.70, 35),
        ],
        analysis_id="ordinary-translation",
        endpoint="paired_semantic_similarity",
    )

    assert summary.equal_weight_r == pytest.approx(0.427321663183899)
    assert summary.fixed_effect.fisher_z == pytest.approx(0.5214287243584279)
    assert summary.fixed_effect.correlation == pytest.approx(0.47880195265759107)
    assert summary.fixed_effect.se_z == pytest.approx(0.10101525445522107)
    assert summary.heterogeneity.q == pytest.approx(7.851410999446063)
    assert summary.heterogeneity.c == pytest.approx(72.22448979591837)
    assert summary.heterogeneity.tau2 == pytest.approx(0.06717127408105597)
    assert summary.heterogeneity.i2 == pytest.approx(0.6179030749744654)
    assert summary.heterogeneity.q_p_value == pytest.approx(0.04918451297857811)
    assert summary.random_effect.fisher_z == pytest.approx(0.4820009399202188)
    assert summary.random_effect.correlation == pytest.approx(0.44784466455136623)
    assert summary.random_effect.se_z == pytest.approx(0.16535734559422272)

    json_path = tmp_path / "meta.json"
    csv_path = tmp_path / "meta.csv"
    write_meta_summary_json(
        summary,
        json_path,
        provenance={
            "cohort_receipt_sha256": _digest("cohort-receipt"),
            "semantic_revision": MINILM_REVISION,
        },
    )
    write_meta_summaries_csv([summary], csv_path)
    payload = json.loads(json_path.read_text())
    assert payload["schema"] == BEHAVIOR_META_SCHEMA
    assert payload["summary"]["study_inputs_sha256"] == summary.study_inputs_sha256
    assert (
        payload["provenance_sha256"]
        == hashlib.sha256(
            (
                json.dumps(
                    payload["provenance"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        ).hexdigest()
    )
    with csv_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["study_inputs_sha256"] == summary.study_inputs_sha256
    assert float(row["random_effect_r"]) == pytest.approx(0.44784466455136623)
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.lock"))

    with pytest.raises(ValueError, match="does not match its study effects"):
        write_meta_summary_json(
            replace(summary, equal_weight_r=0.9),
            tmp_path / "forged.json",
            provenance={"cohort_receipt_sha256": _digest("cohort-receipt")},
        )


def test_meta_analysis_and_pair_correlation_fail_closed() -> None:
    pair_ids = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c")]
    with pytest.raises(ValueError, match="exact same number"):
        pearson_correlation_study(
            "tree",
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            pair_ids=pair_ids,
            direct_artifact_sha256=_digest("direct"),
            semantic_artifact_sha256=_digest("semantic"),
            outcome_definition=SEMANTIC_SIMILARITY_OUTCOME,
        )
    with pytest.raises(ValueError, match="constant or unstable"):
        pearson_correlation_study(
            "tree",
            [1, 1, 1, 1],
            [1, 2, 3, 4],
            pair_ids=pair_ids,
            direct_artifact_sha256=_digest("direct"),
            semantic_artifact_sha256=_digest("semantic"),
            outcome_definition=SEMANTIC_SIMILARITY_OUTCOME,
        )
    with pytest.raises(ValueError, match="study IDs must be unique"):
        fisher_z_dersimonian_laird(
            [
                _study("same", 0.1, 10),
                _study("same", 0.2, 10),
                _study("other", 0.3, 10),
            ],
            analysis_id="analysis",
            endpoint="endpoint",
        )
    with pytest.raises(ValueError, match="input_provenance"):
        fisher_z_dersimonian_laird(
            [
                {"study_id": "a", "r": 0.1, "n": 10},
                {"study_id": "b", "r": 0.2, "n": 10},
                {"study_id": "c", "r": 0.3, "n": 10},
            ],
            analysis_id="analysis",
            endpoint="endpoint",
        )


def test_meta_zero_heterogeneity_is_explicit() -> None:
    summary = fisher_z_dersimonian_laird(
        [
            _study("tree-1", 0.2, 20),
            _study("tree-2", 0.2, 25),
            _study("tree-3", 0.2, 30),
        ],
        analysis_id="analysis",
        endpoint="endpoint",
    )
    assert summary.heterogeneity.q == pytest.approx(0.0)
    assert summary.heterogeneity.tau2 == pytest.approx(0.0)
    assert summary.heterogeneity.i2 == pytest.approx(0.0)
    assert summary.random_effect.correlation == pytest.approx(0.2)
