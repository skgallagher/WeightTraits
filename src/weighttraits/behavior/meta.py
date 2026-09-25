"""Input-bound Fisher-z DerSimonian-Laird meta-analysis.

Each tree is one study. Its outcome is explicitly the paper endpoint,
``semantic similarity = 1 - paired cosine distance``. Model-pair keys are joined
before values are correlated, preventing same-length but differently ordered
upper triangles from being silently combined.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from numbers import Real
import os
from pathlib import Path
from statistics import NormalDist
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from weighttraits.behavior.direct_bridge import array_values_sha256


BEHAVIOR_META_SCHEMA = "weighttraits.behavior.fisher_z_dl.v2"
DEFAULT_CORRELATION_CLIP = 0.999
SEMANTIC_SIMILARITY_OUTCOME = "semantic similarity = 1 - paired cosine distance"

META_SUMMARY_COLUMNS = [
    "analysis_id",
    "endpoint",
    "effect_measure",
    "estimator",
    "n_studies",
    "n_pairs_min",
    "n_pairs_max",
    "n_pairs_total",
    "study_inputs_sha256",
    "correlation_clip",
    "alpha",
    "equal_weight_r",
    "fixed_effect_r",
    "fixed_effect_z",
    "fixed_effect_se_z",
    "fixed_effect_ci_low_r",
    "fixed_effect_ci_high_r",
    "fixed_effect_p_value",
    "random_effect_r",
    "random_effect_z",
    "random_effect_se_z",
    "random_effect_ci_low_r",
    "random_effect_ci_high_r",
    "random_effect_p_value",
    "q",
    "q_df",
    "q_p_value",
    "c",
    "tau2",
    "i2",
    "i2_percent",
]


@dataclass(frozen=True)
class StudyInputProvenance:
    """Hashes binding one study effect to exact pair keys and artifacts."""

    pair_ids_sha256: str
    predictor_values_sha256: str
    outcome_values_sha256: str
    direct_artifact_sha256: str
    semantic_artifact_sha256: str
    outcome_definition: str = SEMANTIC_SIMILARITY_OUTCOME

    def __post_init__(self) -> None:
        for field in (
            "pair_ids_sha256",
            "predictor_values_sha256",
            "outcome_values_sha256",
            "direct_artifact_sha256",
            "semantic_artifact_sha256",
        ):
            _validated_sha256(getattr(self, field), context=field)
        if self.outcome_definition != SEMANTIC_SIMILARITY_OUTCOME:
            raise ValueError(
                "study outcome must be semantic similarity = 1 - paired cosine distance"
            )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CorrelationStudy:
    """One run-level Pearson correlation and its bound input provenance."""

    study_id: str
    correlation: float
    n_pairs: int
    input_provenance: StudyInputProvenance


@dataclass(frozen=True)
class StudyEstimate:
    """Validated Fisher-z study estimate and analysis weights."""

    study_id: str
    n_pairs: int
    correlation: float
    clipped_correlation: float
    fisher_z: float
    variance_z: float
    fixed_weight: float
    random_weight: float
    input_provenance: StudyInputProvenance


@dataclass(frozen=True)
class PooledEffect:
    """One pooled effect on Fisher-z and correlation scales."""

    fisher_z: float
    correlation: float
    se_z: float
    ci_low_r: float
    ci_high_r: float
    z_statistic: float
    p_value: float


@dataclass(frozen=True)
class Heterogeneity:
    """Cochran/DerSimonian-Laird heterogeneity statistics."""

    q: float
    df: int
    q_p_value: float
    c: float
    tau2: float
    i2: float
    i2_percent: float


@dataclass(frozen=True)
class MetaAnalysisSummary:
    """Complete reusable result for one behavioral endpoint."""

    analysis_id: str
    endpoint: str
    effect_measure: str
    estimator: str
    correlation_clip: float
    alpha: float
    equal_weight_r: float
    fixed_effect: PooledEffect
    random_effect: PooledEffect
    heterogeneity: Heterogeneity
    studies: tuple[StudyEstimate, ...]
    study_inputs_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready representation."""

        return {
            "analysis_id": self.analysis_id,
            "endpoint": self.endpoint,
            "effect_measure": self.effect_measure,
            "estimator": self.estimator,
            "correlation_clip": self.correlation_clip,
            "alpha": self.alpha,
            "n_studies": len(self.studies),
            "n_pairs_min": min(study.n_pairs for study in self.studies),
            "n_pairs_max": max(study.n_pairs for study in self.studies),
            "n_pairs_total": sum(study.n_pairs for study in self.studies),
            "study_inputs_sha256": self.study_inputs_sha256,
            "equal_weight_r": self.equal_weight_r,
            "fixed_effect": asdict(self.fixed_effect),
            "random_effect": asdict(self.random_effect),
            "heterogeneity": asdict(self.heterogeneity),
            "studies": [asdict(study) for study in self.studies],
        }

    def to_csv_row(self) -> dict[str, Any]:
        """Flatten the headline fields for table-oriented consumers."""

        payload = self.to_dict()
        fixed = payload["fixed_effect"]
        random = payload["random_effect"]
        heterogeneity = payload["heterogeneity"]
        return {
            "analysis_id": self.analysis_id,
            "endpoint": self.endpoint,
            "effect_measure": self.effect_measure,
            "estimator": self.estimator,
            "n_studies": payload["n_studies"],
            "n_pairs_min": payload["n_pairs_min"],
            "n_pairs_max": payload["n_pairs_max"],
            "n_pairs_total": payload["n_pairs_total"],
            "study_inputs_sha256": self.study_inputs_sha256,
            "correlation_clip": self.correlation_clip,
            "alpha": self.alpha,
            "equal_weight_r": self.equal_weight_r,
            "fixed_effect_r": fixed["correlation"],
            "fixed_effect_z": fixed["fisher_z"],
            "fixed_effect_se_z": fixed["se_z"],
            "fixed_effect_ci_low_r": fixed["ci_low_r"],
            "fixed_effect_ci_high_r": fixed["ci_high_r"],
            "fixed_effect_p_value": fixed["p_value"],
            "random_effect_r": random["correlation"],
            "random_effect_z": random["fisher_z"],
            "random_effect_se_z": random["se_z"],
            "random_effect_ci_low_r": random["ci_low_r"],
            "random_effect_ci_high_r": random["ci_high_r"],
            "random_effect_p_value": random["p_value"],
            "q": heterogeneity["q"],
            "q_df": heterogeneity["df"],
            "q_p_value": heterogeneity["q_p_value"],
            "c": heterogeneity["c"],
            "tau2": heterogeneity["tau2"],
            "i2": heterogeneity["i2"],
            "i2_percent": heterogeneity["i2_percent"],
        }


def paired_semantic_similarity_study(
    study_id: str,
    *,
    direct_pairs: Sequence[object],
    semantic_pairs: Sequence[object],
    direct_artifact_sha256: str,
    semantic_artifact_sha256: str,
) -> CorrelationStudy:
    """Join exact unordered model-pair keys and correlate weight with similarity.

    Both inputs must expose ``left_model_id``, ``right_model_id``, and
    ``distance`` as either attributes or mapping keys. Semantic distances are
    converted to the contracted similarity outcome inside this function.
    """

    direct = _indexed_pairs(direct_pairs, context="direct")
    semantic = _indexed_pairs(semantic_pairs, context="semantic")
    if set(direct) != set(semantic):
        missing = sorted(set(direct) - set(semantic))
        extra = sorted(set(semantic) - set(direct))
        raise ValueError(
            "direct and semantic model-pair identities do not match exactly: "
            f"missing_semantic={missing}, extra_semantic={extra}"
        )
    pair_ids = tuple(sorted(direct))
    predictor = np.asarray([direct[key] for key in pair_ids], dtype=np.float64)
    semantic_distances = np.asarray([semantic[key] for key in pair_ids], dtype=np.float64)
    if np.any(semantic_distances < 0) or np.any(semantic_distances > 2):
        raise ValueError("paired semantic cosine distances must lie within [0, 2]")
    outcome = 1.0 - semantic_distances
    return pearson_correlation_study(
        study_id,
        predictor,
        outcome,
        pair_ids=pair_ids,
        direct_artifact_sha256=direct_artifact_sha256,
        semantic_artifact_sha256=semantic_artifact_sha256,
        outcome_definition=SEMANTIC_SIMILARITY_OUTCOME,
    )


def pearson_correlation_study(
    study_id: str,
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    pair_ids: Sequence[Sequence[str]],
    direct_artifact_sha256: str,
    semantic_artifact_sha256: str,
    outcome_definition: str,
) -> CorrelationStudy:
    """Compute one input-bound Pearson effect with strict pair alignment."""

    _validate_identifier(study_id, field="study_id")
    if outcome_definition != SEMANTIC_SIMILARITY_OUTCOME:
        raise ValueError("correlation outcome must explicitly be semantic similarity, not distance")
    _validated_sha256(direct_artifact_sha256, context="direct_artifact_sha256")
    _validated_sha256(semantic_artifact_sha256, context="semantic_artifact_sha256")
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.ndim != 1 or right_values.ndim != 1:
        raise ValueError("correlation inputs must be one-dimensional")
    if left_values.shape != right_values.shape:
        raise ValueError(
            "correlation inputs must contain the exact same number of aligned pairs; "
            f"got {left_values.shape} and {right_values.shape}"
        )
    normalized_pairs = _validated_pair_ids(pair_ids)
    if len(normalized_pairs) != len(left_values):
        raise ValueError("pair IDs must contain exactly one identity for each aligned value")
    if len(left_values) <= 3:
        raise ValueError("Fisher-z correlation studies require more than three pairs")
    if not np.all(np.isfinite(left_values)) or not np.all(np.isfinite(right_values)):
        raise ValueError("correlation inputs must be finite")
    left_centered = left_values - np.mean(left_values)
    right_centered = right_values - np.mean(right_values)
    denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if not math.isfinite(denominator) or denominator == 0:
        raise ValueError("correlation is undefined for constant or unstable input")
    correlation = float(np.sum(left_centered * right_centered) / denominator)
    provenance = StudyInputProvenance(
        pair_ids_sha256=_pair_ids_sha256(normalized_pairs),
        predictor_values_sha256=array_values_sha256(left_values),
        outcome_values_sha256=array_values_sha256(right_values),
        direct_artifact_sha256=direct_artifact_sha256,
        semantic_artifact_sha256=semantic_artifact_sha256,
        outcome_definition=outcome_definition,
    )
    return CorrelationStudy(
        study_id=study_id,
        correlation=float(np.clip(correlation, -1.0, 1.0)),
        n_pairs=len(left_values),
        input_provenance=provenance,
    )


def fisher_z_dersimonian_laird(
    studies: Sequence[CorrelationStudy | Mapping[str, Any]],
    *,
    analysis_id: str,
    endpoint: str,
    correlation_clip: float = DEFAULT_CORRELATION_CLIP,
    alpha: float = 0.05,
) -> MetaAnalysisSummary:
    """Pool input-bound run correlations using the audited legacy estimator."""

    _validate_identifier(analysis_id, field="analysis_id")
    _validate_identifier(endpoint, field="endpoint")
    if not isinstance(correlation_clip, Real) or not math.isfinite(correlation_clip):
        raise ValueError("correlation_clip must be finite and strictly between zero and one")
    if not 0 < correlation_clip < 1:
        raise ValueError("correlation_clip must be finite and strictly between zero and one")
    if not isinstance(alpha, Real) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one")
    normalized = tuple(_coerce_study(study) for study in studies)
    if len(normalized) < 3:
        raise ValueError("DerSimonian-Laird meta-analysis requires at least three studies")
    study_ids = [study.study_id for study in normalized]
    if len(set(study_ids)) != len(study_ids):
        raise ValueError("meta-analysis study IDs must be unique")

    correlations = np.asarray([study.correlation for study in normalized], dtype=np.float64)
    pair_counts = np.asarray([study.n_pairs for study in normalized], dtype=np.float64)
    clipped = np.clip(correlations, -correlation_clip, correlation_clip)
    fisher_z = np.arctanh(clipped)
    variance = 1.0 / (pair_counts - 3.0)
    fixed_weights = 1.0 / variance
    fixed_z = float(np.sum(fixed_weights * fisher_z) / np.sum(fixed_weights))
    q = float(np.sum(fixed_weights * (fisher_z - fixed_z) ** 2))
    df = len(normalized) - 1
    c_value = float(np.sum(fixed_weights) - np.sum(fixed_weights**2) / np.sum(fixed_weights))
    if c_value <= 0:
        raise ValueError("DerSimonian-Laird C is non-positive for the supplied studies")
    tau2 = max(0.0, (q - df) / c_value)
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    random_weights = 1.0 / (variance + tau2)
    random_z = float(np.sum(random_weights * fisher_z) / np.sum(random_weights))

    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    fixed_effect = _pooled_effect(
        fisher_z=fixed_z,
        se_z=math.sqrt(1.0 / float(np.sum(fixed_weights))),
        critical=critical,
    )
    random_effect = _pooled_effect(
        fisher_z=random_z,
        se_z=math.sqrt(1.0 / float(np.sum(random_weights))),
        critical=critical,
    )
    study_estimates = tuple(
        StudyEstimate(
            study_id=study.study_id,
            n_pairs=study.n_pairs,
            correlation=study.correlation,
            clipped_correlation=float(clipped[index]),
            fisher_z=float(fisher_z[index]),
            variance_z=float(variance[index]),
            fixed_weight=float(fixed_weights[index]),
            random_weight=float(random_weights[index]),
            input_provenance=study.input_provenance,
        )
        for index, study in enumerate(normalized)
    )
    study_inputs_hash = _study_inputs_sha256(normalized)
    return MetaAnalysisSummary(
        analysis_id=analysis_id,
        endpoint=endpoint,
        effect_measure="Pearson r transformed with Fisher z",
        estimator="DerSimonian-Laird random effects",
        correlation_clip=float(correlation_clip),
        alpha=float(alpha),
        equal_weight_r=float(np.tanh(np.mean(fisher_z))),
        fixed_effect=fixed_effect,
        random_effect=random_effect,
        heterogeneity=Heterogeneity(
            q=q,
            df=df,
            q_p_value=_chi_square_survival(q, df),
            c=c_value,
            tau2=tau2,
            i2=i2,
            i2_percent=100.0 * i2,
        ),
        studies=study_estimates,
        study_inputs_sha256=study_inputs_hash,
    )


def write_meta_summary_json(
    summary: MetaAnalysisSummary,
    path: str | Path,
    *,
    provenance: Mapping[str, Any],
) -> None:
    """Atomically write a self-describing, input-bound meta-analysis artifact."""

    _validate_meta_summary(summary)
    normalized_provenance = _validated_artifact_provenance(provenance)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": BEHAVIOR_META_SCHEMA,
        "producer": "weighttraits",
        "summary": summary.to_dict(),
        "provenance": normalized_provenance,
        "provenance_sha256": hashlib.sha256(
            _canonical_json_bytes(normalized_provenance)
        ).hexdigest(),
    }
    with _artifact_write_lock(out.parent, name=f"meta-json-{out.name}"):
        _atomic_write_text(
            out,
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )


def write_meta_summaries_csv(
    summaries: Sequence[MetaAnalysisSummary],
    path: str | Path,
) -> None:
    """Atomically write one input-bound headline row per endpoint."""

    rows = list(summaries)
    if not rows:
        raise ValueError("at least one meta-analysis summary is required")
    for summary in rows:
        _validate_meta_summary(summary)
    keys = [(summary.analysis_id, summary.endpoint) for summary in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("meta-analysis CSV keys (analysis_id, endpoint) must be unique")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=META_SUMMARY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for summary in rows:
        writer.writerow(summary.to_csv_row())
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with _artifact_write_lock(out.parent, name=f"meta-csv-{out.name}"):
        _atomic_write_text(out, buffer.getvalue())


def _indexed_pairs(rows: Sequence[object], *, context: str) -> dict[tuple[str, str], float]:
    indexed: dict[tuple[str, str], float] = {}
    for index, raw in enumerate(rows):
        left = _pair_field(raw, "left_model_id", context=f"{context}[{index}]")
        right = _pair_field(raw, "right_model_id", context=f"{context}[{index}]")
        _validate_identifier(left, field=f"{context} left_model_id")
        _validate_identifier(right, field=f"{context} right_model_id")
        if left == right:
            raise ValueError(f"{context} pair cannot contain the same model twice")
        key = tuple(sorted((left, right)))
        if key in indexed:
            raise ValueError(f"duplicate {context} model pair: {key}")
        value = _pair_field(raw, "distance", context=f"{context}[{index}]")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{context} pair distance must be numeric")
        distance = float(value)
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(f"{context} pair distance must be finite and non-negative")
        indexed[key] = distance
    if len(indexed) <= 3:
        raise ValueError(f"{context} pairs require more than three unique model pairs")
    return indexed


def _pair_field(raw: object, field: str, *, context: str) -> Any:
    if isinstance(raw, Mapping):
        if field not in raw:
            raise ValueError(f"{context} is missing {field}")
        return raw[field]
    if not hasattr(raw, field):
        raise ValueError(f"{context} is missing {field}")
    return getattr(raw, field)


def _validated_pair_ids(raw: Sequence[Sequence[str]]) -> tuple[tuple[str, str], ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("pair IDs must be a sequence")
    normalized: list[tuple[str, str]] = []
    for index, pair in enumerate(raw):
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"pair ID {index} must contain exactly two model IDs")
        left, right = pair
        _validate_identifier(left, field=f"pair_ids[{index}].left")
        _validate_identifier(right, field=f"pair_ids[{index}].right")
        if left == right:
            raise ValueError(f"pair ID {index} cannot contain the same model twice")
        normalized.append(tuple(sorted((left, right))))
    if len(set(normalized)) != len(normalized):
        raise ValueError("pair IDs must be unique")
    return tuple(normalized)


def _pair_ids_sha256(pair_ids: Sequence[tuple[str, str]]) -> str:
    rows = [{"left_model_id": left, "right_model_id": right} for left, right in pair_ids]
    return hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()


def _coerce_study(raw: CorrelationStudy | Mapping[str, Any]) -> CorrelationStudy:
    if isinstance(raw, CorrelationStudy):
        study = raw
    elif isinstance(raw, Mapping):
        study_id = raw.get("study_id", raw.get("tree_id", raw.get("run")))
        correlation = raw.get("correlation", raw.get("r"))
        n_pairs = raw.get("n_pairs", raw.get("n"))
        provenance_raw = raw.get("input_provenance")
        if not isinstance(provenance_raw, Mapping):
            raise ValueError(f"{study_id}: input_provenance must be a mapping")
        try:
            input_provenance = StudyInputProvenance(**dict(provenance_raw))
        except TypeError as exc:
            raise ValueError(f"{study_id}: invalid input provenance: {exc}") from exc
        study = CorrelationStudy(
            study_id=study_id,
            correlation=correlation,
            n_pairs=n_pairs,
            input_provenance=input_provenance,
        )
    else:
        raise ValueError("meta-analysis studies must be CorrelationStudy values or mappings")
    _validate_identifier(study.study_id, field="study_id")
    if isinstance(study.n_pairs, bool) or not isinstance(study.n_pairs, int):
        raise ValueError(f"{study.study_id}: n_pairs must be an integer")
    if study.n_pairs <= 3:
        raise ValueError(f"{study.study_id}: Fisher-z variance requires more than three pairs")
    if isinstance(study.correlation, bool) or not isinstance(study.correlation, Real):
        raise ValueError(f"{study.study_id}: correlation must be numeric")
    correlation = float(study.correlation)
    if not math.isfinite(correlation) or correlation < -1 or correlation > 1:
        raise ValueError(f"{study.study_id}: correlation must be finite and within [-1, 1]")
    if not isinstance(study.input_provenance, StudyInputProvenance):
        raise ValueError(f"{study.study_id}: input provenance is required")
    return CorrelationStudy(
        study_id=study.study_id,
        correlation=correlation,
        n_pairs=study.n_pairs,
        input_provenance=study.input_provenance,
    )


def _study_inputs_sha256(studies: Sequence[CorrelationStudy]) -> str:
    rows = [
        {
            "study_id": study.study_id,
            "n_pairs": study.n_pairs,
            "input_provenance": study.input_provenance.to_dict(),
        }
        for study in sorted(studies, key=lambda value: value.study_id)
    ]
    return hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()


def _validate_meta_summary(summary: object) -> None:
    if not isinstance(summary, MetaAnalysisSummary):
        raise ValueError("summary must be a MetaAnalysisSummary")
    if len(summary.studies) < 3:
        raise ValueError("meta-analysis summary must contain at least three studies")
    reconstructed = tuple(
        CorrelationStudy(
            study_id=study.study_id,
            correlation=study.correlation,
            n_pairs=study.n_pairs,
            input_provenance=study.input_provenance,
        )
        for study in summary.studies
    )
    if summary.study_inputs_sha256 != _study_inputs_sha256(reconstructed):
        raise ValueError("meta-analysis summary input-provenance hash mismatch")
    recalculated = fisher_z_dersimonian_laird(
        reconstructed,
        analysis_id=summary.analysis_id,
        endpoint=summary.endpoint,
        correlation_clip=summary.correlation_clip,
        alpha=summary.alpha,
    )
    if summary.to_dict() != recalculated.to_dict():
        raise ValueError("meta-analysis summary does not match its study effects")
    _canonical_json_bytes(summary.to_dict())


def _validated_artifact_provenance(provenance: object) -> dict[str, Any]:
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("non-empty meta-analysis artifact provenance is required")
    normalized = dict(provenance)
    for key in normalized:
        _validate_identifier(key, field="provenance key")
    _canonical_json_bytes(normalized)
    return normalized


def _pooled_effect(*, fisher_z: float, se_z: float, critical: float) -> PooledEffect:
    z_statistic = fisher_z / se_z
    return PooledEffect(
        fisher_z=fisher_z,
        correlation=math.tanh(fisher_z),
        se_z=se_z,
        ci_low_r=math.tanh(fisher_z - critical * se_z),
        ci_high_r=math.tanh(fisher_z + critical * se_z),
        z_statistic=z_statistic,
        p_value=math.erfc(abs(z_statistic) / math.sqrt(2.0)),
    )


def _chi_square_survival(value: float, df: int) -> float:
    """Chi-square survival function via the regularized upper gamma."""

    if value < 0 or df < 1:
        raise ValueError("chi-square value and degrees of freedom are invalid")
    return _regularized_gamma_q(df / 2.0, value / 2.0)


def _regularized_gamma_q(shape: float, value: float) -> float:
    """Numerically stable regularized upper incomplete gamma for positive inputs."""

    if shape <= 0 or value < 0:
        raise ValueError("regularized gamma inputs are invalid")
    if value == 0:
        return 1.0
    epsilon = 1e-14
    max_iterations = 10_000
    log_prefactor = -value + shape * math.log(value) - math.lgamma(shape)
    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        cursor = shape
        for _ in range(max_iterations):
            cursor += 1.0
            term *= value / cursor
            total += term
            if abs(term) <= abs(total) * epsilon:
                lower = total * math.exp(log_prefactor)
                return min(1.0, max(0.0, 1.0 - lower))
        raise RuntimeError("regularized gamma series did not converge")

    tiny = 1e-300
    b = value + 1.0 - shape
    if abs(b) < tiny:
        b = tiny
    c = 1.0 / tiny
    d = 1.0 / b
    fraction = d
    for iteration in range(1, max_iterations + 1):
        coefficient = -iteration * (iteration - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= epsilon:
            upper = math.exp(log_prefactor) * fraction
            return min(1.0, max(0.0, upper))
    raise RuntimeError("regularized gamma continued fraction did not converge")


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


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
