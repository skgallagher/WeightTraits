"""Surface-form phenotypes for freely generated behavioral responses.

These features deliberately describe the emitted text rather than whether an
answer is correct.  Immediate EOS, label-only output, and repetition are valid
behavioral observations; they are not silently removed from this endpoint.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

import numpy as np

from weighttraits.behavior.responses import BehaviorResponse


SURFACE_FEATURE_NAMES = (
    "empty",
    "label_only",
    "repetition_loop",
    "natural_language",
    "fragment",
    "length_score",
    "token_count_score",
    "alphabetic_fraction",
    "digit_fraction",
    "lexical_diversity",
)
RESPONSE_CLASSES = ("empty", "label_only", "repetition_loop", "natural_language", "fragment")
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*|\d+", re.UNICODE)
_LABEL_TOKENS = {
    "a", "b", "c", "d", "e", "f", "g", "h",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "yes", "no", "true", "false", "entailment", "neutral", "contradiction",
    "positive", "negative",
}


@dataclass(frozen=True)
class SurfaceDistanceResult:
    distances: np.ndarray
    observation_counts: np.ndarray
    model_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    traits_by_model: dict[str, dict[str, Any]]
    audit: dict[str, Any]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def classify_response_text(response: str | None) -> str:
    """Classify an emitted string using auditable, model-independent rules."""
    text = (response or "").strip()
    if not text:
        return "empty"
    tokens = _tokens(text)
    if tokens and all(token in _LABEL_TOKENS for token in tokens):
        return "label_only"
    if len(tokens) >= 4 and max(Counter(tokens).values()) / len(tokens) >= 0.6:
        return "repetition_loop"
    alphabetic = [token for token in tokens if any(character.isalpha() for character in token)]
    alphabetic_characters = sum(character.isalpha() for character in text)
    if len(tokens) >= 4 and len(alphabetic) >= 3 and alphabetic_characters >= 12:
        return "natural_language"
    return "fragment"


def is_natural_language_response(response: str | None) -> bool:
    return classify_response_text(response) == "natural_language"


def response_surface_features(response: str | None) -> np.ndarray:
    """Return fixed-range surface features for one exact generated string."""
    text = (response or "").strip()
    response_class = classify_response_text(text)
    tokens = _tokens(text)
    characters = list(text)
    n_characters = len(characters)
    n_tokens = len(tokens)
    alphabetic_fraction = (
        sum(character.isalpha() for character in characters) / n_characters
        if n_characters
        else 0.0
    )
    digit_fraction = (
        sum(character.isdigit() for character in characters) / n_characters
        if n_characters
        else 0.0
    )
    lexical_diversity = len(set(tokens)) / n_tokens if n_tokens else 0.0
    values = [float(response_class == label) for label in RESPONSE_CLASSES]
    values.extend(
        [
            1.0 - math.exp(-n_characters / 64.0),
            1.0 - math.exp(-n_tokens / 16.0),
            alphabetic_fraction,
            digit_fraction,
            lexical_diversity,
        ]
    )
    return np.asarray(values, dtype=np.float64)


def paired_surface_distances(records: Sequence[BehaviorResponse]) -> SurfaceDistanceResult:
    """Average aligned per-output surface distance for every model pair.

    Completed empty strings and legacy ``empty_generation`` dropped records are
    represented as the empty phenotype. Other errors remain missing and are
    exposed in the pairwise observation-count audit.
    """
    if not records:
        raise ValueError("surface analysis requires at least one behavior response")
    run_ids = {record.run_id for record in records}
    probe_ids = {record.probe_id for record in records}
    if len(run_ids) != 1 or len(probe_ids) != 1:
        raise ValueError("surface analysis requires exactly one run and one probe")
    model_ids = tuple(sorted({record.model_id for record in records}))
    observation_ids = tuple(sorted({f"{record.prompt_id}:{record.sample_id}" for record in records}))
    model_index = {model_id: index for index, model_id in enumerate(model_ids)}
    observation_index = {value: index for index, value in enumerate(observation_ids)}
    features = np.zeros(
        (len(model_ids), len(observation_ids), len(SURFACE_FEATURE_NAMES)), dtype=np.float64
    )
    valid = np.zeros((len(model_ids), len(observation_ids)), dtype=bool)
    classes: dict[str, list[str]] = {model_id: [] for model_id in model_ids}
    for record in records:
        usable = record.status == "completed" or record.reason == "empty_generation"
        if not usable:
            continue
        left = model_index[record.model_id]
        observation_id = f"{record.prompt_id}:{record.sample_id}"
        right = observation_index[observation_id]
        if valid[left, right]:
            raise ValueError(f"duplicate behavior observation: {record.key}")
        text = record.response if record.status == "completed" else ""
        features[left, right] = response_surface_features(text)
        valid[left, right] = True
        classes[record.model_id].append(classify_response_text(text))

    distances = np.zeros((len(model_ids), len(model_ids)), dtype=np.float64)
    counts = np.zeros((len(model_ids), len(model_ids)), dtype=np.int64)
    missing_pairs: list[dict[str, str]] = []
    for left in range(len(model_ids)):
        counts[left, left] = int(np.sum(valid[left]))
        for right in range(left + 1, len(model_ids)):
            shared = valid[left] & valid[right]
            count = int(np.sum(shared))
            counts[left, right] = counts[right, left] = count
            if count == 0:
                distances[left, right] = distances[right, left] = np.nan
                missing_pairs.append({"model_i": model_ids[left], "model_j": model_ids[right]})
                continue
            value = float(np.mean(np.abs(features[left, shared] - features[right, shared])))
            distances[left, right] = distances[right, left] = value

    traits_by_model: dict[str, dict[str, Any]] = {}
    for model_id, index in model_index.items():
        model_features = features[index, valid[index]]
        class_counts = Counter(classes[model_id])
        n = int(model_features.shape[0])
        traits_by_model[model_id] = {
            "n_observations": n,
            "class_counts": {label: class_counts.get(label, 0) for label in RESPONSE_CLASSES},
            "class_fractions": {
                label: class_counts.get(label, 0) / n if n else None for label in RESPONSE_CLASSES
            },
            "mean_features": {
                name: float(np.mean(model_features[:, feature_index])) if n else None
                for feature_index, name in enumerate(SURFACE_FEATURE_NAMES)
            },
            "semantic_coverage": class_counts.get("natural_language", 0) / n if n else None,
        }

    return SurfaceDistanceResult(
        distances=distances,
        observation_counts=counts,
        model_ids=model_ids,
        observation_ids=observation_ids,
        feature_names=SURFACE_FEATURE_NAMES,
        traits_by_model=traits_by_model,
        audit={
            "schema_version": 1,
            "representation": "paired_output_surface_gower",
            "run_id": next(iter(run_ids)),
            "probe_id": next(iter(probe_ids)),
            "n_models": len(model_ids),
            "n_observations": len(observation_ids),
            "feature_names": list(SURFACE_FEATURE_NAMES),
            "empty_generation_policy": "preserved_as_observed_immediate_eos",
            "usable_observations_by_model": {
                model_id: int(np.sum(valid[index])) for model_id, index in model_index.items()
            },
            "n_pairs_without_shared_observations": len(missing_pairs),
            "pairs_without_shared_observations": missing_pairs,
        },
    )
