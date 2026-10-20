from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from weighttraits.behavior.contracts import (
    load_behavior_protocol_registry,
    load_source_fixture,
    render_probe_prompt,
    response_is_eligible,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "examples/behavior/corrected_behavior_protocols_v1.json"


def _registry():
    return load_behavior_protocol_registry(REGISTRY_PATH)


def _template(protocol_id: str, model_task: str, probe_id: str) -> str:
    return _registry().protocol(protocol_id).architectures[model_task].prompt_templates[probe_id]


def test_registry_pins_exact_models_fixtures_and_source_artifacts() -> None:
    registry = _registry()

    assert registry.sha256 == "c13b025802b7fae7b5c1770626865e2bab7d45e5783137b85fd041e2b4689bab"
    assert registry.model_pins["causal_lm"].model_id == "meta-llama/Llama-3.2-1B"
    assert registry.model_pins["causal_lm"].revision == (
        "4e20de362430cd3b72f300e6b0f18e50e7166e08"
    )
    assert registry.model_pins["seq2seq"].model_id == "google/flan-t5-base"
    assert registry.model_pins["seq2seq"].revision == (
        "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
    )
    assert registry.embedding_pin.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert registry.embedding_pin.revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

    assert {
        fixture_id: fixture.sha256 for fixture_id, fixture in registry.fixtures.items()
    } == {
        "legacy_translation_wmt14_fr_en_seed42_n200": (
            "d143e88e9d290e019e9ff1850c8a585a1df37c6daf76c933473e905cce40dffe"
        ),
        "modern_mc_four_probe_seed42_n100_each": (
            "896d21217e5920d3ad41affaa018e391f8bb98b9aa3eb6bad6b830d823defade"
        ),
        "dolly_open_ended_seed42_n30": (
            "3ed1ee9fd92121758934ef0bb31eeacd451305f734ea3dbff076f12207e9caa5"
        ),
    }
    assert (
        registry.fixtures["modern_mc_four_probe_seed42_n100_each"]
        .source_prompt_artifact.sha256
        == "1f783e7081c5a563bb4576e6a491d57e32204ace42896ec923f74126a6c7643d"
    )
    assert (
        registry.fixtures["dolly_open_ended_seed42_n30"].source_prompt_artifact.sha256
        == "10226d2a5ae731ff549d292736968bc9643dd618bc6622207f5b764da659cd4b"
    )


def test_exact_source_indices_and_dataset_revisions_are_frozen() -> None:
    registry = _registry()
    translation = registry.fixtures[
        "legacy_translation_wmt14_fr_en_seed42_n200"
    ].probes["translation"]
    assert translation.dataset.revision == "b199e406369ec1b7634206d3ded5ba45de2fe696"
    assert translation.selection.source_indices == tuple(
        np.random.default_rng(42).permutation(3003)[:200].tolist()
    )
    assert translation.selection.source_indices_sha256 == (
        "883ac2707f1c546e9ff874253979d3213524853910f69277a7775c1ab5abcf99"
    )

    modern = registry.fixtures["modern_mc_four_probe_seed42_n100_each"]
    expected = {
        "hellaswag": (
            10042,
            "218ec52e09a7e7462a5400043bb9a69a41d06b76",
            "eca91ac0d6ba6ecf66c313c449a1832d1a1fab117346caef2b1576566dd22b47",
        ),
        "arc_challenge": (
            1172,
            "210d026faf9955653af8916fad021475a3f00453",
            "33471c036bffa19ed2c0a62bcf051033a9ecccd3324433c64d6c8b583593bd68",
        ),
        "mmlu": (
            14042,
            "c30699e8356da336a370243923dbaf21066bb9fe",
            "f31cd612b191e069d6c34e5775ed23cc8a1f213819f99325fb808b6363036336",
        ),
        "truthfulqa": (
            817,
            "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
            "d77fe6b6a49ded2470c4ad18c4a1c076711a1ae2e4f02366b0e2f0f7583aac8c",
        ),
    }
    for probe_id, (row_count, revision, indices_sha256) in expected.items():
        probe = modern.probes[probe_id]
        expected_indices = tuple(
            np.random.default_rng(42).choice(row_count, size=100, replace=False).tolist()
        )
        assert probe.selection.source_indices == expected_indices
        assert probe.selection.source_indices_sha256 == indices_sha256
        assert probe.dataset.revision == revision

    dolly = registry.fixtures["dolly_open_ended_seed42_n30"].probes["dolly_open_ended"]
    assert dolly.dataset.revision == "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
    assert dolly.selection.source_indices == (
        9637,
        7852,
        6403,
        12823,
        1313,
        1375,
        2722,
        12332,
        11758,
        10439,
        1887,
        3004,
        14575,
        7656,
        12577,
        11695,
        5525,
        11024,
        5975,
        6484,
        6564,
        8148,
        6669,
        7437,
        11560,
        13892,
        10739,
        9777,
        1270,
        11391,
    )
    assert dolly.selection.source_indices_sha256 == (
        "6103d90e8930ffa4feb8db9df7792fe819d0dfebdb1fa4dbdf7f71b6d0c95a86"
    )


def test_exact_generation_draw_contracts() -> None:
    registry = _registry()
    translation = registry.protocol("legacy_translation_200x1_greedy")
    assert translation.prompt_counts == {"translation": 200}
    assert translation.samples_per_prompt == 1
    assert translation.draw_seeds == (42,)
    assert translation.generation_options(model_task="causal_lm", sample_id=0) == {
        "sample_id": 0,
        "seed": 42,
        "generation_batch_size": 16,
        "seed_scope": "probe_draw",
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "min_new_tokens": 0,
        "max_new_tokens": 64,
        "empty_policy": "preserve",
    }
    assert translation.generation_options(model_task="seq2seq", sample_id=0)[
        "max_new_tokens"
    ] == 128

    modern = registry.protocol("modern_mc_100x3_sampled")
    dolly = registry.protocol("dolly_open_ended_30x3_sampled")
    for protocol, maximum in ((modern, 64), (dolly, 96)):
        assert protocol.samples_per_prompt == 3
        assert protocol.draw_seeds == (42, 43, 44)
        for sample_id, seed in enumerate((42, 43, 44)):
            options = protocol.generation_options(
                model_task="causal_lm", sample_id=sample_id
            )
            assert options["seed"] == seed
            assert options["generation_batch_size"] == 32
            assert options["seed_scope"] == "probe_draw"
            assert options["do_sample"] is True
            assert options["temperature"] == 1.0
            assert options["top_p"] == 1.0
            assert options["max_new_tokens"] == maximum

    with pytest.raises(ValueError, match="outside"):
        modern.generation_options(model_task="causal_lm", sample_id=3)


def test_architecture_specific_templates_render_exactly() -> None:
    translation_row = {"translation": {"en": "Hello.", "fr": "Bonjour."}}
    assert render_probe_prompt(
        probe_id="translation",
        model_task="causal_lm",
        source_row=translation_row,
        template=_template(
            "legacy_translation_200x1_greedy", "causal_lm", "translation"
        ),
    ) == (
        "Translate the following English text to French.\nEnglish: Hello.\nFrench:",
        "Bonjour.",
    )
    assert render_probe_prompt(
        probe_id="translation",
        model_task="seq2seq",
        source_row=translation_row,
        template=_template("legacy_translation_200x1_greedy", "seq2seq", "translation"),
    )[0] == "translate English to French: Hello."

    hellaswag = {
        "activity_label": "Cooking",
        "ctx": "A person opens the oven.",
        "endings": ["They bake.", "They swim."],
        "label": "0",
    }
    assert render_probe_prompt(
        probe_id="hellaswag",
        model_task="causal_lm",
        source_row=hellaswag,
        template=_template("modern_mc_100x3_sampled", "causal_lm", "hellaswag"),
    ) == (
        "Complete the following sentence.\n\nCooking: A person opens the oven.",
        "They bake.",
    )
    assert render_probe_prompt(
        probe_id="hellaswag",
        model_task="seq2seq",
        source_row=hellaswag,
        template=_template("modern_mc_100x3_sampled", "seq2seq", "hellaswag"),
    )[0] == "complete: Cooking: A person opens the oven."

    arc = {
        "question": "Which?",
        "choices": {"label": ["1", "2"], "text": ["first", "second"]},
        "answerKey": "2",
    }
    assert render_probe_prompt(
        probe_id="arc_challenge",
        model_task="causal_lm",
        source_row=arc,
        template=_template("modern_mc_100x3_sampled", "causal_lm", "arc_challenge"),
    ) == (
        "Answer the following multiple-choice question in your own words.\n\n"
        "Question: Which?\n1) first\n2) second\n\nAnswer:\n",
        "second",
    )
    assert render_probe_prompt(
        probe_id="arc_challenge",
        model_task="seq2seq",
        source_row=arc,
        template=_template("modern_mc_100x3_sampled", "seq2seq", "arc_challenge"),
    )[0] == "answer: Which?\n1) first\n2) second"

    mmlu = {"question": "Pick?", "choices": ["one", "two"], "answer": 1}
    assert render_probe_prompt(
        probe_id="mmlu",
        model_task="causal_lm",
        source_row=mmlu,
        template=_template("modern_mc_100x3_sampled", "causal_lm", "mmlu"),
    )[0].endswith("Question: Pick?\nA) one\nB) two\n\nAnswer:\n")

    truthful = {"question": "True?", "best_answer": "Yes."}
    assert render_probe_prompt(
        probe_id="truthfulqa",
        model_task="seq2seq",
        source_row=truthful,
        template=_template("modern_mc_100x3_sampled", "seq2seq", "truthfulqa"),
    ) == ("answer truthfully: True?", "Yes.")

    dolly = {
        "instruction": "Explain rain.",
        "context": "For a child.",
        "response": "Water falls from clouds.",
        "category": "general_qa",
    }
    assert render_probe_prompt(
        probe_id="dolly_open_ended",
        model_task="causal_lm",
        source_row=dolly,
        template=_template(
            "dolly_open_ended_30x3_sampled", "causal_lm", "dolly_open_ended"
        ),
    ) == (
        "Respond naturally to the following request. Use complete sentences when appropriate.\n\n"
        "Request: Explain rain.\n\nContext: For a child.\n\nResponse:\n",
        "Water falls from clouds.",
    )
    assert render_probe_prompt(
        probe_id="dolly_open_ended",
        model_task="seq2seq",
        source_row=dolly,
        template=_template(
            "dolly_open_ended_30x3_sampled", "seq2seq", "dolly_open_ended"
        ),
    )[0] == "respond naturally: Explain rain. context: For a child."


def test_model_task_and_revisions_are_explicit_and_fail_closed() -> None:
    registry = _registry()
    with pytest.raises(ValueError, match="required explicitly"):
        registry.validate_model_binding(
            model_task=None,  # type: ignore[arg-type]
            model_id="meta-llama/Llama-3.2-1B",
            model_revision="4e20de362430cd3b72f300e6b0f18e50e7166e08",
        )
    with pytest.raises(ValueError, match="model revision mismatch"):
        registry.validate_model_binding(
            model_task="causal_lm",
            model_id="meta-llama/Llama-3.2-1B",
            model_revision="0" * 40,
        )
    with pytest.raises(ValueError, match="dataset revision mismatch"):
        registry.materialize_probe(
            protocol_id="legacy_translation_200x1_greedy",
            probe_id="translation",
            rows={},
            model_task="causal_lm",
            dataset_name="wmt/wmt14",
            dataset_config="fr-en",
            dataset_revision="0" * 40,
        )
    with pytest.raises(ValueError, match="required explicitly"):
        render_probe_prompt(
            probe_id="translation",
            model_task=None,  # type: ignore[arg-type]
            source_row={"translation": {"en": "x", "fr": "y"}},
            template="{en}",
        )


def test_materialization_is_deterministic_and_hashes_exact_content() -> None:
    registry = _registry()
    fixture = registry.fixtures["legacy_translation_wmt14_fr_en_seed42_n200"]
    indices = fixture.probes["translation"].selection.source_indices
    rows = {
        index: {
            "translation": {"en": f"English {index}", "fr": f"French {index}"},
            "unused": "constant metadata",
        }
        for index in indices
    }
    kwargs = {
        "protocol_id": "legacy_translation_200x1_greedy",
        "probe_id": "translation",
        "rows": rows,
        "model_task": "causal_lm",
        "dataset_name": "wmt/wmt14",
        "dataset_config": "fr-en",
        "dataset_revision": "b199e406369ec1b7634206d3ded5ba45de2fe696",
    }
    first = registry.materialize_probe(**kwargs)
    second = registry.materialize_probe(**kwargs)
    assert first == second
    assert len(first) == 200
    assert [item.source_index for item in first] == list(indices)
    assert first[0].to_dict()["source_row_sha256"] == first[0].source_row_sha256
    assert first[0].metadata["fixture_sha256"] == fixture.sha256
    assert first[0].source_row_sha256 == (
        "eda0b4d9537986331e10cb43955d2be76a81f543ae33856d280c6a2d67d460c6"
    )
    assert first[0].prompt_sha256 == (
        "7e86223882116bcf711a36c35eed06466f70e6d1d51cdde95d395dfcfab302d0"
    )
    assert first[0].prompt_sha256 == hashlib.sha256(first[0].prompt.encode("utf-8")).hexdigest()

    source_changed = {index: dict(row) for index, row in rows.items()}
    changed_index = indices[0]
    source_changed[changed_index] = {
        **source_changed[changed_index],
        "unused": "changed metadata",
    }
    changed_source_prompt = registry.materialize_probe(**{**kwargs, "rows": source_changed})[0]
    assert changed_source_prompt.source_row_sha256 != first[0].source_row_sha256
    assert changed_source_prompt.prompt_sha256 == first[0].prompt_sha256

    prompt_changed = {index: dict(row) for index, row in rows.items()}
    prompt_changed[changed_index] = {
        **prompt_changed[changed_index],
        "translation": {
            **prompt_changed[changed_index]["translation"],
            "en": "Changed prompt text",
        },
    }
    changed_render = registry.materialize_probe(**{**kwargs, "rows": prompt_changed})[0]
    assert changed_render.source_row_sha256 != first[0].source_row_sha256
    assert changed_render.prompt_sha256 != first[0].prompt_sha256


def test_materialization_rejects_missing_source_rows_and_ineligible_dolly() -> None:
    registry = _registry()
    with pytest.raises(ValueError, match="missing row index"):
        registry.materialize_probe(
            protocol_id="legacy_translation_200x1_greedy",
            probe_id="translation",
            rows={},
            model_task="causal_lm",
            dataset_name="wmt/wmt14",
            dataset_config="fr-en",
            dataset_revision="b199e406369ec1b7634206d3ded5ba45de2fe696",
        )

    dolly_indices = registry.fixtures["dolly_open_ended_seed42_n30"].probes[
        "dolly_open_ended"
    ].selection.source_indices
    dolly_rows = {
        index: {
            "instruction": "Do something.",
            "context": "",
            "response": "Done.",
            "category": "classification",
        }
        for index in dolly_indices
    }
    with pytest.raises(ValueError, match="eligibility"):
        registry.materialize_probe(
            protocol_id="dolly_open_ended_30x3_sampled",
            probe_id="dolly_open_ended",
            rows=dolly_rows,
            model_task="causal_lm",
            dataset_name="databricks/databricks-dolly-15k",
            dataset_config=None,
            dataset_revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        )


def test_primary_preserves_empty_and_sensitivity_excludes_only_exact_empty() -> None:
    protocol = _registry().protocol("modern_mc_100x3_sampled")
    assert response_is_eligible(protocol, "", analysis="primary")
    assert response_is_eligible(protocol, "  \n", analysis="primary")
    assert not response_is_eligible(protocol, "", analysis="sensitivity")
    assert not response_is_eligible(protocol, "  \n", analysis="sensitivity")
    assert response_is_eligible(protocol, "A", analysis="sensitivity")
    assert response_is_eligible(protocol, "fragment", analysis="sensitivity")


def test_fixture_sha_and_schema_mismatches_fail_closed(tmp_path: Path) -> None:
    fixture = ROOT / "examples/behavior/fixtures/dolly_open_ended_seed42_n30.json"
    changed = tmp_path / "changed.json"
    changed.write_text(fixture.read_text().replace("9637", "9638", 1))
    with pytest.raises(ValueError, match="fixture sha256 mismatch"):
        load_source_fixture(
            changed,
            expected_sha256="3ed1ee9fd92121758934ef0bb31eeacd451305f734ea3dbff076f12207e9caa5",
        )

    raw = json.loads(fixture.read_text())
    raw["unexpected"] = True
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="schema mismatch"):
        load_source_fixture(unknown)
