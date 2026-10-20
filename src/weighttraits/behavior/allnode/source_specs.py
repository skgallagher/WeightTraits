"""Exact immutable four-cohort source entry points for allnode_v1."""

from __future__ import annotations

from .checkpoints import ArtifactPin
from .loader import CohortSourceSpec
from .symlink_inventory import SourcePathAuthoritySpec, SourcePathMapping


FROZEN_STAGE = "/home/export/sgallagh/WeightTraits-all-results-20260817-clean"
DOWNSTREAM_STAGE = "/home/export/sgallagh/WeightTraits-downstream-20260831-native"
LLAMA_MODEL = "meta-llama/Llama-3.2-1B"
LLAMA_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
FLAN_MODEL = "google/flan-t5-base"
FLAN_REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
COMPLETION_SCHEMA = "weighttraits.training_completion_binding.v1"
FULL_FINETUNE_ARTIFACT_FACTOR = 2
SYMLINK_INVENTORY = f"{FROZEN_STAGE}/.stage_provenance/SYMLINKS.txt"
SYMLINK_INVENTORY_SHA256 = (
    "917c8457151a80eaff1233e4afc49406e8852193013e425f02305ddaab6da422"
)
STAGE_MANIFEST = f"{FROZEN_STAGE}/.stage_provenance/STAGE_SHA256SUMS"
STAGE_MANIFEST_SHA256 = (
    "642912687e67d0a48162a8d5cc0a95164ad92350ea629cd396fcfdd7917cd5d1"
)
LLAMA_FULL_SYMLINK_ENTRY = (
    f"{FROZEN_STAGE}/outputs/confirm_paper_numbers/"
    "llama32_1b_full_finetune_legacy_causal_2000"
)
LLAMA_NOTRANS_SYMLINK_ENTRY = (
    f"{FROZEN_STAGE}/outputs/translation_holdout_20260803/"
    "llama32_1b_full_finetune_legacy_causal_2000"
)
LLAMA_FULL_PHYSICAL_ROOT = (
    "/home/export/sgallagh/WeightTraits-legacy-causal-20260810-clean/"
    "outputs/confirm_paper_numbers/"
    "llama32_1b_full_finetune_legacy_causal_2000"
)
LLAMA_NOTRANS_PHYSICAL_ROOT = (
    "/home/export/sgallagh/WeightTraits-legacy-causal-20260810-clean/"
    "outputs/translation_holdout_20260803/"
    "llama32_1b_full_finetune_legacy_causal_2000"
)
SOURCE_PATH_AUTHORITY = SourcePathAuthoritySpec(
    stage_root=FROZEN_STAGE,
    stage_manifest=ArtifactPin(STAGE_MANIFEST, STAGE_MANIFEST_SHA256),
    symlink_inventory=ArtifactPin(SYMLINK_INVENTORY, SYMLINK_INVENTORY_SHA256),
    mappings=(
        SourcePathMapping(LLAMA_FULL_SYMLINK_ENTRY, LLAMA_FULL_PHYSICAL_ROOT),
        SourcePathMapping(LLAMA_NOTRANS_SYMLINK_ENTRY, LLAMA_NOTRANS_PHYSICAL_ROOT),
    ),
)


def production_source_specs() -> tuple[CohortSourceSpec, ...]:
    """Return the sole ordered all-node source set used by production."""

    return (
        _spec(
            cohort_key="llama_full",
            training_cohort_id="llama32_1b_full_finetune_legacy_causal_2000",
            config="examples/training/confirm_paper_numbers/llama32_1b_full_finetune_legacy_causal_2000.yaml",
            config_sha256="98c899d3f1c31fa2f0047128a0125b53c34fbc05a3104e1d21725a5bccf75248",
            summary="examples/training/confirm_paper_numbers/llama32_1b_full_finetune_legacy_causal_2000_training_run_list_summary.json",
            summary_sha256="b8b9e9d06abcc1f69e9083b91248eb45eaa2481daadea957e61fa406d507acb0",
            completion="runtime_receipts/LLAMA_FULL_COMPLETION_BINDING.json",
            completion_sha256="31628eca9fcc693f0aade432fc5774b0b3e7bb3ab7fb249b80c0e8eef998d5ba",
            model_id=LLAMA_MODEL,
            revision=LLAMA_REVISION,
            allowed_symlink_entry=LLAMA_FULL_SYMLINK_ENTRY,
        ),
        _spec(
            cohort_key="llama_notrans",
            training_cohort_id="llama32_1b_full_finetune_legacy_causal_2000",
            config="examples/training/translation_holdout_20260803/llama32_1b_full_finetune_legacy_causal_2000.yaml",
            config_sha256="ab259dc0af398b2e57b92096e81b67a1a070cb1891b02afdc1f65b02f8072d4a",
            summary="examples/training/translation_holdout_20260803/matched_legacy_causal_2000_training_run_list_summary.json",
            summary_sha256="494cdf2726b38e41811eadc35a50d366b417718269cd77602dd7570e9b8d13ff",
            completion="runtime_receipts/LLAMA_NOTRANS_COMPLETION_BINDING.json",
            completion_sha256="51449fc0e62d1681b0a3073401efc555178def762ef848609e8bd4ebd7ef5e59",
            model_id=LLAMA_MODEL,
            revision=LLAMA_REVISION,
            allowed_symlink_entry=LLAMA_NOTRANS_SYMLINK_ENTRY,
        ),
        _spec(
            cohort_key="flan_full",
            training_cohort_id="flan_t5_full_finetune_legacy_seq2seq_2000",
            config="examples/training/confirm_paper_numbers/flan_t5_full_finetune_legacy_seq2seq_2000.yaml",
            config_sha256="f5ae7e2e4cdec06689f5b31479382d5a2f2b2e6c9e47f88892955554cb9f47e1",
            summary="examples/training/confirm_paper_numbers/flan_t5_full_finetune_legacy_seq2seq_2000_training_run_list_summary.json",
            summary_sha256="2301eaac23551aa9b9489105d0f0f0b3f95611119d8573ecf1ad6da536244f47",
            completion="runtime_receipts/FLAN_FULL_COMPLETION_BINDING.json",
            completion_sha256="648e86ed54a147ce61d4eb67e364f9d992ad8b67fb3d763e62a3e88288933fbf",
            model_id=FLAN_MODEL,
            revision=FLAN_REVISION,
        ),
        _spec(
            cohort_key="flan_notrans",
            training_cohort_id="flan_t5_full_finetune_legacy_seq2seq_2000",
            config="examples/training/translation_holdout_20260803/flan_t5_full_finetune_legacy_seq2seq_2000.yaml",
            config_sha256="603b8b52893a7914ca14b95d876026a1cedf7b67d4795db9a2aa5b2ac26d5ef8",
            summary="examples/training/translation_holdout_20260803/matched_legacy_seq2seq_2000_training_run_list_summary.json",
            summary_sha256="f6d58e94b109bf35b8fa503d2901d96c3f8b884140dcb89b6a8349ea6d862eb9",
            completion="runtime_receipts/FLAN_NOTRANS_COMPLETION_BINDING.json",
            completion_sha256="fc850dd7b8838bb36350a1834dfc5563df3945d2c1c037f7fef712b7a62ace38",
            model_id=FLAN_MODEL,
            revision=FLAN_REVISION,
        ),
    )


def _spec(
    *,
    cohort_key: str,
    training_cohort_id: str,
    config: str,
    config_sha256: str,
    summary: str,
    summary_sha256: str,
    completion: str,
    completion_sha256: str,
    model_id: str,
    revision: str,
    allowed_symlink_entry: str | None = None,
) -> CohortSourceSpec:
    return CohortSourceSpec(
        cohort_key=cohort_key,
        training_cohort_id=training_cohort_id,
        stage_root=FROZEN_STAGE,
        top_config=ArtifactPin(
            path=f"{FROZEN_STAGE}/{config}",
            sha256=config_sha256,
        ),
        training_summary=ArtifactPin(
            path=f"{FROZEN_STAGE}/{summary}",
            sha256=summary_sha256,
        ),
        completion_receipt=ArtifactPin(
            path=f"{DOWNSTREAM_STAGE}/{completion}",
            sha256=completion_sha256,
        ),
        expected_method="full",
        expected_artifact_name="model",
        base_model_id=model_id,
        base_model_revision=revision,
        completion_receipt_root=DOWNSTREAM_STAGE,
        expected_completion_schema=COMPLETION_SCHEMA,
        expected_artifact_factor=FULL_FINETUNE_ARTIFACT_FACTOR,
        allowed_symlink_entries=(
            () if allowed_symlink_entry is None else (allowed_symlink_entry,)
        ),
        source_authority=SOURCE_PATH_AUTHORITY,
    )


__all__ = [
    "DOWNSTREAM_STAGE",
    "COMPLETION_SCHEMA",
    "FLAN_MODEL",
    "FLAN_REVISION",
    "FROZEN_STAGE",
    "FULL_FINETUNE_ARTIFACT_FACTOR",
    "LLAMA_MODEL",
    "LLAMA_REVISION",
    "LLAMA_FULL_SYMLINK_ENTRY",
    "LLAMA_FULL_PHYSICAL_ROOT",
    "LLAMA_NOTRANS_SYMLINK_ENTRY",
    "LLAMA_NOTRANS_PHYSICAL_ROOT",
    "SOURCE_PATH_AUTHORITY",
    "STAGE_MANIFEST",
    "STAGE_MANIFEST_SHA256",
    "SYMLINK_INVENTORY",
    "SYMLINK_INVENTORY_SHA256",
    "production_source_specs",
]
