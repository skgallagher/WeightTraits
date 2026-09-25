"""End-to-end analysis workflows."""

from weighttraits.analysis.completion import (
    audit_training_run_set_completion,
    audit_training_tree_completion,
)
from weighttraits.analysis.branch_ordering import branch_ordering_stats
from weighttraits.analysis.direct import analyze_training_ledger_direct
from weighttraits.analysis.direct_attestation import (
    build_direct_source_contract,
    produce_attested_direct_analysis,
    validate_pinned_direct_replay_receipt,
    verify_direct_analysis_receipt,
    write_direct_analysis_replay_receipt,
)
from weighttraits.analysis.runset_results import summarize_training_run_set_analysis
from weighttraits.analysis.whitebox import analyze_training_ledger, analyze_training_run_set

__all__ = [
    "analyze_training_ledger",
    "analyze_training_ledger_direct",
    "analyze_training_run_set",
    "build_direct_source_contract",
    "branch_ordering_stats",
    "audit_training_run_set_completion",
    "audit_training_tree_completion",
    "summarize_training_run_set_analysis",
    "produce_attested_direct_analysis",
    "validate_pinned_direct_replay_receipt",
    "verify_direct_analysis_receipt",
    "write_direct_analysis_replay_receipt",
]
