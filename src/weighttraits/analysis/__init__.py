"""End-to-end analysis workflows."""

from weighttraits.analysis.completion import (
    audit_training_run_set_completion,
    audit_training_tree_completion,
)
from weighttraits.analysis.branch_ordering import branch_ordering_stats
from weighttraits.analysis.direct import analyze_training_ledger_direct
from weighttraits.analysis.runset_results import summarize_training_run_set_analysis
from weighttraits.analysis.whitebox import analyze_training_ledger, analyze_training_run_set

__all__ = [
    "analyze_training_ledger",
    "analyze_training_ledger_direct",
    "analyze_training_run_set",
    "branch_ordering_stats",
    "audit_training_run_set_completion",
    "audit_training_tree_completion",
    "summarize_training_run_set_analysis",
]
