"""Inventory helpers for the ELLMTrees reference repository."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


_INTERESTING_DIRS = (
    "analysis",
    "blackbox",
    "codex",
    "configs",
    "notes",
    "outputs",
    "paper",
    "pipeline",
    "results",
    "scripts",
    "tests",
    "trees",
)


def _count_files(root: Path, pattern: str) -> int:
    return sum(1 for path in root.rglob(pattern) if path.is_file())


def _count_immediate_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for child in path.iterdir() if child.is_dir())


def inventory_ellmtrees(source: str | Path) -> dict[str, Any]:
    """Return a lightweight, JSON-serializable inventory of an ELLMTrees checkout."""

    root = Path(source).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    suffix_counts: Counter[str] = Counter()
    total_files = 0
    for path in root.rglob("*"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.is_file():
            total_files += 1
            suffix_counts[path.suffix or "<none>"] += 1

    dirs = {
        name: {
            "exists": (root / name).exists(),
            "files": _count_files(root / name, "*") if (root / name).exists() else 0,
            "immediate_dirs": _count_immediate_dirs(root / name),
        }
        for name in _INTERESTING_DIRS
    }

    paper = root / "paper"
    result_groups = sorted(p.name for p in (root / "results").iterdir() if p.is_dir()) if (root / "results").exists() else []
    output_groups = sorted(p.name for p in (root / "outputs").iterdir() if p.is_dir()) if (root / "outputs").exists() else []

    return {
        "source": str(root),
        "total_files_excluding_git_and_pycache": total_files,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "directories": dirs,
        "script_count": _count_files(root / "scripts", "*.py") if (root / "scripts").exists() else 0,
        "shell_script_count": _count_files(root / "scripts", "*.sh") if (root / "scripts").exists() else 0,
        "test_count": _count_files(root / "tests", "test_*.py") if (root / "tests").exists() else 0,
        "paper_tex_exists": (paper / "paper.tex").exists(),
        "paper_pdf_exists": (paper / "paper.pdf").exists(),
        "paper_figure_count": _count_files(paper / "figures", "*") if (paper / "figures").exists() else 0,
        "n_result_groups": len(result_groups),
        "n_output_groups": len(output_groups),
        "result_groups_sample": result_groups[:25],
        "output_groups_sample": output_groups[:25],
    }

