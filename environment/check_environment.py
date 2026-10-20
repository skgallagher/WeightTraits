#!/usr/bin/env python3
"""Compare the running Python environment with ``requirements-lock.txt``.

Prints every missing or mismatched package and exits nonzero if any differ. With ``--core`` only
the numerical stack recorded in the training ledgers is checked (python, torch, transformers,
datasets, accelerate, tokenizers, peft), which is what training reproducibility depends on.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import re
import sys
from pathlib import Path

LOCK = Path(__file__).with_name("requirements-lock.txt")
PYTHON = "3.11.15"
CORE = ("torch", "transformers", "datasets", "accelerate", "tokenizers", "peft")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions(path: Path = LOCK) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")) and "==" in line:
            name, version = line.split("==", 1)
            pins[_normalize(name)] = version.strip()
    return pins


def compare(pins: dict[str, str]) -> list[str]:
    problems = []
    for name, expected in sorted(pins.items()):
        try:
            found = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{name}: missing (locked {expected})")
            continue
        if found != expected:
            problems.append(f"{name}: {found} (locked {expected})")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", action="store_true", help="check only the training stack")
    args = parser.parse_args(argv)
    pins = locked_versions()
    if args.core:
        pins = {name: pins[name] for name in CORE}
    problems = compare(pins)
    if platform.python_version() != PYTHON:
        problems.insert(0, f"python: {platform.python_version()} (locked {PYTHON})")
    for problem in problems:
        print(problem)
    print(f"{len(pins) + 1 - len(problems)}/{len(pins) + 1} match the lock")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
