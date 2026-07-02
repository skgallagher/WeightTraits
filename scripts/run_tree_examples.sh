#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p reports/examples

run_wt() {
  if PYTHONPATH=src python -c "import yaml" >/dev/null 2>&1; then
    PYTHONPATH=src python -m weighttraits.cli "$@"
    return
  fi

  if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "ellmtrees"; then
    conda run -n ellmtrees env PYTHONPATH=src python -m weighttraits.cli "$@"
    return
  fi

  echo "Could not import PyYAML from current Python, and conda env 'ellmtrees' was not found." >&2
  echo "Install the local environment or run: conda run -n ellmtrees scripts/run_tree_examples.sh" >&2
  exit 1
}

for config in examples/trees/*.yaml; do
  name="$(basename "${config}" .yaml)"
  if [[ "${name}" == "README" ]]; then
    continue
  fi
  echo "==> ${name}"
  run_wt generate-tree \
    --config "${config}" \
    --out "reports/examples/${name}.manifest.jsonl"
done
