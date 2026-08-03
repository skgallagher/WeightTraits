# Contributing to WeightTraits

Contributions that improve correctness, reproducibility, portability, documentation, or focused
research workflows are welcome.

## Development setup

```bash
git clone https://github.com/skgallagher/WeightTraits.git
cd WeightTraits
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,dev]"
```

Run the checks before opening a pull request:

```bash
python -m pytest
ruff check src tests
```

## Pull requests

- Keep changes focused and explain the scientific or operational motivation.
- Add or update tests for changed behavior.
- Update user documentation when commands, schemas, metrics, or defaults change.
- Preserve provenance fields when adding artifact-producing commands.
- Report numerical tolerances explicitly; do not replace a statistical definition silently.
- Do not commit model checkpoints, secrets, API responses, private datasets, or cluster caches.

For paper-facing changes, include the command, declared inputs, generated artifact, and verification
performed. Historical ELLMTrees outputs may be used as comparison targets, but they should not
become hidden inputs to native WeightTraits results.

## Reporting problems

Open a GitHub issue with a minimal reproducer, the command run, Python version, relevant optional
extras, and the complete error message. For numerical discrepancies, include the input artifact
schema and expected tolerance when possible.
