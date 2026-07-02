.PHONY: test audit-ellmtrees smoke paper clean

PYTHON ?= python
PYTHONPATH := src

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

audit-ellmtrees:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m weighttraits.cli audit-ellmtrees --source ../ELLMTrees --out reports/ellmtrees_inventory.json

smoke: test audit-ellmtrees

paper:
	@echo "Paper rebuild is intentionally gated until figure/table registries are populated."
	@echo "See docs/PAPER_REPRODUCTION.md."

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete

