# Reproducible environment

The paper's runs (training, distances, behavior, PhyloLM) all used one Python environment: the
`ellmtrees` conda env on Wright. `requirements-lock.txt` pins every package in it at the exact
version, 154 pins in all.

* Python is 3.11.15.
* torch is `2.5.1+cu121`, the CUDA 12.1 build from the PyTorch index; the other 153 pins come from
  PyPI.
* Training reproducibility depends on the numerical stack recorded in every training ledger:
  python, torch, transformers 5.8.0, datasets 4.8.5, accelerate 1.13.0, tokenizers 0.22.2, and
  peft 0.19.1. `python environment/check_environment.py --core` checks just those.

There are three ways to get this environment. All of them run WeightTraits from a checkout with
`PYTHONPATH=src`, which is how the paper's jobs ran.

## 1. venv (any Linux machine with an NVIDIA driver for CUDA 12.1)

```bash
python3.11 -m venv .venv            # Python 3.11.15
. .venv/bin/activate
python -m pip install -r environment/requirements-lock.txt
python environment/check_environment.py
```

With [uv](https://docs.astral.sh/uv/), `uv venv --python 3.11.15` provides the exact interpreter.
Follow it with `uv pip install -r environment/requirements-lock.txt`.

## 2. Apptainer (clusters)

`weighttraits.def` builds the same environment on `python:3.11.15-slim-bookworm`, with the base
image pinned by digest. It installs the lock and fails the build if any version differs. Build it
from the repository root. Point the cache and temp directories at local scratch, because they
need several GB:

```bash
export APPTAINER_CACHEDIR=/mnt/scratch/$USER/apptainer/cache APPTAINER_TMPDIR=/mnt/scratch/$USER/apptainer/tmp
apptainer build --fakeroot weighttraits.sif environment/weighttraits.def
apptainer exec --nv --bind "$PWD" weighttraits.sif env PYTHONPATH="$PWD/src" python -m pytest -q
```

The image carries the dependencies only. The WeightTraits code comes from the bound checkout, so
the code version is whatever commit or tag is checked out. For the training code of record, that
is `training-code-august-20260817`.

## 3. The original conda env on Wright

```bash
/home/export/sgallagh/.conda/envs/ellmtrees/bin/python environment/check_environment.py
```

## Notes

* The lock also pins some CUDA 13 `nvidia-*` wheels that the env happened to contain. torch
  2.5.1+cu121 doesn't use them. They are kept so the lock describes the env exactly.
* Library versions do change results. Biopython 1.88 changed its default Newick branch-length
  format, so WeightTraits pins `%1.5f` itself (see `NEWICK_BRANCH_LENGTH_FORMAT`).
