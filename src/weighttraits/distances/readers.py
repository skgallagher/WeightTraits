"""Tensor readers for streaming distance-cube construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterator, Protocol

import numpy as np


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total


class TensorReader(Protocol):
    model_id: str

    def keys(self) -> list[str]:
        """Return tensor keys in deterministic order."""

    def tensor_info(self, key: str) -> TensorInfo:
        """Return metadata for one tensor without requiring a full checkpoint load."""

    def iter_flat_chunks(self, key: str, chunk_size: int) -> Iterator[np.ndarray]:
        """Yield flattened float64 chunks for one tensor."""

    def read_tensor(self, key: str) -> np.ndarray:
        """Read one full tensor. Intended only for tensor-at-a-time metrics."""

    def close(self) -> None:
        """Release caches or file handles."""


class DictTensorReader:
    """In-memory reader used for golden tests and synthetic smoke runs."""

    def __init__(self, tensors: dict[str, np.ndarray], model_id: str = "model") -> None:
        self.model_id = model_id
        self._tensors = {key: np.asarray(value) for key, value in tensors.items()}
        self.chunk_reads: dict[str, int] = {}
        self.tensor_reads: dict[str, int] = {}

    def keys(self) -> list[str]:
        return list(self._tensors)

    def tensor_info(self, key: str) -> TensorInfo:
        arr = self._tensors[key]
        return TensorInfo(name=key, shape=tuple(arr.shape), dtype=str(arr.dtype))

    def iter_flat_chunks(self, key: str, chunk_size: int):
        self.chunk_reads[key] = self.chunk_reads.get(key, 0) + 1
        flat = self._tensors[key].reshape(-1).astype(np.float64, copy=False)
        for start in range(0, flat.size, chunk_size):
            yield flat[start : start + chunk_size]

    def read_tensor(self, key: str) -> np.ndarray:
        self.tensor_reads[key] = self.tensor_reads.get(key, 0) + 1
        return self._tensors[key].astype(np.float64, copy=False)

    def close(self) -> None:
        return None


class SafetensorsTensorReader:
    """Per-tensor reader for safetensors checkpoints."""

    def __init__(self, path: str | Path, model_id: str | None = None) -> None:
        self.path = Path(path)
        self.model_id = model_id or self.path.stem

    def keys(self) -> list[str]:
        from safetensors import safe_open

        with safe_open(str(self.path), framework="numpy", device="cpu") as handle:
            return sorted(handle.keys())

    def tensor_info(self, key: str) -> TensorInfo:
        return _safetensor_tensor_info(self.path, key)

    def iter_flat_chunks(self, key: str, chunk_size: int):
        yield from _iter_safetensor_flat_chunks(self.path, key, chunk_size)

    def read_tensor(self, key: str) -> np.ndarray:
        from safetensors import safe_open

        with safe_open(str(self.path), framework="numpy", device="cpu") as handle:
            return handle.get_tensor(key).astype(np.float64, copy=False)

    def close(self) -> None:
        return None


class ShardedSafetensorsTensorReader:
    """Per-tensor reader for Hugging Face safetensors shard indexes."""

    def __init__(self, path: str | Path, model_id: str | None = None) -> None:
        path = Path(path)
        if path.is_dir():
            self.root = path
            self.index_path = path / "model.safetensors.index.json"
        else:
            self.root = path.parent
            self.index_path = path
        self.model_id = model_id or self.root.name
        self._weight_map: dict[str, str] | None = None

    def _index(self) -> dict[str, str]:
        if self._weight_map is None:
            raw = json.loads(self.index_path.read_text())
            weight_map = raw.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"safetensors index missing weight_map: {self.index_path}")
            self._weight_map = {str(key): str(value) for key, value in weight_map.items()}
        return self._weight_map

    def _shard_path(self, key: str) -> Path:
        try:
            shard = self._index()[key]
        except KeyError as exc:
            raise KeyError(f"tensor {key!r} not found in {self.index_path}") from exc
        return self.root / shard

    def keys(self) -> list[str]:
        return sorted(self._index())

    def tensor_info(self, key: str) -> TensorInfo:
        return _safetensor_tensor_info(self._shard_path(key), key)

    def iter_flat_chunks(self, key: str, chunk_size: int):
        yield from _iter_safetensor_flat_chunks(self._shard_path(key), key, chunk_size)

    def read_tensor(self, key: str) -> np.ndarray:
        from safetensors import safe_open

        with safe_open(str(self._shard_path(key)), framework="numpy", device="cpu") as handle:
            return handle.get_tensor(key).astype(np.float64, copy=False)

    def close(self) -> None:
        self._weight_map = None


class TorchTensorReader:
    """Torch checkpoint reader using mmap when available.

    This still creates a state-dict object, but with `mmap=True` tensor storages
    are file-backed and lazily paged instead of eagerly copied into CPU RAM.
    """

    def __init__(self, path: str | Path, model_id: str | None = None, mmap: bool = True) -> None:
        self.path = Path(path)
        self.model_id = model_id or self.path.stem
        self.mmap = mmap
        self._state: dict | None = None

    def _load(self) -> dict:
        if self._state is None:
            import torch

            try:
                self._state = torch.load(
                    str(self.path),
                    map_location="cpu",
                    weights_only=True,
                    mmap=self.mmap,
                )
            except TypeError:
                self._state = torch.load(str(self.path), map_location="cpu", weights_only=True)
        return self._state

    def keys(self) -> list[str]:
        return list(self._load().keys())

    def tensor_info(self, key: str) -> TensorInfo:
        tensor = self._load()[key]
        return TensorInfo(name=key, shape=tuple(tensor.shape), dtype=str(tensor.dtype))

    def iter_flat_chunks(self, key: str, chunk_size: int):
        tensor = self._load()[key].detach().cpu().reshape(-1)
        for start in range(0, tensor.numel(), chunk_size):
            chunk = tensor[start : start + chunk_size].float().numpy()
            yield chunk.astype(np.float64, copy=False)

    def read_tensor(self, key: str) -> np.ndarray:
        tensor = self._load()[key]
        return tensor.detach().cpu().float().numpy().astype(np.float64, copy=False)

    def close(self) -> None:
        self._state = None


class LoraFactorReader:
    """Reader that streams LoRA `scale * B @ A` row blocks for vector metrics."""

    def __init__(
        self,
        factors: dict[str, dict[str, np.ndarray]],
        model_id: str = "lora",
        scale: float = 1.0,
    ) -> None:
        self.model_id = model_id
        self.scale = float(scale)
        self.factors = {
            module: {"A": np.asarray(mats["A"]), "B": np.asarray(mats["B"])}
            for module, mats in factors.items()
            if "A" in mats and "B" in mats
        }

    @classmethod
    def from_peft_dir(cls, path: str | Path, model_id: str | None = None) -> "LoraFactorReader":
        path = Path(path)
        cfg_path = path / "adapter_config.json"
        scale = 1.0
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            rank = float(cfg.get("r", 1.0))
            alpha = float(cfg.get("lora_alpha", rank))
            scale = alpha / rank if rank else 1.0

        adapter_file = path / "adapter_model.safetensors"
        if adapter_file.exists():
            raw = _load_safetensor_dict(adapter_file)
        else:
            adapter_file = path / "adapter_model.bin"
            if not adapter_file.exists():
                raise FileNotFoundError(f"no adapter_model.safetensors or adapter_model.bin in {path}")
            raw = _load_torch_dict(adapter_file)
        return cls(_group_lora_factors(raw), model_id=model_id or path.name, scale=scale)

    def keys(self) -> list[str]:
        return sorted(self.factors)

    def tensor_info(self, key: str) -> TensorInfo:
        mats = self.factors[key]
        return TensorInfo(name=key, shape=(mats["B"].shape[0], mats["A"].shape[1]), dtype="float64")

    def iter_flat_chunks(self, key: str, chunk_size: int):
        mats = self.factors[key]
        a = mats["A"]
        b = mats["B"]
        row_width = int(a.shape[1])
        rows_per_slab = max(1, (chunk_size + max(row_width, 1) - 1) // max(row_width, 1))
        blocks = (
            self.scale * (b[start : start + rows_per_slab] @ a)
            for start in range(0, b.shape[0], rows_per_slab)
        )
        yield from _yield_flat_chunks_from_blocks(blocks, chunk_size)

    def read_tensor(self, key: str) -> np.ndarray:
        mats = self.factors[key]
        return (self.scale * (mats["B"] @ mats["A"])).astype(np.float64, copy=False)

    def close(self) -> None:
        return None


class CumulativeLoraReader:
    """Path-summed LoRA deltas for one node."""

    def __init__(self, readers: list[LoraFactorReader], model_id: str = "cumulative_lora") -> None:
        self.readers = readers
        self.model_id = model_id

    def keys(self) -> list[str]:
        key_sets = [set(reader.keys()) for reader in self.readers]
        if not key_sets:
            return []
        common = set.intersection(*key_sets)
        missing = set.union(*key_sets) - common
        if missing:
            raise ValueError(f"cumulative LoRA readers have mismatched keys: {sorted(missing)[:5]}")
        return sorted(common)

    def tensor_info(self, key: str) -> TensorInfo:
        infos = [reader.tensor_info(key) for reader in self.readers]
        shapes = {info.shape for info in infos}
        if len(shapes) != 1:
            raise ValueError(f"cumulative LoRA shape mismatch for {key}: {sorted(shapes)}")
        return TensorInfo(name=key, shape=infos[0].shape, dtype="float64")

    def iter_flat_chunks(self, key: str, chunk_size: int):
        chunk_iters = [reader.iter_flat_chunks(key, chunk_size) for reader in self.readers]
        for chunks in zip(*chunk_iters, strict=True):
            total = np.zeros_like(chunks[0], dtype=np.float64)
            for chunk in chunks:
                total += np.asarray(chunk, dtype=np.float64)
            yield total

    def read_tensor(self, key: str) -> np.ndarray:
        total = None
        for reader in self.readers:
            arr = reader.read_tensor(key)
            total = arr.copy() if total is None else total + arr
        if total is None:
            raise KeyError(key)
        return total

    def close(self) -> None:
        for reader in self.readers:
            reader.close()


def reader_from_path(path: str | Path, model_id: str | None = None) -> TensorReader:
    """Build a reader for a checkpoint path or PEFT adapter directory."""

    path = Path(path)
    if path.is_dir():
        if (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists():
            return LoraFactorReader.from_peft_dir(path, model_id=model_id)
        index_path = path / "model.safetensors.index.json"
        if index_path.exists():
            return ShardedSafetensorsTensorReader(path, model_id=model_id)
        for name in ("model.safetensors", "pytorch_model.bin"):
            candidate = path / name
            if candidate.exists():
                return reader_from_path(candidate, model_id=model_id or path.name)
        safetensors_shards = sorted(path.glob("model-*.safetensors"))
        if safetensors_shards:
            raise NotImplementedError("sharded safetensors reader is planned but not implemented yet")
        raise FileNotFoundError(f"could not find a supported checkpoint file in {path}")

    if path.name.endswith(".safetensors"):
        return SafetensorsTensorReader(path, model_id=model_id)
    if path.name.endswith(".safetensors.index.json"):
        return ShardedSafetensorsTensorReader(path, model_id=model_id)
    if path.suffix in {".bin", ".pt", ".pth"}:
        return TorchTensorReader(path, model_id=model_id)
    raise ValueError(f"unsupported checkpoint path: {path}")


def _safetensor_tensor_info(path: Path, key: str) -> TensorInfo:
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy", device="cpu") as handle:
        tensor_slice = handle.get_slice(key)
        return TensorInfo(
            name=key,
            shape=tuple(tensor_slice.get_shape()),
            dtype=str(tensor_slice.get_dtype()),
        )


def _iter_safetensor_flat_chunks(path: Path, key: str, chunk_size: int):
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy", device="cpu") as handle:
        tensor_slice = handle.get_slice(key)
        shape = tuple(tensor_slice.get_shape())
        if not shape:
            yield np.asarray([handle.get_tensor(key)], dtype=np.float64).reshape(-1)
            return
        if len(shape) == 1:
            for start in range(0, shape[0], chunk_size):
                yield tensor_slice[start : min(start + chunk_size, shape[0])].astype(
                    np.float64, copy=False
                )
            return

        row_width = int(np.prod(shape[1:], dtype=np.int64))
        rows_per_slab = max(1, (chunk_size + max(row_width, 1) - 1) // max(row_width, 1))
        blocks = (
            tensor_slice[start : min(start + rows_per_slab, shape[0])]
            for start in range(0, shape[0], rows_per_slab)
        )
        yield from _yield_flat_chunks_from_blocks(blocks, chunk_size)


def _load_safetensor_dict(path: Path) -> dict[str, np.ndarray]:
    from safetensors import safe_open

    out: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy", device="cpu") as handle:
        for key in handle.keys():
            out[key] = handle.get_tensor(key)
    return out


def _load_torch_dict(path: Path) -> dict[str, np.ndarray]:
    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return {key: value.detach().cpu().float().numpy() for key, value in state.items()}


def _group_lora_factors(raw: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    modules: dict[str, dict[str, np.ndarray]] = {}
    for key, value in raw.items():
        clean = re.sub(r"^base_model\.model\.", "", key)
        if clean.endswith(".lora_A.weight"):
            module = clean[: -len(".lora_A.weight")] + ".weight"
            modules.setdefault(module, {})["A"] = value
        elif clean.endswith(".lora_B.weight"):
            module = clean[: -len(".lora_B.weight")] + ".weight"
            modules.setdefault(module, {})["B"] = value
    return modules


def _yield_flat_chunks_from_blocks(blocks, chunk_size: int):
    carry = np.empty(0, dtype=np.float64)
    for block in blocks:
        flat = np.asarray(block, dtype=np.float64).reshape(-1)
        if carry.size:
            flat = np.concatenate([carry, flat])
            carry = np.empty(0, dtype=np.float64)
        while flat.size >= chunk_size:
            yield flat[:chunk_size]
            flat = flat[chunk_size:]
        carry = flat
    if carry.size:
        yield carry
