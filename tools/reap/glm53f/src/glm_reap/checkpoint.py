from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import struct
import mmap
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np


@dataclass(frozen=True)
class TensorInfo:
    name: str
    file: Path
    dtype: str
    shape: tuple
    offset: int
    nbytes: int


class Checkpoint:
    """Read safetensors headers and only the requested rows of tensor payloads."""

    def __init__(self, root):
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = 0
        self.maps = {}
        self.mode = "mmap"
        self.threads = 1
        self.excluded = set()
        self.root = Path(root)
        self.config = json.loads((self.root / "config.json").read_text())
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.tensors = {}
        for shard in sorted(set(index["weight_map"].values())):
            file = self.root / shard
            with file.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                if length > 128 * 2**20:
                    raise ValueError(f"Implausible safetensors header: {file}")
                header = json.loads(stream.read(length))
            for name, item in header.items():
                if name == "__metadata__":
                    continue
                a, b = item["data_offsets"]
                if name in self.tensors or a < 0 or b < a or 8 + length + b > file.stat().st_size:
                    raise ValueError(f"Invalid tensor entry: {name}")
                self.tensors[name] = TensorInfo(name, file, item["dtype"], tuple(item["shape"]), 8 + length + a, b - a)
        if set(index["weight_map"]) != set(self.tensors):
            raise ValueError("Index and shard tensor names disagree")

    def raw(self, name, start=0, stop=None):
        info = self.tensors[name]
        types = {"F32": "<f4", "F16": "<f2", "BF16": "<u2", "F8_E4M3": "u1", "I64": "<i8", "I32": "<i4"}
        if info.dtype not in types:
            raise ValueError(f"Unsupported source dtype {info.dtype}: {name}")
        stop = info.shape[0] if stop is None else stop
        if not 0 <= start <= stop <= info.shape[0]:
            raise ValueError(f"Invalid row range for {name}: {start}:{stop}")
        dtype = np.dtype(types[info.dtype])
        row = math.prod(info.shape[1:])
        if math.prod(info.shape) * dtype.itemsize != info.nbytes:
            raise ValueError(f"Tensor payload size mismatch: {name}")
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name][start:stop]
        if info.file not in self.maps:
            # Copy-on-write permits torch views without mutating the source file.
            self.maps[info.file] = np.memmap(info.file, dtype=np.uint8, mode="c")
        value = np.ndarray(info.shape, dtype=dtype, buffer=self.maps[info.file], offset=info.offset)[start:stop]
        if self.mode == "ram":
            value = value.copy()
            mapping = self.maps[info.file]
            if hasattr(mapping._mmap, "madvise"):
                begin = (info.offset+start*row*dtype.itemsize)//mmap.PAGESIZE*mmap.PAGESIZE
                end = min(len(mapping), info.offset+stop*row*dtype.itemsize)
                if end > begin:
                    mapping._mmap.madvise(mmap.MADV_DONTNEED, begin, end-begin)
        return value

    def configure(self, config, bank_bytes=0):
        from .common import execution_options, GIB
        options = execution_options(config)
        self.mode, self.threads = options["memory_mode"], options["cpu_threads"]
        self.set_cache_limit(min(options["source_cache_gib"]*GIB, max(0, config.get("host_limit_gib", 130)*GIB-bank_bytes-12*GIB)) if self.mode == "ram" else 0)

    def set_cache_limit(self, limit):
        self.cache_limit = int(limit)
        while self.cache and self.cache_bytes > self.cache_limit:
            _, value = self.cache.popitem(last=False)
            self.cache_bytes -= value.nbytes

    def release_pages(self, name, start=0, stop=None):
        info = self.tensors[name]
        mapping = self.maps.get(info.file)
        if self.mode != "mmap" or mapping is None or not hasattr(mapping._mmap, "madvise"):
            return
        stop = info.shape[0] if stop is None else stop
        row_bytes = info.nbytes//info.shape[0]
        begin = (info.offset+start*row_bytes)//mmap.PAGESIZE*mmap.PAGESIZE
        end = min(len(mapping), info.offset+stop*row_bytes)
        if end > begin:
            mapping._mmap.madvise(mmap.MADV_DONTNEED, begin, end-begin)

    def prefetch_layer(self, layer, selected=None):
        if self.mode != "ram" or not self.cache_limit:
            return
        prefix = f"model.language_model.layers.{layer}."
        names = []
        for name in self.tensors:
            if not name.startswith(prefix) or name in self.cache or name in self.excluded:
                continue
            match = re.search(r"\.mlp\.experts\.(\d+)\.", name)
            if selected and str(layer) in selected and match and int(match[1]) not in selected[str(layer)]:
                continue
            names.append(name)
        needed = sum(self.tensors[name].nbytes for name in names)
        if needed > self.cache_limit:
            return
        while self.cache and self.cache_bytes+needed > self.cache_limit:
            _, value = self.cache.popitem(last=False)
            self.cache_bytes -= value.nbytes
        def load(name):
            # Explicit copies give predictable residency rather than page-cache hints.
            return name, self.raw(name).copy()
        for file in {self.tensors[name].file for name in names}:
            if file not in self.maps:
                self.maps[file] = np.memmap(file, dtype=np.uint8, mode="c")
        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            for name, value in pool.map(load, names):
                self.cache[name] = value
                self.cache_bytes += value.nbytes

    def read(self, name, start=0, stop=None):
        info = self.tensors[name]
        value = self.raw(name, start, stop)
        if info.dtype == "BF16":
            value = (value.astype(np.uint32) << 16).view(np.float32)
        elif info.dtype == "F8_E4M3":
            u = np.arange(256, dtype=np.uint16)
            exp, mantissa = (u >> 3) & 15, u & 7
            table = np.where(exp == 0, mantissa / 8 * 2.0**-6, (1 + mantissa / 8) * np.exp2(exp.astype(np.int32)-7)).astype(np.float32)
            table *= np.where(u & 128, -1, 1)
            table[[127, 255]] = np.nan
            value = table[value]
            scale_name = name + "_scale_inv"
            if scale_name not in self.tensors:
                raise ValueError(f"FP8 tensor lacks scales: {name}")
            block = self.config["quantization_config"]["weight_block_size"]
            if value.ndim != 2:
                raise ValueError(f"Only matrix block FP8 is supported: {name}")
            scales = self.raw(scale_name).astype(np.float32)
            rows = np.arange(start, start + len(value)) // block[0]
            cols = np.arange(value.shape[1]) // block[1]
            value *= scales[rows[:, None], cols[None, :]]
        value = value.astype(np.float32, copy=False)
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite source tensor: {name}")
        self.release_pages(name, start, stop)
        return value

    def tensor(self, name, start, stop, device, dtype):
        """Decode a bounded source tile on CUDA, retaining FP32 scale arithmetic."""
        import torch
        info = self.tensors[name]
        if torch.device(device).type != "cuda" or info.dtype not in ("F8_E4M3", "BF16"):
            return torch.as_tensor(self.read(name, start, stop), device=device, dtype=dtype)
        raw = self.raw(name, start, stop)
        if info.dtype == "BF16":
            value = torch.from_numpy(raw).view(torch.bfloat16).to(device=device, dtype=dtype)
        else:
            scale_name = name + "_scale_inv"
            if scale_name not in self.tensors or raw.ndim != 2:
                raise ValueError(f"Unsupported or unscaled FP8 tensor: {name}")
            value = torch.from_numpy(raw).to(device).view(torch.float8_e4m3fn).float()
            block = self.config["quantization_config"]["weight_block_size"]
            scales = torch.from_numpy(self.raw(scale_name).astype(np.float32)).to(device)
            rows = torch.arange(start, stop, device=device) // block[0]
            cols = torch.arange(raw.shape[1], device=device) // block[1]
            value = (value * scales[rows[:, None], cols[None, :]]).to(dtype)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite source tensor: {name}")
        self.release_pages(name, start, stop)
        return value

    def weight_names(self, vision=False):
        for name in sorted(self.tensors):
            if "weight_scale" in name or ".layers.45." in name:
                continue
            visual = name.startswith("model.visual.")
            if visual == vision:
                yield name

    def identity(self):
        return {"root": str(self.root.resolve()), "config": self.config, "shards": {str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sorted({i.file for i in self.tensors.values()})}}

    def inventory(self, keep=116):
        groups = {k: [0, 0] for k in ("routed", "dense", "protected", "vision", "mtp", "scales")}
        for n, i in self.tensors.items():
            category = "scales" if "weight_scale" in n else "mtp" if ".layers.45." in n else "vision" if n.startswith("model.visual.") else "routed" if ".mlp.experts." in n else "protected" if protected(n, i.shape) else "dense"
            groups[category][0] += math.prod(i.shape)
            groups[category][1] += i.nbytes
        routed = groups["routed"][0] * keep / self.config["text_config"]["n_routed_experts"]
        return {"groups_elements_bytes": groups, "retained_routed_parameters": int(routed), "candidate_bank_estimate_gib": (routed * (2.625 + 3.4375) + groups["dense"][0] * (4.5 + 6.5625 + 8.5)) / 8 / 2**30}


def protected(name, shape):
    return len(shape) < 2 or any(part in name for part in (".hc_", ".mlp.gate.", "conv1d", "kv_b_proj", "index_kpool_compress", ".self_attn.A_log", ".self_attn.dt_bias"))


def expert_key(name):
    match = re.search(r"\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", name)
    return (int(match[1]), int(match[2]), match[3]) if match else None


def group_name(name):
    key = expert_key(name)
    return f"layer.{key[0]}.experts.{key[2]}" if key else name
