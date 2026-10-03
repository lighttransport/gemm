from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .native import decode
from .checkpoint import group_name
from pathlib import Path
from .common import fingerprint, require_disk, GIB


@dataclass
class Candidate:
    qtype: str
    shape: tuple
    raw: np.ndarray


class Bank:
    """Native candidates backed by either RAM or disk memmaps."""
    def __init__(self, mode="ram", directory=None, reserve=4*GIB):
        self.tensors = {}
        self.selection = {}
        self.mode, self.directory, self.reserve = mode, Path(directory) if directory else None, reserve
        if mode not in ("ram", "mmap") or (mode == "mmap" and not directory):
            raise ValueError("mmap candidate storage needs a directory")
        if self.directory and mode == "mmap":
            self.directory.mkdir(parents=True, exist_ok=True)

    def allocate(self, name, qtype, size):
        if self.mode == "ram":
            return np.empty(size, np.uint8)
        path = self.directory / (fingerprint([name, qtype])+".bin")
        # Rebuild from verified tile checkpoints; the disk bank is expendable.
        old = path.stat().st_blocks*512 if path.exists() else 0
        require_disk(self.directory, max(0, size-old), self.reserve)
        return np.memmap(path, dtype=np.uint8, mode="w+", shape=(size,))

    @property
    def bytes(self):
        return sum(c.raw.nbytes for candidates in self.tensors.values() for c in candidates)

    def add(self, name, candidates):
        if self.mode == "mmap":
            for candidate in candidates:
                if not isinstance(candidate.raw, np.memmap):
                    raw = self.allocate(name, candidate.qtype, candidate.raw.nbytes)
                    raw[:] = candidate.raw
                    candidate.raw = raw
                candidate.raw.flush()
                if hasattr(candidate.raw._mmap, "madvise"):
                    import mmap
                    candidate.raw._mmap.madvise(mmap.MADV_DONTNEED)
        self.tensors[name] = candidates

    def retain_selected(self):
        if hasattr(self, "original_selection"):
            return
        self.original_selection = dict(self.selection)
        for name, candidates in self.tensors.items():
            choice = self.selection.get(group_name(name), 0)
            self.tensors[name] = [candidates[choice]]
        self.selection = {group_name(name): 0 for name in self.tensors}

    def read(self, name, start=0, stop=None, choice=None):
        candidates = self.tensors[name]
        i = self.selection.get(group_name(name), 0) if choice is None else choice
        c = candidates[i]
        stop = c.shape[0] if stop is None else stop
        row_bytes = c.raw.nbytes // c.shape[0]
        value = decode(c.raw[start*row_bytes:stop*row_bytes], c.qtype, (stop-start, *c.shape[1:]))
        if self.mode == "mmap" and hasattr(c.raw._mmap, "madvise"):
            import mmap
            begin = start*row_bytes//mmap.PAGESIZE*mmap.PAGESIZE
            end = stop*row_bytes
            if end > begin:
                c.raw._mmap.madvise(mmap.MADV_DONTNEED, begin, end-begin)
        return value

    def costs(self):
        result = {}
        for name, candidates in self.tensors.items():
            group = group_name(name)
            costs = np.array([c.raw.nbytes for c in candidates], dtype=np.int64)
            if group in result and len(result[group]) != len(costs):
                raise ValueError("Grouped tensors have incompatible candidate counts")
            result[group] = result.get(group, 0) + costs
        return result
