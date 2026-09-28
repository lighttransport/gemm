"""Minimal safetensors read/write (F32/I32/U32), no dependency: an 8-byte
little-endian header length, a JSON header, then raw tensors. The format of
LightRig's .lrm models and common/safetensors.h."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

DTYPES = {np.dtype("<f4"): "F32", np.dtype("<i4"): "I32", np.dtype("<u4"): "U32"}
NP = {v: k for k, v in DTYPES.items()}


def save(path, tensors: dict, metadata: dict | None = None) -> int:
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        a = np.asarray(arr)
        if a.dtype.kind == "f":
            a = a.astype("<f4")
        elif a.dtype.kind == "u":
            a = a.astype("<u4")
        else:
            a = a.astype("<i4")
        a = np.ascontiguousarray(a)
        b = a.tobytes()
        header[name] = {"dtype": DTYPES[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    data = struct.pack("<Q", len(h)) + h + b"".join(blobs)
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(data)
    tmp.replace(path)
    return len(data)


def load(path) -> tuple[dict, dict]:
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    meta = header.pop("__metadata__", {})
    base = 8 + n
    out = {}
    for name, h in header.items():
        a, b = h["data_offsets"]
        out[name] = np.frombuffer(raw[base + a:base + b], dtype=NP[h["dtype"]]).reshape(h["shape"])
    return out, meta
