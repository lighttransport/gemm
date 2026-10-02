"""Framework-free MHR adapter. TorchScript belongs to the offline exporter only."""
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .. import gpu
from ..service import ROOT

STEM = "sam3d_body_mhr_jit"
ASSET_FILES = (STEM + ".safetensors", STEM + ".json",
               STEM + "_rig.safetensors", STEM + "_rig.json")


def resolve_assets(assets=None, model=None):
    if assets is not None:
        return Path(assets)
    if model is not None:
        model = Path(model)
        if model.name != "mhr_model.pt" or model.parent.name != "assets":
            raise ValueError("Use --mhr-assets DIR for native MHR exports")
        return model.parents[2] / "safetensors"
    return gpu.model_path("sam3d-body/safetensors")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor(path, name):
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("truncated MHR metadata")
        size, = struct.unpack("<Q", raw)
        if size > 1024 * 1024:
            raise ValueError("oversized MHR metadata header")
        header = json.loads(f.read(size))
        spec = header[name]
        dtype = {"I32": "<i4", "F32": "<f4"}[spec["dtype"]]
        shape = tuple(spec["shape"])
        if not shape or len(shape) > 2 or any(not isinstance(n, int) or n < 1 or n > 18439 for n in shape):
            raise ValueError("invalid MHR metadata shape")
        start, stop = spec["data_offsets"]
        count = int(np.prod(shape))
        if start < 0 or stop - start != count * 4 or 8 + size + stop > path.stat().st_size:
            raise ValueError("invalid MHR metadata offsets")
        f.seek(8 + size + start)
        return np.frombuffer(f.read(stop - start), dtype=dtype).reshape(shape).copy()


def metadata(assets):
    assets = Path(assets)
    missing = [name for name in ASSET_FILES if not (assets / name).is_file()]
    if missing:
        raise ValueError("Missing native MHR assets: " + ", ".join(missing) +
                         "; run ref/sam3d-body/dump_mhr_assets.py --rig-only in the export environment")
    meta = json.loads((assets / (STEM + "_rig.json")).read_text())
    if meta.get("version") != 1 or meta.get("units") != "cm" or meta.get("quaternion_order") != "xyzw":
        raise ValueError("unsupported MHR rig metadata")
    for suffix, key in ((".safetensors", "weights_sha256"), (".json", "constants_sha256"),
                        ("_rig.safetensors", "rig_sha256")):
        if _sha256(assets / (STEM + suffix)) != meta.get(key):
            raise ValueError("MHR asset checksum mismatch: " + suffix)
    names = meta.get("joint_names", [])
    if len(names) != 127 or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != 127:
        raise ValueError("invalid MHR joint names")
    path = assets / (STEM + "_rig.safetensors")
    parents = _tensor(path, "joint_parents")
    indices = _tensor(path, "joint_indices")
    weights = _tensor(path, "skin_weights")
    if (parents.dtype != np.dtype("int32") or parents.shape != (127,) or
            np.any(parents < -1) or np.any(parents >= np.arange(127)) or
            indices.dtype != np.dtype("int32") or indices.ndim != 2 or indices.shape[0] != 18439 or
            indices.shape[1] < 4 or weights.shape != indices.shape or weights.dtype != np.dtype("float32") or
            np.any(indices < 0) or np.any(indices >= 127) or not np.isfinite(weights).all() or
            np.any(weights < 0) or np.any(weights.sum(axis=1) <= 0)):
        raise ValueError("invalid MHR skeleton or skin weights")
    return {"names": ["mhr_" + n for n in names], "parents": parents,
            "joint_indices": indices, "skin_weights": weights, "provenance": meta}


def binary(backend):
    directory = ROOT / backend / "sam3d_body"
    path = directory / "mhr_decode"
    # make also handles a stale binary after a source update.
    subprocess.run(["make", "-s", "-C", str(directory), "mhr_decode"],
                   check=True, capture_output=True, text=True)
    return path


def decode(assets, params, identity, face=None, *, skeleton_only=False, backend=None, device=None, threads=1):
    params = np.asarray(params, dtype=np.float32)
    identity = np.asarray(identity, dtype=np.float32)
    if params.ndim != 2 or params.shape[1] != 204 or not 1 <= len(params) <= 65536:
        raise ValueError("MHR pose must have shape [B,204], 1 <= B <= 65536")
    if identity.shape == (45,):
        identity = np.broadcast_to(identity, (len(params), 45))
    if identity.shape != (len(params), 45):
        raise ValueError("MHR identity must have shape [45] or [B,45]")
    if face is not None:
        face = np.asarray(face, dtype=np.float32)
        if face.shape != (len(params), 72):
            raise ValueError("MHR facial coefficients must have shape [B,72]")
    if not all(np.isfinite(v).all() for v in (params, identity, face) if v is not None):
        raise ValueError("MHR coefficients must be finite")
    rig = metadata(assets)
    requested = backend if backend is not None else gpu.backend()
    if requested not in ("cpu", "cuda", "rocm"):
        raise ValueError("unsupported MHR backend")
    selected = "cuda" if requested == "cuda" and not skeleton_only else "cpu"
    runner = binary(selected)
    temp = ROOT / "tmp/vhuman-runtime"
    temp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mhr-", dir=temp) as directory:
        directory = Path(directory)
        np.save(directory / "params.npy", np.ascontiguousarray(params), allow_pickle=False)
        np.save(directory / "shape.npy", np.ascontiguousarray(identity), allow_pickle=False)
        cmd = [str(runner), "--mhr-assets", str(Path(assets).resolve()),
               "--params", str(directory / "params.npy"), "--shape", str(directory / "shape.npy"),
               "--output-dir", str(directory), "--backend", selected,
               "--device", str(gpu.device_index() if device is None else device), "--threads", str(threads)]
        if face is not None:
            np.save(directory / "face.npy", np.ascontiguousarray(face), allow_pickle=False)
            cmd += ["--face", str(directory / "face.npy")]
        if skeleton_only:
            cmd += ["--skeleton-only"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if proc.returncode:
            raise RuntimeError("Native MHR decode failed: " + proc.stderr[-4000:])
        state = np.load(directory / "skeleton.npy", allow_pickle=False)
        vertices = None if skeleton_only else np.load(directory / "vertices.npy", allow_pickle=False)
        if state.shape != (len(params), 127, 8) or not np.isfinite(state).all():
            raise ValueError("invalid native MHR skeleton output")
        if vertices is not None and (vertices.shape != (len(params), 18439, 3) or not np.isfinite(vertices).all()):
            raise ValueError("invalid native MHR vertex output")
        report = json.loads(proc.stdout)
    report["requested_backend"] = requested
    return vertices, state, rig, report
