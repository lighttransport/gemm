"""SOMA-X implementation of the BodyModel backend (MHR identity -> SOMA topology).

torch/Warp live only in the SOMA interpreter (VHUMAN_SOMA_PYTHON); this module
stays framework-free and talks to soma_runner.py through .npy files.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .. import gpu
from ..service import ROOT

PREFIX = "soma_"
ASSET_FILES = ()  # fetched into the Hugging Face cache by soma-x itself unless --mhr-assets DIR is given
LANDMARKS = Path(__file__).with_name("soma_landmarks.json")
RUNNER = Path(__file__).with_name("soma_runner.py")
_VERTICES, _JOINTS = 18056, 78


def python() -> str:
    exe = os.environ.get("VHUMAN_SOMA_PYTHON")
    if not exe or not Path(exe).is_file():
        raise RuntimeError("SOMA-X needs a Python with py-soma-x installed: "
                           "set VHUMAN_SOMA_PYTHON=/path/to/python")
    return exe


def resolve_assets(assets=None, model=None):
    """Local SOMA-X asset directory, or None to use soma-x's Hugging Face cache.

    ``assets`` is the MHR asset directory shared by the CLI/job plumbing and is
    deliberately ignored; override SOMA-X assets with VHUMAN_SOMA_ASSETS.
    """
    env = os.environ.get("VHUMAN_SOMA_ASSETS")
    if env:
        return Path(env)
    local = gpu.model_path("soma-x")
    return local if Path(local).is_dir() else None


def _run(args, timeout=1800):
    proc = subprocess.run([python(), "-I", str(RUNNER), *args], capture_output=True, text=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError("SOMA-X runner failed: " + proc.stderr[-4000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def decode(assets, params, identity, face=None, *, skeleton_only=False, backend=None, device=None, threads=1):
    params = np.asarray(params, dtype=np.float32)
    identity = np.asarray(identity, dtype=np.float32)
    if params.ndim != 2 or params.shape[1] != 204 or not 1 <= len(params) <= 65536:
        raise ValueError("SOMA pose must have shape [B,204], 1 <= B <= 65536")
    if identity.shape == (45,):
        identity = np.broadcast_to(identity, (len(params), 45))
    if identity.shape != (len(params), 45):
        raise ValueError("SOMA identity must have shape [45] or [B,45]")
    if face is not None:
        raise ValueError("SOMA-X has no facial coefficients; faces use the vhuman face rig")
    if not all(np.isfinite(v).all() for v in (params, identity)):
        raise ValueError("SOMA coefficients must be finite")
    temp = ROOT / "tmp/vhuman-runtime"
    temp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="soma-", dir=temp) as directory:
        directory = Path(directory)
        np.save(directory / "params.npy", np.ascontiguousarray(params), allow_pickle=False)
        np.save(directory / "shape.npy", np.ascontiguousarray(identity), allow_pickle=False)
        cmd = ["--params", str(directory / "params.npy"), "--shape", str(directory / "shape.npy"),
               "--output-dir", str(directory), "--device", os.environ.get("VHUMAN_SOMA_DEVICE", "cpu")]
        if assets is not None:
            cmd += ["--data-root", str(Path(assets).resolve())]
        if skeleton_only:
            cmd += ["--skeleton-only"]
        report = _run(cmd)
        state = np.load(directory / "skeleton.npy", allow_pickle=False)
        vertices = None if skeleton_only else np.load(directory / "vertices.npy", allow_pickle=False)
        z = np.load(directory / "rig.npz", allow_pickle=False)
        rig = {"names": [PREFIX + n for n in report.pop("names")], "parents": z["parents"],
               "joint_indices": z["joint_indices"], "skin_weights": z["skin_weights"],
               "faces": z["faces"], "provenance": {"model": "SOMA-X", "unit": "m"}}
    if state.shape != (len(params), _JOINTS, 8) or not np.isfinite(state).all():
        raise ValueError("invalid SOMA skeleton output")
    if vertices is not None and (vertices.shape != (len(params), _VERTICES, 3) or not np.isfinite(vertices).all()):
        raise ValueError("invalid SOMA vertex output")
    report["requested_backend"] = backend
    return vertices, state, rig, report


def faces(head_assets):
    raise NotImplementedError("SOMA faces come from decode(): use rig['faces']")


def landmark_mapping(head_assets=None):
    """One-hot [5, V+J] mapping onto SOMA vertices (see soma_landmarks.json)."""
    idx = json.loads(LANDMARKS.read_text())["indices"]
    out = np.zeros((5, _VERTICES + _JOINTS))
    out[np.arange(5), idx] = 1.0
    return out
