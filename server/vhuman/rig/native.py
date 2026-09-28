"""The rig package for the native deformer (ryzen/vhuman_deformer.c):
rig_deformer.safetensors, the welded head (skin + mouth bag) with its
blendshapes, the ML correctives, skin weights and the linear rig as dense
tensors. `Native` loads the shared library through ctypes."""
from __future__ import annotations

import ctypes
import json
import subprocess
from pathlib import Path

import numpy as np

from . import rigdef
from . import safetensors as st
from .rigdef import ATTRS

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ("ryzen/vhuman_deformer.c", "ryzen/lightrig_mlp2.c", "ryzen/gemm_avx2.c")


def write_package(path, rig_def: dict, rest: np.ndarray, shapes: dict, joints: np.ndarray, weights: np.ndarray,
                  ml=None) -> dict:
    R = rigdef.Rig(rig_def)
    C, I, J = len(R.controls), len(R.inputs), R.J
    jm = np.zeros((J * 6, I), np.float32)
    np.add.at(jm, (R.M_rows, R.M_cols), R.M_vals)
    kmax = max([len(c["inputs"]) for c in rig_def["correctives"]] + [1])
    corr = np.full((max(len(rig_def["correctives"]), 1), kmax), -1, np.int32)
    for p, c in enumerate(rig_def["correctives"]):
        corr[p, :len(c["inputs"])] = [R.cidx[n] for n in c["inputs"]]
    names = [n for n in R.shape_names if n in shapes and not n.startswith("ml_")]
    src = np.array([R.iidx[next(b["input"] for b in rig_def["blendshapes"] if b["name"] == n)] for n in names],
                   np.int32)
    morph = [shapes[n] for n in names]
    t = {"controls.range": np.stack([R.lo, R.hi], 1), "joints.matrix": jm,
         "joints.parent": np.asarray(R.parent, np.int32), "joints.rest_t": R.rest_t, "joints.rest_R": R.rest_R,
         "joints.inv_bind": R.inv_bind, "shapes.src": src, "rest": rest,
         "skin.joints": joints.astype(np.int32), "skin.weights": weights / weights.sum(1, keepdims=True)}
    if rig_def["correctives"]:
        t["corr.inputs"] = corr
        t["corr.weight"] = np.array([c.get("weight", 1.0) for c in rig_def["correctives"]], np.float32)
    if ml is not None:
        deltas = ml.target_deltas()
        morph += [deltas[n] for n in ml.target_names()]
        for k in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "input.mean", "input.scale",
                  "output.mean", "output.scale"):
            t[f"ml.{k}"] = ml.m[k]
    t["morph"] = np.stack(morph).astype(np.float32)
    meta = {"format": "vhuman-rig-deformer", "version": "1", "controls": json.dumps(R.controls),
            "morphs": json.dumps(names + (ml.target_names() if ml is not None else [])),
            "units": "metres; head frame +Y up, face +Z"}
    size = st.save(path, t, meta)
    return {"bytes": size, "vertices": int(len(rest)), "morphs": int(len(morph)), "controls": C}


def build_library(out_dir: Path) -> Path:
    """Compile libvhuman_deformer.so (gcc, AVX2/FMA) into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = out_dir / "libvhuman_deformer.so"
    cmd = ["gcc", "-O3", "-mavx2", "-mfma", "-ffast-math", "-fPIC", "-shared", "-o", str(lib),
           *[str(ROOT / s) for s in SOURCES], "-lm"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return lib


class Native:
    def __init__(self, lib_path, package):
        self.lib = ctypes.CDLL(str(lib_path))
        L = self.lib
        L.vh_deformer_load.restype = ctypes.c_void_p
        L.vh_deformer_load.argtypes = [ctypes.c_char_p]
        L.vh_deformer_free.argtypes = [ctypes.c_void_p]
        for f in ("vh_deformer_controls", "vh_deformer_vertices", "vh_deformer_morphs"):
            getattr(L, f).restype = ctypes.c_size_t
            getattr(L, f).argtypes = [ctypes.c_void_p]
        fp = ctypes.POINTER(ctypes.c_float)
        L.vh_deformer_eval.argtypes = [ctypes.c_void_p, fp, ctypes.c_int, fp]
        L.vh_deformer_weights.argtypes = [ctypes.c_void_p, fp, fp]
        L.vh_deformer_batch_scratch.restype = ctypes.c_size_t
        L.vh_deformer_batch_scratch.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        L.vh_deformer_eval_batch.argtypes = [ctypes.c_void_p, fp, ctypes.c_size_t, ctypes.c_int, fp, fp]
        self.h = L.vh_deformer_load(str(package).encode())
        if not self.h:
            raise ValueError(f"cannot load {package}")
        self.C = L.vh_deformer_controls(self.h)
        self.V = L.vh_deformer_vertices(self.h)
        self.M = L.vh_deformer_morphs(self.h)

    @staticmethod
    def _p(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def eval(self, controls: np.ndarray, use_ml: bool = True) -> np.ndarray:
        x = np.ascontiguousarray(controls, np.float32)
        out = np.zeros((self.V, 3), np.float32)
        self.lib.vh_deformer_eval(self.h, self._p(x), int(use_ml), self._p(out))
        return out

    def eval_batch(self, controls: np.ndarray, use_ml: bool = True) -> np.ndarray:
        x = np.ascontiguousarray(controls, np.float32)
        F = len(x)
        scratch = np.zeros(self.lib.vh_deformer_batch_scratch(self.h, F), np.float32)
        out = np.zeros((F, self.V, 3), np.float32)
        self.lib.vh_deformer_eval_batch(self.h, self._p(x), F, int(use_ml), self._p(scratch), self._p(out))
        return out

    def close(self):
        if self.h:
            self.lib.vh_deformer_free(self.h)
            self.h = None
