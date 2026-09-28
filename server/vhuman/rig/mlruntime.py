"""numpy runtime of the ML corrective deformer (no PyTorch needed)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import safetensors as st


class MLDeformer:
    """numpy runtime: controls -> pre-skinning corrective offsets (V, 3) metres."""

    def __init__(self, folder):
        folder = Path(folder)
        self.m, meta = st.load(folder / "deformer.lrm")
        b, _ = st.load(folder / "deformer_basis.safetensors")
        self.inputs = json.loads(meta["inputs"])
        self.basis = b["basis"].astype(np.float64)                        # (K, V, 3), per metre of coefficient
        self.mean = b["mean"].astype(np.float64)

    def coefficients(self, controls: dict | np.ndarray) -> np.ndarray:
        if isinstance(controls, dict):
            x = np.array([float(controls.get(n, 0.0)) for n in self.inputs])
        else:
            x = np.asarray(controls, np.float64)
        m = self.m
        x = (x - m["input.mean"]) * m["input.scale"]
        h = np.maximum(m["fc1.weight"] @ x + m["fc1.bias"], 0)
        y = m["fc2.weight"] @ h + m["fc2.bias"]
        return y * m["output.scale"] + m["output.mean"]

    def offsets(self, controls) -> np.ndarray:
        c = self.coefficients(controls)
        return self.mean + np.tensordot(c, self.basis, 1)

    # morph-target form: "ml_mean" (weight 1) and "ml_NN" = basis_k * output.scale_k,
    # weight c_k / output.scale_k, so weights stay O(1) in viewers
    def target_names(self) -> list[str]:
        return ["ml_mean"] + [f"ml_{k:02d}" for k in range(len(self.basis))]

    def target_deltas(self) -> dict:
        sc = self.m["output.scale"].astype(np.float64)
        out = {"ml_mean": self.mean}
        for k in range(len(self.basis)):
            out[f"ml_{k:02d}"] = self.basis[k] * sc[k]
        return out

    def target_weights(self, controls) -> np.ndarray:
        c = self.coefficients(controls)
        return np.concatenate([[1.0], c / self.m["output.scale"]])
