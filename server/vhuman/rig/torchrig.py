"""The linear rig (rigdef.py) in PyTorch, batched over control vectors, on the
welded template (skin + mouth bag): the deformer's training-time reference.

    TorchRig(rig_def, rest (V,3), shapes {name: (V,3)}, joints (V,4), weights (V,4))
    out = tr(controls (S, C))  ->  {"pos": (S,V,3), "skin": (S,J,4,4), "blend": (S,V,3,3)}
"""
from __future__ import annotations

import numpy as np
import torch

from .rigdef import ATTRS


def euler_zyx(r: torch.Tensor) -> torch.Tensor:
    """(..., 3) radians -> (..., 3, 3) = Rz @ Ry @ Rx."""
    rx, ry, rz = r[..., 0], r[..., 1], r[..., 2]
    cx, sx, cy, sy, cz, sz = rx.cos(), rx.sin(), ry.cos(), ry.sin(), rz.cos(), rz.sin()
    o, z = torch.ones_like(rx), torch.zeros_like(rx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], -1).reshape(*rx.shape, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).reshape(*rx.shape, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], -1).reshape(*rx.shape, 3, 3)
    return Rz @ Ry @ Rx


class TorchRig(torch.nn.Module):
    def __init__(self, d: dict, rest: np.ndarray, shapes: dict, joints: np.ndarray, weights: np.ndarray,
                 device="cpu", dtype=torch.float32):
        super().__init__()
        self.d = d
        self.controls = [c["name"] for c in d["controls"]]
        cidx = {n: i for i, n in enumerate(self.controls)}
        self.inputs = self.controls + [c["name"] for c in d["correctives"]]
        iidx = {n: i for i, n in enumerate(self.inputs)}
        names = [j["name"] for j in d["joints"]]
        jidx = {n: i for i, n in enumerate(names)}
        J, I = len(names), len(self.inputs)
        M = torch.zeros(J * 6, I, dtype=dtype)
        for e in d["joint_matrix"]:
            M[jidx[e["joint"]] * 6 + ATTRS.index(e["attr"]), iidx[e["input"]]] += e["value"]
        self.register_buffer("M", M)
        self.register_buffer("lo", torch.tensor([c["min"] for c in d["controls"]], dtype=dtype))
        self.register_buffer("hi", torch.tensor([c["max"] for c in d["controls"]], dtype=dtype))
        self.corr = [([cidx[n] for n in c["inputs"]], float(c.get("weight", 1.0))) for c in d["correctives"]]
        self.parent = [jidx.get(j["parent"], -1) if j["parent"] else -1 for j in d["joints"]]
        self.register_buffer("rest_t", torch.tensor([j["rest_translation"] for j in d["joints"]], dtype=dtype))
        self.register_buffer("rest_R", torch.tensor([j["rest_rotation"] for j in d["joints"]], dtype=dtype))
        bind = torch.tensor(np.array([j["bind"] for j in d["joints"]]), dtype=torch.float64)
        self.register_buffer("inv_bind", torch.linalg.inv(bind).to(dtype))
        self.shape_names = [b["name"] for b in d["blendshapes"] if b["name"] in shapes]
        self.register_buffer("shape_src", torch.tensor([iidx[b["input"]] for b in d["blendshapes"]
                                                        if b["name"] in shapes]))
        D = np.stack([shapes[n] for n in self.shape_names]).reshape(len(self.shape_names), -1)
        self.register_buffer("D", torch.tensor(D, dtype=dtype))                 # (B, 3V)
        self.register_buffer("rest", torch.tensor(rest, dtype=dtype))
        self.register_buffer("jn", torch.tensor(joints, dtype=torch.long))
        w = weights / np.maximum(weights.sum(1, keepdims=True), 1e-12)
        self.register_buffer("w", torch.tensor(w, dtype=dtype))
        self.J = J
        self.to(device)

    def input_vector(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.maximum(torch.minimum(x, self.hi), self.lo)
        c = []
        for ids, wt in self.corr:
            p = torch.full_like(x[:, 0], wt)
            for i in ids:
                p = p * x[:, i].clamp(0, 1)
            c.append(p.clamp(max=1.0))
        return torch.cat([x, torch.stack(c, 1)], 1) if c else x

    def skinning(self, inp: torch.Tensor) -> torch.Tensor:
        S = inp.shape[0]
        delta = (inp @ self.M.T).reshape(S, self.J, 6)
        R = self.rest_R[None] @ euler_zyx(delta[..., 3:])
        t = self.rest_t[None] + delta[..., :3]
        local = torch.zeros(S, self.J, 4, 4, dtype=inp.dtype, device=inp.device)
        local[..., :3, :3] = R
        local[..., :3, 3] = t
        local[..., 3, 3] = 1
        world = []
        for j in range(self.J):
            p = self.parent[j]
            world.append(local[:, j] if p < 0 else world[p] @ local[:, j])
        world = torch.stack(world, 1)
        return world @ self.inv_bind[None]

    def forward(self, controls: torch.Tensor, pre: torch.Tensor | None = None) -> dict:
        """controls (S, C). `pre` (S, V, 3): an extra pre-skinning offset (the
        ML correctives)."""
        inp = self.input_vector(controls)
        skin = self.skinning(inp)
        bw = inp[:, self.shape_src]
        p = self.rest[None] + (bw @ self.D).reshape(-1, *self.rest.shape)
        if pre is not None:
            p = p + pre
        m = skin[:, self.jn]                                         # (S, V, 4, 4, 4)
        blend = (self.w[None, :, :, None, None] * m).sum(2)            # (S, V, 4, 4)
        pos = (blend[..., :3, :3] @ p[..., None])[..., 0] + blend[..., :3, 3]
        return {"pos": pos, "skin": skin, "blend": blend[..., :3, :3], "inputs": inp, "pre": p}
