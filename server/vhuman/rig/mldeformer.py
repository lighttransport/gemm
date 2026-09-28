"""ML corrective deformer: learn what the linear rig gets wrong.

1. Ground truth (offline, PyTorch). For sampled control vectors, the linear
   rig's surface is relaxed by an energy that the linear model cannot express:
   - anchoring to the linear result (the rig's intent),
   - as-rigid-as-possible edges relative to the rest shape (soft tissue keeps
     its local shape instead of LBS/blendshape shearing and volume loss),
   - contact: the lids stay outside the (rotating) eyeballs, the lips and the
     mouth's vestibule outside the teeth (spheres per tooth, moving with the
     teeth joints), and the upper lip does not pass through the lower one.
2. The residual (solved - linear) is mapped back before skinning (inverse
   blended rotation per vertex), so it composes with the rig like any
   corrective blendshape. PCA gives K components.
3. A two-layer ReLU MLP (controls -> PCA coefficients) is trained in PyTorch
   and exported as a LightRig-style .lrm (safetensors: fc1/fc2, input and
   output normalisation), evaluated at runtime by ryzen/lightrig_mlp2.c, the
   web viewer or numpy; the PCA basis ships as morph targets / a tensor.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import rigdef
from . import safetensors as st
from . import template as T
from .common import edges
from .torchrig import TorchRig

EYE_MARGIN_MM = 0.25
TOOTH_MARGIN_MM = 0.6


# ---- sampling -----------------------------------------------------------------------------

def _mirror(name: str) -> str:
    for a, b in (("Left", "Right"), ("Right", "Left")):
        if name.endswith(a):
            return name[:-len(a)] + b
    return name


def sample_controls(names: list[str], n: int, seed: int = 0) -> np.ndarray:
    """Sparse, plausible combinations: a few expressions at a time, often
    symmetric, with many jaw-open + lip combinations (where contact matters)."""
    rng = np.random.default_rng(seed)
    idx = {k: i for i, k in enumerate(names)}
    pool = [k for k in rigdef.LR_FACE_V1]
    lips = [k for k in pool if k.startswith("mouth")]
    lids = [k for k in pool if k.startswith("eye") or k.startswith("cheekSquint")]
    tongue = [k for k in names if k.startswith("tongue")]
    X = np.zeros((n, len(names)), np.float32)
    for s in range(n):
        r = rng.random()
        picks = list(rng.choice(pool, size=1 + rng.poisson(2.5), replace=False))
        if r < 0.35:
            picks += ["jawOpen"] + list(rng.choice(lips, size=1 + rng.poisson(1.2), replace=False))
            if rng.random() < 0.4:
                picks += list(rng.choice(tongue, size=1 + rng.poisson(0.5), replace=False))
        elif r < 0.55:
            picks += list(rng.choice(lids, size=2 + rng.poisson(1.0), replace=False))
        for k in picks:
            v = float(rng.uniform(0.25, 1.0) ** 0.7)
            X[s, idx[k]] = max(X[s, idx[k]], v)
            if k.startswith("tongue") and rng.random() < 0.6:
                X[s, idx[k]] = min(X[s, idx[k]], float(rng.uniform(0.1, 0.6)))
            if rng.random() < 0.5 and _mirror(k) in idx:
                X[s, idx[_mirror(k)]] = max(X[s, idx[_mirror(k)]], v * rng.uniform(0.85, 1.0))
    X[: max(1, n // 50)] = 0                            # a few neutral samples
    return X


# ---- contact geometry -------------------------------------------------------------------

def tooth_spheres(part, joint_index: int, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Per tooth: centre and radius (m), from the procedural crown blocks."""
    P = part.positions
    n = len(P) // block
    c, r = [], []
    for t in range(n):
        q = P[t * block:(t + 1) * block]
        cen = q.mean(0)
        d = np.linalg.norm(q - cen, axis=1)
        c.append(cen)
        r.append(0.8 * float(np.median(d)))
    return np.asarray(c), np.asarray(r)


def tongue_spheres(part, nu: int = 28, nv: int = 20):
    """Spheres filling the lofted tongue: per cross-section row, three across
    its width (radius ~ the half thickness), on that row's two joints."""
    P = part.positions[:nu * nv].reshape(nu, nv, 3)
    J = part.joints[:nu * nv].reshape(nu, nv, 4)[:, 0, :2]
    W = part.weights[:nu * nv].reshape(nu, nv, 4)[:, 0, :2]
    c, r, jj, ww = [], [], [], []
    for i in range(1, nu - 1):
        row = P[i]
        cen = row.mean(0)
        side = row[np.argmax(row[:, 0])] - row[np.argmin(row[:, 0])]
        half_w = 0.5 * float(np.linalg.norm(side))
        side = side / max(np.linalg.norm(side), 1e-9)
        half_t = 0.5 * float(np.ptp(row @ np.cross(side, P[i + 1].mean(0) - P[i - 1].mean(0))
                                    / max(np.linalg.norm(P[i + 1].mean(0) - P[i - 1].mean(0)), 1e-9)))
        rad = max(min(half_t, half_w) * 0.95, 0.0015)
        for f in (-1.0, 0.0, 1.0):
            off = f * max(half_w - rad, 0.0)
            c.append(cen + side * off)
            r.append(rad)
            jj.append(J[i])
            ww.append(W[i])
    return np.asarray(c), np.asarray(r), np.asarray(jj), np.asarray(ww)


class Contacts:
    """Contacts on the welded template, thresholds relative to the rest pose
    (a vertex may never get closer than its rest distance or the margin,
    whichever is smaller), so the neutral face needs no correction:
    - lid vertices vs the eyeballs (centres on the eye joints);
    - lip/vestibule vertices vs sphere sets: teeth (rigid on the teeth
      joints) and the tongue (spheres on its blended chain joints);
    - upper vs lower lip, per sample pair on rings 1..-2, along the mean of
      the head's and the jaw's up axes."""

    PAIR_RINGS = (1, 0, -1, -2)

    def __init__(self, tmpl: T.Template, feat, skel: dict, teeth: list, device, rest: np.ndarray,
                 tongue=None, dtype=torch.float32):
        names = [j["name"] for j in skel["joints"]]
        ji = {n: i for i, n in enumerate(names)}
        dev = device
        self.eye = []
        self._export = {"eye": [], "spheres": {}, "margins_mm": {"eye": EYE_MARGIN_MM, "sphere": TOOTH_MARGIN_MM}}
        eyes = {e["side"]: e for e in feat.eyes}
        for g, side, jn in ((0, "right", "eye_R"), (1, "left", "eye_L")):
            ids = np.flatnonzero((tmpl.group == g) & (tmpl.ring <= 4))
            e = eyes[side]
            d_rest = np.linalg.norm(rest[ids] - e["center"], axis=1) * 1000
            thr = np.minimum(e["radius"] * 1000 + EYE_MARGIN_MM, d_rest)
            self.eye.append((torch.tensor(ids, device=dev), ji[jn],
                             torch.tensor([*e["center"], 1.0], device=dev, dtype=dtype),
                             torch.tensor(thr, device=dev, dtype=dtype)))
            self._export["eye"].append({"ids": ids.tolist(), "joint": ji[jn], "center": list(map(float, e["center"])),
                                        "radius_mm": float(e["radius"] * 1000)})
        lip_np = np.flatnonzero((tmpl.group == 2) & (tmpl.ring >= -2) & (tmpl.ring <= 4))
        self.lip_ids = torch.tensor(lip_np, device=dev)
        self._export["lip_ids"] = lip_np.tolist()
        lip = rest[lip_np]
        self.spheres = {}
        sets = []
        cs, rs, js = [], [], []
        for part, jn in teeth:
            block = 10 * 18 + 1                     # mouthparts._crown(lat=10, lon=18) + the tip centre
            c, r = tooth_spheres(part, ji[jn], block)
            cs.append(c)
            rs.append(r)
            js.append(np.full(len(c), ji[jn]))
        C = np.concatenate(cs)
        sets.append(("teeth", C, np.concatenate(rs), np.stack([np.concatenate(js), np.zeros(len(C), int)], 1),
                     np.stack([np.ones(len(C)), np.zeros(len(C))], 1)))
        if tongue is not None:
            sets.append(("tongue", *tongue_spheres(tongue)))
        for name, C, R, J2, W2 in sets:
            d_rest = np.linalg.norm(lip[:, None] - C[None], axis=-1) * 1000            # (L, T)
            thr = np.minimum(R[None] * 1000 + TOOTH_MARGIN_MM, d_rest)
            self.spheres[name] = (torch.tensor(np.concatenate([C, np.ones((len(C), 1))], 1), device=dev, dtype=dtype),
                                  torch.tensor(thr, device=dev, dtype=dtype),
                                  torch.tensor(J2, device=dev), torch.tensor(W2, device=dev, dtype=dtype))
            self._export["spheres"][name] = {"centers": C.tolist(), "radius_mm": (R * 1000).tolist(),
                                             "joints": J2.tolist(), "weights": W2.tolist()}
        H, N = T.MOUTH_HALF, T.MOUTH_N
        up, lo = [], []
        for k in self.PAIR_RINGS:
            ids = tmpl.ring_ids("mouth", k)
            for j in range(1, H):
                up.append(ids[j])
                lo.append(ids[N - j])
        self.pair_u = torch.tensor(up, device=dev)
        self.pair_l = torch.tensor(lo, device=dev)
        sep0 = (rest[up] - rest[lo])[:, 1] * 1000
        self.sep0 = torch.tensor(np.minimum(sep0, 0.0), device=dev, dtype=dtype)
        self.head = ji["head"]
        self.jaw = ji["jaw"]
        self._export.update({"pairs_upper": up, "pairs_lower": lo, "head": ji["head"], "jaw": ji["jaw"]})

    def export(self) -> dict:
        """For the viewer's contact heat map (welded vertex ids; bind-space centres)."""
        return self._export

    def energy(self, x: torch.Tensor, skin: torch.Tensor, per_vertex: bool = False):
        """x (S, V, 3) in mm; skin (S, J, 4, 4) in metres. Squared penetration
        depths; with per_vertex, also the depth per vertex (S, V) in mm."""
        S = x.shape[0]
        e = x.new_zeros(S)
        depth = torch.zeros(x.shape[:2], device=x.device, dtype=x.dtype) if per_vertex else None
        stats = {}
        pen = 0
        for ids, j, c, R in self.eye:
            cen = (skin[:, j] @ c)[:, :3] * 1000                        # (S, 3)
            d = (x[:, ids] - cen[:, None]).norm(dim=-1)
            p = torch.relu(R - d)
            e = e + (p ** 2).sum(1)
            pen = pen + (p > 0.05).sum(1)
            if per_vertex:
                depth[:, ids] = torch.maximum(depth[:, ids], p)
        stats["eye"] = pen
        q = x[:, self.lip_ids]                                          # (S, L, 3)
        for name, (c, thr, J2, W2) in self.spheres.items():
            M = (W2[None, :, :, None, None] * skin[:, J2]).sum(2)        # (S, T, 4, 4) blended
            tc = (M @ c[None, :, :, None])[..., :3, 0] * 1000            # (S, T, 3)
            p = torch.relu(thr[None] - torch.cdist(q, tc))              # (S, L, T)
            e = e + (p ** 2).sum((1, 2))
            stats[name] = (p.amax(2) > 0.05).sum(1)
            if per_vertex:
                depth[:, self.lip_ids] = torch.maximum(depth[:, self.lip_ids], p.amax(2))
        up = skin[:, self.head, :3, 1] + skin[:, self.jaw, :3, 1]
        up = up / up.norm(dim=-1, keepdim=True)
        sep = ((x[:, self.pair_u] - x[:, self.pair_l]) * up[:, None]).sum(-1)
        p = torch.relu(self.sep0[None] - sep)
        e = e + (p ** 2).sum(1)
        stats["lips"] = (p > 0.05).sum(1)
        if per_vertex:
            for ids in (self.pair_u, self.pair_l):
                depth[:, ids] = torch.maximum(depth[:, ids], p)
            return e, stats, depth
        return e, stats


# ---- the solve ---------------------------------------------------------------------------

class Solver:
    def __init__(self, tr: TorchRig, tmpl: T.Template, contacts: Contacts, w_arap: float = 0.4,
                 w_contact: float = 400.0, iters: int = 80):
        self.tr, self.c = tr, contacts
        E = edges(tmpl.tris)
        dev = tr.rest.device
        self.ei = torch.tensor(E[:, 0], device=dev)
        self.ej = torch.tensor(E[:, 1], device=dev)
        self.rest_mm = tr.rest * 1000
        self.e0 = self.rest_mm[self.ej] - self.rest_mm[self.ei]
        self.w_arap, self.w_contact, self.iters = w_arap, w_contact, iters
        V = tr.rest.shape[0]
        self.faces = torch.tensor(tmpl.tris, device=dev)
        self.n0 = self._normals(self.rest_mm[None])[0]
        e2 = (self.e0 ** 2).sum(-1)
        acc = torch.zeros(V, device=dev, dtype=e2.dtype)
        acc.index_add_(0, self.ei, e2)
        acc.index_add_(0, self.ej, e2)
        self.nscale = acc / torch.bincount(torch.cat([self.ei, self.ej]), minlength=V).clamp(min=1)

    def _normals(self, x):
        f = self.faces
        n = torch.cross(x[:, f[:, 1]] - x[:, f[:, 0]], x[:, f[:, 2]] - x[:, f[:, 0]], dim=-1)
        acc = torch.zeros_like(x)
        for c in range(3):
            acc.index_add_(1, f[:, c], n)
        return acc / acc.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def rotations(self, x):
        """Per-vertex best rotations (ARAP local step), (S, V, 3, 3): the
        orthogonal polar factor of the edge covariance (plus a normal-to-normal
        term, so flat fans stay full rank), by Newton's iteration
        X <- (X + X^-T) / 2 with closed-form 3x3 inverses (no batched SVD)."""
        e = x[:, self.ej] - x[:, self.ei]                                    # (S, E, 3)
        cov = e[:, :, :, None] * self.e0[None, :, None, :]                   # (S, E, 3, 3): e e0^T
        Cv = torch.zeros(x.shape[0], x.shape[1], 3, 3, device=x.device, dtype=x.dtype)
        Cv.index_add_(1, self.ei, cov)
        Cv.index_add_(1, self.ej, cov)
        n = self._normals(x)
        Cv = Cv + self.nscale[None, :, None, None] * n[..., :, None] * self.n0[None, :, None, :]
        X = Cv / Cv.flatten(-2).norm(dim=-1)[..., None, None].clamp(min=1e-12)
        for _ in range(12):
            a, b, c = X[..., 0, :], X[..., 1, :], X[..., 2, :]
            cof = torch.stack([torch.cross(b, c, dim=-1), torch.cross(c, a, dim=-1), torch.cross(a, b, dim=-1)], -2)
            det = (a * cof[..., 0, :]).sum(-1)[..., None, None]
            inv_t = cof / torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)   # X^-T
            X = 0.5 * (X + inv_t)
        return X

    def solve(self, controls: torch.Tensor) -> dict:
        with torch.no_grad():
            lin = self.tr(controls)
        x_lin = lin["pos"] * 1000
        skin = lin["skin"]
        off = torch.zeros_like(x_lin, requires_grad=True)
        opt = torch.optim.Adam([off], lr=0.1)
        with torch.no_grad():
            _, before = self.c.energy(x_lin, skin)
        R = None
        for it in range(self.iters):
            x = x_lin + off
            if it % 10 == 0:
                with torch.no_grad():
                    R = self.rotations(x)
            e = x[:, self.ej] - x[:, self.ei]
            Ri = R[:, self.ei]
            target = (Ri @ self.e0[None, :, :, None])[..., 0]            # R e0 ~ e
            arap = ((e - target) ** 2).sum(-1).mean(1)
            anchor = (off ** 2).sum(-1).mean(1)
            contact, _ = self.c.energy(x, skin)
            loss = (anchor + self.w_arap * arap + self.w_contact * contact / x.shape[1]).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            x = x_lin + off
            _, after = self.c.energy(x, skin)
            # back to the rest frame, before skinning: A r_pre = r_posed
            r_pre = torch.linalg.solve(lin["blend"], off[..., None])[..., 0]
        return {"residual_mm": r_pre, "before": before, "after": after, "offset_mm": off.detach()}


# ---- the model ---------------------------------------------------------------------------

class MLP2(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_in, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def train(tr: TorchRig, tmpl: T.Template, contacts: Contacts, out_dir: Path, samples: int = 2048,
          k: int = 64, hidden: int = 256, batch: int = 128, iters: int = 100, epochs: int = 2500, seed: int = 0,
          log=print, progress=None, w_contact: float = 1500.0) -> dict:
    t0 = time.perf_counter()
    dev = tr.rest.device
    torch.manual_seed(seed)
    X = sample_controls(tr.controls, samples, seed)
    solver = Solver(tr, tmpl, contacts, iters=iters, w_contact=w_contact)
    res, before, after = [], [], []
    for s in range(0, samples, batch):
        c = torch.tensor(X[s:s + batch], device=dev)
        o = solver.solve(c)
        res.append(o["residual_mm"].cpu())
        before.append({k2: v.cpu() for k2, v in o["before"].items()})
        after.append({k2: v.cpu() for k2, v in o["after"].items()})
        if progress and (s // batch) % 8 == 0:
            progress(s / samples, f"deformer ground truth {s}/{samples}")
    Rm = torch.cat(res)                                                   # (N, V, 3) mm
    N, V = Rm.shape[:2]
    t_solve = time.perf_counter() - t0
    agg = lambda lst, key: int(sum(int(d[key].sum()) for d in lst))       # noqa: E731
    contact = {key: {"linear": agg(before, key), "solved": agg(after, key)} for key in before[0]}
    # PCA of the residuals
    Y = Rm.reshape(N, -1).to(dev)
    mean = Y.mean(0)
    U, Sv, Vh = torch.linalg.svd(Y - mean, full_matrices=False)
    k = min(k, Vh.shape[0])
    basis = Vh[:k]                                                         # (K, 3V)
    coeff = (Y - mean) @ basis.T
    var = (Sv ** 2)
    explained = float(var[:k].sum() / var.sum()) if var.sum() > 0 else 1.0
    # MLP: controls -> coefficients (normalised both ways), vertex-space loss
    Xt = torch.tensor(X, device=dev)
    xm, xs = Xt.mean(0), Xt.std(0)
    xs = torch.where(xs > 1e-6, 1.0 / xs, torch.ones_like(xs))
    cm, cs = coeff.mean(0), coeff.std(0).clamp(min=1e-6)
    perm = torch.randperm(N, device=dev)
    n_val = max(1, N // 8)
    va, trn = perm[:n_val], perm[n_val:]
    net = MLP2(X.shape[1], hidden, k).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xin = (Xt - xm) * xs
    target = (coeff - cm) / cs

    def vert_err(idx):
        with torch.no_grad():
            pred = net(xin[idx]) * cs + cm
            r = pred @ basis + mean
            return (r - Y[idx]).reshape(len(idx), V, 3).norm(dim=-1)

    for ep in range(epochs):
        opt.zero_grad()
        pred = net(xin[trn])
        loss = (((pred - target[trn]) * (cs / cs.max())) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if log and ep % 500 == 0:
            log(f"deformer mlp epoch {ep}: loss {loss.item():.5f}, val mean err {vert_err(va).mean().item():.4f} mm")
    ev = vert_err(va)
    mag = Y[va].reshape(len(va), V, 3).norm(dim=-1)
    # the runtime question: contacts on held-out controls, linear rig vs linear + ML correctives
    with torch.no_grad():
        pre = ((net(xin[va]) * cs + cm) @ basis + mean).reshape(len(va), V, 3) / 1000
        cv = torch.tensor(X, device=dev)[va]
        lin = tr(cv)
        ml = tr(cv, pre=pre)
        _, c_lin = contacts.energy(lin["pos"] * 1000, lin["skin"])
        _, c_ml = contacts.energy(ml["pos"] * 1000, ml["skin"])
    held_out = {key: {"linear": int(c_lin[key].sum()), "ml": int(c_ml[key].sum())} for key in c_lin}
    # export: LightRig .lrm (output.* denormalises straight to PCA coefficients in metres)
    out_dir = Path(out_dir)
    lrm = {"fc1.weight": net.fc1.weight.detach().cpu().numpy(), "fc1.bias": net.fc1.bias.detach().cpu().numpy(),
           "fc2.weight": net.fc2.weight.detach().cpu().numpy(), "fc2.bias": net.fc2.bias.detach().cpu().numpy(),
           "input.mean": xm.cpu().numpy(), "input.scale": xs.cpu().numpy(),
           "output.mean": (cm / 1000).cpu().numpy(), "output.scale": (cs / 1000).cpu().numpy()}
    meta = {"format": "vhuman-ml-deformer", "version": "1", "inputs": json.dumps(tr.controls),
            "outputs": "pca coefficients (metres) of pre-skinning corrective offsets",
            "kind": "face-corrective"}
    st.save(out_dir / "deformer.lrm", lrm, meta)
    st.save(out_dir / "deformer_basis.safetensors",
            {"basis": basis.reshape(k, V, 3).cpu().numpy(), "mean": (mean / 1000).reshape(V, 3).cpu().numpy()},
            {"vertices": V, "components": k, "units": "basis is unit-norm; offsets = mean + sum_k c_k basis_k"})
    stats = {"samples": N, "components": k, "hidden": hidden, "explained_variance": round(explained, 4),
             "residual_mean_mm": round(float(mag.mean()), 4), "residual_p99_mm": round(float(mag.quantile(0.99)), 4),
             "val_error_mean_mm": round(float(ev.mean()), 4), "val_error_p99_mm": round(float(ev.quantile(0.99)), 4),
             "contact_vertices": contact, "held_out_contacts": held_out, "solve_seconds": round(t_solve, 1),
             "seconds": round(time.perf_counter() - t0, 1)}
    (out_dir / "deformer.json").write_text(json.dumps(stats, indent=1))
    return stats


from .mlruntime import MLDeformer  # noqa: E402,F401  (re-exported)
