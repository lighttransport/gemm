"""Optional CPU/GPU Torch oracle, isolated from production corrective training.

Preserved reference equations for comparisons; default training uses native C++.
ML corrective deformer: learn what the linear rig gets wrong.

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

from server.vhuman.rig import rigdef
from server.vhuman.rig import safetensors as st
from server.vhuman.rig import template as T
from server.vhuman.rig.common import edges
from server.vhuman.rig.torchrig import TorchRig

from server.vhuman.rig.contact_setup import (EYE_MARGIN_MM, TOOTH_MARGIN_MM, sample_controls,
                            tooth_spheres, tongue_spheres, region_mask)


# ---- sampling -----------------------------------------------------------------------------

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
            if torch.version.hip and off.is_cuda:
                # ROCm's batched getrs allocates per-matrix pointer workspace;
                # bound each launch for the many tiny skinning systems.
                a = lin["blend"].reshape(-1, 3, 3)
                b = off.reshape(-1, 3, 1)
                r_pre = torch.cat([torch.linalg.solve(a[i:i+4096], b[i:i+4096])
                                   for i in range(0, len(a), 4096)]).reshape_as(off)
            else:
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


def train(tr: TorchRig, tmpl: T.Template, contacts: Contacts, out_dir: Path, samples: int = 4096,
          k: int = 48, k_mouth: int = 48, hidden: int = 256, batch: int = 128, iters: int = 100, epochs: int = 2500, seed: int = 0,
          log=print, progress=None, w_contact: float = 1500.0, tongue_weight: float = 4.0,
          tongue_share: float = 0.3, weight_decay: float = 1e-3, solve_cache=None, lip_weight: float = 10.0) -> dict:
    t0 = time.perf_counter()
    dev = tr.rest.device
    torch.manual_seed(seed)
    X = sample_controls(tr.controls, samples, seed, tongue_share=tongue_share)
    cache = Path(solve_cache) if solve_cache else None
    if cache is not None and cache.exists():                # experiments: reuse a solve
        blob = torch.load(cache, weights_only=False)
        X, res, before, after = blob["X"], blob["res"], blob["before"], blob["after"]
    else:
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
        if cache is not None:
            torch.save({"X": X, "res": res, "before": before, "after": after}, cache)
    Rm = torch.cat(res)                                                   # (N, V, 3) mm
    N, V = Rm.shape[:2]
    t_solve = time.perf_counter() - t0
    agg = lambda lst, key: int(sum(int(d[key].sum()) for d in lst))       # noqa: E731
    contact = {key: {"linear": agg(before, key), "solved": agg(after, key)} for key in before[0]}
    # PCA of the residuals, per region: the mouth (lips, vestibule, bag and
    # their surroundings: teeth/tongue contacts) and the rest (mostly lids).
    # Disjoint supports keep the stacked basis orthonormal, and the rare mouth
    # corrections get their own capacity instead of a global tail.
    Y = Rm.reshape(N, -1).to(dev)
    mean = Y.mean(0)
    mouth_v = torch.tensor(region_mask(tmpl, tr.rest.cpu().numpy()), device=dev)
    masks = [mouth_v.repeat_interleave(3), ~mouth_v.repeat_interleave(3)]
    ks = [k_mouth, k]
    bases, explained = [], []
    for m, kk in zip(masks, ks):
        Z = (Y - mean) * m
        U, Sv, Vh = torch.linalg.svd(Z, full_matrices=False)
        kk = min(kk, Vh.shape[0])
        bases.append(Vh[:kk] * m)
        var = Sv ** 2
        explained.append(float(var[:kk].sum() / var.sum()) if var.sum() > 0 else 1.0)
    basis = torch.cat(bases)                                                 # (K, 3V)
    k = basis.shape[0]
    coeff = (Y - mean) @ basis.T
    # MLP: controls -> coefficients (normalised both ways), vertex-space loss
    # network inputs: the rig's input vector (controls and their corrective
    # products, e.g. tongueOut x jawOpen, which contacts depend on)
    with torch.no_grad():
        Xt = tr.input_vector(torch.tensor(X, device=dev))
    xm, xs = Xt.mean(0), Xt.std(0)
    xs = torch.where(xs > 1e-6, 1.0 / xs, torch.ones_like(xs))
    cm, cs = coeff.mean(0), coeff.std(0).clamp(min=1e-6)
    perm = torch.randperm(N, device=dev)
    n_val = max(1, N // 8)
    va, trn = perm[:n_val], perm[n_val:]
    net = MLP2(Xt.shape[1], hidden, k).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xin = (Xt - xm) * xs
    target = (coeff - cm) / cs

    def vert_err(idx):
        with torch.no_grad():
            pred = net(xin[idx]) * cs + cm
            r = pred @ basis + mean
            return (r - Y[idx]).reshape(len(idx), V, 3).norm(dim=-1)

    # lips must not cross through the correction either: per upper/lower pair,
    # the posed separation (linear rig + the predicted pre-skinning offset,
    # carried by each vertex's blended skinning) stays >= min(rest gap, 0)
    pu, pl = contacts.pair_u, contacts.pair_l
    sep_lin, Au, Al = [], [], []
    with torch.no_grad():
        for i in range(0, N, 128):
            o = tr(torch.tensor(X[i:i + 128], device=dev))
            up = o["skin"][:, contacts.head, :3, 1] + o["skin"][:, contacts.jaw, :3, 1]
            up = up / up.norm(dim=-1, keepdim=True)                          # (b, 3)
            x = o["pos"] * 1000
            sep_lin.append(((x[:, pu] - x[:, pl]) * up[:, None]).sum(-1))
            Au.append((up[:, None, None, :] @ o["blend"][:, pu])[..., 0, :])  # (b, P, 3): up^T A_u
            Al.append((up[:, None, None, :] @ o["blend"][:, pl])[..., 0, :])
    sep_lin, Au, Al = torch.cat(sep_lin), torch.cat(Au), torch.cat(Al)
    sep_floor = contacts.sep0[None]
    Bu = basis.reshape(k, V, 3)[:, pu]                                      # (K, P, 3)
    Bl = basis.reshape(k, V, 3)[:, pl]
    mu, ml_ = mean.reshape(V, 3)[pu], mean.reshape(V, 3)[pl]

    def lip_penalty(pred_norm, idx):
        c = pred_norm * cs + cm                                              # (n, K) mm
        du = torch.einsum("nk,kpc->npc", c, Bu) + mu
        dl = torch.einsum("nk,kpc->npc", c, Bl) + ml_
        sep = sep_lin[idx] + (Au[idx] * du).sum(-1) - (Al[idx] * dl).sum(-1)
        return (torch.relu(sep_floor - sep) ** 2).mean()

    # rare contact classes (the tongue) weigh more: they are all residual, and few
    rare = torch.cat([d["tongue"] for d in before]).to(dev) if "tongue" in before[0] else torch.zeros(N, device=dev)
    sw = (1.0 + tongue_weight * (rare > 0).float())[:, None]
    for ep in range(epochs):
        opt.zero_grad()
        pred = net(xin[trn])
        loss = ((((pred - target[trn]) * (cs / cs.max())) ** 2) * sw[trn]).mean()
        loss = loss + lip_weight * lip_penalty(pred, trn) / float(cs.max()) ** 2
        loss.backward()
        opt.step()
        sched.step()
        if log and ep % 500 == 0:
            log(f"deformer mlp epoch {ep}: loss {loss.item():.5f}, val mean err {vert_err(va).mean().item():.4f} mm")
    ev = vert_err(va)
    mag = Y[va].reshape(len(va), V, 3).norm(dim=-1)
    # the runtime question: contacts on held-out (and as many training)
    # controls, linear rig vs linear + ML correctives (chunked: the rig's
    # per-vertex matrices are large)
    def contact_counts(ids):
        tot_l, tot_m = {}, {}
        for i in range(0, len(ids), 128):
            b = ids[i:i + 128]
            with torch.no_grad():
                pre = ((net(xin[b]) * cs + cm) @ basis + mean).reshape(len(b), V, 3) / 1000
                cv = torch.tensor(X, device=dev)[b]
                lin = tr(cv)
                _, cl = contacts.energy(lin["pos"] * 1000, lin["skin"])
                ml = tr(cv, pre=pre)
                _, cm_ = contacts.energy(ml["pos"] * 1000, ml["skin"])
            for key in cl:
                tot_l[key] = tot_l.get(key, 0) + int(cl[key].sum())
                tot_m[key] = tot_m.get(key, 0) + int(cm_[key].sum())
        return {key: {"linear": tot_l[key], "ml": tot_m[key]} for key in tot_l}

    held_out = contact_counts(va)
    train_contacts = contact_counts(trn[:len(va)])
    # export: LightRig .lrm (output.* denormalises straight to PCA coefficients in metres)
    out_dir = Path(out_dir)
    lrm = {"fc1.weight": net.fc1.weight.detach().cpu().numpy(), "fc1.bias": net.fc1.bias.detach().cpu().numpy(),
           "fc2.weight": net.fc2.weight.detach().cpu().numpy(), "fc2.bias": net.fc2.bias.detach().cpu().numpy(),
           "input.mean": xm.cpu().numpy(), "input.scale": xs.cpu().numpy(),
           "output.mean": (cm / 1000).cpu().numpy(), "output.scale": (cs / 1000).cpu().numpy()}
    meta = {"format": "vhuman-ml-deformer", "version": "2", "inputs": json.dumps(tr.inputs),
            "outputs": "pca coefficients (metres) of pre-skinning corrective offsets",
            "kind": "face-corrective"}
    st.save(out_dir / "deformer.lrm", lrm, meta)
    st.save(out_dir / "deformer_basis.safetensors",
            {"basis": basis.reshape(k, V, 3).cpu().numpy(), "mean": (mean / 1000).reshape(V, 3).cpu().numpy()},
            {"vertices": V, "components": k, "units": "basis is unit-norm; offsets = mean + sum_k c_k basis_k"})
    stats = {"samples": N, "components": k, "hidden": hidden,
             "explained_variance": {"mouth": round(explained[0], 4), "rest": round(explained[1], 4)},
             "residual_mean_mm": round(float(mag.mean()), 4), "residual_p99_mm": round(float(mag.quantile(0.99)), 4),
             "val_error_mean_mm": round(float(ev.mean()), 4), "val_error_p99_mm": round(float(ev.quantile(0.99)), 4),
             "contact_vertices": contact, "held_out_contacts": held_out, "train_contacts": train_contacts, "solve_seconds": round(t_solve, 1),
             "seconds": round(time.perf_counter() - t0, 1)}
    (out_dir / "deformer.json").write_text(json.dumps(stats, indent=1))
    return stats


from server.vhuman.rig.mlruntime import MLDeformer  # noqa: E402,F401  (re-exported)
