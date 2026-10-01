"""Fit the template topology to a subject head (features.py -> 3D vertices).

1. Rings: the subject's ring-0 loops (lid margins on the eyeballs, the lip
   seam, the neck cut) give every ring vertex's chart position through the
   same construction as the template's (template.rings_chart).
2. Free vertices: the canonical chart is warped onto the subject's with a
   compactly supported radial basis (Wendland C2) interpolating the ring
   and landmark correspondences; far regions do not move.
3. Surface: every skin vertex is cast from the centre C along its direction
   onto the subject's outermost surface (a max-radius map over the chart).
4. Refinement (PyTorch): vertex positions minimise point-to-plane distance
   to the subject surface, edge lengths matching the template's (scaled),
   and a bending term, with ring 0 fixed. Nearest surface points are
   re-queried periodically (a k-d tree over the dense subject vertices).
5. Inner strips: the lid margins/linings on the eyeballs, the lips' inner
   surface and the oral cavity are placed analytically from the rings.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import template as T
from .common import normalize, smoothstep
from .features import Features

WENDLAND_SUPPORT_MM = 70.0
MAP_RES_MM = 1.0                  # max-radius map resolution (chart mm)
MAP_EXTENT_MM = 320.0
LID_FRONT_M = 0.0010              # anterior lid edge (ring 0) off the eyeball
LID_BACK_M = 0.00015              # posterior lid edge (ring -1) off the eyeball
LINING_ARC_M = 0.004              # conjunctiva: ring -2 this far under the lid


def wendland(r):
    r = np.clip(r, 0, 1)
    return (1 - r) ** 4 * (4 * r + 1)


def rbf_warp(src: np.ndarray, dst: np.ndarray, query: np.ndarray, support: float) -> np.ndarray:
    """2D displacement interpolation: query + sum_j w_j phi(|query - src_j|)."""
    d = np.linalg.norm(src[:, None] - src[None], axis=-1) / support
    # a small ridge: nearby controls (1 mm apart on the rings) make A ill-conditioned
    A = wendland(d) + 1e-4 * np.eye(len(src))
    w = np.linalg.solve(A, dst - src)
    out = query.copy()
    for i in range(0, len(query), 4096):
        q = query[i:i + 4096]
        out[i:i + 4096] += wendland(np.linalg.norm(q[:, None] - src[None], axis=-1) / support) @ w
    return out


class RadiusMap:
    """Outermost distance from C over the chart (vertex splats, max-filtered)."""

    def __init__(self, positions: np.ndarray, center: np.ndarray):
        n = int(2 * MAP_EXTENT_MM / MAP_RES_MM)
        c = T.to_chart(positions - center)
        r = np.linalg.norm(positions - center, axis=1)
        ij = np.floor((c + MAP_EXTENT_MM) / MAP_RES_MM).astype(np.int64)
        ok = ((ij >= 0) & (ij < n)).all(1)
        grid = np.zeros((n, n))
        np.maximum.at(grid, (ij[ok, 1], ij[ok, 0]), r[ok])
        g = grid
        for _ in range(2):                       # close the gaps between splats
            g = np.max(np.stack([np.roll(np.roll(g, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), 0)
        self.grid = np.where(grid > 0, grid, g)
        self.center, self.n = center, n

    def sample(self, chart: np.ndarray) -> np.ndarray:
        f = (chart + MAP_EXTENT_MM) / MAP_RES_MM - 0.5
        x0 = np.clip(np.floor(f[:, 0]).astype(int), 0, self.n - 2)
        y0 = np.clip(np.floor(f[:, 1]).astype(int), 0, self.n - 2)
        tx, ty = np.clip(f[:, 0] - x0, 0, 1), np.clip(f[:, 1] - y0, 0, 1)
        g = self.grid
        vals = np.stack([g[y0, x0], g[y0, x0 + 1], g[y0 + 1, x0], g[y0 + 1, x0 + 1]], 1)
        wts = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], 1)
        wts = wts * (vals > 0)
        return (vals * wts).sum(1) / np.maximum(wts.sum(1), 1e-9)

    def cast(self, chart: np.ndarray) -> np.ndarray:
        return self.center + T.from_chart(chart) * self.sample(chart)[:, None]


def subject_rings(feat: Features, center=T.CENTER) -> tuple[dict, dict]:
    """Subject ring charts and the ring-0 positions in 3D."""
    eyes = {e["side"]: e["contour"] for e in feat.eyes}
    seam_c = T.chart_of_points(feat.seam, center)
    loop_c = T.mouth_loop(seam_c, seam_c)
    rc = T.rings_chart(eyes, loop_c, feat.neck["ring"], center)
    ring0 = {"eye_right": eyes["right"], "eye_left": eyes["left"], "mouth": T.mouth_loop(feat.seam, feat.seam),
             "neck": feat.neck["ring"]}
    return rc, ring0


def _eye_inner(ring0: np.ndarray, eye: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anterior lid edge (ring 0), posterior edge (-1) and lining end (-2)
    from the lid-margin contour on the eyeball."""
    c, R = eye["center"], eye["radius"]
    gaze = eye["rotation"][:, 2]
    apex = c + R * gaze
    u = normalize(ring0 - c)
    back = c + u * (R + LID_BACK_M)
    # tangent away from the fissure centre (under the lid)
    t = (ring0 - apex)
    t = normalize(t - (t * u).sum(1, keepdims=True) * u)
    front = c + u * (R + LID_FRONT_M) + 0.0003 * t
    ang = LINING_ARC_M / R
    lin = c + (u * math.cos(ang) + t * math.sin(ang)) * (R + LID_BACK_M * 0.5)
    return front, back, lin


def _mouth_inner(seam: np.ndarray, feat: Features) -> list[np.ndarray]:
    """Rings -1..-MOUTH_INNER of the mouth: the lips' contact (closed), the
    vestibule folds, then the cavity; and the cap point."""
    N, H = T.MOUTH_N, T.MOUTH_HALF
    loop = T.mouth_loop(seam, seam)
    upper = np.zeros(N, bool)
    upper[1:H] = True
    lower = np.zeros(N, bool)
    lower[H + 1:] = True
    mid = 0.5 * (seam[0] + seam[-1])
    fwd = normalize(np.array([0.0, 0.0, 1.0]))
    width = float(np.linalg.norm(seam[-1] - seam[0]))
    # position along the mouth, -1 (right corner) .. 1 (left corner)
    xs = (loop[:, 0] - mid[0]) / (0.5 * width)
    sgn_v = np.where(upper, 1.0, np.where(lower, -1.0, 0.0))
    arc = np.sqrt(np.clip(1 - xs ** 2, 0, 1))
    rings = []
    # -1: the lips' contact surface, 4 mm in (closed: both lips together)
    rings.append(loop - fwd * 0.004 * (0.4 + 0.6 * arc)[:, None])
    # -2: the vestibule fold behind each lip (above/below the teeth's front)
    r2 = loop - fwd * (0.007 * (0.5 + 0.5 * arc))[:, None]
    r2[:, 1] += sgn_v * (0.006 + 0.002 * arc) * np.where(upper, 1.0, 1.15)
    rings.append(r2)
    # -3..-6: the cavity: ellipses around the teeth and tongue, deeper and narrower
    depth = (0.012, 0.026, 0.040, 0.052)
    half_w = (0.5 * width + 0.006, 0.5 * width + 0.004, 0.5 * width - 0.002, 0.5 * width - 0.010)
    up_h = (0.013, 0.014, 0.012, 0.008)
    dn_h = (0.017, 0.020, 0.018, 0.012)
    th = np.zeros(N)
    # angle around the ring: upper lip 0..pi (right -> left over the top), lower pi..2pi
    th[:H + 1] = np.linspace(math.pi, 0, H + 1)
    th[H + 1:] = np.linspace(0, -math.pi, N - H + 1)[1:-1]
    for dz, hw, uh, dh in zip(depth, half_w, up_h, dn_h):
        s = np.sin(th)
        y = np.where(s >= 0, uh * s, dh * s)
        ring = np.stack([mid[0] + hw * np.cos(th), mid[1] - 0.002 + y, np.full(N, mid[2] - dz)], 1)
        rings.append(ring)
    cap = np.array([mid[0], mid[1] - 0.004, mid[2] - 0.060])
    return rings, cap


def fit(tmpl: T.Template, subj, feat: Features, iters: int = 400, device: str | None = None,
        log=None) -> dict:
    t0 = time.perf_counter()
    C = T.CENTER
    rc_s, ring0 = subject_rings(feat, C)
    cf = T.canonical_features()
    V = tmpl.n
    pos = np.full((V, 3), np.nan)
    fixed = np.zeros(V, bool)
    subj_chart = np.full((V, 2), np.nan)
    names = ["eye_right", "eye_left", "mouth", "neck"]
    ring_src, ring_dst = [], []
    for name in names:
        K = rc_s[name].shape[0]
        for k in range(K):
            ids = tmpl.ring_ids(name, k)
            subj_chart[ids] = rc_s[name][k]
            ring_src.append(tmpl.chart[ids])
            ring_dst.append(rc_s[name][k])
    # landmarks
    lm_src = T.chart_of_points(np.stack([cf["points"][k] for k in cf["points"]]), C)
    lm_dst = T.chart_of_points(np.stack([feat.points[k] for k in cf["points"]]), C)
    src = np.concatenate(ring_src + [lm_src])
    dst = np.concatenate(ring_dst + [lm_dst])
    # anchors: far away, fixed
    th = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    anchors = np.stack([235 * np.cos(th), 235 * np.sin(th)], 1)
    src = np.concatenate([src[::2], anchors])       # every other ring vertex is plenty
    dst = np.concatenate([dst[::2], anchors])
    free = tmpl.kind == T.KIND["free"]
    subj_chart[free] = rbf_warp(src, dst, tmpl.chart[free], WENDLAND_SUPPORT_MM)
    rmap = RadiusMap(subj.positions, C)
    skin = np.isin(tmpl.kind, [T.KIND["free"], T.KIND["eye"], T.KIND["mouth"], T.KIND["neck"]])
    pos[skin] = rmap.cast(subj_chart[skin])
    # ring 0 exact
    for name in names:
        ids = tmpl.ring_ids(name, 0)
        pos[ids] = ring0[name]
        fixed[ids] = True
    eye_by_side = {e["side"]: e for e in feat.eyes}
    for side in ("right", "left"):
        name = f"eye_{side}"
        front, back, lin = _eye_inner(ring0[name], eye_by_side[side])
        pos[tmpl.ring_ids(name, 0)] = front
        pos[tmpl.ring_ids(name, -1)] = back
        pos[tmpl.ring_ids(name, -2)] = lin
        fixed[tmpl.ring_ids(name, -1)] = fixed[tmpl.ring_ids(name, -2)] = True
        # keep the first skin rings off the eyeball (the lid's thickness)
        e = eye_by_side[side]
        for k, lift in ((1, LID_FRONT_M), (2, 0.6 * LID_FRONT_M), (3, 0.3 * LID_FRONT_M)):
            ids = tmpl.ring_ids(name, k)
            d = pos[ids] - e["center"]
            r = np.linalg.norm(d, axis=1, keepdims=True)
            pos[ids] = e["center"] + d / r * np.maximum(r, e["radius"] + lift)
    inner, cap = _mouth_inner(feat.seam, feat)
    for k, ring in enumerate(inner, start=1):
        ids = tmpl.ring_ids("mouth", -k)
        pos[ids] = ring
        fixed[ids] = True
    cap_id = np.flatnonzero(tmpl.kind == T.KIND["cap"])
    pos[cap_id] = cap
    fixed[cap_id] = True
    init = pos.copy()
    t_init = time.perf_counter() - t0
    refined, stats = refine(tmpl, subj, pos, fixed, skin, iters=iters, device=device, log=log)
    stats.update({"init_seconds": round(t_init, 2), "seconds": round(time.perf_counter() - t0, 2)})
    return {"positions": refined, "init": init, "fixed": fixed, "skin": skin, "chart": subj_chart, "stats": stats}


def _template_edge_lengths(tmpl: T.Template, pos: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Target edge lengths: the template chart's (a well-shaped layout),
    scaled by the local distance from C (the chart's radius)."""
    d = T.from_chart(tmpl.chart)
    chord = np.linalg.norm(d[edges[:, 0]] - d[edges[:, 1]], axis=1)
    r = np.linalg.norm(pos - T.CENTER, axis=1)
    return chord * 0.5 * (r[edges[:, 0]] + r[edges[:, 1]])


def refine(tmpl: T.Template, subj, pos: np.ndarray, fixed: np.ndarray, skin: np.ndarray, iters: int = 400,
           device: str | None = None, log=None) -> tuple[np.ndarray, dict]:
    import torch
    from scipy.spatial import cKDTree
    from .common import edges as edge_list
    from ..runtime import torch_device
    dev = torch.device(device if device is not None else torch_device(torch))
    tris = tmpl.tris[tmpl.tri_mat == 0]
    E = edge_list(tris)
    L0 = _template_edge_lengths(tmpl, pos, E)
    # the ring vertices' rest spacing comes from the subject's own contours
    tree = cKDTree(subj.positions)
    movable = skin & ~fixed
    x = torch.tensor(pos, dtype=torch.float64, device=dev)
    mv = torch.tensor(movable, device=dev)
    var = torch.zeros((int(movable.sum()), 3), dtype=torch.float64, device=dev, requires_grad=True)
    idx = torch.tensor(np.flatnonzero(movable), device=dev)
    Et = torch.tensor(E, device=dev)
    L0t = torch.tensor(L0, device=dev)
    # the chart layout's shape is the target; the subject's scale is taken from the init
    cur = np.linalg.norm(pos[E[:, 0]] - pos[E[:, 1]], axis=1)
    ok = np.isfinite(cur)
    scale = float(np.median(cur[ok] / np.maximum(L0[ok], 1e-9)))
    L0t = L0t * scale
    # uniform Laplacian (bending): neighbours' mean
    n = len(pos)
    deg = np.bincount(E.reshape(-1), minlength=n).astype(np.float64)
    degt = torch.tensor(deg, device=dev).clamp(min=1)[:, None]
    opt = torch.optim.Adam([var], lr=3e-4)
    w_plane, w_point, w_edge, w_bend = 1.0, 0.05, 0.3, 0.05
    q = n_ = None
    actual_backend = "cpu" if dev.type == "cpu" else ("rocm" if torch.version.hip else "cuda")
    stats = {"backend": actual_backend, "device": str(dev), "iters": iters}
    for it in range(iters):
        if it % 25 == 0:
            with torch.no_grad():
                cur_x = x.clone()
                cur_x[idx] += var
                p = cur_x[idx].cpu().numpy()
            # nearest subject point whose normal agrees with the template's
            # (concavities such as the nostrils otherwise pull vertices inside)
            from .common import vertex_normals
            cur_n = vertex_normals(cur_x.cpu().numpy(), tris)[movable]
            dk, jk = tree.query(p, k=8)
            agree = (subj.normals[jk] * cur_n[:, None]).sum(-1) > 0.3
            pick = np.argmin(np.where(agree, dk, dk + 1.0), 1)
            j = jk[np.arange(len(p)), pick]
            dist = dk[np.arange(len(p)), pick]
            q = torch.tensor(subj.positions[j], device=dev)
            n_ = torch.tensor(subj.normals[j], device=dev)
            if it == 0:
                stats["initial_mean_dist_mm"] = round(float(dist.mean() * 1000), 3)
        opt.zero_grad()
        X = x.clone()
        X[idx] = X[idx] + var
        P = X[idx]
        e_plane = (((P - q) * n_).sum(1) ** 2).mean()
        e_point = ((P - q) ** 2).sum(1).mean()
        el = (X[Et[:, 0]] - X[Et[:, 1]]).norm(dim=1)
        e_edge = ((el - L0t) ** 2).mean()
        acc = torch.zeros_like(X)
        acc.index_add_(0, Et[:, 0], X[Et[:, 1]])
        acc.index_add_(0, Et[:, 1], X[Et[:, 0]])
        lap = acc / degt - X
        e_bend = (lap[idx] ** 2).sum(1).mean()
        loss = (w_plane * e_plane + w_point * e_point + w_edge * e_edge + w_bend * e_bend) * 1e6
        loss.backward()
        opt.step()
        if log and it % 100 == 0:
            log(f"refine {it}: plane {e_plane.item() ** .5 * 1e3:.3f} mm, edge {e_edge.item() ** .5 * 1e3:.3f} mm")
    with torch.no_grad():
        X = x.clone()
        X[idx] += var
        out = X.cpu().numpy()
    dist, _ = tree.query(out[movable])
    stats.update({"final_mean_dist_mm": round(float(dist.mean() * 1000), 3),
                  "final_p95_dist_mm": round(float(np.percentile(dist, 95) * 1000), 3),
                  "edge_scale": round(scale, 4), "movable": int(movable.sum())})
    return out, stats
