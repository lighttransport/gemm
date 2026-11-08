"""One-time registration of the GNM skin surface onto the MetaHuman archetype head, and the GNM-UV -> MH-UV texel map.

Stages (CPU):
  A  similarity (Umeyama) from paired landmarks: eye corners, mouth corners, lip midpoints (GNM iBUG-68 barycentric
     landmarks vs named MetaHuman identity-template vertices), plus ear-region centroids at low weight;
  B  GNM identity coefficients (170 head modes) + similarity by Adam on landmark + point-to-plane + reverse-coverage terms;
  C  non-rigid ICP: free per-vertex displacement with a graph-Laplacian regulariser, point-to-surface targets, decreasing
     stiffness, interior/mouth-sock vertices excluded from the data term.
Outputs (derived geometry only, MODELS_ROOT/binding_v1/): gnm_on_mh.npz (registered GNM skin vertices, residuals) and
texel_map_<res>.npz (per GNM texel: MetaHuman UV, MH triangle, distance, validity). Because every MetaHuman character
shares the archetype topology and UV layout, the texel map is identity independent: any MH texture resamples to GNM UV
by a lookup at mh_uv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from . import MH_PLUGIN, MODELS_ROOT
from . import template as mh_template
from ..face_assets import asset_path

ROOT = Path(__file__).resolve().parents[3]
GNM_LM68 = ROOT / 'server/vhuman/reconstruction/data/gnm_landmarks68.txt'
FACE_LANDMARKS = MH_PLUGIN / 'Content/Face/IdentityTemplate/face_landmarks.json'
OUT_DIR = MODELS_ROOT / 'binding_v1'
# (GNM iBUG-68 index, MetaHuman named point)
PAIRS = [(45, 'pt_eye_corner_outer_l'), (36, 'pt_eye_corner_outer_r'), (42, 'pt_eye_corner_inner_l'), (39, 'pt_eye_corner_inner_r'),
         (54, 'pt_mouth_corner_l'), (48, 'pt_mouth_corner_r'), (51, 'pt_lip_upper_outer_m'), (57, 'pt_lip_lower_outer_m')]
MH_EAR_CURVES = ('crv_ear_outer_helix', 'crv_ear_inner_helix', 'ear_concha', 'ear_tragus', 'ear_antitragus', 'crv_ear_central_upper', 'crv_ear_central_lower')


# ---------------------------------------------------------------- geometry helpers
def vertex_normals(v, f):
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for i in range(3):
        np.add.at(n, f[:, i], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def closest_point_on_triangles(p, a, b, c):
    """Vectorised closest point (Ericson, Real-Time Collision Detection 5.1.5) for paired points/triangles.
    Returns closest points (N,3) and barycentrics (N,3)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b; d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c; d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    n = len(p); bary = np.zeros((n, 3))
    denom = va + vb + vc
    with np.errstate(divide='ignore', invalid='ignore'):
        v = vb / denom; w = vc / denom
    bary[:] = np.stack([1 - v - w, v, w], 1)
    # regions (order matters: later assignments override)
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = d1 / (d1 - d3)
    bary[m] = np.stack([1 - t[m], t[m], 0 * t[m]], 1)
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = d2 / (d2 - d6)
    bary[m] = np.stack([1 - t[m], 0 * t[m], t[m]], 1)
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    bary[m] = np.stack([0 * t[m], 1 - t[m], t[m]], 1)
    m = (d1 <= 0) & (d2 <= 0); bary[m] = [1, 0, 0]
    m = (d3 >= 0) & (d4 <= d3); bary[m] = [0, 1, 0]
    m = (d6 >= 0) & (d5 <= d6); bary[m] = [0, 0, 1]
    bary = np.nan_to_num(bary, nan=1 / 3)
    q = a * bary[:, :1] + b * bary[:, 1:2] + c * bary[:, 2:]
    return q, bary


class MeshQuery:
    """Closest point on a triangle mesh via k nearest triangle centroids (exact within the candidate set)."""

    def __init__(self, v, f, k=16):
        self.v, self.f, self.k = np.asarray(v, np.float64), np.asarray(f), k
        self.tree = cKDTree(self.v[self.f].mean(1))
        self.fn = np.cross(self.v[self.f[:, 1]] - self.v[self.f[:, 0]], self.v[self.f[:, 2]] - self.v[self.f[:, 0]])
        self.fn /= np.maximum(np.linalg.norm(self.fn, axis=1, keepdims=True), 1e-12)

    def query(self, p):
        p = np.asarray(p, np.float64)
        _, cand = self.tree.query(p, k=self.k)
        best_d = np.full(len(p), np.inf); best_q = np.zeros_like(p); best_t = np.zeros(len(p), np.int64); best_b = np.zeros((len(p), 3))
        for j in range(cand.shape[1]):
            t = cand[:, j]; tri = self.f[t]
            q, b = closest_point_on_triangles(p, self.v[tri[:, 0]], self.v[tri[:, 1]], self.v[tri[:, 2]])
            d = np.linalg.norm(q - p, axis=1)
            m = d < best_d
            best_d[m], best_q[m], best_t[m], best_b[m] = d[m], q[m], t[m], b[m]
        return best_q, best_d, best_t, best_b


def umeyama(src, dst, w=None):
    w = np.ones(len(src)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    ms, md = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    xs, xd = src - ms, dst - md
    cov = (w[:, None] * xd).T @ xs
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(3); d[2, 2] = np.sign(np.linalg.det(u @ vt))
    r = u @ d @ vt
    var = (w * (xs ** 2).sum(1)).sum()
    scale = np.trace(np.diag(s) @ d) / var
    t = md - scale * r @ ms
    return scale, r, t


def graph_laplacian(n, f):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    a = (a + a.T).tocsr(); a.data[:] = 1
    deg = np.asarray(a.sum(1)).ravel()
    return sparse.diags(deg) - a


# ---------------------------------------------------------------- data
def load_gnm():
    z = np.load(asset_path('gnm'), allow_pickle=True)
    names = [str(n) for n in z['vertex_group_names']]
    groups = {n: z['vertex_groups'][i] > .5 for i, n in enumerate(names)}
    skin = groups['skin']
    tris = z['triangles']; tm = skin[tris].all(1)
    ids = np.flatnonzero(skin); remap = -np.ones(len(skin), np.int64); remap[ids] = np.arange(len(ids))
    rows = [l.split() for l in GNM_LM68.read_text().splitlines() if l.strip()]
    lm = [(np.array([int(r[0]), int(r[2]), int(r[4])]), np.array([float(r[1]), float(r[3]), float(r[5])])) for r in rows]
    return dict(template=z['template_vertex_positions'].astype(np.float64), basis=z['vertex_identity_basis'][:170].astype(np.float64),
                triangles=tris, triangle_uvs=z['triangle_uvs'], skin=skin, skin_ids=ids, skin_tris=remap[tris[tm]], skin_tri_mask=tm,
                groups=groups, lm68=lm)


def mh_points():
    fl = json.loads(FACE_LANDMARKS.read_text())

    def ids(name):
        e = fl[name]; v = e.get('vIDs', e.get('vID'))
        return np.atleast_1d(np.asarray(v, np.int64))
    return fl, ids


def gnm_landmark(v, lm, i):
    vid, w = lm[i]
    return (v[vid] * w[:, None]).sum(0)


# ---------------------------------------------------------------- registration
def register(iters_b=300, stiffness=(50., 20., 8., 3., 1.5), verbose=True):
    import torch
    mh = mh_template.load()
    mv, mf = mh['vertices'].astype(np.float64), mh['triangles']
    query = MeshQuery(mv, mf)
    mnorm = vertex_normals(mv, mf)
    g = load_gnm()
    fl, mh_ids = mh_points()
    src = np.array([gnm_landmark(g['template'], g['lm68'], i) for i, _ in PAIRS])
    dst = np.array([mv[mh_ids(n)].mean(0) for _, n in PAIRS])
    gl, gr = g['groups']['ears'] & g['groups']['left'], g['groups']['ears'] & g['groups']['right']
    mh_ear_l = np.concatenate([mh_ids(f'{c}_l') for c in MH_EAR_CURVES]); mh_ear_r = np.concatenate([mh_ids(f'{c}_r') for c in MH_EAR_CURVES])
    src_ear = np.array([g['template'][gl].mean(0), g['template'][gr].mean(0)]); dst_ear = np.array([mv[mh_ear_l].mean(0), mv[mh_ear_r].mean(0)])
    # A: similarity
    s, r, t = umeyama(np.concatenate([src, src_ear]), np.concatenate([dst, dst_ear]), w=np.r_[np.ones(len(src)), .3 * np.ones(2)])
    res_a = np.linalg.norm((s * (r @ src.T)).T + t - dst, axis=1)
    if verbose:
        print(f'A similarity scale {s:.4f}; landmark residual mm mean {1e3 * res_a.mean():.2f} max {1e3 * res_a.max():.2f}')
    # B: identity + similarity (torch, CPU)
    skin = g['skin_ids']; ext = g['groups']['skin_exterior'][skin] & ~g['groups']['mouth_sock'][skin]
    tmpl = torch.tensor(g['template'][skin]); basis = torch.tensor(g['basis'][:, skin])          # (170, Ns, 3)
    beta = torch.zeros(170, dtype=torch.float64, requires_grad=True)
    logs = torch.tensor(np.log(s), requires_grad=True)
    rotv = torch.tensor(rotation_to_axis_angle(r), requires_grad=True)
    trans = torch.tensor(t, requires_grad=True)
    lm_rows = [g['lm68'][i] for i, _ in PAIRS]
    lm_idx = torch.tensor(np.stack([np.searchsorted(skin, v) for v, _ in lm_rows])); lm_w = torch.tensor(np.stack([w for _, w in lm_rows]))
    dst_t = torch.tensor(dst)
    opt = torch.optim.Adam([beta, logs, rotv, trans], lr=.01)
    face_mask_mh = facial_mask_mh(mv, dst)           # MH vertices used for the reverse (coverage) term
    for it in range(iters_b):
        if it % 25 == 0:
            with torch.no_grad():
                x = similarity(tmpl + torch.einsum('k,kvc->vc', beta, basis), logs, rotv, trans).numpy()
            q, d, tri, _ = query.query(x[ext])
            qn = query.fn[tri]
            q_t, qn_t = torch.tensor(q), torch.tensor(qn)
            rev_tree = cKDTree(x[ext]); _, nn = rev_tree.query(mv[face_mask_mh]); rev_src = torch.tensor(np.flatnonzero(ext)[nn]); rev_dst = torch.tensor(mv[face_mask_mh])
        x = similarity(tmpl + torch.einsum('k,kvc->vc', beta, basis), logs, rotv, trans)
        lmx = (x[lm_idx] * lm_w[..., None]).sum(1)
        e_lm = ((lmx - dst_t) ** 2).sum(1).mean()
        xe = x[torch.tensor(np.flatnonzero(ext))]
        e_pl = torch.nn.functional.huber_loss(((xe - q_t) * qn_t).sum(1), torch.zeros(len(q_t), dtype=torch.float64), delta=.005)
        e_rev = torch.nn.functional.huber_loss(x[rev_src], rev_dst, delta=.005)
        e_reg = (beta ** 2).mean()
        loss = 10 * e_lm + e_pl + .5 * e_rev + 1e-6 * e_reg
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and it % 100 == 0:
            print(f'B it {it}: lm {1e3 * e_lm.sqrt().item():.2f} mm rms, plane {e_pl.item():.2e}, rev {e_rev.item():.2e}, |beta| {beta.abs().max().item():.2f}')
    with torch.no_grad():
        x0 = similarity(tmpl + torch.einsum('k,kvc->vc', beta, basis), logs, rotv, trans).numpy()
    # C: non-rigid ICP with Laplacian regulariser (separable per coordinate)
    lap = graph_laplacian(len(skin), g['skin_tris'])
    lap2 = (lap.T @ lap).tocsc()
    data_mask = ext.copy()
    x = x0.copy()
    gnm_norm = lambda v: vertex_normals(v, g['skin_tris'])
    lm_sel = [np.searchsorted(skin, v) for v, _ in lm_rows]
    for alpha in stiffness:
        for _ in range(3):
            q, d, tri, _ = query.query(x)
            nx = gnm_norm(x); agree = (nx * query.fn[tri]).sum(1)
            w = data_mask & (d < .02) & (agree > .3)
            wd = w.astype(float)
            a = alpha * lap2 + sparse.diags(wd)
            # landmark soft constraints on their barycentric vertices
            lm_mat = np.zeros((len(PAIRS), len(skin)))
            for k, (ids_k, (_, wk)) in enumerate(zip(lm_sel, lm_rows)):
                lm_mat[k, ids_k] = wk
            lm_sp = sparse.csr_matrix(lm_mat); wl = 50.
            a = (a + wl * (lm_sp.T @ lm_sp)).tocsc()
            solver = splu(a)
            newx = np.empty_like(x)
            for c in range(3):
                b = alpha * (lap2 @ x0[:, c]) + wd * q[:, c] + wl * (lm_sp.T @ dst[:, c])
                newx[:, c] = solver.solve(b)
            x = newx
        q, d, _, _ = query.query(x)
        if verbose:
            print(f'C alpha {alpha}: surface distance mm median {1e3 * np.median(d[ext]):.2f} p95 {1e3 * np.percentile(d[ext], 95):.2f} max {1e3 * d[ext].max():.1f}')
    q, d, tri, bary = query.query(x)
    report = dict(similarity_scale=float(s), stage_a_landmark_mm=(1e3 * res_a).tolist(), stiffness=list(stiffness),
                  surface_distance_mm=dict(median=float(1e3 * np.median(d[ext])), p95=float(1e3 * np.percentile(d[ext], 95)), max=float(1e3 * d[ext].max())),
                  per_region_mm={k: float(1e3 * np.median(d[g['groups'][k][skin] & ext])) for k in ('ears', 'nose_region', 'upper_lip', 'lower_lip', 'eye_sockets',
                                                                                                        'forehead_region', 'left_cheek_region', 'chin_region')
                                 if (g['groups'][k][skin] & ext).any()},
                  beta_abs_max=float(beta.detach().abs().max()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / 'gnm_on_mh.npz', skin_ids=skin, vertices=x.astype(np.float32), stage_b=x0.astype(np.float32),
                        distance=d.astype(np.float32), mh_tri=tri.astype(np.int32), mh_bary=bary.astype(np.float32), beta=beta.detach().numpy())
    (OUT_DIR / 'gnm_on_mh.json').write_text(json.dumps(report, indent=1) + '\n')
    return report


def facial_mask_mh(mv, dst):
    """MetaHuman vertices in front of the ears (used for the reverse coverage term): within 11 cm of the landmark centroid
    and on the face side (z above the ear plane)."""
    c = dst.mean(0)
    return (np.linalg.norm(mv - c, axis=1) < .11) & (mv[:, 2] > c[2] - .05)


def rotation_to_axis_angle(r):
    ang = np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))
    if ang < 1e-8:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2 * np.sin(ang))
    return axis * ang


def similarity(x, logs, rotv, trans):
    import torch
    theta = rotv.norm() + 1e-12; k = rotv / theta
    kx = torch.zeros(3, 3, dtype=x.dtype)
    kx[0, 1], kx[0, 2], kx[1, 0], kx[1, 2], kx[2, 0], kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    r = torch.eye(3, dtype=x.dtype) + torch.sin(theta) * kx + (1 - torch.cos(theta)) * kx @ kx
    return torch.exp(logs) * x @ r.T + trans


# ---------------------------------------------------------------- texel map
def texel_map(res, max_dist=.004, min_agree=.5):
    from ..rig.bake import rasterize_uv
    g = load_gnm(); mh = mh_template.load()
    reg = np.load(OUT_DIR / 'gnm_on_mh.npz')
    x = reg['vertices'].astype(np.float64)
    tris = g['skin_tris']
    uv = g['triangle_uvs'][g['skin_tri_mask']].reshape(-1, 2)
    corner = np.arange(len(uv)).reshape(-1, 3)
    tid, bary = rasterize_uv(uv, corner, res)
    yy, xx = np.nonzero(tid >= 0)
    t = tid[yy, xx]; b = bary[yy, xx].astype(np.float64)
    p = (x[tris[t]] * b[..., None]).sum(1)
    nx = vertex_normals(x, tris); n = (nx[tris[t]] * b[..., None]).sum(1); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    query = MeshQuery(mh['vertices'], mh['triangles'])
    q, d, mt, mb = query.query(p)
    agree = (n * query.fn[mt]).sum(1)
    muv = (mh['triangle_uvs'][mt] * mb[..., None]).sum(1)
    sock = g['groups']['mouth_sock'][g['skin_ids']][tris[t]].any(1)
    valid = (d < max_dist) & (agree > min_agree) & ~sock
    mh_uv = np.full((res, res, 2), np.nan, np.float32); mh_uv[yy, xx] = muv
    dist = np.full((res, res), np.nan, np.float32); dist[yy, xx] = d
    val = np.zeros((res, res), bool); val[yy, xx] = valid
    covered = np.zeros((res, res), bool); covered[yy, xx] = True
    out = OUT_DIR / f'texel_map_{res}.npz'
    np.savez_compressed(out, mh_uv=mh_uv, distance=dist, valid=val, covered=covered, mh_tri=_scatter(res, yy, xx, mt, -1, np.int32),
                        mh_bary=_scatter3(res, yy, xx, mb))
    rep = dict(res=res, covered=int(covered.sum()), valid=int(val.sum()), valid_fraction=float(val.sum() / max(covered.sum(), 1)),
               distance_mm_p95=float(1e3 * np.nanpercentile(dist[covered], 95)))
    (OUT_DIR / f'texel_map_{res}.json').write_text(json.dumps(rep, indent=1) + '\n')
    return rep


def _scatter(res, yy, xx, v, fill, dtype):
    a = np.full((res, res), fill, dtype); a[yy, xx] = v; return a


def _scatter3(res, yy, xx, v):
    a = np.zeros((res, res, 3), np.float16); a[yy, xx] = v; return a


def resample(texture, tmap, *, linear=False):
    """Sample an MH-UV texture (H,W,C uint8 or float) at the texel map's mh_uv (bilinear); returns (res,res,C) float32 and valid."""
    img = np.asarray(texture, np.float32)
    if img.dtype == np.float32 and img.max() > 1.5:
        img = img / 255.
    h, w = img.shape[:2]
    uv = tmap['mh_uv']; ok = tmap['valid'] & np.isfinite(uv).all(-1)
    u = np.nan_to_num(uv[..., 0]) * w - .5; v = np.nan_to_num(uv[..., 1]) * h - .5
    x0 = np.clip(np.floor(u).astype(int), 0, w - 1); y0 = np.clip(np.floor(v).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    fx = np.clip(u - np.floor(u), 0, 1)[..., None]; fy = np.clip(v - np.floor(v), 0, 1)[..., None]
    out = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    out[~ok] = 0
    return out.astype(np.float32), ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--skip-register', action='store_true'); ap.add_argument('--res', type=int, nargs='*', default=[1024])
    a = ap.parse_args()
    if not a.skip_register:
        print(json.dumps(register(), indent=1))
    for r in a.res:
        print(json.dumps(texel_map(r), indent=1))


if __name__ == '__main__':
    main()
