"""Dense-ear shape and albedo prior from the MetaHuman dataset, in vhuman's refined-ear topology.

usage: python -m server.vhuman.metahuman.ear_prior_fit [--topology binding_v1/ear_topology_ref.npz] [--k 20] [--out PATH]
The refined ear (ear_mesh.py) is a subdivision of the native GNM ear quads plus a head-skin band whose extent varies per
subject; the 4,108 ear vertices are identical across subjects (same subdivision-operator rows), so the prior is built on a
reference topology and remapped per subject by ear_mesh.adapt_prior. Its zero-offset base points are placed on the registered GNM (binding_v1/gnm_on_mh.npz) and projected once onto
the MetaHuman archetype surface, giving a fixed dense-ear -> MH binding (triangle + barycentrics). Evaluating that binding
on each generated head samples MetaHuman's ear relief at dense resolution. Per side the ears are similarity-aligned to the
mean (generalised Procrustes), then PCA gives mean + basis (per side: the dense topology's L/R correspondence is not one-to-one); per-vertex log albedo (sampled from each sample's
basecolor at the bound MH UVs) gives an ear colour mean + basis relative to the cheek tone.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from . import MODELS_ROOT
from . import template as mh_template
from .manifest import prior_report, sha256
from .register import MeshQuery, OUT_DIR, load_gnm, umeyama
from .samples import V1, load_sample, sample_dirs, to_vhuman, texture
from .skin_prior_fit import srgb_to_linear
from ..reconstruction.skin_prior import pca_small_n

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOPOLOGY = MODELS_ROOT / 'binding_v1' / 'ear_topology_ref.npz'   # refined-ear topology (subdivision operator) dumped from the s1 ear workspace
RELIEF_ITERATIONS = 60
import os as _os
PROJECTION = _os.environ.get('EAR_PRIOR_PROJECTION', 'closest')     # 'closest' or 'ray' (normal-ray, nearest hit)


def load_topology(path=DEFAULT_TOPOLOGY):
    from scipy.sparse import csr_matrix
    z = np.load(path)
    op = csr_matrix((z['op_data'], z['op_indices'], z['op_indptr']), shape=tuple(z['op_shape']))
    return dict(operator=op, side=z['side'].astype(int), band=z['band'].astype(bool), triangles=z['triangles'], native_count=int(z['native_count']))


def dense_binding(topology=DEFAULT_TOPOLOGY):
    """Fixed binding of the dense-ear base points to the MH archetype surface (computed on the archetype).

    Dense base = subdivision operator @ registered GNM (exactly what the ear fitter's topo['base'] is for a subject);
    template_points = operator @ GNM template is the subject-independent vertex key used by ear_mesh.adapt_prior."""
    topo = load_topology(topology)
    gnm = load_gnm(); reg = np.load(OUT_DIR / 'gnm_on_mh.npz')
    full = gnm['template'].copy(); full[reg['skin_ids']] = reg['vertices']            # registered GNM (skin) on MH archetype
    base = topo['operator'] @ full
    template_points = topo['operator'] @ gnm['template']
    mh = mh_template.load()
    q, d, tri, bary = MeshQuery(mh['vertices'], mh['triangles']).query(base)          # closest point (default)
    nrm = dense_normals(base, topo['triangles'], topo['native_count'])
    if PROJECTION == 'ray':
        hit, rq, rd, rtri, rbary = ray_project(base, nrm, mh['vertices'], mh['triangles'])
        q[hit], d[hit], tri[hit], bary[hit] = rq[hit], rd[hit], rtri[hit], rbary[hit]
    else:
        hit = np.zeros(len(base), bool)
    side = topo['side']; band = topo['band']
    lid, rid = np.flatnonzero(side == 0), np.flatnonzero(side == 1)
    if base[lid][:, 0].mean() < 0:            # 'left' = +x (subject-left)
        lid, rid = rid, lid
    lx = q[lid]; rx = q[rid] * np.array([-1, 1, 1])
    dm, nn = cKDTree(lx).query(rx)
    return dict(template_points=template_points, ray_hit=hit, mh_tri=mh['triangles'][tri], mh_tri_index=tri, mh_bary=bary, distance=d, left=lid, right=rid,
                right_to_left=nn, mirror_err=dm, band=band, uv=(mh['triangle_uvs'][tri] * bary[..., None]).sum(1), dense_triangles=topo['triangles'],
                native_count=topo['native_count'])


def dense_normals(points, dense_triangles, native_count):
    """Vertex normals of the dense ear from its dense-only triangles (native-indexed corners are ignored)."""
    t = np.asarray(dense_triangles); t = t[(t >= native_count).all(1)] - native_count
    fn = np.cross(points[t[:, 1]] - points[t[:, 0]], points[t[:, 2]] - points[t[:, 0]])
    n = np.zeros_like(points)
    for i in range(3):
        np.add.at(n, t[:, i], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def ray_project(points, normals, v, f, max_dist=.008, k=96, min_facing=-1.):
    """Nearest intersection of the lines p + t n (|t| <= max_dist) with the mesh (Moller-Trumbore over the k triangles whose
    centroids are nearest to p), accepting only triangles facing the same way as n (dot > min_facing). Returns hit mask,
    points, |t|, triangle index, barycentrics."""
    tree = cKDTree(v[f].mean(1))
    _, cand = tree.query(points, k=k)
    n = len(points); best = np.full(n, np.inf); bq = np.zeros_like(points); bt = np.zeros(n, np.int64); bb = np.zeros((n, 3))
    for j in range(cand.shape[1]):
        tri = cand[:, j]; a, b, c = v[f[tri, 0]], v[f[tri, 1]], v[f[tri, 2]]
        e1, e2 = b - a, c - a
        pv = np.cross(normals, e2); det = (e1 * pv).sum(1)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        tv = points - a
        u = (tv * pv).sum(1) * inv
        qv = np.cross(tv, e1)
        w = (normals * qv).sum(1) * inv
        t = (e2 * qv).sum(1) * inv
        fn = np.cross(e1, e2); fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-18)
        facing = (fn * normals).sum(1) > min_facing          # same-facing sheet only: no hopping onto the back of a fold
        m = ok & facing & (u >= -1e-6) & (w >= -1e-6) & (u + w <= 1 + 1e-6) & (np.abs(t) <= max_dist) & (np.abs(t) < best)
        best[m] = np.abs(t[m]); bq[m] = points[m] + t[m, None] * normals[m]; bt[m] = tri[m]
        bb[m] = np.stack([1 - u[m] - w[m], u[m], w[m]], 1)
    return np.isfinite(best), bq, best, bt, bb


def side_triangles(dense_triangles, native_count, index):
    """Dense-only triangles of one side, re-indexed to the side's ascending vertex order."""
    t = np.asarray(dense_triangles); t = t[(t >= native_count).all(1)] - native_count
    remap = -np.ones(int(t.max()) + 1, np.int64); remap[index] = np.arange(len(index))
    t = remap[t]
    return t[(t >= 0).all(1)]


def relief_heights(shape, tris, fixed, iterations=60):
    """Per-vertex height of `shape` above its own Laplacian low-pass (band vertices fixed), along the low-pass normals."""
    from scipy.sparse import csr_matrix
    e = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), 1), axis=0)
    n = len(shape)
    adj = csr_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    deg = np.maximum(np.asarray(adj.sum(1)).ravel(), 1)[:, None]
    low = shape.copy()
    for _ in range(iterations):
        nxt = .5 * low + .5 * (adj @ low) / deg
        nxt[fixed] = shape[fixed]
        low = nxt
    fn = np.cross(low[tris[:, 1]] - low[tris[:, 0]], low[tris[:, 2]] - low[tris[:, 0]])
    nl = np.zeros_like(low)
    for i in range(3):
        np.add.at(nl, tris[:, i], fn)
    nl /= np.maximum(np.linalg.norm(nl, axis=1, keepdims=True), 1e-12)
    return ((shape - low) * nl).sum(1)


def sample_ear(db, head_vh):
    return (head_vh[db['mh_tri']] * db['mh_bary'][..., None]).sum(1)


def sample_colour(db, base_png_rgb):
    img = srgb_to_linear(np.asarray(base_png_rgb, np.float32) / 255.)
    h, w = img.shape[:2]
    u = np.clip((db['uv'][:, 0] * w - .5).round().astype(int), 0, w - 1); v = np.clip((db['uv'][:, 1] * h - .5).round().astype(int), 0, h - 1)
    return img[v, u]


def fit(topology=DEFAULT_TOPOLOGY, k_modes=20, out=None):
    db = dense_binding(topology)
    print(f'dense binding: ray hits {db["ray_hit"].mean():.3f}; projection distance mm median {1e3 * np.median(db["distance"]):.2f} p95 {1e3 * np.percentile(db["distance"], 95):.2f}; '
          f'L/R mirror match mm median {1e3 * np.median(db["mirror_err"]):.2f} p95 {1e3 * np.percentile(db["mirror_err"], 95):.2f}', flush=True)
    sampling = json.loads((V1 / 'sampling.json').read_text()); held = set(sampling['held_out_ids'])
    sides = dict(left=db['left'], right=db['right'])
    data = {k: dict(train=[], test=[], cols=[], h=[], htest=[]) for k in sides}
    native = db['native_count']
    side_tris = {k: side_triangles(db['dense_triangles'], native, idx) for k, idx in sides.items()}
    ids = []
    for d in sample_dirs():
        s = load_sample(d)
        head = to_vhuman(s['face']['head'][0])
        ear = sample_ear(db, head)
        col = np.log(np.maximum(sample_colour(db, texture(d, 'basecolor')[..., :3]), 1e-3))
        for k, idx in sides.items():
            h = relief_heights(ear[idx], side_tris[k], db['band'][idx], RELIEF_ITERATIONS)
            if d.name in held:
                data[k]['test'].append(ear[idx]); data[k]['htest'].append(h)
            else:
                data[k]['train'].append(ear[idx]); data[k]['h'].append(h); data[k]['cols'].append(col[idx] - np.median(col[idx], 0))
        if d.name not in held:
            ids.append(d.name)
    fitted, ev = {}, {}
    for k in sides:                          # per-side PCA: the L/R vertex correspondence of the dense topology is not bijective
        shapes = np.array(data[k]['train']); test = np.array(data[k]['test'])
        ref = shapes.mean(0)
        def align(a, ref=ref):
            sc, r, t = umeyama(a, ref); return (sc * (r @ a.T)).T + t
        for _ in range(3):                   # generalised Procrustes
            shapes = np.array([align(a, ref) for a in shapes]); ref = shapes.mean(0)
        mean, basis, sigma, expl = pca_small_n(shapes.reshape(len(shapes), -1), min(k_modes, len(shapes) - 1))
        errs = []
        for a in test:
            a = align(a, ref); dvec = a.reshape(-1) - mean
            errs.append(np.linalg.norm((mean + (dvec @ basis.T) @ basis - a.reshape(-1)).reshape(-1, 3), axis=1))
        errs = np.array(errs)
        cols = np.array(data[k]['cols'])
        cmean, cbasis, csigma, cexpl = pca_small_n(cols.reshape(len(cols), -1), min(10, len(cols) - 1))
        hs = np.array(data[k]['h']); ht = np.array(data[k]['htest'])
        hmean, hbasis, hsigma, hexpl = pca_small_n(hs, min(k_modes, len(hs) - 1))
        herr = [np.abs(hmean + ((h - hmean) @ hbasis.T) @ hbasis - h) for h in ht]
        fitted[k] = dict(mean=mean.reshape(-1, 3).astype(np.float32), basis=basis.astype(np.float32), sigma=sigma, explained=expl,
                         hmean=hmean.astype(np.float32), hbasis=hbasis.astype(np.float32), hsigma=hsigma, hexplained=hexpl, triangles=side_tris[k],
                         colour_mean=cmean.reshape(-1, 3).astype(np.float32), colour_basis=cbasis.astype(np.float16), colour_sigma=csigma)
        ev[k] = dict(held_out=int(len(errs)), recon_mm_mean=float(1e3 * errs.mean()) if len(errs) else None,
                     recon_mm_p95=float(1e3 * np.percentile(errs, 95)) if len(errs) else None,
                     mean_only_mm=float(1e3 * np.mean([np.linalg.norm(align(a, ref) - ref, axis=1).mean() for a in test])) if len(test) else None,
                     explained=float(expl.sum()), colour_explained=float(cexpl.sum()),
                     relief=dict(explained=float(hexpl.sum()), height_mm_abs_mean=float(1e3 * np.abs(hmean).mean()), height_mm_range=[float(1e3 * hmean.min()), float(1e3 * hmean.max())],
                                 held_out_mm_mean=float(1e3 * np.mean(herr)) if herr else None, mean_only_mm=float(1e3 * np.mean([np.abs(h - hmean).mean() for h in ht])) if len(ht) else None))
    out = Path(out or MODELS_ROOT / 'ear_prior_v1.npz')
    arrays = {f'{k}_{n}': v for k, f in fitted.items() for n, v in f.items()}
    np.savez(out, template_points=db['template_points'].astype(np.float64), left=db['left'], right=db['right'], band=db['band'], mh_tri_index=db['mh_tri_index'], mh_bary=db['mh_bary'].astype(np.float32), **arrays)
    rep = prior_report('ear_prior_v1', dataset_manifest_sha256=sha256(V1 / 'manifest.json') if (V1 / 'manifest.json').is_file() else None,
                       sample_ids_used=sorted(set(ids)), held_out_ids=sorted(held), weights_path=out,
                       derived_from='MetaHuman head geometry sampled at the vhuman dense-ear topology (binding_v1); basecolor at bound MH UVs',
                       limitations=['projection of the coarse registered GNM ear onto MH: deep concha/helix folds are under-sampled',
                                    'colour is MetaHuman synthesized basecolor relative to the per-ear median'],
                       extra=dict(k=k_modes, relief_iterations=RELIEF_ITERATIONS, ray_hit_fraction=float(db['ray_hit'].mean()), evaluation=ev,
                                  projection_mm_p95=float(1e3 * np.percentile(db['distance'], 95)), mirror_mm_p95=float(1e3 * np.percentile(db['mirror_err'], 95)),
                                  topology=str(topology)))
    (out.with_name(out.stem + '.report.json')).write_text(json.dumps(rep, indent=1) + '\n')
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--topology', default=str(DEFAULT_TOPOLOGY)); ap.add_argument('--k', type=int, default=20); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rep = fit(a.topology, a.k, a.out)
    print(json.dumps({k: v for k, v in rep.items() if k in ('k', 'evaluation', 'projection_mm_p95', 'mirror_mm_p95')}, indent=1))


if __name__ == '__main__':
    main()
