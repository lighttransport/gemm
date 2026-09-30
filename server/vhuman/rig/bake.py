"""Transfer the subject's skin maps onto the template's UV atlas.

Per atlas texel: the template triangle and barycentrics (UV rasterization),
its 3D point on the fitted template, the nearest subject surface point with
a compatible normal (k nearest subject vertices, then the closest point on
their triangles), and the subject's UV there. Base colour and ORM are
resampled; the normal map is re-expressed from the subject's tangent frame
(with its surface normal) into the template's, so the subject's geometric
detail the coarser template lacks is carried in the normal map. The lid
margins and linings get the fit's lining colour.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import template as T
from .common import normalize
from .meshes import Part


def rasterize_uv(uv: np.ndarray, tris: np.ndarray, res: int):
    """Triangle id (-1 empty) and barycentrics per texel centre; rows = v * res."""
    tid = np.full((res, res), -1, np.int64)
    bary = np.zeros((res, res, 3), np.float32)
    px = uv * res - 0.5
    for i, (a, b, c) in enumerate(tris):
        pa, pb, pc = px[a], px[b], px[c]
        lo = np.maximum(np.floor(np.minimum(np.minimum(pa, pb), pc)).astype(int), 0)
        hi = np.minimum(np.ceil(np.maximum(np.maximum(pa, pb), pc)).astype(int), res - 1)
        if (hi < lo).any():
            continue
        yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        den = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
        if abs(den) < 1e-14:
            continue
        w0 = ((pb[1] - pc[1]) * (xx - pc[0]) + (pc[0] - pb[0]) * (yy - pc[1])) / den
        w1 = ((pc[1] - pa[1]) * (xx - pc[0]) + (pa[0] - pc[0]) * (yy - pc[1])) / den
        w2 = 1 - w0 - w1
        m = (w0 >= -1e-3) & (w1 >= -1e-3) & (w2 >= -1e-3)
        if not m.any():
            continue
        sub = tid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        sub[m] = i
        bsub = bary[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        bsub[m] = np.stack([w0[m], w1[m], w2[m]], 1)
    return tid, bary


def dilate(img: np.ndarray, covered: np.ndarray, steps: int) -> np.ndarray:
    img = img.astype(np.float32).copy()
    cov = covered.copy()
    for _ in range(steps):
        acc = np.zeros_like(img)
        cnt = np.zeros(cov.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            m = np.roll(np.roll(cov, dy, 0), dx, 1)
            if dy > 0: m[:dy] = False
            if dy < 0: m[dy:] = False
            if dx > 0: m[:, :dx] = False
            if dx < 0: m[:, dx:] = False
            acc += np.roll(np.roll(img, dy, 0), dx, 1) * m[..., None]
            cnt += m
        new = (~cov) & (cnt > 0)
        img[new] = acc[new] / cnt[new][:, None]
        cov = cov | new
    return img


def _sample(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear, glTF UVs (rows = v * H), clamped."""
    h, w = img.shape[:2]
    x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1.001)
    y = np.clip(uv[:, 1] * h - 0.5, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    tx, ty = (x - x0)[:, None], (y - y0)[:, None]
    f = img.astype(np.float32)
    return ((f[y0, x0] * (1 - tx) + f[y0, x0 + 1] * tx) * (1 - ty)
            + (f[y0 + 1, x0] * (1 - tx) + f[y0 + 1, x0 + 1] * tx) * ty)


def _closest_on_triangles(p, a, b, c):
    """Closest points of p (N,3) on triangles (N,3)x3 and barycentrics (Ericson)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-30, 1e-30, va + vb + vc)
    v = vb / den
    w = vc / den
    # clamp into the triangle (approximate: project barycentrics, renormalize)
    u = 1 - v - w
    bc = np.stack([u, v, w], 1)
    bc = np.clip(bc, 0, None)
    bc /= np.maximum(bc.sum(1, keepdims=True), 1e-12)
    q = bc[:, :1] * a + bc[:, 1:2] * b + bc[:, 2:3] * c
    return q, bc


def bake(tmpl: T.Template, skin: Part, pos: np.ndarray, subj, out_dir: Path, res: int = 2048,
         lining_srgb=(150, 100, 85), log=None) -> dict:
    from scipy.spatial import cKDTree
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    tid, bary = rasterize_uv(skin.uv, skin.tris, res)
    cov = tid >= 0
    ys, xs = np.nonzero(cov)
    ti = tid[ys, xs]
    bc = bary[ys, xs].astype(np.float64)
    corners = skin.tris[ti]                                   # unwelded ids
    P = pos[skin.vmap]
    p = (bc[:, :, None] * P[corners]).sum(1)
    n = normalize((bc[:, :, None] * skin.normals[corners]).sum(1))
    tan = (bc[:, :, None] * skin.tangents[corners, :3]).sum(1)
    tan = normalize(tan - (tan * n).sum(1, keepdims=True) * n)
    w = np.sign(skin.tangents[corners[:, 0], 3])
    bit = np.cross(n, tan) * w[:, None]
    # lid margins/linings: template-only surfaces
    wv = skin.vmap[corners]
    lining = (tmpl.kind[wv] == T.KIND["eye_inner"]).all(1)
    # nearest compatible subject vertex, then its triangles
    tree = cKDTree(subj.positions)
    k = 8
    d, j = tree.query(p, k=k)
    agree = (subj.normals[j] * n[:, None]).sum(-1) > 0.2
    # Prefer a similarly oriented sample near the texel, but do not let the
    # normal test select a distant point across an open neck or scan boundary.
    # Imported full-head atlases expose more of those boundaries than the
    # procedural face atlas does.
    score = np.where(agree, d, d + 0.004)
    best = j[np.arange(len(p)), np.argmin(score, 1)]
    # vertex -> incident triangles (CSR, capped)
    ST = subj.triangles
    order = np.argsort(ST.reshape(-1), kind="stable")
    vt = (order // 3)
    counts = np.bincount(ST.reshape(-1), minlength=len(subj.positions))
    start = np.concatenate([[0], np.cumsum(counts)])
    cap = 10
    cand = np.full((len(p), cap), -1, np.int64)
    for c in range(cap):
        has = counts[best] > c
        cand[has, c] = vt[start[best[has]] + c]
    bestq = np.zeros_like(p)
    bestb = np.zeros_like(p)
    bestt = np.zeros(len(p), np.int64)
    bestd = np.full(len(p), np.inf)
    for c in range(cap):
        ok = cand[:, c] >= 0
        if not ok.any():
            continue
        tt = ST[cand[ok, c]]
        q, b = _closest_on_triangles(p[ok], subj.positions[tt[:, 0]], subj.positions[tt[:, 1]],
                                     subj.positions[tt[:, 2]])
        dd = np.linalg.norm(q - p[ok], axis=1)
        better = dd < bestd[ok]
        idx = np.flatnonzero(ok)[better]
        bestd[idx], bestq[idx], bestb[idx], bestt[idx] = dd[better], q[better], b[better], cand[ok, c][better]
    # a vertex without triangles (unreferenced): fall back to the vertex itself
    lone = ~np.isfinite(bestd)
    bestt[lone] = 0
    bestb[lone] = [1.0, 0.0, 0.0]
    bestd[lone] = np.linalg.norm(subj.positions[best[lone]] - p[lone], axis=1)
    stri = ST[bestt]
    stri[lone] = best[lone, None]
    suv = (bestb[:, :, None] * subj.uvs[stri]).sum(1)
    sn = normalize((bestb[:, :, None] * subj.normals[stri]).sum(1))
    st = (bestb[:, :, None] * subj.tangents[stri, :3]).sum(1)
    st = normalize(st - (st * sn).sum(1, keepdims=True) * sn)
    sw = np.sign(subj.tangents[stri[:, 0], 3])
    sw[sw == 0] = 1
    sb = np.cross(sn, st) * sw[:, None]
    folder = subj.folder
    maps = {}
    for key, name in (("basecolor", "skin_basecolor.png"), ("orm", "skin_orm.png"), ("normal", "skin_normal.png")):
        f = folder / name
        maps[key] = np.asarray(Image.open(f).convert("RGB")) if f.exists() else None
    if maps["basecolor"] is None:                   # fall back to the GLB's base colour
        mat = subj.glb.doc["materials"][0]
        tex = mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        maps["basecolor"] = subj.glb.image(subj.glb.doc["textures"][tex]["source"])[..., :3]
    base = _sample(maps["basecolor"], suv)
    orm = _sample(maps["orm"], suv) if maps["orm"] is not None else np.tile([255, 115, 0], (len(p), 1))
    if maps["normal"] is not None:
        nm = _sample(maps["normal"], suv) / 127.5 - 1.0
    else:
        nm = np.tile([0.0, 0.0, 1.0], (len(p), 1))
    obj = normalize(nm[:, :1] * st + nm[:, 1:2] * sb + nm[:, 2:3] * sn)
    tn = np.stack([(obj * tan).sum(1), (obj * bit).sum(1), (obj * n).sum(1)], 1)
    tn[:, 2] = np.maximum(tn[:, 2], 0.05)
    tn = normalize(tn)
    lrgb = np.asarray(lining_srgb, np.float64)
    base[lining] = lrgb
    orm[lining] = [255, 90, 0]
    tn[lining] = [0, 0, 1]
    far = bestd > 0.004                          # nothing of the subject nearby (cap/neck ends)
    img_b = np.zeros((res, res, 3), np.float32)
    img_o = np.zeros((res, res, 3), np.float32)
    img_n = np.zeros((res, res, 3), np.float32)
    img_b[ys, xs] = base
    img_o[ys, xs] = orm
    img_n[ys, xs] = (tn * 0.5 + 0.5) * 255
    # A full-head source topology can cover the back of a scan whose subject
    # mesh contains only a small face atlas. Extend measured colours over
    # those texels in UV space, and report their count for quality control.
    measured = cov.copy()
    measured[ys[far], xs[far]] = False
    if far.any() and measured.any():
        from scipy.ndimage import distance_transform_edt
        _, nearest = distance_transform_edt(~measured, return_indices=True)
        for img in (img_b, img_o):
            img[ys[far], xs[far]] = img[nearest[0, ys[far], xs[far]],
                                        nearest[1, ys[far], xs[far]]]
        img_n[ys[far], xs[far]] = [127.5, 127.5, 255]
    img_b = dilate(img_b, cov, 12)
    img_o = dilate(img_o, cov, 12)
    img_n = dilate(img_n, cov, 12)
    uncovered = ~dilate(cov[..., None].astype(np.float32), cov, 12)[..., 0].astype(bool)
    img_n[uncovered] = [127.5, 127.5, 255]
    out = {}
    for name, img in (("rig_basecolor.png", img_b), ("rig_orm.png", img_o), ("rig_normal.png", img_n)):
        Image.fromarray(np.clip(np.round(img), 0, 255).astype(np.uint8)).save(out_dir / name)
        out[name] = str(out_dir / name)
    for channel in ("coverage", "confidence", "specular"):
        path = subj.folder / f"skin_{channel}.png"
        if not path.is_file():
            continue
        source = np.asarray(Image.open(path).convert("RGB"))
        values = _sample(source, suv)
        if channel in ("coverage", "confidence"):
            values[far | lining] = 0
        image = np.zeros((res, res, 3), np.float32)
        image[ys, xs] = values
        if channel == "specular":
            image = dilate(image, cov, 12)
        name = f"rig_{channel}.png"
        Image.fromarray(np.uint8(np.clip(image, 0, 255))).save(out_dir / name)
        out[name] = str(out_dir / name)
    stats = {"res": res, "texels": int(cov.sum()), "far_texels": int(far.sum()),
             "inferred_texels": int(far.sum()),
             "mean_transfer_mm": round(float(bestd[~lining].mean() * 1000), 3),
             "p95_transfer_mm": round(float(np.percentile(bestd[~lining], 95) * 1000), 3),
             "seconds": round(time.perf_counter() - t0, 2)}
    if log:
        log(f"bake: {stats}")
    return {"files": out, "stats": stats}


def mouth_texture(res: int = 256) -> np.ndarray:
    """The mouth interior: lip-inner pink at the lips (v small) to a dark,
    wet red deep in the cavity."""
    v = np.linspace(0, 1, res)[:, None]
    lip = np.array([176, 92, 90], np.float64)
    deep = np.array([58, 16, 18], np.float64)
    t = np.clip((v - 0.12) / 0.6, 0, 1) ** 0.8
    rgb = lip * (1 - t) + deep * t
    rgb = np.repeat(rgb[:, None, :], res, 1)
    return np.clip(rgb, 0, 255).astype(np.uint8)
