"""Cut the eye fissures out of Pixal3D's head and drape the lids onto the eyeballs.

Pixal3D paints the eye (sclera, iris) onto the face surface, and after
carving (fit.carve) whole triangles remain around the opening: a jagged,
triangle-level lid margin with painted sclera on it, floating in front of
the eyeball. Here, per eye:
1. the palpebral fissure's contour (landmarks.fissure) is smoothed: its
   polar radius around the iris centre, r(theta), low-passed to a few
   Fourier terms (an almond);
2. the local surface is clipped along that contour in the portrait's
   projection (marching triangles on f = r/r(theta) - 1; the cut points are
   interpolated along the edges, with their UVs, so the new lid margin is a
   smooth polyline on the contour), and everything inside is dropped.
   Where the camera ray misses the eyeball (the tips of the canthi) the
   surface is kept, and reported as `lining` for recolouring (it is painted
   as sclera);
3. the cut vertices of the front surface move onto the eyeball (plus a small
   margin) along their camera ray: the lid margin rests on the eye exactly
   where the portrait shows it, so the front view is unchanged;
4. the lid band behind the cut follows with a falloff over a few
   millimetres of surface (a shortest-path walk across welded UV seams), and any
   band vertex left inside the eyeball is pushed out onto it;
5. where the margin still stands off the eyeball (a move over MAX_MOVE_M
   drags the band too far and uncovers the socket's walls), a lid lining
   closes the gap: a ribbon around the fissure from the front-most cut
   point in each direction (with median-filtered depth) back along the
   camera ray onto the eyeball;
6. `lining` also gets the lid surface near the fissure that the portrait
   saw edge-on (its texture is stretched, mostly sclera), to recolour. It is invisible in the portrait's view and
   is the lid's thickness from any other (exported as its own mesh with a
   flat, shadowed skin material).
"""
from __future__ import annotations

import math
import heapq

import numpy as np

from ..eye import optics
from .camera import PixalCamera
from . import eyeedge

LID_WRAP_MARGIN_M = 0.00025     # the lid margin sits this far off the eyeball surface
BAND_M = 0.003                  # the margin's move fades out over this much surface
FOURIER_TERMS = 6
LOCAL_RADIUS_M = 0.022          # the region around each eye that is edited
MAX_MOVE_M = 0.004              # a cut vertex needing a bigger move is not on the lid margin
GRAZING_COS = 0.4               # |cos| between a lid triangle and the portrait's view ray...
GRAZING_REACH = 1.3             # ... within this multiple of the fissure contour: recoloured
SKIRT_MAX_M = 0.012             # longest lid-lining strip (margin to eyeball)


def smooth_contour(opening: np.ndarray, cx: float, cy: float, bins: int = 180, terms: int = FOURIER_TERMS):
    """r(theta) of the opening around (cx, cy), low-passed (periodic)."""
    ys, xs = np.nonzero(opening)
    if len(xs) == 0:
        return None
    ang = np.arctan2(ys - cy, xs - cx)
    rad = np.hypot(xs - cx, ys - cy)
    k = ((ang + math.pi) / (2 * math.pi) * bins).astype(int) % bins
    r = np.zeros(bins)
    np.maximum.at(r, k, rad)
    empty = r == 0
    if empty.all():
        return None
    if empty.any():                       # fill gaps periodically
        idx = np.arange(bins)
        r[empty] = np.interp(idx[empty], idx[~empty], r[~empty], period=bins)
    spec = np.fft.rfft(r)
    spec[terms + 1:] = 0
    smooth = np.fft.irfft(spec, n=bins)
    return lambda theta: np.interp((np.asarray(theta) + math.pi) / (2 * math.pi) * bins, np.arange(bins + 1),
                                   np.append(smooth, smooth[0]))


def eye_contour(eye):
    """The smoothed fissure contour of a landmarks.Eye (its opening if no fissure)."""
    return smooth_contour(eye.fissure if eye.fissure is not None else eye.opening, eye.cx, eye.cy)


def _ray_sphere_t(o: np.ndarray, d: np.ndarray, c: np.ndarray, r: float):
    """Near hit distance per ray (unit directions d, (N, 3)); nan on a miss."""
    oc = o - c
    b = d @ oc
    disc = b * b - (oc @ oc - r * r)
    return np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), np.nan)


def wrap(mesh: dict, keep: np.ndarray, cam: PixalCamera, eyes, poses) -> tuple:
    """Cut the fissures and drape the lids. Returns (mesh, keep, info,
    lining): `mesh` is a copy with vertices and triangles added (positions,
    normals, uvs, triangles), `keep` covers the new triangle list, and
    `lining` is a vertex mask of the surface kept inside a fissure (the
    canthi's tips, painted as sclera)."""
    pos = [mesh["positions"]]
    nrm = [mesh["normals"]]
    uvs = [mesh["uvs"]]
    tris = mesh["triangles"]
    keep = keep.copy()
    n_vert = len(mesh["positions"])
    info = {}
    lining_v: list[np.ndarray] = []
    skirt_p: list[np.ndarray] = []
    skirt_t: list[np.ndarray] = []
    skirt_uv: list[np.ndarray] = []
    touched_v: list[np.ndarray] = []
    wet_meshes = []
    prof = optics.ANATOMICAL
    o = np.asarray(cam.origin, np.float64)
    for eye, pose in zip(eyes, poses):
        P_ = np.concatenate(pos)
        k = pose.units_per_m
        centre = pose.center
        contour = eye_contour(eye)
        if contour is None:
            info[eye.side] = {"cut_triangles": 0}
            continue
        local_v = np.linalg.norm(P_ - centre, axis=1) < LOCAL_RADIUS_M * k
        local_t = np.flatnonzero(keep & local_v[tris].all(1))
        # f < 0 inside the fissure (and in front of or behind the eyeball, where
        # the ray meets it); the rest, the canthi's tips included, is kept
        vid = np.unique(tris[local_t])
        pix = cam.project(P_[vid])
        theta = np.arctan2(pix[:, 1] - eye.cy, pix[:, 0] - eye.cx)
        q = np.hypot(pix[:, 0] - eye.cx, pix[:, 1] - eye.cy) / np.maximum(contour(theta), 1e-6)
        d = P_[vid] - o
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        misses = np.isnan(_ray_sphere_t(o, d, centre, prof.sclera_radius * k))
        f = np.zeros(len(P_))
        f[vid] = np.where(misses, np.maximum(q - 1.0, 1e-3), q - 1.0)
        lining_v.append(vid[misses & (q < 1.0)])
        new_p, new_n, new_uv, new_t, cut_ids, dropped = _clip(tris, local_t, f, P_, np.concatenate(nrm),
                                                              np.concatenate(uvs), n_vert)
        keep[local_t[dropped]] = False
        n_vert += len(new_p)
        pos.append(new_p)
        nrm.append(new_n)
        uvs.append(new_uv)
        tris = np.concatenate([tris, new_t]) if len(new_t) else tris
        keep = np.concatenate([keep, np.ones(len(new_t), bool)])
        P_ = np.concatenate(pos)
        # the cut on the front surface drapes onto the eyeball, along its ray
        fn = np.cross(P_[tris[:, 1]] - P_[tris[:, 0]], P_[tris[:, 2]] - P_[tris[:, 0]])
        facing_t = np.einsum("ij,ij->i", fn, o - P_[tris].mean(1)) > 0
        local_v = np.linalg.norm(P_ - centre, axis=1) < LOCAL_RADIUS_M * k
        front_t = tris[keep & facing_t & local_v[tris].all(1)]
        front_v = np.zeros(len(P_), bool)
        front_v[front_t.ravel()] = True
        cut = cut_ids[front_v[cut_ids]]
        radius = (prof.sclera_radius + LID_WRAP_MARGIN_M) * k
        d = P_[cut] - o
        t_v = np.linalg.norm(d, axis=1)
        d /= t_v[:, None]
        t_hit = _ray_sphere_t(o, d, centre, radius)
        ok = ~np.isnan(t_hit) & (np.abs(t_hit - t_v) < MAX_MOVE_M * k)
        moved = {int(v): d[i] * (t_hit[i] - t_v[i]) for i, v in enumerate(cut) if ok[i]}
        # UV charts duplicate positions. Propagate through geometric seams
        # and both faces of a fold so draping cannot pull the copies apart.
        band_t = tris[keep & local_v[tris].all(1)]
        band = _band(band_t, P_, moved, BAND_M * k)
        pushed = _drape(P_, band, o, centre, radius)
        # the lid lining: where the margin still stands off the eyeball, a strip
        # from each front cut segment back along the camera rays onto it
        # (from every cut point, not just the front surface's: where the lid
        # curls under, its visible edge is a fold and the cut lies beneath)
        sk_p, sk_t, sk_uv = _ribbon(cut_ids, P_, o, cam, eye, centre, radius, SKIRT_MAX_M * k)
        if len(sk_t):
            skirt_t.append(sk_t + sum(len(x) for x in skirt_p))
            skirt_p.append(sk_p)
            skirt_uv.append(sk_uv)
        edge = eyeedge.margin(P_[cut_ids], cam, eye, contour)
        tear = eyeedge.tearline(edge, pose, cam)
        # A support triangle may cross the local patch boundary. The ray
        # hit itself is distance-bounded by caruncle(), so retain whole faces.
        local_tri = tris[keep]
        tissue = eyeedge.caruncle(eye, pose, cam, contour, P_, local_tri)
        wet_meshes.extend(m for m in (tear, tissue) if m is not None)
        # Lid surface near the fissure that the portrait saw edge-on (the lid
        # margins' tops and undersides, the walls) has a stretched, unreliable
        # texture, mostly sclera: add it to the lining to recolour.
        t_near = np.flatnonzero(keep & local_v[tris].all(1))
        cen = P_[tris[t_near]].mean(1)
        fn = np.cross(P_[tris[t_near, 1]] - P_[tris[t_near, 0]], P_[tris[t_near, 2]] - P_[tris[t_near, 0]])
        view = o - cen
        cos = np.abs(np.einsum("ij,ij->i", fn, view)) / np.maximum(
            np.linalg.norm(fn, axis=1) * np.linalg.norm(view, axis=1), 1e-20)
        pc = cam.project(cen)
        th = np.arctan2(pc[:, 1] - eye.cy, pc[:, 0] - eye.cx)
        qc = np.hypot(pc[:, 0] - eye.cx, pc[:, 1] - eye.cy) / np.maximum(contour(th), 1e-6)
        grazing = t_near[(cos < GRAZING_COS) & (qc < GRAZING_REACH)]
        lining_v.append(np.unique(tris[grazing]))
        pos = [P_]
        nrm = [np.concatenate(nrm)]
        uvs = [np.concatenate(uvs)]
        touched_v.append(np.array(list(band) + [int(v) for v in cut_ids], np.int64))
        info[eye.side] = {"cut_triangles": int(dropped.size and dropped.sum()), "new_triangles": int(len(new_t)),
                          "cut_vertices": int(len(cut_ids)), "draped": len(moved), "band_vertices": len(band),
                          "pushed_out": pushed, "lining_vertices": int(len(lining_v[-2]) + len(lining_v[-1])),
                          "grazing_triangles": int(len(grazing)),
                          "skirt_triangles": int(len(sk_t)),
                          "mean_drape_mm": round(float(np.mean([np.linalg.norm(m) for m in moved.values()]) / k
                                                       * 1000), 3) if moved else 0.0}
    P_, N_, UV = np.concatenate(pos), np.concatenate(nrm), np.concatenate(uvs)
    touched = np.concatenate(touched_v) if touched_v else np.zeros(0, np.int64)
    if len(touched):
        fresh = vertex_normals(P_, tris[keep])
        good = np.linalg.norm(fresh[touched], axis=1) > 0
        N_[touched[good]] = fresh[touched[good]]
    lining = np.zeros(len(P_), bool)
    for v in lining_v:
        lining[v] = True
    skirt = None
    if skirt_t:
        sp, st = np.concatenate(skirt_p), np.concatenate(skirt_t)
        # normals towards the portrait camera: the strips lie along its rays
        # (their geometric normals are edge-on and flip between neighbours)
        toward = o - sp
        skirt = {"positions": sp, "triangles": st, "uvs": np.concatenate(skirt_uv),
                 "normals": toward / np.linalg.norm(toward, axis=1, keepdims=True)}
    return dict(mesh, positions=P_, normals=N_, uvs=UV, triangles=tris, lid_lining=skirt, eye_edges=wet_meshes), keep, info, lining


def _drape(pos, band, origin, centre, radius):
    """Move each band vertex along its own portrait ray, preserving coverage.

    Transferring a seed's XYZ vector to a neighbouring ray shifts triangles
    sideways. Radial sphere push-out does too. Depth-only motion preserves
    projected winding and the portrait's texture alignment even on thin
    folded triangles.
    """
    if not band:
        return 0
    ids = np.array(list(band), np.int64)
    ray = pos[ids] - origin
    depth = np.linalg.norm(ray, axis=1)
    ray /= depth[:, None]
    disp = np.array([band[int(v)][0] * band[int(v)][1] for v in ids])
    depth += np.einsum("ij,ij->i", disp, ray)
    result = origin + ray * depth[:, None]
    inside = np.linalg.norm(result - centre, axis=1) < radius
    hit = _ray_sphere_t(origin, ray, centre, radius)
    push = inside & np.isfinite(hit)
    result[push] = origin + ray[push] * hit[push, None]
    pos[ids] = result
    return int(push.sum())


def _ribbon(cut_ids, P_, o, cam, eye, centre, radius: float, max_len: float, bins: int = 180):
    """Ribbon from the smoothed front margin to the eyeball, along camera rays.

    Return positions, triangles and UVs (v=0 at the margin, v=1 deep in
    the lining). Sphere misses and excessive stand-off remain open.
    """
    empty = np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 2))
    top = eyeedge.margin(P_[cut_ids], cam, eye, eye_contour(eye), bins)
    if top is None:
        return empty
    dist = np.linalg.norm(top - o, axis=1)
    d = (top - o) / dist[:, None]
    hit = _ray_sphere_t(o, d, centre, radius)
    valid = np.isfinite(hit) & (hit > dist) & (hit - dist < max_len)
    bottom = o + d * hit[:, None]
    top[~valid] = np.nan
    ok = ~np.isnan(top[:, 0])
    out_p = np.concatenate([top, bottom])          # vertex k: top, bins + k: bottom
    # The median-filtered margin may still switch between source layers.
    # Match the tear line's discontinuity rule: a lining quad spanning
    # that jump fabricates a wall where no continuous tissue was found.
    units_per_m = radius / (optics.ANATOMICAL.sclera_radius + LID_WRAP_MARGIN_M)
    continuous = np.linalg.norm(np.roll(top, -1, axis=0) - top, axis=1) <= eyeedge.MAX_MARGIN_STEP_M * units_per_m
    tri = []
    for k_ in range(bins):
        n = (k_ + 1) % bins
        if ok[k_] and ok[n] and continuous[k_]:
            tri += [(k_, n, bins + n), (k_, bins + n, bins + k_)]
    if not tri:
        return empty
    tri = np.array(tri, np.int64)
    used = np.unique(tri)
    remap = np.full(2 * bins, -1)
    remap[used] = np.arange(len(used))
    uv = np.zeros((2 * bins, 2))
    uv[bins:, 1] = 1.0
    return out_p[used], remap[tri], uv[used]


def _clip(tris, idx, f, P_, N_, UV, base: int):
    """Marching triangles over `tris[idx]` on the vertex field f (< 0 is cut
    away). Returns the new vertices (positions, normals, uvs), the new
    triangles (the kept parts of the crossing triangles, indices from
    `base`), the new vertex ids on the cut, and a mask over `idx` of the
    triangles replaced or dropped."""
    t = tris[idx]
    ft = f[t]
    inside = ft < 0
    n_in = inside.sum(1)
    dropped = n_in > 0
    cross = np.flatnonzero((n_in > 0) & (n_in < 3))
    edge_id: dict = {}
    new_p, new_n, new_uv = [], [], []

    def cut(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in edge_id:
            s = f[a] / (f[a] - f[b])
            new_p.append(P_[a] + s * (P_[b] - P_[a]))
            nn = N_[a] + s * (N_[b] - N_[a])
            new_n.append(nn / max(float(np.linalg.norm(nn)), 1e-12))
            new_uv.append(UV[a] + s * (UV[b] - UV[a]))
            edge_id[key] = base + len(new_p) - 1
        return edge_id[key]

    out = []
    for i in cross:
        a, b, c = (int(v) for v in t[i])
        ins = inside[i]
        # rotate so the odd one out is first, keeping the winding
        while not (ins[0] != ins[1] and ins[0] != ins[2]):
            a, b, c = b, c, a
            ins = np.roll(ins, -1)
        ab, ac = cut(a, b), cut(a, c)
        if ins[0]:          # a inside: keep the quad b, c, ac, ab
            out += [(ab, b, c), (ab, c, ac)]
        else:               # a outside: keep the triangle a, ab, ac
            out.append((a, ab, ac))
    z3 = np.zeros((0, 3))
    return (np.array(new_p) if new_p else z3, np.array(new_n) if new_n else z3,
            np.array(new_uv) if new_uv else np.zeros((0, 2)),
            np.array(out, np.int64).reshape(-1, 3), np.array(sorted(edge_id.values()), np.int64), dropped)


def _neighbours(tris: np.ndarray, verts: np.ndarray) -> dict:
    """Vertex adjacency restricted to `verts` (a set of indices)."""
    nb: dict[int, set] = {int(v): set() for v in verts}
    for a, b, c in tris:
        for x, y in ((a, b), (b, c), (c, a)):
            if x in nb and y in nb:
                nb[x].add(int(y))
                nb[y].add(int(x))
    return nb


def _band(tris: np.ndarray, pos: np.ndarray, moved: dict, reach: float) -> dict:
    """Geodesic falloff with coincident UV-seam vertices moving together.

    Welding is only for deformation adjacency: original UVs and triangle
    indices remain separate. The tiny tolerance also joins cut points
    independently interpolated on the two sides of an atlas seam.
    """
    if not moved or len(tris) == 0:
        return {}
    verts = np.unique(tris)
    tolerance = reach * 1e-5
    _, first, group = np.unique(np.rint(pos[verts] / tolerance).astype(np.int64),
                                axis=0, return_index=True, return_inverse=True)
    points = pos[verts[first]]
    local_tri = group[np.searchsorted(verts, tris)]
    nb = _neighbours(local_tri, np.arange(len(points)))
    seeds = {}
    for v, disp in moved.items():
        i = np.searchsorted(verts, v)
        if i < len(verts) and verts[i] == v:
            seeds.setdefault(int(group[i]), []).append(disp)
    best = {v: (0.0, np.mean(ds, axis=0)) for v, ds in seeds.items()}
    queue = [(0.0, v) for v in best]
    heapq.heapify(queue)
    while queue:
        dist, v = heapq.heappop(queue)
        if dist != best[v][0]:
            continue
        disp = best[v][1]
        for u in sorted(nb[v]):
            nd = dist + float(np.linalg.norm(points[u] - points[v]))
            if nd < reach and (u not in best or nd < best[u][0]):
                best[u] = (nd, disp)
                heapq.heappush(queue, (nd, u))
    out = {}
    for v, g in zip(verts, group):
        if g in best:
            dist, disp = best[g]
            x = 1.0 - dist / reach
            out[int(v)] = (disp, x * x * (3 - 2 * x))
    return out


def vertex_normals(pos: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted normals, continuous across coincident atlas vertices."""
    fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    n = np.zeros_like(pos)
    for c in range(3):
        np.add.at(n, tris[:, c], fn)
    if len(pos):
        # Cut edges on opposite UV charts can differ by interpolation roundoff.
        tolerance = max(float(np.ptp(pos, axis=0).max()) * 1e-9, 1e-12)
        _, group = np.unique(np.rint(pos / tolerance).astype(np.int64), axis=0, return_inverse=True)
        shared = np.zeros((int(group.max()) + 1, 3))
        np.add.at(shared, group, n)
        n = shared[group]
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(norm > 1e-20, n / np.maximum(norm, 1e-20), 0.0)
