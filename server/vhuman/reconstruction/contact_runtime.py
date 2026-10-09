"""NumPy-only per-pose lip/arch contact deformer (shared by Python tools and the Blender worker).

See ``oral_contact`` for the model. No SciPy or project imports so Blender's
bundled Python can load it directly.
"""
import numpy as np


def closest_on_triangles(p, A, B, C):
    """Exact closest points of p (V,3) on candidate triangles (V,K,3); returns points and unit normals."""
    p = np.asarray(p, float)[:, None, :]
    n = np.cross(B-A, C-A)
    nn = n/np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-15)
    q = p-((p-A)*nn).sum(-1, keepdims=True)*nn
    ins = np.ones(q.shape[:2], bool)
    for a, b in ((A, B), (B, C), (C, A)):
        ins &= (np.cross(b-a, q-a)*n).sum(-1) >= 0
    best = q.copy()
    dist = np.where(ins, np.linalg.norm(p-q, axis=-1), np.inf)
    for a, b in ((A, B), (B, C), (C, A)):
        e = b-a
        t = np.clip(((p-a)*e).sum(-1)/np.maximum((e*e).sum(-1), 1e-15), 0, 1)
        r = a+t[..., None]*e
        d = np.linalg.norm(p-r, axis=-1)
        take = ~ins & (d < dist)
        best = np.where(take[..., None], r, best)
        dist = np.where(take, d, dist)
    return best, nn, dist


TIE_SCALE = 2e-4


def signed_contact(p, A, B, C):
    """Closest point, smooth pseudo-normal and signed distance of p (V,3) to the candidate set.

    The closest point on a shared edge/vertex ties between triangles; picking one face normal
    flips the push direction under tiny perturbations, so normals are blended with weights
    exp(-(d-d_min)/TIE_SCALE), which is continuous across ties.
    """
    q, nn, dist = closest_on_triangles(p, A, B, C)
    r = np.arange(len(p))
    k = dist.argmin(1)
    w = np.exp(-(dist-dist[r, k][:, None])/TIE_SCALE)
    m = (w[..., None]*nn).sum(1)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-15)
    point = q[r, k]
    return point, m, ((np.asarray(p, float)-point)*m).sum(-1)


def contact_deform(positions, native, spec):
    """Press lining vertices onto the labial arch surface at ``clearance`` (pull gaps, push penetration).

    positions: (V,3) lining part positions for this pose (modified copy returned); native: (N,3) posed
    native vertices (arch). Displacements are weighted, capped and smoothed over ``neighbours``.
    """
    out = np.asarray(positions, float).copy()
    native = np.asarray(native, float)
    act, cand = spec['active'], np.asarray(spec['candidates'])
    A, B, C = native[cand[..., 0]], native[cand[..., 1]], native[cand[..., 2]]
    _, normal, sd = signed_contact(out[act], A, B, C)
    disp = np.zeros_like(out)
    disp[act] = -(np.asarray(spec['weight'])*(sd-spec['clearance']))[:, None]*normal
    length = np.linalg.norm(disp, axis=1, keepdims=True)
    disp *= np.minimum(1, spec['max_move']/np.maximum(length, 1e-12))
    nb = spec.get('neighbours')
    if nb is not None:
        alpha, iterations = spec['smoothing']
        rows, cols = nb
        count = np.bincount(rows, minlength=len(out)).astype(float)
        for _ in range(int(iterations)):
            avg = np.zeros_like(disp)
            np.add.at(avg, rows, disp[cols])
            avg /= np.maximum(count, 1)[:, None]
            disp = np.where((count > 0)[:, None], (1-alpha)*disp+alpha*avg, disp)
    out = out+disp
    # Final one-sided pass: smoothing must not leave visible lining inside the arch.
    _, normal, sd = signed_contact(out[act], A, B, C)
    push = np.clip(spec['clearance']-sd, 0, spec['max_move'])*np.asarray(spec['weight'])
    out[act] += push[:, None]*normal
    return out
