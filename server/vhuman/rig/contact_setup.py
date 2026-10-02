"""Framework-free contact asset construction shared by export and training."""
import numpy as np
from . import rigdef
from . import template as T

EYE_MARGIN_MM = 0.25
TOOTH_MARGIN_MM = 0.6

def _mirror(name: str) -> str:
    for a, b in (("Left", "Right"), ("Right", "Left")):
        if name.endswith(a):
            return name[:-len(a)] + b
    return name


def sample_controls(names: list[str], n: int, seed: int = 0, tongue_share: float = 0.18) -> np.ndarray:
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
        if r > 1.0 - tongue_share and tongue:                       # tongue scenarios: rare in use, but all contact
            X[s, idx["jawOpen"]] = float(rng.uniform(0.15, 1.0))
            for k in rng.choice(tongue, size=min(len(tongue), 1 + rng.poisson(0.8)), replace=False):
                X[s, idx[k]] = float(rng.uniform(0.3, 1.0))
            for k in rng.choice(lips, size=rng.poisson(0.8), replace=False):
                X[s, idx[k]] = float(rng.uniform(0.2, 1.0))
            continue
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


def region_mask(tmpl: T.Template, rest: np.ndarray, reach_m: float = 0.022) -> np.ndarray:
    """Vertices of the mouth region: the mouth group and skin within reach of
    the lip seam (ring 0)."""
    seam = rest[tmpl.ring_ids("mouth", 0)]
    d = np.min(np.linalg.norm(rest[:, None] - seam[None], axis=-1), 1)
    return (tmpl.group == 2) | (d < reach_m)


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



def export_contacts(tmpl, feat, skel, teeth, rest, tongue=None):
    joints = {j['name']: i for i, j in enumerate(skel['joints'])}
    eyes = {e['side']: e for e in feat.eyes}
    result = {'eye': [], 'spheres': {},
              'margins_mm': {'eye': EYE_MARGIN_MM, 'sphere': TOOTH_MARGIN_MM}}
    for group, side, joint in ((0, 'right', 'eye_R'), (1, 'left', 'eye_L')):
        ids = np.flatnonzero((tmpl.group == group) & (tmpl.ring <= 4))
        eye = eyes[side]
        result['eye'].append({'ids': ids.tolist(), 'joint': joints[joint],
                              'center': list(map(float, eye['center'])),
                              'radius_mm': float(eye['radius'] * 1000)})
    result['lip_ids'] = np.flatnonzero((tmpl.group == 2) & (tmpl.ring >= -2) & (tmpl.ring <= 4)).tolist()
    centers, radii, ids = [], [], []
    for part, joint in teeth:
        c, r = tooth_spheres(part, joints[joint], 10 * 18 + 1)
        centers.append(c); radii.append(r); ids.append(np.full(len(c), joints[joint]))
    c = np.concatenate(centers)
    sets = [('teeth', c, np.concatenate(radii),
             np.stack([np.concatenate(ids), np.zeros(len(c), int)], 1),
             np.stack([np.ones(len(c)), np.zeros(len(c))], 1))]
    if tongue is not None:
        sets.append(('tongue', *tongue_spheres(tongue)))
    for name, c, r, j, w in sets:
        result['spheres'][name] = {'centers': c.tolist(), 'radius_mm': (r * 1000).tolist(),
                                    'joints': j.tolist(), 'weights': w.tolist()}
    up, lo = [], []
    for ring in (1, 0, -1, -2):
        ids = tmpl.ring_ids('mouth', ring)
        for j in range(1, T.MOUTH_HALF):
            up.append(ids[j]); lo.append(ids[T.MOUTH_N - j])
    result.update(pairs_upper=up, pairs_lower=lo, head=joints['head'], jaw=joints['jaw'])
    return result
