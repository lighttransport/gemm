"""Region-template skin maps on the GNM UV atlas for the Cycles quality candidate (stage 2c).

Fixed topology lets one authored template serve every subject: per-region roughness and specular level,
a geometric cavity map (concavity of the fitted neutral mesh), a lip mask for the wet band and a micro-normal
strength mask, plus a tileable pore micro-normal. Per-subject variation is limited to scalars (oiliness, age)
so the maps are priors, not measurements, and are labelled as such in the manifest.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image

# roughness, specular level, micro-normal strength per GNM vertex group (first match wins, listed by priority)
REGION_TEMPLATE = (
    ('upper_lip', .40, .60, .25), ('lower_lip', .40, .60, .25),
    ('eye_sockets', .48, .45, .20),
    ('nose_region', .36, .55, 1.0),
    ('forehead_region', .42, .55, .80), ('middle_brow_region', .42, .55, .80),
    ('left_brow_region', .46, .50, .60), ('right_brow_region', .46, .50, .60),
    ('left_orbital_region', .48, .45, .40), ('right_orbital_region', .48, .45, .40),
    ('left_infraorbital_region', .47, .47, .50), ('right_infraorbital_region', .47, .47, .50),
    ('left_zygomatic_region', .50, .45, .90), ('right_zygomatic_region', .50, .45, .90),
    ('left_cheek_region', .50, .45, 1.0), ('right_cheek_region', .50, .45, 1.0),
    ('chin_region', .45, .50, .90),
    ('left_parotid_region', .52, .42, .70), ('right_parotid_region', .52, .42, .70),
    ('left_temple_region', .50, .45, .60), ('right_temple_region', .50, .45, .60),
    ('ears', .52, .42, .50),
)
DEFAULT = (.55, .42, .50)   # scalp, neck and any unlisted exterior skin


def micro_normal(res=512, seed=7, pore_density=28., strength=.35):
    """Tileable pore micro-normal (tangent space, RGB8) from periodic cellular + fractal noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:res, 0:res] / res
    height = np.zeros((res, res))
    # periodic Gabor-like pore bumps: sum of wrapped Gaussians at random cells
    n = int(pore_density ** 2)
    cx, cy = rng.random(n), rng.random(n); radius = rng.uniform(.004, .011, n); depth = rng.uniform(.4, 1., n)
    for px, py, r, d in zip(cx, cy, radius, depth):
        dx = (x - px + .5) % 1 - .5; dy = (y - py + .5) % 1 - .5
        height -= d * np.exp(-(dx * dx + dy * dy) / (2 * r * r))
    # fine periodic grain
    for k, amp in ((48, .08), (96, .05), (192, .03)):
        height += amp * (np.sin(2 * np.pi * (k * x + rng.random())) * np.sin(2 * np.pi * (k * y + rng.random())))
    height = (height - height.mean()) / (height.std() + 1e-9) * strength
    gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * res / 2
    gy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * res / 2
    normal = np.stack((-gx, -gy, np.full_like(gx, res / 8.)), -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return ((normal * .5 + .5) * 255).round().astype(np.uint8)


def concavity(vertices, triangles):
    """Per-vertex concavity in [0,1]: how far the neighbour mean lies inside the surface along the normal."""
    v = np.asarray(vertices, np.float64); t = np.asarray(triangles)
    n = np.zeros_like(v); fn = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    for k in range(3):
        np.add.at(n, t[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    acc = np.zeros_like(v); count = np.zeros(len(v))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        np.add.at(acc, t[:, a], v[t[:, b]]); np.add.at(count, t[:, a], 1)
        np.add.at(acc, t[:, b], v[t[:, a]]); np.add.at(count, t[:, b], 1)
    mean = acc / np.maximum(count[:, None], 1)
    edge = np.linalg.norm(v[t[:, 1]] - v[t[:, 0]], axis=1).mean()
    signed = ((mean - v) * n).sum(1) / max(edge, 1e-9)     # positive = concave
    return np.clip(signed / .25, 0, 1)


def build(candidate, out, *, res=2048, oiliness=1., age=1., seed=7):
    """Write out/skin_templates.npz + skin_templates.json. oiliness scales specular level (T-zone most),
    age scales micro-normal strength and roughness slightly."""
    from scipy.ndimage import gaussian_filter
    from ..rig.bake import rasterize_uv
    from ..rig.gnm_model import GNMModel
    candidate, out = Path(candidate), Path(out)
    with np.load(candidate / 'geometry.npz', allow_pickle=False) as z:
        vertices = z['neutral']; tri = z['triangles']; uv = z['triangle_uvs']
    model = GNMModel(); exterior = model.group('skin_exterior')
    tid, bary = rasterize_uv(uv.reshape(-1, 2), np.arange(len(uv) * 3).reshape(-1, 3), res)
    yy, xx = np.nonzero(tid >= 0); ids = tid[yy, xx]; weights = bary[yy, xx]
    def splat(per_vertex, fill=0.):
        img = np.full((res, res), fill, np.float32)
        img[yy, xx] = (np.asarray(per_vertex, np.float32)[tri[ids]] * weights).sum(-1)
        return img
    rough = np.full(len(vertices), DEFAULT[0], np.float32); spec = np.full(len(vertices), DEFAULT[1], np.float32)
    micro = np.full(len(vertices), DEFAULT[2], np.float32); assigned = np.zeros(len(vertices), bool)
    regions = {}
    for name, r, s, m in REGION_TEMPLATE:
        mask = model.group(name)[exterior] & ~assigned
        rough[mask] = r; spec[mask] = s; micro[mask] = m; assigned |= mask
        regions[name] = int(mask.sum())
    tzone = (model.group('forehead_region') | model.group('middle_brow_region') | model.group('nose_region') | model.group('chin_region'))[exterior]
    spec = spec * np.where(tzone, oiliness, .5 + .5 * oiliness)
    rough = np.clip(rough * (1 + .08 * (age - 1)), .2, .9); micro = np.clip(micro * age, 0, 1.5)
    lips = (model.group('upper_lip') | model.group('lower_lip'))[exterior].astype(np.float32)
    cav = concavity(vertices, tri)
    sigma = max(res / 1024, .5)
    maps = dict(roughness=gaussian_filter(splat(rough, DEFAULT[0]), sigma), specular=gaussian_filter(splat(spec, DEFAULT[1]), sigma),
                micro_strength=gaussian_filter(splat(micro, DEFAULT[2]), sigma), lips=gaussian_filter(splat(lips), sigma * 2),
                cavity=gaussian_filter(splat(cav), sigma * 3))
    for k in maps:
        maps[k][tid < 0] = {'roughness': DEFAULT[0], 'specular': DEFAULT[1], 'micro_strength': 0, 'lips': 0, 'cavity': 0}[k]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'skin_templates.npz', **{k: v.astype(np.float32) for k, v in maps.items()})
    Image.fromarray(micro_normal(seed=seed)).save(out / 'skin_micro_normal.png')
    manifest = dict(schema='vhuman.skin_templates.v1', res=res, oiliness=oiliness, age=age, seed=seed,
                    regions=regions, default=dict(zip(('roughness', 'specular', 'micro_strength'), DEFAULT)),
                    status='authored region priors on fixed GNM topology; cavity is geometric concavity of the fitted mesh, not a measurement',
                    files=dict(maps='skin_templates.npz', micro_normal='skin_micro_normal.png'),
                    shader=dict(roughness='template * (1 + 0.6 cavity)', specular_ior_level='template * (1 - 0.7 cavity)',
                                coat='0.10 (1 - cavity) + 0.35 lips, roughness 0.30 -> 0.12 on lips',
                                subsurface='random_walk_skin, radius (3.67, 1.37, 0.68) mm, scale 0.5, weight 0.6',
                                micro_normal='tiled 12x, strength 0.6 * micro_strength', displacement='true displacement, adaptive subdivision'))
    (out / 'skin_templates.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('candidate'); ap.add_argument('--out', required=True); ap.add_argument('--res', type=int, default=2048)
    ap.add_argument('--oiliness', type=float, default=1.); ap.add_argument('--age', type=float, default=1.)
    a = ap.parse_args()
    print(json.dumps(build(a.candidate, a.out, res=a.res, oiliness=a.oiliness, age=a.age), indent=1))
