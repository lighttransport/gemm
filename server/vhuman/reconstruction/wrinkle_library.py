"""Region wrinkle library: authored displacement channels driven by the 12 regional strain activations (stage 2f).

skin_detail.npz already carries 12 strain-driven groove channels (one per GNM region, photo-crease shaped).
This module appends six anatomically placed wrinkle fields on the same UV atlas, each with its own driver row
(a linear combination of the 12 regional compressions), so cycles_scene's existing wrinkle_i drivers extend
without changes: forehead horizontal lines, glabellar vertical lines, crow's feet, nasolabial fold deepening,
nasal-root (bunny) lines and chin dimpling. Amplitudes are priors; a held-out expression gate
(expression_detail-style) decides whether the library stays active for a subject.
"""
import json
from pathlib import Path
import numpy as np

from .skin_detail import REGIONS

# name: (regions for the spatial mask, driver weights over REGIONS, depth metres at full activation)
WRINKLES = {
    'forehead':   (('forehead_region',), {'forehead_region': 1.0, 'middle_brow_region': .3}, 4.5e-4),
    'glabella':   (('middle_brow_region',), {'middle_brow_region': 1.0}, 5.0e-4),
    'crows_feet': (('left_orbital_region', 'right_orbital_region'),
                   {'left_orbital_region': .5, 'right_orbital_region': .5, 'left_cheek_region': .25, 'right_cheek_region': .25}, 3.5e-4),
    'nasolabial': (('left_cheek_region', 'right_cheek_region'), {'left_cheek_region': .5, 'right_cheek_region': .5, 'upper_lip_region': .4}, 6.0e-4),
    'nasal_root': (('nose_region',), {'nose_region': .7, 'middle_brow_region': .5}, 3.0e-4),
    'chin':       (('chin_region',), {'chin_region': 1.0, 'lower_lip_region': .3}, 3.0e-4),
}


def _lines(mask, res, *, spacing_px, angle_deg, jitter, sigma_px, rng):
    """Parallel wavy grooves (Gaussian cross-section) inside a soft mask; returns 0..1 depth field."""
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float64)
    a = np.deg2rad(angle_deg); u = xx * np.cos(a) + yy * np.sin(a); v = -xx * np.sin(a) + yy * np.cos(a)
    wave = jitter * spacing_px * (np.sin(v / (spacing_px * 2.3) + rng.uniform(0, 6)) + .5 * np.sin(v / (spacing_px * .9) + rng.uniform(0, 6)))
    phase = ((u + wave) / spacing_px) % 1.
    d = np.minimum(phase, 1 - phase) * spacing_px
    depth = np.exp(-(d * d) / (2 * sigma_px * sigma_px))
    strength = .6 + .4 * np.sin(v / (spacing_px * 3.1) + rng.uniform(0, 6)) ** 2   # broken, uneven lines
    return depth * strength * mask


def _pebbles(mask, res, *, scale_px, rng):
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float64)
    field = np.zeros((res, res))
    for k, amp in ((1., .5), (2., .3), (4., .2)):
        field += amp * np.sin(xx / scale_px * k + rng.uniform(0, 6)) * np.sin(yy / scale_px * k * 1.3 + rng.uniform(0, 6))
    return np.clip(field, 0, 1) * mask


def _boundary_band(inner, outer, px):
    from scipy.ndimage import gaussian_filter, binary_dilation
    a = binary_dilation(inner > .5, iterations=px) & (outer > .5)
    b = binary_dilation(outer > .5, iterations=px) & (inner > .5)
    return gaussian_filter((a | b).astype(np.float64), px / 2)


def build_fields(region_masks, res, *, mm_px, seed=5):
    """region_masks: (12,res,res) soft masks in REGIONS order. Returns {name: field 0..1}."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    m = {name: region_masks[i] for i, name in enumerate(REGIONS)}
    soft = {k: gaussian_filter(v, mm_px) for k, v in m.items()}
    fields = {}
    fields['forehead'] = _lines(soft['forehead_region'], res, spacing_px=7 * mm_px, angle_deg=90, jitter=.35, sigma_px=.9 * mm_px, rng=rng)
    fields['glabella'] = _lines(soft['middle_brow_region'], res, spacing_px=6 * mm_px, angle_deg=0, jitter=.15, sigma_px=.8 * mm_px, rng=rng)
    crow = np.maximum(_lines(soft['left_orbital_region'], res, spacing_px=4 * mm_px, angle_deg=70, jitter=.4, sigma_px=.6 * mm_px, rng=rng),
                      _lines(soft['right_orbital_region'], res, spacing_px=4 * mm_px, angle_deg=-70, jitter=.4, sigma_px=.6 * mm_px, rng=rng))
    fields['crows_feet'] = crow
    nose_lip = np.maximum(m['nose_region'], m['upper_lip_region'])
    band = np.maximum(_boundary_band(m['left_cheek_region'], nose_lip, max(1, int(2 * mm_px))),
                      _boundary_band(m['right_cheek_region'], nose_lip, max(1, int(2 * mm_px))))
    fields['nasolabial'] = np.clip(band / max(band.max(), 1e-6), 0, 1)
    root = gaussian_filter(_boundary_band(m['nose_region'], m['middle_brow_region'], max(1, int(3 * mm_px))), mm_px)
    fields['nasal_root'] = _lines(root / max(root.max(), 1e-6), res, spacing_px=3.5 * mm_px, angle_deg=20, jitter=.3, sigma_px=.6 * mm_px, rng=rng)
    fields['chin'] = _pebbles(soft['chin_region'], res, scale_px=2.5 * mm_px, rng=rng)
    return fields


def augment(detail_path, out=None, *, head_height_m=.24, protect=None, seed=5):
    """Append the library channels to a skin_detail.npz (height channels + driver rows). Returns the manifest."""
    detail_path = Path(detail_path); out = Path(out or detail_path)
    with np.load(detail_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    masks = data['region_masks']; res = masks.shape[-1]
    if data['dynamic_height_m'].shape[0] != len(REGIONS):
        raise ValueError('skin_detail already augmented or not a 12-region detail file')
    # UV atlas spans roughly the head height; 1 mm in texels
    mm_px = max(1., res / (head_height_m * 1000.) * .9)
    fields = build_fields(masks, res, mm_px=mm_px, seed=seed)
    matrix = data['coefficient_to_activation']
    heights, rows, names = [], [], []
    for name, (_, weights, depth) in WRINKLES.items():
        h = -(fields[name] * depth).astype(np.float32)
        if protect is not None:
            h *= protect
        row = np.zeros(matrix.shape[1], np.float32)
        for region, w in weights.items():
            row += w * matrix[REGIONS.index(region)]
        heights.append(h); rows.append(np.clip(row, -1, 1)); names.append(name)
    data['dynamic_height_m'] = np.concatenate([data['dynamic_height_m'], np.stack(heights)], 0)
    data['coefficient_to_activation'] = np.concatenate([matrix, np.stack(rows)], 0)
    data['library_names'] = np.array(names)
    np.savez_compressed(out, **data)
    manifest = dict(schema='vhuman.wrinkle_library.v1', channels=names, depth_m={k: v[2] for k, v in WRINKLES.items()},
                    drivers={k: v[1] for k, v in WRINKLES.items()}, mm_px=mm_px, res=res, seed=seed,
                    status='authored displacement priors driven by regional strain; subject to the held-out expression gate')
    out.with_name('wrinkle_library.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('detail'); ap.add_argument('--out'); ap.add_argument('--preview', help='write per-channel PNG previews to this dir')
    a = ap.parse_args()
    manifest = augment(a.detail, a.out)
    print(json.dumps(manifest, indent=1))
    if a.preview:
        from PIL import Image
        p = Path(a.preview); p.mkdir(parents=True, exist_ok=True)
        with np.load(a.out or a.detail, allow_pickle=False) as z:
            dyn = z['dynamic_height_m']; names = list(z['library_names'])
        for i, name in enumerate(names):
            h = -dyn[len(REGIONS) + i]
            Image.fromarray((np.clip(h / max(h.max(), 1e-9), 0, 1) * 255).astype(np.uint8)).save(p / f'{name}.png')
