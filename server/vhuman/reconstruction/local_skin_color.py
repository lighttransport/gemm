"""Repair localized skin color outliers with explicit surface regions and provenance.

This is an authored appearance prior. Euclidean neighborhoods are not geodesic
and may mix nearby folds; inspect matched renders before selecting an output.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .observations import sha256
from .provenance import validate_candidate
from .reference import linear_to_srgb, srgb_to_linear


def correct_regions(points, colors, regions, *, protected=None, strength=.9):
    """Return corrected linear RGB, edit weights and per-region diagnostics."""
    points, colors = np.asarray(points, float), np.asarray(colors, float)
    if (points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape
            or not len(points) or not np.isfinite(points).all()
            or not np.isfinite(colors).all() or np.any(colors < 0)
            or np.any(colors > 1) or not np.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('invalid surface colors or strength')
    protected = np.zeros(len(points), bool) if protected is None else np.asarray(protected, bool)
    if protected.shape != (len(points),):
        raise ValueError('protected mask differs from surface')
    if not isinstance(regions, list) or not regions:
        raise ValueError('explicit nonempty region list required')
    logrgb = np.log(np.maximum(colors, 1e-5))
    luma = colors @ np.array([.2126, .7152, .0722])
    output, weights = colors.copy(), np.zeros(len(points))
    claimed = np.zeros(len(points), bool)
    reports, names = [], set()
    for region in regions:
        name = region['name']
        center = np.asarray(region['center'], float)
        radii = np.asarray(region['radii'], float)
        if (not isinstance(name, str) or not name or name in names
                or center.shape != (3,) or radii.shape != (3,)
                or not np.isfinite(center).all() or not np.isfinite(radii).all()
                or np.any(radii <= 0)):
            raise ValueError('invalid or duplicate surface region')
        names.add(name)
        distance = np.linalg.norm((points - center) / radii, axis=1)
        mask = distance < 1
        if np.any(mask & claimed):
            raise ValueError('overlapping regions require an explicit combined recipe')
        claimed |= mask
        ids = np.flatnonzero(mask)
        row = dict(name=name, center=center.tolist(), radii=radii.tolist(),
                   region_texels=len(ids), trusted_samples=0, edited_weights=0,
                   maximum_weight=0.)
        reports.append(row)
        if not len(ids):
            continue
        p = points[ids]
        _, coarse = np.unique(np.floor(p / .0015).astype(np.int64), axis=0,
                              return_index=True)
        ratio = np.median(logrgb[ids, 2] - logrgb[ids, 0])
        trusted = ((logrgb[ids[coarse], 2] - logrgb[ids[coarse], 0] < ratio + .08)
                   & (luma[ids[coarse]] < np.quantile(luma[ids], .9)))
        coarse = coarse[trusted]
        row['trusted_samples'] = len(coarse)
        if not len(coarse):
            # Uniform regions have no lower-luminance reference and need no edit.
            continue
        k = min(32, len(coarse))
        _, near = cKDTree(p[coarse]).query(p, k=k)
        near = np.asarray(near).reshape(len(ids), k)
        median = np.median(logrgb[ids[coarse[near]]], axis=1)
        target_luma = np.exp(median) @ np.array([.2126, .7152, .0722])
        chroma = logrgb[ids] - logrgb[ids].mean(1)[:, None]
        blue = np.clip((chroma[:, 2] - chroma[:, 0] - ratio - .08) / .20, 0, 1)
        bright = np.clip((np.log(np.maximum(luma[ids], 1e-5) / target_luma)
                          - np.log(1.3)) / .35, 0, 1)
        feather = np.clip((1 - distance[ids]) / .25, 0, 1)
        feather = feather * feather * (3 - 2 * feather)
        w = strength * np.maximum(blue, bright) * feather
        w[protected[ids]] = 0
        weights[ids] = w
        edited = w > 0
        output[ids[edited]] = np.exp(logrgb[ids[edited]] * (1 - w[edited, None])
                                    + median[edited] * w[edited, None])
        row.update(edited_weights=int(edited.sum()), maximum_weight=float(w.max()))
    return output, weights, reports


def cleanup(candidate, audit, out, regions, *, edit_observed=False, strength=.9):
    candidate, audit, out = map(lambda p: Path(p).resolve(), (candidate, audit, out))
    manifest = validate_candidate(candidate)
    request = json.loads((audit / 'request.json').read_text())
    if (manifest['geometry_sha256'] != request['geometry_sha256']
            or sha256(audit / 'surface.npz') != request['surface_sha256']):
        raise ValueError('surface audit provenance mismatch')
    original = Path(request['source'])
    reference = validate_candidate(original)
    if (reference['geometry_sha256'] != manifest['geometry_sha256']
            or reference['portrait_sha256'] != manifest['portrait_sha256']
            or sha256(original / 'skin_basecolor.png') != request['source_basecolor_sha256']):
        raise ValueError('photographic reference provenance mismatch')
    if out.exists() and any(out.iterdir()):
        raise ValueError('cleanup output must be empty')
    with np.load(audit / 'surface.npz', allow_pickle=False) as surface:
        valid, observed = surface['valid'].astype(bool), surface['observed'].astype(bool)
        points = surface['points'].copy()
    base = np.asarray(Image.open(candidate / 'skin_basecolor.png').convert('RGB'))
    original_image = np.asarray(Image.open(original / 'skin_basecolor.png').convert('RGB'))
    if valid.shape != base.shape[:2] or original_image.shape != base.shape:
        raise ValueError('surface and texture dimensions differ')
    if int(valid.sum()) != len(points) or observed.shape != (len(points),):
        raise ValueError('surface sample counts differ')
    corrected, weights, rows = correct_regions(points, srgb_to_linear(base[valid] / 255.),
        regions, protected=None if edit_observed else observed, strength=strength)
    image = base.copy()
    image[valid] = np.uint8(np.clip(linear_to_srgb(corrected) * 255 + .5, 0, 255))
    changed = np.any(image != base, axis=2)
    if np.any(changed[valid] & (weights == 0)):
        raise ValueError('cleanup changed a protected texel')
    out.mkdir(parents=True, exist_ok=True)
    for path in candidate.iterdir():
        if path.is_file():
            shutil.copyfile(path, out / path.name)
    Image.fromarray(image).save(out / 'skin_basecolor.png')
    mask = np.zeros(valid.shape, np.uint8)
    mask[valid] = np.uint8(weights * 255 + .5)
    # A quantized feather must still identify every changed texel.
    mask[changed & (mask == 0)] = 1
    Image.fromarray(mask).save(out / 'local_color_edit_mask.png')
    report = dict(schema='vhuman.local_skin_color.v1', source=str(candidate),
        source_basecolor_sha256=sha256(candidate / 'skin_basecolor.png'),
        geometry_sha256=manifest['geometry_sha256'], surface_sha256=request['surface_sha256'],
        regions=rows, strength=strength, edit_observed=edit_observed,
        edited_texels=int(changed.sum()), edited_photographed_texels=int(changed[valid][observed].sum()),
        outside_weight_exact=True, new_view_evidence=False,
        mask_sha256=sha256(out / 'local_color_edit_mask.png'),
        limitation='Authored local color prior; may smooth real pigmentation or mix nearby folds. No new observation or geometric repair.')
    completion = json.loads((candidate / 'generated_skin.json').read_text())
    completion['basecolor_sha256'] = sha256(out / 'skin_basecolor.png')
    completion.setdefault('local_color_cleanup', []).append(report)
    completion['photographed_texels_changed'] = int(np.any(
        image[valid][observed] != original_image[valid][observed], axis=1).sum())
    completion['limitations'] = [s for s in completion['limitations']
                                 if not (edit_observed and 'remain exact' in s)] + [report['limitation']]
    manifest['id'] = out.name
    manifest['material']['synthetic_completion'] = completion
    manifest['material_refinement'] = dict(source=str(candidate), geometry_unchanged=True,
        projection_repair=edit_observed, scope='explicit local surface regions')
    for name, data in [('local_color_report.json', report), ('generated_skin.json', completion),
                       ('manifest.json', manifest), ('skin_material.json', manifest['material'])]:
        (out / name).write_text(json.dumps(data, indent=2) + '\n')
    validate_candidate(out)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'audit', 'out', 'regions'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--edit-observed', action='store_true')
    parser.add_argument('--strength', type=float, default=.9)
    args = vars(parser.parse_args())
    args['regions'] = json.loads(Path(args['regions']).read_text())
    print(json.dumps(cleanup(**args), indent=2))
