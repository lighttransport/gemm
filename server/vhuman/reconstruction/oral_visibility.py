"""Source-view oral visibility audit; image heuristics are not dental ground truth."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .reference import Camera, rasterize


INNER_LIP = (78, 81, 13, 311, 308, 402, 14, 178)


def source_tooth_mask(image, polygon, *, minimum=.48, chroma=.24):
    """Conservative bright, low-chroma pixels inside the observed mouth polygon."""
    image = np.asarray(image)
    polygon = np.asarray(polygon, dtype=float)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('source must be an RGB uint8 image')
    if polygon.shape != (8, 2) or not np.isfinite(polygon).all():
        raise ValueError('eight finite inner-lip points required')
    if not 0 <= minimum <= 1 or not 0 <= chroma <= 1:
        raise ValueError('invalid tooth mask thresholds')
    h, w = image.shape[:2]
    interior_image = Image.new('L', (w, h))
    ImageDraw.Draw(interior_image).polygon([tuple(p) for p in polygon], fill=255)
    interior = np.asarray(interior_image) > 0
    rgb = image.astype(float)/255
    mask = interior & (rgb.min(-1) > minimum) & (np.ptp(rgb, axis=-1) < chroma)
    return mask, interior


def overlap_metrics(predicted, target, interior):
    if predicted.shape != target.shape or target.shape != interior.shape:
        raise ValueError('oral masks must have matching shapes')
    intersection = int((predicted & target).sum())
    union = int((predicted | target).sum())
    return dict(target_pixels=int(target.sum()), visible_teeth_pixels=int(predicted.sum()),
                intersection_pixels=intersection, iou=intersection/union if union else None,
                target_recall=intersection/int(target.sum()) if target.any() else None,
                teeth_outside_mouth_pixels=int((predicted & ~interior).sum()))


def audit(candidate, out, *, minimum=.48, chroma=.24):
    from ..rig.gnm_model import GNMModel

    candidate, out = Path(candidate).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory must be empty')
    manifest = json.loads((candidate/'manifest.json').read_text())
    observations = json.loads((candidate/'observations.json').read_text())
    if len(observations['views']) != 1:
        raise ValueError('oral visibility audit requires one source view')
    view = observations['views'][0]
    polygon = np.array([view['anchors'][f'mp_{i:03d}']['xy'] for i in INNER_LIP])
    rgb = np.asarray(Image.open(candidate/'portrait.png').convert('RGB'))
    target, interior = source_tooth_mask(rgb, polygon, minimum=minimum, chroma=chroma)
    if not target.any():
        raise ValueError('no reliable bright tooth pixels; audit is not applicable')
    with np.load(candidate/'geometry.npz', allow_pickle=False) as g:
        vertices, triangles = g['full_captured'][0], g['full_triangles']
    # Observation cameras precede fitting; use the final fitted camera.
    camera = Camera.from_dict(manifest['geometry']['fitted_cameras'][0])
    tid, _, depth = rasterize(vertices, triangles, camera, (rgb.shape[1], rgb.shape[0]))
    model = GNMModel()
    teeth = ((model.group('upper_teeth_and_gums') | model.group('lower_teeth_and_gums'))[triangles].all(1)
             & ~model.group('gums')[triangles].all(1))
    valid = tid >= 0
    predicted = valid & teeth[np.maximum(tid, 0)]
    groups = ('skin_exterior', 'upper_lip', 'lower_lip', 'mouth_sock',
              'upper_teeth_and_gums', 'lower_teeth_and_gums', 'gums', 'tongue')
    visible_groups = {}
    for name in groups:
        membership = model.group(name)[triangles].all(1)
        visible_groups[name] = int((target & valid & membership[np.maximum(tid, 0)]).sum())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(schema='vhuman.oral_visibility.v1', candidate=str(candidate),
                  geometry_sha256=digest(candidate/'geometry.npz'),
                  portrait_sha256=digest(candidate/'portrait.png'), camera=camera.as_dict(),
                  mask=dict(minimum_srgb=minimum, maximum_chroma=chroma, inner_lip_indices=INNER_LIP),
                  metrics=overlap_metrics(predicted, target, interior),
                  target_visible_groups=visible_groups,
                  limitations=['Bright low-chroma tooth pixels are a heuristic, not ground truth',
                               'Group counts overlap; upper/lower lips are subsets of skin',
                               'Single source view cannot validate unseen anatomy or motion',
                               'Coverage does not measure tooth shape, contact, or shading'])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'visibility.npz', target=target, predicted=predicted,
                        interior=interior, triangle_ids=tid, depth=depth)
    overlay = rgb.copy()
    overlay[target, 1] = 255
    overlay[predicted, 0] = 255
    Image.fromarray(overlay).save(out/'overlay.png')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--minimum', type=float, default=.48)
    parser.add_argument('--chroma', type=float, default=.24)
    args = parser.parse_args()
    print(json.dumps(audit(args.candidate, args.out, minimum=args.minimum, chroma=args.chroma), indent=2))


if __name__ == '__main__':
    main()
