"""Apply the UE-calibrated optical contact profile to a matched static Cycles view.

Requires a peri_passes_blender capture, its staged scene assets, and request.json
with an explicit camera. The source beauty can include diffuse-only skin SSS.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .images import read_exr, write_exr
from .peri_scatter import load_passes
from ..reconstruction.eye_contact import projected_eye_mask, soften_contact


def apply(beauty, capture, assets, request, width_m=.0012, sigma_m=.00025, strength=.35739718109334984,
          *, skin_width_m=.0012, skin_strength=0.):
    image = read_exr(beauty)
    passes = load_passes(capture)
    meta = json.loads(Path(capture).with_suffix('.json').read_text())
    req = json.loads(Path(request).read_text())
    camera = req.get('explicit_camera')
    if camera is None:
        raise ValueError('contact diagnostic requires a staged explicit camera')
    if image.shape[:2] != passes['Combined'].shape[:2]:
        raise ValueError('beauty and capture resolutions must match')
    focal = image.shape[0] * float(meta['lens_mm']) / float(meta['sensor_height_mm'])
    expected_focal = camera['focal_px'] * image.shape[0] / camera['size'][1]
    if not np.allclose(camera['origin'], meta['camera_origin'], atol=1e-5) or not np.isclose(focal, expected_focal, rtol=1e-5):
        raise ValueError('request and capture cameras do not match')
    with np.load(assets, allow_pickle=False) as data:
        mask = projected_eye_mask(data, camera, image.shape[:2], skin_mask=passes['skin_mask'][..., 0])
    depth = ((passes['Position'][..., :3] - np.asarray(meta['camera_origin'])) * np.asarray(meta['camera_forward'])).sum(-1)
    result = soften_contact(image, mask, depth, focal, width_m, sigma_m, strength,
                            skin_mask=passes['skin_mask'][...,0],
                            skin_width_m=skin_width_m, skin_strength=skin_strength)
    record = dict(method='UE-inspired optical contact layer; optional skin-side band; central iris/alpha preserved',
                  width_m=width_m, sigma_m=sigma_m, strength=strength, tint=[1., .904661, .896269],
                  skin_width_m=skin_width_m, skin_strength=skin_strength,
                  beauty=str(beauty), capture=str(capture), assets=str(assets), request=str(request),
                  eye_pixels=int((mask > .5).sum()), resolution=list(image.shape[:2]))
    return result, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('beauty', type=Path); parser.add_argument('out', type=Path)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--width-mm', type=float, default=1.2)
    parser.add_argument('--sigma-mm', type=float, default=.25)
    parser.add_argument('--strength', type=float, default=.35739718109334984)
    parser.add_argument('--skin-width-mm', type=float, default=1.2)
    parser.add_argument('--skin-strength', type=float, default=0.)
    args = parser.parse_args()
    image, record = apply(args.beauty, args.capture, args.assets, args.request,
                          args.width_mm * .001, args.sigma_mm * .001, args.strength,
                          skin_width_m=args.skin_width_mm*.001, skin_strength=args.skin_strength)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_exr(args.out, image)
    args.out.with_suffix('.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
