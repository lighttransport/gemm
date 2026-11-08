"""Apply the experimental diffuse-only skin-scattering model to Cycles pass captures.

python -m server.vhuman.metahuman.peri_scatter capture.exr output.exr --scale 1 --weight .6
Input is peri_passes_blender.py's multilayer EXR, captured with native skin SSS=0.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .images import write_exr
from ..reconstruction.skin_scatter import scatter_lighting, recombine_skin


def load_passes(path):
    import OpenEXR
    with OpenEXR.File(str(path)) as exr:
        channels = {}
        for part in exr.parts:
            part_name = part.header.get('name', '')
            for key, channel in part.channels.items():
                full_key = part_name if key in ('RGB', 'RGBA') and part_name else key
                if full_key in channels:
                    full_key = part_name + '.' + key
                channels[full_key] = channel
        def get(name):
            aliases = ('Image', 'Combined') if name == 'Combined' else (name,)
            found = [c.pixels for key, c in channels.items() if any(key == alias or key.endswith('.' + alias) for alias in aliases)]
            if len(found) != 1:
                raise ValueError(f'expected one {name} pass; available {list(channels)}')
            return np.asarray(found[0], np.float32).copy()
        result = {name: get(name) for name in ('Combined', 'DiffDir', 'DiffInd', 'DiffCol', 'Position', 'skin_mask', 'skin_geometric_normal')}
        for name in ('skin_profile', 'eye_mask', 'DiffIndDenoised', 'DiffDirDenoised'):
            if any(key == name or key.endswith('.' + name) for key in channels):
                result[name] = get(name)
    shape = result['Combined'].shape[:2]
    if any(value.shape[:2] != shape or value.ndim != 3 or value.shape[-1] < 3
           or not np.isfinite(value).all() for value in result.values()):
        raise ValueError('invalid or mismatched lighting passes')
    mask = result['skin_mask'][..., 0]
    interior = mask > .99
    if not interior.any() or mask.min() < -1e-5 or mask.max() > 1.00001:
        raise ValueError('skin mask is not a valid material AOV')
    lengths = np.linalg.norm(result['skin_geometric_normal'][..., :3][interior], axis=-1)
    if np.median(np.abs(lengths - 1.)) > .05:
        raise ValueError('skin geometric normal AOV is not normalized; check compositor links')
    return result


def apply(path, scale=1., weight=1., samples=48):
    p = load_passes(path)
    meta = json.loads(Path(path).with_suffix('.json').read_text())
    if meta['native_sss_weight'] != 0:
        raise ValueError('diffuse scattering requires native SSS disabled to avoid double diffusion')
    height = p['Combined'].shape[0]
    focal_px = height * float(meta['lens_mm']) / float(meta['sensor_height_mm'])
    depth = ((p['Position'][..., :3] - np.asarray(meta['camera_origin'])) * np.asarray(meta['camera_forward'])).sum(-1)
    lighting = p['DiffDir'][..., :3] + p['DiffInd'][..., :3]
    mask = p['skin_mask'][..., 0]
    radii = np.array([1.05, .4069, .2625]) * scale
    scattered = scatter_lighting(lighting, mask, depth, p['skin_geometric_normal'][..., :3], focal_px, radii, samples)
    result = recombine_skin(p['Combined'], lighting, scattered, p['DiffCol'][..., :3], mask, weight)
    return result, dict(method='normalized Burley radial lighting gather; albedo/specular preserved',
                        capture=meta, model_version=1,
                        model_source_sha256=hashlib.sha256((Path(__file__).parents[1]/'reconstruction/skin_scatter.py').read_bytes()).hexdigest(),
                        radius_mm=radii.tolist(), weight=weight, samples=samples, source=str(path), focal_px=focal_px)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path); parser.add_argument('out', type=Path)
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--weight', type=float, default=1.)
    parser.add_argument('--samples', type=int, default=48)
    args = parser.parse_args()
    image, record = apply(args.capture, args.scale, args.weight, args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_exr(args.out, image)
    args.out.with_suffix('.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
