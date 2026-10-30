"""Bake calibrated close-up views (e.g. generated ear close-ups) into a candidate's skin atlas (stage 2b).

Unlike mv_texture (six fixed orthographic MV-Adapter cameras) this baker takes an explicit list of
(image, Camera, optional mask) in candidate space. Guards follow the existing completion code:
depth/facing visibility tests against the fitted mesh, silhouette erosion, grazing-angle confidence
reduction, multi-view agreement (multiview_skin.fuse_views), metric feathering into existing texels and
byte-exact protection of photographed texels. The result is a new candidate directory whose
generated_skin.json / manifest record the bake as synthetic (method 'closeup_view_bake').
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .reference import Camera, rasterize, srgb_to_linear, linear_to_srgb
from .observations import sha256

PHOTOGRAPHED = .99   # skin_confidence >= this is treated as photographed evidence (never modified)


def _sample(img, uv):
    h, w = img.shape[:2]
    x = np.clip(uv[:, 0] * w - .5, 0, w - 1.001); y = np.clip(uv[:, 1] * h - .5, 0, h - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def project(points, normals, mesh_points, triangles, image, camera, mask=None, *, min_facing=.25, depth_tolerance=.002,
            grazing=.4, erode=2):
    """Colors (N,3 linear) and weights (N) for atlas surface points seen in one view."""
    from scipy.ndimage import binary_erosion
    res_y, res_x = np.asarray(image).shape[:2]
    ids, _, depth = rasterize(mesh_points, triangles, camera, (res_x, res_y))
    valid = ids >= 0
    if mask is not None:
        valid &= np.asarray(mask, bool)
    interior = binary_erosion(valid, iterations=erode)
    xy, z = camera.project(points)
    ix = np.clip(xy[:, 0].astype(int), 0, res_x - 1); iy = np.clip(xy[:, 1].astype(int), 0, res_y - 1)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < res_x) & (xy[:, 1] >= 0) & (xy[:, 1] < res_y)
    facing = np.maximum((normals * camera.rotation[2]).sum(1), 0)
    seen = inside & interior[iy, ix] & (np.abs(depth[iy, ix] - z) < depth_tolerance) & (facing > min_facing)
    rgb = srgb_to_linear(np.asarray(image, float) / 255.)
    uv = np.stack((xy[:, 0] / res_x, xy[:, 1] / res_y), 1)
    colors = _sample(rgb, np.clip(uv, 0, 1))
    weight = seen * facing ** 2 * np.where(facing < grazing, (facing / grazing) ** 2, 1.)   # grazing texels count less
    return colors, weight, dict(seen_texels=int(seen.sum()), facing_median=float(np.median(facing[seen])) if seen.any() else None)


def bake_cameras(candidate, views, out, *, min_facing=.25, depth_tolerance=.002, tolerance=.012, feather_m=.006,
                 support_cap=.25, generator='', license='', notes=None):
    """views: list of dict(image=path, camera=Camera|dict, mask=path|None, name=str). Writes a new candidate at out."""
    from .generated_skin import atlas_surface, unseen_weight
    from .multiview_skin import fuse_views
    from scipy.ndimage import gaussian_filter
    candidate, out = Path(candidate), Path(out)
    if out.exists():
        raise ValueError('use a new output directory')
    shutil.copytree(candidate, out)
    manifest = json.loads((out / 'manifest.json').read_text())
    with np.load(out / 'geometry.npz', allow_pickle=False) as z:
        geometry = {k: z[k] for k in ('captured', 'triangles', 'triangle_uvs')}
    base_img = Image.open(out / 'skin_basecolor.png').convert('RGB'); res = base_img.width
    base8 = np.asarray(base_img).copy(); base = srgb_to_linear(base8.astype(float) / 255.)
    conf = np.asarray(Image.open(out / 'skin_confidence.png').convert('L'), float) / 255.
    valid, points, normals = atlas_surface(geometry, res)
    photographed = conf >= PHOTOGRAPHED
    colors, weights, reports = [], [], []
    for view in views:
        cam = view['camera'] if isinstance(view['camera'], Camera) else Camera.from_dict(view['camera'])
        image = Image.open(view['image']).convert('RGB')
        mask = None if not view.get('mask') else np.asarray(Image.open(view['mask']).convert('L')) > 127
        c, w, rep = project(points, normals, geometry['captured'][0], geometry['triangles'], image, cam, mask,
                            min_facing=min_facing, depth_tolerance=depth_tolerance)
        colors.append(c); weights.append(w)
        rep.update(name=view.get('name'), image=str(Path(view['image']).resolve()), image_sha256=sha256(view['image']),
                   mask=None if not view.get('mask') else str(Path(view['mask']).resolve()), camera=cam.as_dict())
        reports.append(rep)
    fused, support, agreement = fuse_views(np.stack(colors), np.stack(weights), tolerance=tolerance)
    # texels this bake supports (never photographed ones)
    yy, xx = np.nonzero(valid)
    texel_support = np.zeros((res, res)); texel_support[yy, xx] = support
    texel_support[photographed] = 0
    fused_img = np.zeros((res, res, 3)); fused_img[yy, xx] = fused
    # metric feather from the existing photographed anchors keeps seams soft; support scales the blend
    feather = np.ones((res, res))
    if photographed[yy, xx].any():
        feather_vals = unseen_weight(points, photographed[yy, xx], distance=feather_m); feather[yy, xx] = feather_vals
    blend = np.clip(texel_support / support_cap, 0, 1) * feather
    blend = gaussian_filter(blend, 1.) * (texel_support > 0)
    result = base * (1 - blend[..., None]) + fused_img * blend[..., None]
    result8 = np.clip(linear_to_srgb(np.clip(result, 0, 1)) * 255 + .5, 0, 255).astype(np.uint8)
    result8[photographed] = base8[photographed]           # byte-exact protection
    changed = int((result8 != base8).any(-1).sum())
    Image.fromarray(result8).save(out / 'skin_basecolor.png')
    new_conf = np.maximum(conf, np.minimum(texel_support, support_cap)); new_conf[photographed] = conf[photographed]
    Image.fromarray(np.clip(new_conf * 255 + .5, 0, 255).astype(np.uint8)).save(out / 'skin_confidence.png')
    Image.fromarray(np.clip(blend * 255 + .5, 0, 255).astype(np.uint8)).save(out / 'closeup_bake_blend.png')
    previous = manifest.get('material', {}).get('synthetic_completion')
    report = dict(schema='vhuman.synthetic_skin_completion.v1', method='closeup_view_bake', synthetic=True,
                  source_geometry_sha256=manifest['geometry_sha256'], generator=generator, license=license,
                  basecolor_sha256=sha256(out / 'skin_basecolor.png'), confidence_sha256=sha256(out / 'skin_confidence.png'),
                  blend_mask_sha256=sha256(out / 'closeup_bake_blend.png'),
                  photographed_texels=int(photographed.sum()), photographed_texels_changed=0, changed_texels=changed,
                  supported_texels=int((texel_support > 0).sum()), views=reports, agreement=agreement,
                  parameters=dict(min_facing=min_facing, depth_tolerance=depth_tolerance, tolerance=tolerance, feather_m=feather_m,
                                  support_cap=support_cap, grazing_confidence='facing<0.4 weight x (facing/0.4)^2'),
                  previous_completion=previous, notes=notes or 'generated close-up views are appearance priors, not photographic evidence')
    (out / 'generated_skin.json').write_text(json.dumps(report, indent=2) + '\n')
    manifest.setdefault('material', {})['synthetic_completion'] = report
    manifest['basecolor_sha256'] = report['basecolor_sha256']
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'closeup_bake.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('candidate'); ap.add_argument('--out', required=True)
    ap.add_argument('--views', required=True, help='json list of {name,image,camera,mask}')
    ap.add_argument('--generator', default=''); ap.add_argument('--license', default='')
    ap.add_argument('--min-facing', type=float, default=.25); ap.add_argument('--depth-tolerance', type=float, default=.002)
    ap.add_argument('--tolerance', type=float, default=.012); ap.add_argument('--feather-m', type=float, default=.006)
    a = ap.parse_args(argv)
    views = json.loads(Path(a.views).read_text())
    report = bake_cameras(a.candidate, views, a.out, min_facing=a.min_facing, depth_tolerance=a.depth_tolerance,
                          tolerance=a.tolerance, feather_m=a.feather_m, generator=a.generator, license=a.license)
    print(json.dumps({k: report[k] for k in ('changed_texels', 'supported_texels', 'photographed_texels_changed', 'agreement')}, indent=1))


if __name__ == '__main__':
    raise SystemExit(main())
