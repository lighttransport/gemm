"""Transfer an existing synthetic appearance prior onto a fresh portrait bake.

Identical topology and UVs are required. This is material correspondence, not
new image evidence or a rerun of the old multiview generation.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from .generated_skin import atlas_surface, unseen_weight
from .observations import sha256
from .provenance import validate_candidate
from .reference import srgb_to_linear, linear_to_srgb


def blend_prior(base, prior, observed, eligible, weight):
    base, prior = np.asarray(base), np.asarray(prior)
    observed, eligible, weight = np.asarray(observed), np.asarray(eligible), np.asarray(weight)
    shape = base.shape[:2]
    if (base.ndim != 3 or base.shape[2] != 3 or prior.shape != base.shape
            or base.dtype != np.uint8 or prior.dtype != np.uint8
            or observed.shape != shape or eligible.shape != shape or weight.shape != shape
            or observed.dtype != bool or eligible.dtype != bool
            or not np.isfinite(weight).all() or (weight < 0).any() or (weight > 1).any()):
        raise ValueError('invalid prior transfer arrays')
    active = eligible & ~observed & (weight > 0)
    result = base.copy()
    w = weight[active, None]
    colors = srgb_to_linear(base[active]/255)*(1-w)+srgb_to_linear(prior[active]/255)*w
    result[active] = np.uint8(np.clip(linear_to_srgb(colors)*255+.5, 0, 255))
    return result


def transfer(candidate, prior, out, *, feather_mm=6., maximum_displacement_mm=3.):
    candidate, prior, out = (Path(p).resolve() for p in (candidate, prior, out))
    if (not np.isfinite([feather_mm, maximum_displacement_mm]).all()
            or feather_mm <= 0 or maximum_displacement_mm <= 0):
        raise ValueError('positive finite transfer limits required')
    manifest, source = validate_candidate(candidate), validate_candidate(prior)
    if manifest['portrait_sha256'] != source['portrait_sha256']:
        raise ValueError('prior portrait differs from target portrait')
    if manifest['material'].get('synthetic_completion'):
        raise ValueError('target must be a fresh portrait bake')
    completion = source['material'].get('synthetic_completion')
    if not completion:
        raise ValueError('prior must have a validated synthetic completion')
    if out.exists() and any(out.iterdir()):
        raise ValueError('transfer output must be empty')
    geometry, old = (dict(np.load(p/'geometry.npz', allow_pickle=False)) for p in (candidate, prior))
    for name in ('triangles', 'triangle_uvs'):
        if not np.array_equal(geometry[name], old[name]):
            raise ValueError('identical skin topology and UVs required')
    if geometry['captured'].shape != old['captured'].shape:
        raise ValueError('matching captured geometry required')
    displacement = np.linalg.norm(geometry['captured']-old['captured'], axis=-1)*1000
    if not np.isfinite(displacement).all() or displacement.max() > maximum_displacement_mm:
        raise ValueError('skin displacement exceeds transfer bound')
    def pixels(root, name, mode):
        return np.asarray(Image.open(root/name).convert(mode))
    base, previous = (pixels(p, 'skin_basecolor.png', 'RGB') for p in (candidate, prior))
    observed = pixels(candidate, 'skin_coverage.png', 'L') > 0
    old_observed = pixels(prior, 'skin_coverage.png', 'L') > 0
    support = pixels(prior, 'skin_generated_support.png', 'L')
    if (base.shape != previous.shape or base.shape[0] != base.shape[1]
            or any(a.shape != base.shape[:2] for a in (observed, old_observed, support))):
        raise ValueError('matching square texture atlases required')
    valid, points, _ = atlas_surface(geometry, len(base))
    eligible = valid & ~observed & ~old_observed & (support > 0)
    weight = np.zeros(valid.shape)
    weight[valid] = unseen_weight(points, observed[valid], distance=feather_mm*.001)
    weight[~eligible] = 0
    result = blend_prior(base, previous, observed, eligible, weight)
    distance, near = distance_transform_edt(~valid, return_indices=True)
    gutter = ~valid & (distance <= 4) & ~observed
    result[gutter] = result[near[0][gutter], near[1][gutter]]
    out.mkdir(parents=True, exist_ok=True)
    for path in candidate.iterdir():
        if path.is_file():
            shutil.copyfile(path, out/path.name)
    Image.fromarray(result).save(out/'skin_basecolor.png')
    # Retain prior support only as a discounted appearance heuristic, not confidence.
    transferred_support = np.uint8(np.clip(support*weight, 0, 255)+.5)
    Image.fromarray(transferred_support).save(out/'skin_generated_support.png')
    Image.fromarray(np.uint8(weight*255+.5)).save(out/'skin_prior_transfer.png')
    count = int((weight > 0).sum())
    report = dict(schema='vhuman.synthetic_skin_completion.v1', method='same_uv_prior_transfer',
        source_geometry_sha256=manifest['geometry_sha256'], synthetic=True,
        generator='Transferred existing synthetic appearance; no generation performed',
        license=completion.get('license', 'See source completion'),
        basecolor_sha256=sha256(out/'skin_basecolor.png'),
        generated_support_sha256=sha256(out/'skin_generated_support.png'),
        photographed_texels_changed=int(np.any(result[observed] != base[observed], axis=1).sum()),
        generated_texels=count, unseen_texels=int((valid & ~observed).sum()),
        prior_transfer=dict(source=str(prior), source_geometry_sha256=source['geometry_sha256'],
            source_completion_sha256=sha256(prior/'generated_skin.json'),
            source_basecolor_sha256=sha256(prior/'skin_basecolor.png'),
            source_support_sha256=sha256(prior/'skin_generated_support.png'),
            source_coverage_sha256=sha256(prior/'skin_coverage.png'),
            target_bake=str(candidate), target_basecolor_sha256=sha256(candidate/'skin_basecolor.png'),
            target_coverage_sha256=sha256(candidate/'skin_coverage.png'),
            transfer_mask_sha256=sha256(out/'skin_prior_transfer.png'),
            maximum_skin_displacement_mm=float(displacement.max()),
            displacement_limit_mm=maximum_displacement_mm, feather_mm=feather_mm,
            prior_texels_used=count, observed_texels_protected=int(observed.sum()),
            old_observed_texels_excluded=int((valid & ~observed & old_observed).sum()),
            topology_and_uvs_identical=True, new_view_evidence=False),
        limitations=['Transferred color is an appearance prior, not newly measured hidden skin',
                     'Old multiview coverage and consistency metrics do not apply to this geometry',
                     'Prior photographed texels are excluded; new photographed texels remain exact',
                     'Same UV correspondence does not establish shading or material accuracy'])
    manifest['id'] = out.name
    manifest['material']['synthetic_completion'] = report
    manifest['material_refinement'] = dict(source=str(candidate), geometry_unchanged=True,
                                          appearance_prior=str(prior))
    for name, data in (('manifest.json', manifest), ('skin_material.json', manifest['material']),
                       ('generated_skin.json', report)):
        (out/name).write_text(json.dumps(data, indent=2)+'\n')
    validate_candidate(out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'prior', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--feather-mm', type=float, default=6.)
    parser.add_argument('--maximum-displacement-mm', type=float, default=3.)
    print(json.dumps(transfer(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
