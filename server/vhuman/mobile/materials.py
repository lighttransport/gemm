"""Bake the prepared short-hair undercoat into the mobile skin atlas.

The visible parsing mask and inferred crown stay separate inputs. Hair is an
opaque material region on the existing scalp, so no intersecting shell appears.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from ..rig.bake import rasterize_uv
from ..reconstruction.reference import srgb_to_linear, linear_to_srgb


def skin_textures(candidate, scene, assets, description, out):
    candidate, scene, out = map(Path, (candidate, scene, out))
    result = {name: candidate/('skin_'+name+'.png') for name in ('basecolor', 'normal', 'orm')}
    if 'skin_hair_crown' not in assets or not (scene/'hair_coverage.png').is_file(): return result
    base = np.array(Image.open(result['basecolor']).convert('RGB'))
    normal = np.array(Image.open(result['normal']).convert('RGB'))
    orm = np.array(Image.open(result['orm']).convert('RGB'))
    if base.shape != normal.shape or base.shape != orm.shape or base.shape[0] != base.shape[1]:
        raise ValueError('mobile skin atlases must have matching square dimensions')
    resolution = len(base); uv = assets['skin_uvs']; triangles = assets['skin_triangles']
    ids, bary = rasterize_uv(uv.reshape(-1, 2), np.arange(uv.size//2).reshape(-1, 3), resolution)
    valid = ids >= 0; y, x = np.nonzero(valid); tid = ids[y, x]; weight = bary[y, x]
    crown = (assets['skin_hair_crown'][triangles[tid]]*weight).sum(1)
    source = (assets['skin_source_uvs'][tid]*weight[..., None]).sum(1)
    mask = np.asarray(Image.open(scene/'hair_coverage.png').convert('L'), float)/255
    sx = np.floor(source[:, 0]*mask.shape[1]).astype(int); sy = np.floor(source[:, 1]*mask.shape[0]).astype(int)
    inside = (sx>=0)&(sy>=0)&(sx<mask.shape[1])&(sy<mask.shape[0])
    measured = np.zeros(len(y)); measured[inside] = mask[sy[inside], sx[inside]]
    coverage = np.maximum(crown, measured).clip(0, 1)
    alpha = coverage[:, None]
    color = srgb_to_linear(base[y, x]/255)
    color = color*(1-alpha)+np.asarray(description['hair']['color_linear'])*alpha
    base[y, x] = np.uint8(np.clip(linear_to_srgb(color)*255+.5, 0, 255))
    orm[y, x, 1] = np.uint8(orm[y, x, 1]*(1-coverage)+.85*255*coverage)
    normal[y, x] = np.uint8(normal[y, x]*(1-alpha)+np.array([128,128,255])*alpha)
    # Four-texel island gutters keep bilinear sampling from picking an unrelated
    # UV island/background, matching the existing offline atlas convention.
    distance, nearest = distance_transform_edt(~valid, return_indices=True)
    gutter = (~valid)&(distance<=4)
    for name, image in (('basecolor', base), ('normal', normal), ('orm', orm)):
        image[gutter] = image[nearest[0][gutter], nearest[1][gutter]]
        target = out/('mobile_skin_'+name+'.png'); Image.fromarray(image).save(target); result[name] = target
    return result
