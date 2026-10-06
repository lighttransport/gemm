"""Bake the prepared short-hair undercoat into the mobile skin atlas.

The visible parsing mask and inferred crown stay separate inputs. Hair is an
opaque material region on the existing scalp, so no intersecting shell appears.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from ..rig.bake import rasterize_uv
from ..reconstruction.reference import srgb_to_linear, linear_to_srgb
from .baking import sample, seam_correct


def skin_textures(candidate, scene, assets, description, out):
    candidate, scene, out = map(Path, (candidate, scene, out))
    result = {name: candidate/('skin_'+name+'.png') for name in ('basecolor', 'normal', 'orm')}
    base = np.array(Image.open(result['basecolor']).convert('RGB'))
    normal = np.array(Image.open(result['normal']).convert('RGB'))
    orm = np.array(Image.open(result['orm']).convert('RGB'))
    if base.shape != normal.shape or base.shape != orm.shape or base.shape[0] != base.shape[1]:
        raise ValueError('mobile skin atlases must have matching square dimensions')
    resolution = len(base); uv = assets['skin_uvs']; triangles = assets['skin_triangles']
    ids, bary = rasterize_uv(uv.reshape(-1, 2), np.arange(uv.size//2).reshape(-1, 3), resolution)
    valid = ids >= 0; y, x = np.nonzero(valid); tid = ids[y, x]; weight = bary[y, x]
    confidence=np.asarray(Image.open(candidate/'skin_confidence.png').convert('L'),float)/255
    color,seams=seam_correct(srgb_to_linear(base/255),confidence,triangles,uv,ids)
    if (candidate/'generated_skin.json').is_file():
        # A synthetic completion must not pull new colours across a seam into
        # photographed skin. The existing scalp material is applied below.
        observed=np.asarray(Image.open(candidate/'skin_coverage.png'))>0
        color[observed]=srgb_to_linear(base[observed]/255)
        seams['photographed_colors_restored_after_seam_filter']=True
    coverage=np.zeros(len(y));hair_report=dict(enabled=False)
    if 'skin_hair_crown' in assets and (scene/'hair_coverage.png').is_file():
        crown = (assets['skin_hair_crown'][triangles[tid]]*weight).sum(1)
        mask = np.asarray(Image.open(scene/'hair_coverage.png').convert('L'), float)/255
        source = (assets['skin_source_uvs'][tid]*weight[..., None]).sum(1)
        inside=(source.min(1)>=0)&(source.max(1)<1)
        measured=sample(mask,source)*inside
        coverage=np.maximum(crown,measured).clip(0,1)
        hair_report=dict(enabled=True,projected_samples=int((measured>.1).sum()),
            method='bilinear source mask plus separate crown prior',
            limitation='single-view projection; scalp occlusion not resolved')
    alpha = coverage[:, None]
    if hair_report['enabled']:
        color[y,x]=color[y,x]*(1-alpha)+np.asarray(description['hair']['color_linear'])*alpha
    base=np.uint8(np.clip(linear_to_srgb(color)*255+.5,0,255))
    orm[y, x, 1] = np.uint8(orm[y, x, 1]*(1-coverage)+.85*255*coverage)
    vectors=normal[y,x].astype(float)/255*2-1
    vectors=vectors*(1-alpha)+np.array([0.,0.,1.])*alpha
    vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
    normal[y,x]=np.uint8(np.clip((vectors*.5+.5)*255+.5,0,255))
    # Four-texel island gutters keep bilinear sampling from picking an unrelated
    # UV island/background, matching the existing offline atlas convention.
    distance, nearest = distance_transform_edt(~valid, return_indices=True)
    gutter = (~valid)&(distance<=4)
    for name, image in (('basecolor', base), ('normal', normal), ('orm', orm)):
        image[gutter] = image[nearest[0][gutter], nearest[1][gutter]]
        target = out/('mobile_skin_'+name+'.png'); Image.fromarray(image).save(target); result[name] = target
    (out/'bake_quality.json').write_text(json.dumps(dict(schema='vhuman.bake_quality.v1',
        seams=seams,hair=hair_report,normal_blending='normalized tangent-space vectors',
        limitations=['unseen albedo remains completed, not recovered','scalp crown remains an authored prior']),indent=2))
    return result
