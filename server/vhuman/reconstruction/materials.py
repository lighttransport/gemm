"""Conservative diffuse estimator and camera-visible UV transfer.

Exposure gauge: median diffuse illumination equals one. Roughness/F0 are
artist priors; one portrait cannot identify them. Unobserved texels are labelled.
"""
import json
import numpy as np
from PIL import Image
from ..rig import bake
from ..rig.common import vertex_normals, normalize
from .reference import srgb_to_linear, linear_to_srgb, rasterize


def estimate(rgb, normals, confidence):
    rgb = srgb_to_linear(np.asarray(rgb)/255.)
    n = np.asarray(normals, float)
    keep = (confidence > .5) & (rgb.mean(-1) > .025) & (rgb.max(-1) < .9)
    light = np.array([1., 0., 0., 0.])
    if keep.sum() >= 64:
        a = np.column_stack((np.ones(keep.sum()), n[keep]))
        y = np.log(np.maximum(rgb[keep].mean(-1), 1e-4))
        # Low-order achromatic lighting in log space, ridge regularized.
        for _ in range(4):
            err = y-a@light
            w = np.minimum(1., .15/np.maximum(abs(err), 1e-6))
            light = np.linalg.solve(a.T@(w[:, None]*a)+np.diag([1e-6, 20, 20, 20]), a.T@(w*y))
    illum = np.exp(np.clip(n @ light[1:], -.5, .5))
    if keep.any():
        illum /= np.median(illum[keep])
    illum = np.clip(illum,.5,2.)
    albedo = np.clip(rgb / illum[..., None], 0, 1)
    return albedo, dict(log_direction=light[1:].tolist(), gauge='median visible diffuse irradiance = 1',
                         fitted_pixels=int(keep.sum()), max_gain=float((1/illum).max()))


def complete_surface(points, normals, colors, measured, max_distance=.02):
    """Bounded geometric completion; far/normal-incompatible skin gets a median.

    Never copies nearest UV-island colors into ears/neck. Completion is low-detail
    and remains unobserved; no hidden texture is claimed to have been recovered.
    """
    from scipy.spatial import cKDTree
    visible = np.flatnonzero(measured)
    if not len(visible):
        raise ValueError('surface completion needs visible samples')
    out = colors.copy()
    missing = np.flatnonzero(~measured)
    fallback = np.median(colors[visible],axis=0)
    if not len(missing):
        return out
    tree = cKDTree(points[visible])
    distance,local = tree.query(points[missing],k=min(8,len(visible)))
    distance,local = distance.reshape(len(missing),-1),local.reshape(len(missing),-1)
    ids = visible[local]
    agreement = (normals[missing,None]*normals[ids]).sum(-1)
    good = (distance<max_distance)&(agreement>.5)
    weights = good*np.maximum(agreement,0)**4/np.maximum(distance,.001)**2
    total = weights.sum(1)
    near = (colors[ids]*weights[...,None]).sum(1)/np.maximum(total[:,None],1e-12)
    nearest = np.min(np.where(good,distance,np.inf),axis=1)
    blend = np.clip(1-nearest/max_distance,0,1)
    out[missing] = near*blend[:,None]+fallback*(1-blend[:,None])
    return out


def bake_portrait(vertices, triangles, triangle_uvs, views, cameras, out, res=512, roughness=.55, f0=.028):
    if not .08<=roughness<=1. or not .005<=f0<=.04:
        raise ValueError("roughness must be .08..1; F0 .005...04")
    from scipy.ndimage import distance_transform_edt
    out.mkdir(parents=True, exist_ok=True)
    # Split UV corners, preserving geometric correspondence.
    uv = triangle_uvs.reshape(-1, 2)
    tt = np.arange(len(uv)).reshape(-1, 3)
    ids, bary = bake.rasterize_uv(uv, tt, res)
    covered = ids >= 0
    yy, xx = np.nonzero(covered)
    t = ids[yy, xx]
    b = bary[yy, xx]
    neutral = vertices[0] if vertices.ndim == 3 else vertices
    p = (neutral[triangles[t]]*b[..., None]).sum(1)
    neutral_points = p.copy()
    neutral_normals = normalize((vertex_normals(neutral,triangles)[triangles[t]]*b[...,None]).sum(1))
    accum = np.zeros((len(p), 3))
    weight = np.zeros(len(p))
    lighting = []
    observations, normals, directions, confidences = [], [], [], []
    for vi, (view, cam) in enumerate(zip(views, cameras)):
        surface = vertices[vi] if vertices.ndim == 3 else vertices
        p = (surface[triangles[t]]*b[..., None]).sum(1)
        vn = vertex_normals(surface, triangles)
        n = normalize((vn[triangles[t]]*b[..., None]).sum(1))
        image = np.asarray(Image.open(view['image_path']).convert('RGBA'))
        h, w = image.shape[:2]
        # Bounded reference raster resolution; depth agrees in metric frame.
        size = (min(w, 512), min(h, 512))
        from .reference import Camera
        sx, sy = size[0]/w, size[1]/h
        if abs(sx-sy) > .01:
            size = (max(1, round(w*min(sx,sy))), max(1, round(h*min(sx,sy))))
        s = size[0]/w
        smallcam = cam.scaled(s)
        _, _, depth = rasterize(surface, triangles, smallcam, size)
        pixels, z = cam.project(p)
        ix, iy = np.floor(pixels[:,0]).astype(int), np.floor(pixels[:,1]).astype(int)
        valid = (ix>=0)&(ix<w)&(iy>=0)&(iy<h)&(z>0)
        ix, iy = np.clip(ix,0,w-1), np.clip(iy,0,h-1)
        dx = np.clip((pixels[:,0]*s).astype(int),0,size[0]-1)
        dy = np.clip((pixels[:,1]*s).astype(int),0,size[1]-1)
        viewdir = normalize(cam.origin-p)
        agree = np.maximum((n*viewdir).sum(1), 0)
        conf = valid*(abs(depth[dy,dx]-z)<.002)*agree*(image[iy,ix,3]/255.)
        if view.get('exclusion_mask_path'):
            mask = np.asarray(Image.open(view['exclusion_mask_path']).convert('L'))
            if mask.shape != (h,w):
                raise ValueError('exclusion mask dimensions mismatch')
            conf *= 1-mask[iy,ix]/255.
        rgb, light = estimate(image[iy,ix,:3], n, conf)
        calibration=view.get("lighting")
        observations.append(srgb_to_linear(image[iy,ix,:3]/255.) / (calibration.get("exposure",1) if calibration else 1))
        normals.append(n);directions.append(viewdir);confidences.append(conf)
        accum += rgb*conf[:,None]
        weight += conf
        lighting.append(light)
    reflectance = dict(status='artist prior',reason='calibrated multi-light observations absent')
    if len(views)>=3 and all(v.get('lighting') for v in views):
        from .reflectance import fit
        try:
            fitted,roughness,f0,reflectance=fit(np.array(observations),np.array(normals),np.array(directions),
                np.array(confidences),[v['lighting'] for v in views],roughness,f0)
            accum=fitted*weight[:,None]
        except ValueError as exc:
            reflectance=dict(status='artist prior',reason=str(exc))
    measured = weight > .1
    if not measured.any():
        raise ValueError('no visible skin texels; check camera and winding')
    color = np.zeros((res,res,3), float)
    color[yy,xx] = complete_surface(neutral_points,neutral_normals,
                                  accum/np.maximum(weight[:,None],1e-9),measured)
    observed = np.zeros((res,res), bool)
    observed[yy[measured],xx[measured]] = True
    # UV gutter padding only. Covered hidden skin was completed in metric space.
    _, nearest = distance_transform_edt(~covered, return_indices=True)
    color[~covered] = color[nearest[0][~covered],nearest[1][~covered]]
    base = np.uint8(np.clip(linear_to_srgb(color)*255+.5,0,255))
    orm = np.tile(np.array([255,round(roughness*255),0],np.uint8),(res,res,1))
    normal = np.tile(np.array([128,128,255],np.uint8),(res,res,1))
    spec = np.full((res,res),round(f0/.04*255),np.uint8)
    confidence = np.zeros((res,res),np.uint8)
    confidence[yy,xx] = np.uint8(np.clip(weight,0,1)*255)
    for name, data in [('basecolor',base),('orm',orm),('normal',normal),('specular',spec),
                       ('coverage',observed.astype(np.uint8)*255),('confidence',confidence)]:
        Image.fromarray(data).save(out/f'skin_{name}.png')
    manifest = dict(format='vhuman.skin_material.v1', units='metres',
                    maps={k:f'skin_{k}.png' for k in ['basecolor','orm','normal','specular','coverage','confidence']},
                    semantics=dict(basecolor='sRGB albedo estimate',orm='linear R=AO G=roughness B=metallic',
                                   normal='linear tangent-space +Y',specular='linear scalar intensity; exporter packs glTF alpha, F0=.04*A',
                                   coverage='observed only, not completed',confidence='visible fit confidence'),
                    lighting=lighting, reflectance=reflectance,roughness=dict(value=roughness,status=reflectance['status']),
                    f0=dict(value=f0,status=reflectance['status']), sss=dict(enabled=False,radii_m=[.0012,.0006,.0003],
                    status='authored profile, not measured anatomy'),
                    observed_texels=int(observed.sum()),covered_texels=int(covered.sum()),
                    completion=dict(method='normal-aware surface-space weighted completion to median skin',max_distance_m=.02),
                    limitations=['single-view lighting/albedo ambiguity','bounded low-detail completion; no hidden detail recovery',
                                 'pores/wrinkles are separate authored detail; no material predictor'])
    (out/'skin_material.json').write_text(json.dumps(manifest,indent=2))
    return manifest
