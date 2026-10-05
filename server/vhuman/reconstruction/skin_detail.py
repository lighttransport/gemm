"""Metric skin height and compact strain-driven expression detail.

Photo crease masks shape bounded artist groove profiles. Luminance is never
interpreted as measured depth. The expression strain drivers are geometric
priors, retained separately from observations and any later data fitting.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image

REGIONS = ('forehead_region','middle_brow_region','left_orbital_region','right_orbital_region',
    'left_infraorbital_region','right_infraorbital_region','nose_region','left_cheek_region',
    'right_cheek_region','upper_lip_region','lower_lip_region','chin_region')


def driver_matrix(vertices, triangles, basis, regions):
    """Linearized regional areal compression, with unobservable modes zero."""
    v=vertices[triangles];a=v[:,1]-v[:,0];b=v[:,2]-v[:,0]
    n=np.cross(a,b);area=np.linalg.norm(n,axis=-1);unit=n/np.maximum(area[:,None],1e-12)
    weights=regions[:,triangles].mean(-1)*area[None]
    result=np.zeros((len(regions),len(basis)),np.float32)
    for e,delta in enumerate(basis):
        d=delta[triangles]
        derivative=(unit*(np.cross(d[:,1]-d[:,0],b)+np.cross(a,d[:,2]-d[:,0]))).sum(-1)/np.maximum(area,1e-12)
        result[:,e]=-6*(weights@derivative)/np.maximum(weights.sum(-1),1e-12)
    return np.clip(result,-1,1)


def evaluate(weights, coefficients, reference):
    activation=np.clip(np.asarray(weights)@(np.asarray(coefficients)-reference),-1,1)
    if not np.isfinite(activation).all():raise ValueError('nonfinite wrinkle activation')
    return activation


def build(candidate, out=None, *, preset='mature', res=None):
    from scipy.ndimage import gaussian_filter
    from ..rig.bake import rasterize_uv
    from ..rig.gnm_model import GNMModel
    from .reference import srgb_to_linear
    candidate=Path(candidate);out=Path(out or candidate)
    if preset not in ('source','mature'):raise ValueError('detail preset must be source or mature')
    out.mkdir(parents=True,exist_ok=True)
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:
        vertices=z['neutral'];tri=z['triangles'];uv=z['triangle_uvs']
        reference=z['gnm_expressions'][0] if 'gnm_expressions' in z else np.zeros(383)
        scale=float(z['scale']);rotation=z['rotation']
    base=Image.open(candidate/'skin_basecolor.png').convert('RGB')
    res=res or base.width
    if res not in (256,512,1024,2048):raise ValueError('invalid detail atlas size')
    color=srgb_to_linear(np.asarray(base.resize((res,res)),np.float32)/255)
    confidence=np.asarray(Image.open(candidate/'skin_confidence.png').resize((res,res)),np.float32)/255
    tid,bary=rasterize_uv(uv.reshape(-1,2),np.arange(len(uv)*3).reshape(-1,3),res)
    yy,xx=np.nonzero(tid>=0);ids=tid[yy,xx]
    points=(vertices[tri[ids]]*bary[yy,xx,:,None]).sum(1)
    model=GNMModel();exterior=model.group('skin_exterior')
    regions=np.stack([model.group(name)[exterior].astype(float) for name in REGIONS])
    masks=np.zeros((len(REGIONS),res,res),np.float32)
    masks[:,yy,xx]=(regions[:,tri[ids]]*bary[yy,xx][None]).sum(-1)
    # Preserve contacts: no height across lids or lip margins.
    protected=(model.group('eye_sockets')|model.group('upper_lip')|model.group('lower_lip'))[exterior]
    protection=np.ones((res,res),np.float32)
    protection[yy,xx]=1-(protected[tri[ids]]*bary[yy,xx]).sum(-1)
    luma=color@np.array([.2126,.7152,.0722])
    crease=np.clip((gaussian_filter(luma,4)-gaussian_filter(luma,1)-.015)/.08,0,1)
    crease*=confidence*protection
    # Groove profile amplitudes are authored, not photometric measurements.
    height=(-crease*1.5e-4).astype(np.float32)
    if preset=='mature':
        rng=np.random.default_rng(19);directions=rng.normal(size=(12,3));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        pore=np.zeros(len(points))
        wavelength=max(.0004,.8/res)
        for direction in directions:pore+=np.sin(points@direction*(2*np.pi/wavelength)+rng.uniform(0,2*np.pi))*1e-5/12
        height[yy,xx]+=pore*protection[yy,xx]
    height[tid<0]=0
    dynamic=-masks*(crease[None]*5e-5+1e-4)*protection[None]
    matrix=driver_matrix(vertices,tri,scale*model.data['expression_basis'][:,exterior]@rotation.T,regions)
    np.savez_compressed(out/'skin_detail.npz',height_m=height,dynamic_height_m=dynamic,
                        coefficient_to_activation=matrix,reference_expression=reference,region_masks=masks)
    manifest=dict(schema='vhuman.skin_detail.v1',file='skin_detail.npz',regions=list(REGIONS),
        units='metres',preset=preset,res=res,pore_amplitude_m=1e-5 if preset=='mature' else 0,
        total_height_limit_m=.0005,driver='clamp(M @ (GNM383 - reference), -1, 1)',
        provenance=dict(crease_pattern='photo-shaped authored groove estimate',pores='metric-space artist prior',
                        dynamic_wrinkles='regional GNM areal-compression prior; not trained from I2V'),
        reference_expression=reference.tolist(),limitations=['single-image depth is unobservable',
        'shadow and pigment may resemble creases; groove amplitude remains an artist control'])
    (out/'skin_detail.json').write_text(json.dumps(manifest,indent=2))
    return manifest
