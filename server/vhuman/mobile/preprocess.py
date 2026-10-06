"""Fit geometric wrinkle activation and bake bounded tangent-space slopes.

Supervision is GNM regional area strain, not image-derived wrinkle depth.
Train/test coefficients use disjoint deterministic random seeds. No captured
or generated-video pixels become ground-truth displacement.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from .export import validate_package
from ..reconstruction.provenance import validate_candidate
from ..reconstruction.observations import sha256
from ..reconstruction.skin_detail import REGIONS
from ..rig.gnm_model import GNMModel
from ..rig.bake import rasterize_uv
from .baking import chart_labels, chart_gradient


def regional_strain(reference, basis, triangles, regions, coefficients):
    mask = regions[:, triangles].mean(-1)
    def areas(vertices):
        p = vertices[triangles]
        return np.linalg.norm(np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0]), axis=-1)
    denominator = mask @ areas(reference)
    if np.any(denominator < 1e-10): raise ValueError('unobservable detail region')
    result = []
    for start in range(0, len(coefficients), 8):
        positions = coefficients[start:start+8] @ basis.reshape(len(basis), -1)
        for p in positions.reshape(-1, *reference.shape):
            result.append(np.clip(6*(1-(mask @ areas(reference+p))/denominator), -1, 1))
    return np.asarray(result)


def features(delta, prior):
    return np.concatenate((delta, (delta @ prior.T)**2), axis=-1)


def fit_driver(train, target, test, expected, prior):
    x = features(train, prior)
    scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-5)
    normalized = x/scale
    # Zero intercept keeps the captured reference's dynamic detail exactly zero.
    ridge = len(x)*.002
    weights = np.linalg.solve(normalized.T@normalized+ridge*np.eye(x.shape[1]), normalized.T@target)/scale[:,None]
    predict = np.clip(features(test, prior)@weights, -1, 1)
    baseline = np.clip(test@prior.T, -1, 1)
    metric = lambda a: dict(mae=float(np.mean(abs(a-expected))), p95=float(np.percentile(abs(a-expected),95)))
    scores = dict(trained=metric(predict), analytic=metric(baseline))
    scores['accepted'] = scores['trained']['mae'] < scores['analytic']['mae'] and scores['trained']['p95'] < .05
    return weights, scores


def tangent_slopes(geometry, fields, resolution):
    """dh/ds along orthonormal UV tangents, accounting for UV metric/shear."""
    uv = geometry['triangle_uvs']; triangles = geometry['triangles']
    ids, _ = rasterize_uv(uv.reshape(-1,2), np.arange(uv.size//2).reshape(-1,3), resolution)
    p = geometry['neutral'][triangles]; e1=p[:,1]-p[:,0]; e2=p[:,2]-p[:,0]
    u=uv[:,1]-uv[:,0]; v=uv[:,2]-uv[:,0]; determinant=u[:,0]*v[:,1]-u[:,1]*v[:,0]
    safe=np.where(abs(determinant)>1e-12,determinant,1)
    dpdu=(e1*v[:,1,None]-e2*u[:,1,None])/safe[:,None]
    dpdv=(-e1*v[:,0,None]+e2*u[:,0,None])/safe[:,None]
    tu=np.linalg.norm(dpdu,axis=1); tangent=dpdu/np.maximum(tu[:,None],1e-10)
    shear=(dpdv*tangent).sum(1); tv=np.linalg.norm(dpdv-shear[:,None]*tangent,axis=1)
    valid=(ids>=0); selected=ids[valid]
    valid[valid]=(abs(determinant[selected])>1e-12)&(tu[selected]>1e-6)&(tv[selected]>1e-6)
    selected=ids[valid]
    face_labels=chart_labels(triangles,uv)
    labels=np.where(valid,face_labels[np.maximum(ids,0)],-1)
    source_resolution=fields.shape[-1]
    source_ids,_=rasterize_uv(uv.reshape(-1,2),np.arange(uv.size//2).reshape(-1,3),source_resolution)
    source_coverage=(source_ids>=0).astype(np.float32)
    filtered_coverage=np.asarray(Image.fromarray(source_coverage).resize((resolution,resolution),Image.Resampling.BILINEAR))
    _, near=distance_transform_edt(~valid,return_indices=True)
    output=np.zeros((len(fields),resolution,resolution,2),np.float32)
    for index, field in enumerate(fields):
        # Normalize coverage before differentiating to avoid height-to-zero
        # discontinuities along atlas boundaries.
        height=np.asarray(Image.fromarray((field*source_coverage).astype(np.float32)).resize((resolution,resolution),Image.Resampling.BILINEAR))
        height=height/np.maximum(filtered_coverage,1e-8)
        dv,du=chart_gradient(height,labels,1/resolution)
        sx=du[valid]/tu[selected]
        sy=(dv[valid]-shear[selected]*sx)/tv[selected]
        output[index,valid,0]=sx;output[index,valid,1]=sy
        output[index,~valid]=output[index,near[0][~valid],near[1][~valid]]
    return np.clip(output,-.35,.35)


def encode_slopes(slopes, limit=.35):
    """Filterable RG8, exact zero at 128, bounded signed slope range."""
    return np.uint8(np.clip(np.round(np.clip(slopes,-limit,limit)/limit*127)+128,1,255))


def preprocess(candidate, package, out, samples=768, resolution=256):
    if not 512<=samples<=4096 or resolution not in (128,256,512): raise ValueError('invalid preprocessing budget')
    candidate,package,out=map(Path,(candidate,package,out))
    source=validate_candidate(candidate); manifest=validate_package(package)
    if source['geometry_sha256']!=manifest['source_geometry_sha256']: raise ValueError('identity mismatch')
    if out.exists() and any(out.iterdir()): raise ValueError('preprocess output must be empty')
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z: geometry=dict(z)
    with np.load(package/'skin_detail.npz',allow_pickle=False) as z: detail=dict(z)
    if (detail['coefficient_to_activation'].shape!=(12,383) or detail['reference_expression'].shape!=(383,)
            or detail['dynamic_height_m'].ndim!=3 or len(detail['dynamic_height_m'])!=12
            or not all(np.isfinite(detail[name]).all() for name in ('coefficient_to_activation','reference_expression','dynamic_height_m'))
            or not np.allclose(detail['reference_expression'],geometry['gnm_expressions'][0],atol=1e-6)):
        raise ValueError('skin detail does not match the native reference/control layout')
    m=GNMModel();exterior=m.group('skin_exterior')
    regions=np.stack([m.group(name)[exterior] for name in REGIONS])
    basis=float(geometry['scale'])*m.data['expression_basis'][:,exterior]@geometry['rotation'].T
    reference=detail['reference_expression'];prior=detail['coefficient_to_activation'].astype(float)
    def sample(seed,count):
        delta=np.random.default_rng(seed).normal(0,.15,(count,383))
        return np.clip(reference+delta,-3,3)-reference
    train=sample(317,samples);test=sample(911,192)
    target=regional_strain(geometry['captured'][0],basis,geometry['triangles'],regions,train)
    expected=regional_strain(geometry['captured'][0],basis,geometry['triangles'],regions,test)
    weights,scores=fit_driver(train,target,test,expected,prior)
    slopes=tangent_slopes(geometry,detail['dynamic_height_m'],resolution)
    out.mkdir(parents=True,exist_ok=True)
    # Core WebGL2 filterable RG8; biased signed quantization preserves exact
    # zero. Unlike RG32F this needs no optional float-linear texture extension.
    encoded=encode_slopes(slopes)
    encoded.tofile(out/'wrinkle_slopes.u8')
    model=dict(schema='vhuman.strain_driver.v1',source_geometry_sha256=source['geometry_sha256'],
        package_sha256=sha256(package/'avatar.json'),source_detail_sha256=sha256(package/'skin_detail.npz'),
        reference=reference.tolist(),prior=prior.tolist(),weights=weights.T.tolist(),regions=list(REGIONS),
        feature_recipe='concat(delta383, square(prior12 @ delta383))',resolution=resolution,
        slope_limit=.35,activation_limit=1.,max_combined_slope=.5,trained_samples=samples,heldout_samples=len(test),
        slope_encoding='rg8_snorm_bias128',slope_file='wrinkle_slopes.u8',
        slope_quantization_max_error=float(abs((encoded.astype(float)-128)*(.35/127)-slopes).max()),
        gradient_method='coverage-normalized reduction; within-chart central/one-sided metric derivatives',
        train_seed=317,test_seed=911,coefficient_sigma=.15,metrics=scores,
        selected_driver='trained' if scores['accepted'] else 'analytic',
        supervision='synthetic native GNM regional surface-area strain; no photometric depth labels',
        detail_source='authored bounded groove/height prior; not learned wrinkle depth',
        limitations=['validation covers the sampled coefficient distribution, not all expressions',
            'no new identity, anatomical depth, or image-observed wrinkle accuracy is established'])
    model['files']={'wrinkle_slopes.u8':dict(sha256=sha256(out/'wrinkle_slopes.u8'),bytes=(out/'wrinkle_slopes.u8').stat().st_size)}
    (out/'detail.json').write_text(json.dumps(model,separators=(',',':')))
    np.savez_compressed(out/'heldout.npz',delta=test,target=expected,predicted=np.clip(features(test,prior)@weights,-1,1))
    return dict(out=str(out),selected_driver=model['selected_driver'],metrics=scores,texture_bytes=encoded.nbytes,
        slope_quantization_max_error=model['slope_quantization_max_error'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','package','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--samples',type=int,default=768);p.add_argument('--resolution',type=int,default=256)
    print(json.dumps(preprocess(**vars(p.parse_args())),indent=2))


if __name__=='__main__':main()
