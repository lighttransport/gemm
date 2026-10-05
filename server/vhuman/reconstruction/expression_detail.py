"""Registered I2V appearance residuals with a withheld-expression acceptance gate.

These maps are synthetic appearance estimates. They do not measure skin depth,
reflectance or wrinkles. Physical groove heights remain separate artist priors.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .reference import Camera, srgb_to_linear, rasterize
from .skin_detail import build, evaluate
from .observations import sha256


def learn(candidate, motion_root, catalog, out, *, res=256, heldout='disgust'):
    import cv2
    from scipy.ndimage import gaussian_filter
    from ..rig.bake import rasterize_uv
    from ..rig.gnm_model import GNMModel
    from ..rig.common import vertex_normals, normalize
    candidate,motion_root,catalog,out=map(Path,(candidate,motion_root,catalog,out))
    from .provenance import validate_candidate
    validate_candidate(candidate)
    if res not in (256,512):raise ValueError('appearance fitting atlas must be 256 or 512')
    out.mkdir(parents=True,exist_ok=True)
    detail=build(candidate,out,preset='mature',res=res)
    driver=np.load(out/'skin_detail.npz',allow_pickle=False)
    model=GNMModel();skin=np.flatnonzero(model.group('skin_exterior'))
    with np.load(candidate/'geometry.npz',allow_pickle=False) as data:
        tri=data['triangles'];uv=data['triangle_uvs'];identity=data['gnm_identity']
    tid,bary=rasterize_uv(uv.reshape(-1,2),np.arange(len(uv)*3).reshape(-1,3),res)
    yy,xx=np.nonzero(tid>=0);triangles=tri[tid[yy,xx]];barycentric=bary[yy,xx]
    samples=[];weights=[];features=[];groups=[];sources=[]
    expressions=('neutral','happy','sad','angry','fear','surprise','disgust')
    for name in expressions:
        path=motion_root/name
        if not (path/'motion.json').is_file():path=path/'native_motion'
        track=json.loads((path/'motion.json').read_text())
        if not track['quality_gate']:continue
        if track['candidate_geometry_sha256']!=sha256(candidate/'geometry.npz'):
            raise ValueError('motion reconstruction mismatch')
        motion=np.load(path/'motion.npz',allow_pickle=False)
        if not np.array_equal(motion['identity'],identity):raise ValueError('expression identity changed')
        camera=Camera.from_dict(track['camera']);w,h=track['size']
        parsing=np.load(catalog/name/'face_parsing.npz',allow_pickle=False)['labels']
        cap=cv2.VideoCapture(track['clip']);i=0
        try:
            while True:
                ok,image=cap.read()
                if not ok:break
                if i%3:
                    i+=1;continue
                vertices=motion['vertices'][i,skin]
                points=(vertices[triangles]*barycentric[:,:,None]).sum(1)
                normals=normalize((vertex_normals(vertices,tri)[triangles]*barycentric[:,:,None]).sum(1))
                projected,z=camera.project(points)
                pxi=np.clip(projected[:,0].astype(int),0,w-1);pyi=np.clip(projected[:,1].astype(int),0,h-1)
                valid=(projected[:,0]>=0)&(projected[:,0]<w)&(projected[:,1]>=0)&(projected[:,1]<h)&(z>.02)
                small=(max(1,round(w*128/h)),128);factor=128/h
                _,_,depth=rasterize(vertices,tri,camera.scaled(factor),small)
                dx=np.clip((projected[:,0]*factor).astype(int),0,small[0]-1);dy=np.clip((projected[:,1]*factor).astype(int),0,127)
                confidence=valid*(abs(depth[dy,dx]-z)<.004)*np.maximum((normals*normalize(camera.origin-points)).sum(-1),0)
                confidence*=np.isin(parsing[i,pyi,pxi],[1,2,3,10,12,13])
                rgb=srgb_to_linear(image[pyi,pxi,::-1]/255)
                texture=np.zeros((res,res,3));texture[yy,xx]=rgb
                conf=np.zeros((res,res));conf[yy,xx]=confidence
                # Remove broad color/exposure changes; retain only registered
                # fine appearance. Normalize blur by observed support at borders.
                smooth=gaussian_filter(texture*conf[:,:,None],(4,4,0))/np.maximum(gaussian_filter(conf,4)[:,:,None],1e-6)
                residual=np.clip((texture-smooth)[yy,xx],-.1,.1)
                samples.append(residual);weights.append(confidence)
                features.append(evaluate(driver['coefficient_to_activation'],motion['expression'][i],driver['reference_expression']))
                groups.append(name);i+=1
        finally:cap.release()
        sources.append(dict(expression=name,motion_sha256=sha256(path/'motion.npz'),clip_sha256=track['clip_sha256']))
        print(f'{name}: registered appearance samples',flush=True)
    values=np.asarray(samples);weight=np.asarray(weights);x=np.asarray(features);groups=np.asarray(groups)
    train=groups!=heldout;test=groups==heldout
    if not train.any() or not test.any():raise ValueError('appearance fitting requires training and withheld expression')
    # Remove the weighted static texture; it is not an expression channel.
    static=(values[train]*weight[train,:,None]).sum(0)/np.maximum(weight[train].sum(0)[:,None],1e-6)
    residual=values-static[None]
    gram=np.einsum('fr,fp,fs->prs',x[train],weight[train],x[train])
    rhs=np.einsum('fr,fp,fpc->prc',x[train],weight[train],residual[train])
    ridge=np.eye(12)[None]*2
    coefficients=np.clip(np.linalg.solve(gram+ridge,rhs),-.04,.04)
    prediction=np.einsum('fr,prc->fpc',x[test],coefficients)
    baseline=float((residual[test]**2*weight[test,:,None]).sum()/max(weight[test].sum()*3,1e-6))
    fitted=float(((residual[test]-prediction)**2*weight[test,:,None]).sum()/max(weight[test].sum()*3,1e-6))
    accepted=bool(fitted<baseline*.95 and weight[test].sum()>100)
    maps=np.zeros((12,res,res,3),np.float32)
    if accepted:maps[:,yy,xx]=np.clip(coefficients.transpose(1,0,2),-.04,.04)
    np.savez_compressed(out/'expression_appearance.npz',linear_albedo_delta=maps,
        coefficient_to_activation=driver['coefficient_to_activation'],reference_expression=driver['reference_expression'])
    result=dict(schema='vhuman.expression_appearance.v1',file='expression_appearance.npz',accepted=accepted,
        candidate_geometry_sha256=sha256(candidate/'geometry.npz'),res=res,channels=12,
        heldout_expression=heldout,baseline_mse=baseline,fitted_mse=fitted,minimum_improvement=.05,
        sources=sources,observations='synthetic I2V RGB registered to native GNM',
        units='linear RGB albedo delta estimate',physical_depth_measured=False,
        fallback='zero appearance deltas when withheld gate fails; static portrait plus authored groove drivers retained',
        limitations=['generated lighting and identity drift can contaminate appearance',
        'single-source synthetic holdout is a consistency test, not photometric ground truth'])
    (out/'expression_appearance.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate');parser.add_argument('--motion-root',required=True)
    parser.add_argument('--catalog',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();print(json.dumps(learn(**vars(args)),indent=2))


if __name__=='__main__':main()
