"""Linear EXR and polarized Emily reference processing with explicit light gauge.

Camera matrices are treated as camera-to-world, looking down local -Z. Their
centimetre translations match the OBJ's authored scale. Flash power/exposure is
not supplied by the camera files: relative fits are explicitly diagnostics.
"""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image
from .observations import sha256
from .reference import Camera,linear_to_srgb,ggx,rasterize
from .prepare_datasets import exr_header,unique
from .artifacts import artifact_path


def read_linear(path,max_side=1024):
    import OpenEXR
    window=exr_header(path)['data_window'];pixels=(window[2]-window[0]+1)*(window[3]-window[1]+1)
    if pixels<=0 or pixels>32*1024**2:raise ValueError('EXR exceeds 32M-pixel decode budget')
    with OpenEXR.File(str(path)) as f:
        if len(f.parts)!=1:raise ValueError('multipart EXR not supported')
        channels=f.channels();key='RGB' if 'RGB' in channels else 'RGBA'
        image=channels[key].pixels
        if image.ndim!=3 or image.shape[-1]<3:raise ValueError('RGB EXR required')
        stride=max(1,int(np.ceil(max(image.shape[:2])/max_side)))
        rgb=image[::stride,::stride,:3].astype(np.float32)
    if not np.isfinite(rgb).all():raise ValueError('nonfinite linear EXR values')
    return rgb,stride,dict(file=str(Path(path).resolve()),sha256=sha256(path),data_window=window,
                           negative_fraction=float((rgb<0).mean()),decoder='OpenEXR 3.4.4; no sRGB decode')


def calibration(path):
    text=Path(path).read_text()
    def section(name,count):
        match=re.search(r'#'+re.escape(name)+r'\s*\n([^#]+)',text)
        if not match:raise ValueError('missing Emily camera section '+name)
        values=list(map(float,match[1].split()))
        if len(values)!=count:raise ValueError('invalid Emily camera section '+name)
        return np.array(values)
    focal=section('focal length',2);size=section('resolution',2).astype(int)
    centre=section('principal point',2);distortion=section('distortion coeffs',4)
    matrix=np.array([list(map(float,line.split())) for line in text.split('MATRIX :')[1].strip().splitlines()[:4]])
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1]):
        raise ValueError('invalid Emily camera-to-world matrix')
    cam=Camera.from_dict(dict(focal=focal[0],focal_y=focal[1],cx=centre[0],cy=centre[1],
                             origin=(matrix[:3,3]*.01).tolist(),rotation=matrix[:3,:3].T.tolist()))
    return cam,size,distortion


def distort(xy,camera,coefficients):
    k1,k2,p1,p2=coefficients
    fy=camera.focal if camera.focal_y is None else camera.focal_y
    x=(xy[...,0]-camera.cx)/camera.focal;y=(xy[...,1]-camera.cy)/fy
    r=x*x+y*y;gain=1+k1*r+k2*r*r
    xd=x*gain+2*p1*x*y+p2*(r+2*x*x)
    yd=y*gain+p1*(r+2*y*y)+2*p2*x*y
    return np.stack((xd*camera.focal+camera.cx,yd*fy+camera.cy),-1)


def skin_mesh(path):
    positions=[];uvs=[];tri=[];uv=[];skin=False
    with Path(path).open() as f:
        for line in f:
            parts=line.split()
            if not parts:continue
            if parts[0]=='v':positions.append(list(map(float,parts[1:4])))
            elif parts[0]=='vt':uvs.append(list(map(float,parts[1:3])))
            elif parts[0]=='usemtl':skin=parts[1]=='Skin_Blend_01'
            elif parts[0]=='f' and skin:
                corners=[p.split('/') for p in parts[1:]]
                for i in range(1,len(corners)-1):
                    c=[corners[j] for j in (0,i,i+1)]
                    tri.append([int(p[0])-1 if int(p[0])>0 else len(positions)+int(p[0]) for p in c])
                    uv.append([int(p[1])-1 if int(p[1])>0 else len(uvs)+int(p[1]) for p in c])
    p=np.asarray(positions,np.float32)*.01;t=np.asarray(tri,np.int32)
    if not len(t):raise ValueError('Emily skin material missing')
    ids,remap=np.unique(t,return_inverse=True)
    return p[ids],remap.reshape(-1,3).astype(np.int32),np.asarray(uvs,np.float32)[np.asarray(uv,np.int32)]


def fit_regions(rgb,normals,viewdirs,confidence,lights,regions,roughness=.55,f0=.028):
    """Spatially varying calibrated GGX, only when physical radiance is supplied."""
    from .reflectance import fit
    result=np.tile([roughness,f0],(rgb.shape[1],1));albedo=np.zeros_like(rgb[0]);reports={}
    for region in np.unique(regions):
        ids=np.flatnonzero(regions==region)
        try:
            a,r,f,report=fit(rgb[:,ids],normals[:,ids],viewdirs[:,ids],confidence[:,ids],lights,roughness,f0)
            albedo[ids]=a;result[ids]=[r,f]
        except ValueError as exc:report=dict(status='artist prior',reason=str(exc))
        reports[str(int(region))]=report
    return albedo,result,reports


def run(root,out,camera_ids=(1,3,5)):
    from ..rig.common import vertex_normals
    from ..rig.source_lod import simplify
    from scipy.optimize import least_squares
    root,out=Path(root),artifact_path(out)
    if out.exists():raise ValueError('choose a fresh Emily output directory')
    source=root/'emily/extracted'
    mesh=unique(source/'Emily_2_1_OBJ','*.obj');p,t,uv=skin_mesh(mesh)
    normals=vertex_normals(p,t)
    indices=np.arange(0,len(t),max(1,len(t)//6000))
    points=p[t[indices]].mean(1);n=normals[t[indices]].mean(1);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
    tex=uv[indices].mean(1)
    coarse,ct,_,_,lod=simplify(p,t,.004)
    out.mkdir(parents=True)
    sources=[];samples=[];confidence=[];directions=[];camera_checks=[]
    for camera_id in camera_ids:
        cam,size,dist=calibration(source/'DigitalEmily2_Calibration'/f'camera{camera_id:02d}.txt')
        xy,z=cam.project(points);raw_xy=distort(xy,cam,dist)
        resolution=tuple(round(s*256/max(size)) for s in size)
        tid,_,depth=rasterize(coarse,ct,cam.scaled(256/max(size)),resolution)
        ix=np.clip((xy[:,0]*256/max(size)).astype(int),0,resolution[0]-1)
        iy=np.clip((xy[:,1]*256/max(size)).astype(int),0,resolution[1]-1)
        v=cam.origin-points;distance=np.linalg.norm(v,axis=1);v/=distance[:,None]
        visible=(z>0)&(raw_xy[:,0]>=0)&(raw_xy[:,0]<size[0])&(raw_xy[:,1]>=0)&(raw_xy[:,1]<size[1])
        visible&=(abs(depth[iy,ix]-z)<.008)&((n*v).sum(1)>.15)
        channels={}
        for kind in ('FlashCross','FlashParallel','SpecularOnly'):
            path=unique(source/('DigitalEmily2_'+kind),f'cam{camera_id}_*.exr')
            image,stride,record=read_linear(path);sources.append(record)
            x=np.clip((raw_xy[:,0]/stride).astype(int),0,image.shape[1]-1)
            y=np.clip((raw_xy[:,1]/stride).astype(int),0,image.shape[0]-1)
            channels[kind]=image[y,x]
            if kind=='FlashCross':
                preview=np.uint8(np.clip(linear_to_srgb(np.maximum(image,0)/max(np.quantile(image,.995),1e-6)),0,1)*255)
                # Overlay projected visible mesh samples in raw distorted coordinates.
                preview[y[visible],x[visible]]=[0,255,0]
                Image.fromarray(preview).save(out/f'camera_{camera_id}_projection.png')
        samples.append(channels);confidence.append(visible);directions.append(v)
        camera_checks.append(dict(camera_id=camera_id,visible_samples=int(visible.sum()),
                                  distortion=dist.tolist(),camera=cam.as_dict(),calibration_sha256=sha256(source/'DigitalEmily2_Calibration'/f'camera{camera_id:02d}.txt')))
    confidence=np.asarray(confidence);directions=np.asarray(directions)
    rgb_cross=np.array([s['FlashCross'] for s in samples]);rgb_parallel=np.array([s['FlashParallel'] for s in samples]);rgb_spec=np.array([s['SpecularOnly'] for s in samples])
    # Point-light power and exposure are unknown: retain an explicit relative gauge.
    # Fit a view-dependent cross/spec ratio with regional gains; no absolute F0 claim.
    regions=(tex[:,0]>.5).astype(int)+2*(tex[:,1]>.5).astype(int)
    material=np.tile([.55,.028],(len(points),1));reports={}
    nl=np.maximum((n[None]*directions).sum(-1),1e-5)
    for region in np.unique(regions):
        ids=np.flatnonzero((regions==region)&confidence.all(0)&(rgb_cross.mean(-1)>.03).all(0)&(n[:,2]>.1))
        if len(ids)<64:
            reports[str(region)]=dict(status='insufficient common visible samples',samples=len(ids));continue
        observed=np.maximum(rgb_spec[:,ids].mean(-1),0)/np.maximum(rgb_cross[:,ids].mean(-1),.01)
        # Gains fit only the first two cameras; the third camera is untouched.
        def predict(parameters):
            _,spec=ggx(np.ones_like(n[ids]),n[ids][None],directions[:,ids],directions[:,ids],parameters[0],.028)
            shape=spec[...,0]/nl[:,ids]
            return parameters[1]*shape
        solve=least_squares(lambda a:(predict(a)[:-1]-observed[:-1]).ravel(),[.55,1.],
                            bounds=([.08,.001],[1.,100.]),loss='soft_l1',f_scale=.05,max_nfev=100)
        prior_gain=least_squares(lambda a:(predict([.55,a[0]])[:-1]-observed[:-1]).ravel(),[1.],
                                bounds=([.001],[100.]),loss='soft_l1',f_scale=.05,max_nfev=100)
        baseline=float(np.mean((predict([.55,prior_gain.x[0]])[-1]-observed[-1])**2))
        error=float(np.mean((predict(solve.x)[-1]-observed[-1])**2))
        accepted=bool(solve.success and error<baseline*.98 and np.linalg.cond(solve.jac.T@solve.jac)<1e7)
        if accepted:material[regions==region,0]=solve.x[0]
        reports[str(region)]=dict(status='relative roughness diagnostic accepted' if accepted else 'artist prior retained',
                                   samples=len(ids),roughness=float(solve.x[0]),relative_gain=float(solve.x[1]),
                                   baseline_gain=float(prior_gain.x[0]),
                                   heldout_before=baseline,heldout_after=error,f0=.028,
                                   gauge='fixed artist F0; unknown flash power/exposure/polarization gains')
    np.savez_compressed(out/'linear_samples.npz',positions=points,normals=n,uv=tex,confidence=confidence,
                        cross=rgb_cross,parallel=rgb_parallel,specular=rgb_spec,material=material,regions=regions)
    result=dict(format='vhuman.emily_material_diagnostic.v1',status='linear references processed',sources=sources,
                mesh_sha256=sha256(mesh),camera_checks=camera_checks,coarse_visibility_mesh=lod,regions=reports,
                linear_samples_sha256=sha256(out/'linear_samples.npz'),
                limitations=['camera-to-world/-Z/centimetres convention is a tested dataset interpretation, not new measured calibration',
                             'no physical flash radiance/exposure supplied; absolute F0 and calibrated SSS are not recovered',
                             'relative roughness uses authored polarization differences and fixed F0; acceptance is diagnostic',
                             'raw EXR negatives recorded; nonnegative clipping is restricted to fitting/preview',
                             'coarse visibility mesh and sampled triangles bound the experiment'])
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('/mnt/nvme02/data/vhuman'));parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();print(json.dumps(run(a.root,a.out)))


if __name__=='__main__':main()
