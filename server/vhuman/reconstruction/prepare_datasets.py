"""Prepare bounded local evaluation fixtures; never download or train on media.

Multiface calibration follows the publisher's K @ [R_C R_f, R_C t_f+t_C]
projection. Same-topology reference surfaces are separate from predictions.
SpeakingFaces uses estimated annotations/cameras, not scan ground truth.
"""
import argparse
import json
from pathlib import Path
import struct
import numpy as np
from PIL import Image
from . import observations
from .reference import Camera, rasterize


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def unique(root, pattern):
    rows=sorted(Path(root).rglob(pattern))
    if len(rows)!=1:
        raise ValueError(f'expected one {pattern} under {root}, found {len(rows)}')
    return rows[0]


def load_krt(path):
    rows=[line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows)%8:
        raise ValueError('malformed KRT blocks')
    cameras={}
    for i in range(0,len(rows),8):
        k=np.array([list(map(float,line.split())) for line in rows[i+1:i+4]])
        dist=np.array(list(map(float,rows[i+4].split())))
        rt=np.array([list(map(float,line.split())) for line in rows[i+5:i+8]])
        if k.shape!=(3,3) or rt.shape!=(3,4) or not np.isfinite(k).all() or not np.isfinite(rt).all():
            raise ValueError('invalid KRT matrices')
        if not np.allclose(k[2],[0,0,1]) or not np.isclose(k[1,0],0) or not np.isfinite(dist).all():
            raise ValueError('unsupported KRT intrinsics')
        if np.any(dist!=0):
            raise ValueError('nonzero lens distortion requires explicit undistortion')
        if rows[i] in cameras:
            raise ValueError('duplicate KRT camera')
        cameras[rows[i]]=(k,rt)
    return cameras


def multiface_camera(k, extrinsic, transform):
    # OpenCV +Z forward/+Y down -> renderer -Z forward/+Y up, proper rotation.
    flip=np.diag([1.,-1.,-1.])
    r=flip@extrinsic[:,:3]@transform[:,:3]
    t=flip@(extrinsic[:,:3]@transform[:,3]+extrinsic[:,3])*.001
    return Camera.from_dict(dict(focal=k[0,0],focal_y=k[1,1],skew=k[0,1],
                                 cx=k[0,2],cy=k[1,2],origin=(-r.T@t).tolist(),rotation=r.tolist()))


def multiface(root,out,expression,annotate=False):
    from ..rig.face_models import _obj
    out.mkdir(parents=True)
    krt=unique(root/'multiface/extracted/metadata','KRT')
    cameras=load_krt(krt)
    mesh_root=root/'multiface/extracted'/('tracked_mesh--'+expression)
    meshes=sorted(mesh_root.rglob('*.obj'))
    if len(meshes)<3:
        raise ValueError('three tracked frames required')
    selected=[meshes[0],meshes[len(meshes)//2],meshes[-1]]
    geometry=[];views=[];checks=[];chosen=None;triangles=None
    for fi,mesh in enumerate(selected):
        world,tri,uv=_obj(mesh)
        aligned=np.fromfile(mesh.with_suffix('.bin'),dtype=np.float32).reshape(-1,3)
        transform=np.loadtxt(mesh.with_name(mesh.stem+'_transform.txt'))
        if aligned.shape!=world.shape or transform.shape!=(3,4) or not np.isfinite(aligned).all():
            raise ValueError('invalid Multiface frame')
        world_error=float(np.max(abs(world-(aligned@transform[:,:3].T+transform[:,3]))))
        if world_error>.01:
            raise ValueError('tracked OBJ/bin/head-transform disagreement')
        if triangles is not None and not np.array_equal(triangles,tri):
            raise ValueError('tracked topology changed')
        triangles=tri
        positions=aligned*.001
        frame_cameras={name:multiface_camera(k,rt,transform) for name,(k,rt) in cameras.items()}
        if chosen is None:
            # Prefer frontal cameras for robust detector annotations; hold out one.
            ranked=sorted(frame_cameras,key=lambda n: -frame_cameras[n].origin[2]/np.linalg.norm(frame_cameras[n].origin))
            available=[]
            for name in ranked:
                matches=list((root/'multiface/extracted'/('images--'+expression)).rglob(f'{name}/{mesh.stem}.png'))
                if matches:available.append(name)
            if len(available)<3:raise ValueError('three RGB cameras required')
            chosen=available[:3]
        for ci,name in enumerate(chosen):
            image=unique(root/'multiface/extracted'/('images--'+expression),f'{name}/{mesh.stem}.png')
            cam=frame_cameras[name];k,rt=cameras[name]
            cv=(aligned@transform[:,:3].T+transform[:,3])@rt[:,:3].T+rt[:,3]
            direct=cv@k.T;direct=direct[:,:2]/direct[:,2:]
            projected,z=cam.project(positions)
            # Float32 conversion to metres is included in this tolerance.
            error=float(np.max(abs(projected-direct)))
            if error>.01 or not (z>0).all():raise ValueError('camera projection conversion failed')
            with Image.open(image) as im:size=list(im.size)
            view=dict(image=str(image.resolve()),sha256=observations.sha256(image),size=size,
                      camera=cam.as_dict(),frame_id=mesh.stem,camera_id=name,
                      split='held-out' if ci==2 else 'fit',anchors={})
            if annotate:
                view['anchors']=observations.observe(image,cam)['views'][0]['anchors']
            views.append(view);geometry.append(positions)
            checks.append(dict(frame=mesh.stem,camera=name,world_transform_max_mm=world_error,
                               projection_max_px=error,positive_depth_fraction=float((z>0).mean()),
                               mesh_sha256=observations.sha256(mesh),
                               aligned_sha256=observations.sha256(mesh.with_suffix('.bin')),
                               transform_sha256=observations.sha256(mesh.with_name(mesh.stem+'_transform.txt'))))
            if fi==0:
                s=256/max(size);small=cam.scaled(s);resolution=tuple(round(n*s) for n in size)
                tid,bary,_=rasterize(positions,tri,small,resolution)
                atlas_path=unique(root/'multiface/extracted'/('unwrapped_uv_1024--'+expression),f'average/{mesh.stem}.png')
                atlas=np.asarray(Image.open(atlas_path).convert('RGB'))[::-1]
                render=np.full((resolution[1],resolution[0],3),32,np.uint8)
                yy,xx=np.nonzero(tid>=0)
                tex=(uv[tid[yy,xx]]*bary[yy,xx,:,None]).sum(1)
                ax=np.clip((tex[:,0]*atlas.shape[1]).astype(int),0,atlas.shape[1]-1)
                ay=np.clip((tex[:,1]*atlas.shape[0]).astype(int),0,atlas.shape[0]-1)
                render[yy,xx]=atlas[ay,ax]
                photo=np.asarray(Image.open(image).convert('RGB').resize(resolution))
                Image.fromarray(np.concatenate((photo,render),axis=1)).save(out/f'camera_{name}_reference.png')
                Image.fromarray((tid>=0).astype(np.uint8)*255).resize(tuple(size),Image.Resampling.NEAREST).save(out/f'camera_{name}_reference_mask.png')
    provenance=dict(dataset='Multiface',license='CC-BY-NC-4.0',expression=expression,
                    source='https://github.com/facebookresearch/multiface/blob/main/dataset.py',
                    krt_sha256=observations.sha256(krt),units='millimetres converted to metres',
                    annotations='MediaPipe estimates' if annotate else 'none',
                    frame='publisher aligned head-local frame; no GNM anatomical correspondence',
                    limitations=['tracked meshes are reference only; never label them predictions',
                                 'no independent scan metric for another topology without explicit correspondence',
                                 'reference texture is captured appearance, not intrinsic albedo'])
    for split in ('fit','held-out'):
        ids=[i for i,v in enumerate(views) if v['split']==split]
        write_json(out/(split+'.json'),dict(format=observations.FORMAT,provenance=provenance,views=[views[i] for i in ids]))
        np.savez_compressed(out/(split+'_reference.npz'),positions=np.asarray(geometry)[ids],triangles=triangles)
        observations.load(out/(split+'.json'))
    result=dict(status='prepared',views=len(views),cameras=chosen,frames=[p.stem for p in selected],
                checks=checks,provenance=provenance)
    write_json(out/'report.json',result)
    return result


def speakingfaces(root,out,fit=False):
    from . import pipeline
    from .reference import pixal_camera
    from ..rig.common import load_subject
    from ..rig.face_models import MODEL_CACHE
    if not (MODEL_CACHE/'gnm-v3/gnm_head.npz').is_file() or not (MODEL_CACHE/'face_landmarker.task').is_file():
        raise ValueError('cached GNM and MediaPipe weights required; run model setup first')
    source=root/'speakingfaces';out.mkdir(parents=True)
    seed=out/'head';seed.mkdir()
    pipeline.direct_seed(source/'frames/00000.png',seed,'gnm_v3')
    camera=pixal_camera(load_subject(seed))
    provenance=json.loads((source/'download_manifest.json').read_text())
    provenance.update(annotations='MediaPipe estimates, not manual ground truth',
                      calibration='assumed intrinsics/IPD; monocular smoke evaluation')
    for split,indices in [('fit',[0,24,48,71]),('held-out',[12,36,60])]:
        views=[]
        for i in indices:
            view=observations.observe(source/'frames'/f'{i:05d}.png',camera)['views'][0]
            view['timestamp_s']=i/28;views.append(view)
        write_json(out/(split+'.json'),dict(format=observations.FORMAT,views=views,provenance=provenance))
    result=dict(status='prepared',provenance=provenance)
    if fit:
        from .evaluate import evaluate
        pipeline.run(seed,observation_file=out/'fit.json',res=256,iterations=50,run_id='publicstarter')
        candidate=seed/'reconstruction/publicstarter'
        evaluate(candidate,out/'fit.json',out/'training-diagnostic',allow_training=True)
        raw=evaluate(candidate,out/'held-out.json',out/'held-out')
        aligned=evaluate(candidate,out/'held-out.json',out/'pose-aligned-diagnostic',pose_align=True)
        from .sequence import predict
        prediction=predict(candidate,np.array([12,36,60])/28,out/'predicted_surfaces.npz')
        animated=evaluate(candidate,out/'held-out.json',out/'animated-held-out',surfaces=out/'predicted_surfaces.npz')
        result.update(status='evaluated',heldout_rms_px=[v['landmarks']['rms_px'] for v in raw['views']],
                      aligned_mouth_rms_px=[v['pose_alignment_diagnostic'].get('expression_landmarks',{}).get('rms_px') for v in aligned['views']],
                      animated_rms_px=[v['landmarks']['rms_px'] for v in animated['views']],prediction=prediction)
    write_json(out/'report.json',result)
    return result


def exr_header(path):
    """Read bounded EXR metadata only; no pixel decoder or tone mapping claims."""
    def string(stream):
        buf=bytearray()
        for _ in range(256):
            c=stream.read(1)
            if c==b'\0':return buf.decode('ascii')
            if not c:raise ValueError('truncated EXR header')
            buf.extend(c)
        raise ValueError('oversized EXR attribute')
    with Path(path).open('rb') as f:
        if f.read(4)!=struct.pack('<I',20000630):raise ValueError('invalid EXR magic')
        version=struct.unpack('<I',f.read(4))[0];attrs={}
        while name:=string(f):
            typ=string(f);size=struct.unpack('<I',f.read(4))[0]
            if size>1024**2 or f.tell()+size>1024**2:raise ValueError('EXR header exceeds limit')
            data=f.read(size)
            if len(data)!=size:raise ValueError('truncated EXR attribute')
            if name=='dataWindow' and typ=='box2i':attrs['data_window']=list(struct.unpack('<4i',data))
        return dict(version=version,**attrs)


def emily(root,out):
    out.mkdir(parents=True)
    source=root/'emily/extracted';references=[]
    for kind in ('Unpolarized','FlashParallel','FlashCross','SpecularOnly'):
        files=sorted((source/('DigitalEmily2_'+kind)).rglob('*.exr'))
        if len(files)!=7:raise ValueError('Emily reference cameras incomplete')
        for p in files:references.append(dict(file=str(p.resolve()),kind=kind,**exr_header(p)))
    maps=[dict(file=str(p.resolve()),**exr_header(p)) for p in sorted((source/'Emily_2_1_Textures').rglob('*.exr'))]
    cameras=sorted((source/'DigitalEmily2_Calibration').glob('camera*.txt'))
    if len(cameras)!=7:raise ValueError('Emily camera calibration incomplete')
    result=dict(status='references indexed',references=references,maps=maps,
                cameras=[dict(file=str(p.resolve()),sha256=observations.sha256(p)) for p in cameras],
                mesh=str(unique(source/'Emily_2_1_OBJ','*.obj').resolve()),
                license='research/illustration, noncommercial, no redistribution',
                limitations=['EXR headers validated; linear pixels not decoded or fitted',
                             'camera distortion must be handled before image comparison',
                             'point-light radiance/exposure calibration is not established; do not infer a calibrated GGX fit'])
    write_json(out/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/nvme02/data/vhuman'))
    parser.add_argument('--out',type=Path,required=True,help='new local output directory; generated media must stay outside Git')
    parser.add_argument('--datasets',nargs='+',choices=['multiface','speakingfaces','emily'],default=['multiface','speakingfaces','emily'])
    parser.add_argument('--expression',choices=['E057_Cheeks_Puffed','E061_Lips_Puffed'],default='E057_Cheeks_Puffed')
    parser.add_argument('--annotate-multiface',action='store_true')
    parser.add_argument('--fit-speakingfaces',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[3]
    destination=args.out.resolve()
    if destination.is_relative_to(repo) and not destination.is_relative_to(repo/'tmp'):
        parser.error('generated dataset media must be outside the repository or in ignored tmp/')
    if args.out.exists():parser.error('output directory already exists; choose a fresh run directory')
    args.out.mkdir(parents=True)
    result={}
    for name in args.datasets:
        print('Preparing '+name,flush=True)
        result[name]=multiface(args.root,args.out/name,args.expression,args.annotate_multiface) if name=='multiface' else speakingfaces(args.root,args.out/name,args.fit_speakingfaces) if name=='speakingfaces' else emily(args.root,args.out/name)
        write_json(args.out/'report.json',result)
        print(name+': '+result[name]['status'],flush=True)


if __name__=='__main__':main()
