"""Reproducible CPU GNM comparisons on existing real video frames or scan splits.

Only training images enter optimization. Held-out scan geometry is read after
fitting and is never a solver loss or correspondence target. Reusing a scan
case while developing the solver makes it a development regression check,
not an untouched test set.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .. import gpu
from ..face_assets import asset_path
from ..native_landmarks import FaceLandmarker
from ..face_parsing import FaceParser
from ..rig.gnm_model import GNMModel
from .gnm_fit import solve,replay,cpu_threads
from .gnm_motion import candidate_bind,fit_views,views_from_landmarks,overlay
from .reference import Camera
from .artifacts import artifact_path
from .observations import load,sha256,pixel_sha256


def observe(paths,cameras,synthetic=False,threads=4):
    parser=FaceParser(threads=threads);views=[];valid=[]
    with FaceLandmarker(asset_path('mediapipe'),threads=threads) as tracker:
        for path,camera in zip(paths,cameras):
            with Image.open(path) as image: rgb=np.asarray(image.convert('RGB'));size=image.size
            faces=tracker.detect(rgb);visible=len(faces)==1
            points=np.asarray(faces[0]['landmarks']) if visible else np.zeros((478,3))
            labels=parser.predict(rgb)[0]
            views.extend(views_from_landmarks(points[None],np.array([visible]),camera,size,labels=labels[None],synthetic=synthetic))
            valid.append(visible)
    if not any(valid): raise ValueError('no faces in benchmark inputs')
    return views


def video(candidate,frames,out, *, stride=3,count=24,fps=28,iterations=120,threads=4):
    candidate,frames,out=map(Path,(candidate,frames,out));out=artifact_path(out)
    if stride<1 or count<2 or not np.isfinite(fps) or fps<=0: raise ValueError('invalid frame sampling')
    out.mkdir(parents=True,exist_ok=False)
    paths=sorted(frames.glob('*.png'))[::stride][:count]
    if len(paths)<2: raise ValueError('video benchmark needs at least two frames')
    if pixel_sha256(paths[0])!=pixel_sha256(candidate/'portrait.png'):
        raise ValueError('benchmark first frame must be the candidate source portrait')
    bind=candidate_bind(candidate);camera=Camera.from_dict(bind['metadata']['camera'])
    views=observe(paths,[camera]*len(paths),threads=threads)
    timestamps=np.arange(len(paths))*stride/fps
    result=fit_views(bind,views,timestamps,out=out,iterations=iterations,threads=threads)
    overlay(out,paths,result,fps=fps/stride)
    report=dict(result['track'],benchmark='real video frames; no generated images',
                source_frames=[dict(file=str(p.resolve()),sha256=sha256(p)) for p in paths],
                identity_bytes_unchanged=result['identity'].tobytes()==bind['identity'].tobytes())
    (out/'benchmark.json').write_text(json.dumps(report,indent=2));return report


def scan(prepared,baseline,out, *, iterations=120,threads=4):
    from .scan import surface_metrics
    prepared,baseline,out=map(Path,(prepared,baseline,out));out=artifact_path(out);out.mkdir(parents=True,exist_ok=False)
    train=load(prepared/'fit.json');held=load(prepared/'held-out.json')
    manifest=json.loads((baseline/'pca/manifest.json').read_text());alignment_report=json.loads((baseline/'report.json').read_text())
    with np.load(baseline/'pca/geometry.npz',allow_pickle=False) as z:
        old_neutral=z['neutral'];old_captured=z['captured'];scale=float(z['scale']);alignment=z['rotation']
    model=GNMModel(device='cpu');beta=np.zeros(253)
    coef=manifest['geometry']['identity_coefficients'];beta[:len(coef)]=coef
    offset=np.asarray(alignment_report['translation_m'])
    expression=np.zeros((len(train['views']),383));expression[:,manifest['geometry']['expression_modes']]=manifest['geometry']['expression_coefficients']
    cameras=[Camera.from_dict(c) for c in manifest['geometry']['fitted_cameras']]
    paths=[Path(v['image_path']) for v in train['views']]
    views=observe(paths,cameras,threads=threads)
    # Sparse manual semantic points remain training evidence, never holdouts.
    for view,original in zip(views,train['views']): view['anchors'].update(original.get('anchors',{}))
    with cpu_threads(threads):
        result=solve(views,identity=beta,expression=expression,scale=scale,alignment=alignment,offset=offset,
                     identity_modes=64,iterations=iterations,threads=threads,freeze_pose=True,
                     groups=[v['frame_id'] for v in train['views']],model=model)
    skin=model.group('skin_exterior');new=result['vertices'][:,skin]
    centre=scale*model.data['template_joint_positions'][[2,3]].mean(0)@alignment.T+offset
    local=(old_neutral-centre)@alignment
    roi=(abs(local[:,0])<.073)&(local[:,1]>-.105)&(local[:,1]<.045)&(local[:,2]>-.04)
    ids=np.flatnonzero(roi)[::max(1,int(roi.sum())//256)]
    # This is the first read of held-out geometry. Optimization is complete.
    with np.load(prepared/'held-out_reference.npz',allow_pickle=False) as z:
        truth=z['positions'];triangles=z['triangles']
    rows=[]
    for i,view in enumerate(held['views']):
        frame=next(j for j,v in enumerate(train['views']) if v['frame_id']==view['frame_id'])
        rows.append(dict(frame=view['frame_id'],before=surface_metrics(old_captured[frame],truth[i],triangles,ids),
                         after=surface_metrics(new[frame],truth[i],triangles,ids)))
    before=float(np.mean([r['before']['rms_mm'] for r in rows]));after=float(np.mean([r['after']['rms_mm'] for r in rows]))
    report=dict(benchmark='independent held-out scan cameras; same-frame expressions',backend='cpu',
                fitting=result['report'],geometry=rows,before_rms_mm=before,after_rms_mm=after,
                scan_gate=after<=before*1.02,heldout_geometry_used_for_fitting=False,
                baseline_geometry_sha256=sha256(baseline/'pca/geometry.npz'))
    np.savez_compressed(out/'predictions.npz',positions=new,identity=result['identity'],expression=result['expression'])
    (out/'benchmark.json').write_text(json.dumps(report,indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    v=sub.add_parser('video');v.add_argument('--candidate',required=True);v.add_argument('--frames',required=True)
    v.add_argument('--stride',type=int,default=3);v.add_argument('--count',type=int,default=24);v.add_argument('--fps',type=float,default=28)
    s=sub.add_parser('scan');s.add_argument('--prepared',required=True);s.add_argument('--baseline',required=True)
    for parser in (v,s):
        parser.add_argument('--out',required=True);parser.add_argument('--iterations',type=int,default=120);parser.add_argument('--threads',type=int,default=4)
    args=vars(p.parse_args());mode=args.pop('mode')
    with gpu.execution('cpu'):
        result=(video if mode=='video' else scan)(**args)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
