"""Predict held-out motion from training captures without reading target anchors.

This deterministic baseline interpolates captured deformation and rigid pose.
It is not an audio-conditioned motion generator or held-out landmark fitting.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .reference import Camera
from .observations import sha256
from .artifacts import artifact_path


def predict(candidate,timestamps,out):
    from scipy.spatial.transform import Rotation, Slerp
    from .fitting import safe_geometry
    candidate,out=Path(candidate),artifact_path(out)
    manifest=json.loads((candidate/'manifest.json').read_text())
    document=json.loads((candidate/'observations.json').read_text())
    if sha256(candidate/'geometry.npz')!=manifest['geometry_sha256']:
        raise ValueError('candidate geometry changed')
    times=np.array([v['timestamp_s'] for v in document['views']],float)
    target=np.asarray(timestamps,float)
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or (np.diff(times)<=0).any():
        raise ValueError('training timestamps must increase')
    if target.ndim!=1 or not len(target) or not np.isfinite(target).all() or (np.diff(target)<=0).any():
        raise ValueError('prediction timestamps must increase')
    if target.min()<times[0] or target.max()>times[-1]:
        raise ValueError('target timestamps outside training interval; no implicit extrapolation')
    rotations=[];translations=[]
    for view,fitted in zip(document['views'],manifest['geometry']['fitted_cameras']):
        original=Camera.from_dict(view['camera']);fitted=Camera.from_dict(fitted)
        r=original.rotation.T@fitted.rotation
        rotations.append(r);translations.append(original.origin-r@fitted.origin)
    if len(rotations)!=len(times):raise ValueError('training camera count mismatch')
    rotation=Slerp(times,Rotation.from_matrix(rotations))(target).as_matrix()
    translations=np.asarray(translations)
    translation=np.column_stack([np.interp(target,times,translations[:,i]) for i in range(3)])
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:
        neutral,tri,captured=z['neutral'],z['triangles'],z['captured']
    if captured.shape!=(len(times),len(neutral),3):raise ValueError('training capture count mismatch')
    surfaces=[];steps=[]
    for t,r,offset in zip(target,rotation,translation):
        hi=min(int(np.searchsorted(times,t,side='right')),len(times)-1);lo=max(0,hi-1)
        a=(t-times[lo])/(times[hi]-times[lo])
        surface=(1-a)*captured[lo]+a*captured[hi]
        step=1.
        while not safe_geometry(neutral,surface,tri) and step>1/128:
            step*=.5;surface=neutral+step*((1-a)*captured[lo]+a*captured[hi]-neutral)
        if not safe_geometry(neutral,surface,tri):raise ValueError('unsafe interpolated surface')
        surfaces.append(surface@r.T+offset);steps.append(step)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,positions=np.asarray(surfaces,np.float32),triangles=tri,timestamps=target)
    report=dict(format='vhuman.sequence_prediction.v1',method='linear captured deformation + quaternion pose interpolation',
                training_observations_sha256=sha256(candidate/'observations.json'),geometry_sha256=manifest['geometry_sha256'],
                training_timestamps=times.tolist(),target_timestamps=target.tolist(),safe_steps=steps,
                output_sha256=sha256(out),target_annotations_used=False,
                limitations=['interpolation baseline; not audio-conditioned motion',
                             'rapid expressions between training samples cannot be recovered',
                             'camera/metric scale inherits source calibration'])
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--timestamps',type=float,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(predict(args.candidate,args.timestamps,args.out)))


if __name__=='__main__':main()
