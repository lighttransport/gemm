#!/usr/bin/env python3
"""Offline real-checkpoint parity for native cue CNN and streaming motion GRU."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cues',type=Path,required=True);ap.add_argument('--motion',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    import torch
    from server.vhuman.reconstruction.learned_cues import network
    from server.vhuman.rig import safetensors
    from server.vhuman.native_models import run_image_model
    from server.vhuman.realtime.src.animation.causal_model import ReferenceMotionAdapter
    from server.vhuman.realtime.src.animation.native_motion import MotionAdapter
    from server.vhuman.realtime.src.pipeline.protocol import TTSFeatureFrame
    torch.set_num_threads(4);rng=np.random.default_rng(813)
    tensors,_=safetensors.load(a.cues/'normal_cue.safetensors')
    model=network(tensors['prior'].shape[-1]);model.load_state_dict({k:torch.tensor(v) for k,v in tensors.items()});model.eval()
    report={'cues':[]}
    for h,w in ((64,64),(57,83)):
        rgb=rng.random((3,h,w),dtype=np.float32);prior=rng.normal(size=(3,h,w)).astype(np.float32)
        prior/=np.linalg.norm(prior,axis=0)
        with torch.inference_mode():normals,logits=model(torch.from_numpy(rgb)[None],torch.from_numpy(prior)[None])
        ref=np.concatenate((normals[0].numpy(),logits.sigmoid()[0].numpy()))
        got=run_image_model('cues',a.cues/'normal_cue.safetensors',np.concatenate((rgb,prior)))
        error=abs(got-ref)
        check=dict(shape=[h,w],max_abs=float(error.max()),mean_abs=float(error.mean()),passed=bool(error.max()<2e-5))
        report['cues'].append(check)
        np.savez(a.out/f'cues-{h}-{w}.npz',input=np.concatenate((rgb,prior)),reference=ref,native=got)
    data=torch.load(a.motion,map_location='cpu',weights_only=True);revision=data['tts_revision']
    ref=ReferenceMotionAdapter(a.motion,revision,allow_diagnostic=True)
    native=MotionAdapter(a.motion,revision,allow_diagnostic=True)
    mean=ref.model.hidden_mean.numpy();scale=ref.model.hidden_scale.numpy()
    hidden=(rng.normal(size=(40,len(mean))).astype(np.float32)*scale+mean).astype(np.float32)
    codes=rng.integers(0,2048,size=(40,16),dtype=np.int32);expected=[];actual=[]
    for i in range(40):
        frame=TTSFeatureFrame(0,i*1920,codes[i],hidden[i],revision)
        expected.append(np.stack([x.controls for x in ref.push(frame)]));actual.append(np.stack([x.controls for x in native.push(frame)]))
    error=abs(np.asarray(expected)-actual)
    native.reset(3)
    assert native.push(TTSFeatureFrame(2,0,codes[0],hidden[0],revision))==[]
    reset=np.stack([x.controls for x in native.push(TTSFeatureFrame(3,0,codes[0],hidden[0],revision))])
    np.testing.assert_array_equal(reset,actual[0])
    try:native.push(TTSFeatureFrame(3,3840,codes[1],hidden[1],revision))
    except ValueError:pass
    else:raise AssertionError('noncontiguous frame accepted')
    native.close()
    report['motion']=dict(steps=40,max_abs=float(error.max()),mean_abs=float(error.mean()),reset_exact=True,passed=bool(error.max()<2e-5))
    np.savez(a.out/'motion.npz',hidden=hidden,codes=codes,reference=expected,native=actual)
    report['pass']=report['motion']['passed'] and all(c['passed'] for c in report['cues'])
    (a.out/'parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    return 0 if report['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
