#!/usr/bin/env python3
"""Short native GPU fitting/visual checks under contention, without frameworks.

Uses cleared local corpora only. Timings are synchronized wall times, not
isolated benchmarks. Small step counts deliberately do not certify fit quality.
"""
import argparse
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


class NoFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','onnx','onnxruntime','tensorflow','gsplat','diffusers','mediapipe'}:
            raise AssertionError('unexpected model runtime: '+fullname)


sys.meta_path.insert(0,NoFrameworks())
import numpy as np
from PIL import Image,ImageDraw
from server.vhuman.native_training import ROOT
from server.vhuman.reconstruction.artifacts import artifact_path
from server.vhuman.reconstruction.observations import sha256
from server.vhuman.reconstruction.reference import linear_to_srgb
from server.vhuman.reconstruction.learned_cues import metrics
from server.vhuman.reconstruction.native_cue_training import CueNet
from server.vhuman.realtime.src.avatar.bundle import bind,GaussianAvatar
from server.vhuman.realtime.src.avatar.native_training import AppearanceTrainer,initialize_rgb
from server.vhuman.realtime.src.avatar.provenance import verify_files
from server.vhuman.rig import safetensors


def inventory():
    result=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],
                          text=True,capture_output=True,check=True)
    return result.stdout.strip()


def sheet(path, panels, display='linear'):
    images=[]
    for label,value in panels:
        if display=='linear':value=linear_to_srgb(np.clip(value,0,1))
        images.append((label,Image.fromarray(np.uint8(np.clip(value,0,1)*255))))
    height=max(image.height for _,image in images)
    canvas=Image.new('RGB',(sum(image.width for _,image in images),height+26),(24,24,24))
    draw=ImageDraw.Draw(canvas);x=0
    for label,image in images:
        draw.text((x+4,5),label,fill='white');canvas.paste(image,(x,26));x+=image.width
    canvas.save(path)


def resize(value,size):
    if value.ndim==2:return np.asarray(Image.fromarray(value).resize(size,Image.Resampling.BILINEAR),np.float32).copy()
    return np.stack([resize(value[...,i],size) for i in range(value.shape[-1])],-1)


def appearance(args,out,report):
    manifest=Path(args.corpus);spec=json.loads(manifest.read_text());verify_files(spec['provenance'],manifest.parent)
    if spec.get('format')!='vhuman.appearance_corpus.v1' or spec['data'] not in {r['path'] for r in spec['provenance']}:
        raise ValueError('verified appearance corpus required')
    with np.load(manifest.parent/spec['data'],allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    original=data['images'][0];height,width=original.shape[:2];size=(args.side,args.side)
    target=resize(original,size);mask=resize(data['masks'][0],size);k=data['intrinsics'][0].copy()
    k[0]*=args.side/width;k[1]*=args.side/height
    v,tri,ctrl,view=data['vertices'][0],data['triangles'],data['controls'][0],data['view'][0]
    avatar=bind(v,tri,spec['control_names'],args.count,7,purpose=spec.get('purpose','diagnostic'),provenance=spec['provenance'])
    points=(v[tri[avatar.arrays['triangle']]]*avatar.arrays['barycentric'][...,None]).sum(1)
    initial=initialize_rgb(points,target,view,k)
    model=AppearanceTrainer(avatar,tri,initial,device=args.device,resident=True,memory_mb=args.memory_mb)
    try:
        before=model.compute(v,ctrl,view,k,size,target,mask)[0]
        # One matched CPU step supplies a comparison without more GPU work.
        cpu=AppearanceTrainer(avatar,tri,initial)
        start=time.perf_counter();cpu_result=cpu.compute(v,ctrl,view,k,size,target,mask,update=True)
        cpu_ms=(time.perf_counter()-start)*1000
        losses=[];times=[]
        for _ in range(args.steps):
            start=time.perf_counter()
            _,loss,_=model.compute(v,ctrl,view,k,size,target,mask,update=True,return_gradient=False,return_output=False)
            times.append((time.perf_counter()-start)*1000);losses.append(float(loss[1]))
        after=model.compute(v,ctrl,view,k,size,target,mask)[0]
        receipt=dict(count=args.count,size=list(size),step_ms=times,pre_update_l1=losses,
            before_l1=float(abs(before[...,:3]-target).mean()),after_l1=float(abs(after[...,:3]-target).mean()),
            allocated_peak_mib=model._gpu.peak_bytes/2**20,tile_overlaps=model._gpu.overlaps,
            matched_cpu_step_ms=cpu_ms,cpu_gpu_initial_rgba_max_error=float(abs(cpu_result[0]-before).max()),
            cpu_initial_l1=float(cpu_result[1][1]),
            visual='appearance-cold.png',quality_accepted=False)
        sheet(out/'appearance-cold.png',[('Reference',target),('Cold start',before[...,:3]),(f'After {args.steps} steps',after[...,:3])])
        model.export(avatar);avatar.metadata.update(covariance_policy='trace-v1',trained=True,training_steps=args.steps,
            training_backend='repository_cuda_gemm',training_device=args.device,training_l1=receipt['after_l1'],
            validation_only=True,visual_quality_accepted=False,corpus_sha256=sha256(manifest.parent/spec['data']))
        avatar.save(out/'appearance-cold.npz');report['appearance_cold']=receipt
        print(json.dumps({'appearance_cold':receipt}),flush=True)
    finally:model.close()
    # Warm start tests preservation of an already fitted appearance; it is not
    # a fresh quality benchmark or optimizer-state resume.
    if args.avatar:
        avatar=GaussianAvatar.load(args.avatar,tri)
        if avatar.metadata['control_names']!=spec['control_names']:raise ValueError('warm-start control order mismatch')
        model=AppearanceTrainer(avatar,tri,device=args.device,resident=True,memory_mb=args.memory_mb)
        model.load_avatar(avatar);model.optimizer.lr=.0001
        try:
            before=model.compute(v,ctrl,view,k,size,target,mask)[0];start=time.perf_counter()
            model.compute(v,ctrl,view,k,size,target,mask,update=True,return_gradient=False,return_output=False)
            elapsed=(time.perf_counter()-start)*1000;after=model.compute(v,ctrl,view,k,size,target,mask)[0]
            model.export(avatar)
            avatar.metadata.update(training_backend='repository_cuda_gemm',training_device=args.device,training_steps=1,
                warm_start_sha256=sha256(args.avatar),validation_only=True,visual_quality_accepted=False)
            avatar.save(out/'appearance-warm.npz')
            # Validate exported checkpoint through the independent native FP32
            # inference renderer, with the training frameworks still blocked.
            from server.vhuman.realtime.src.renderer.native import NativeGaussianRenderer
            renderer=NativeGaussianRenderer(avatar,tri,device=args.device)
            try:
                frame=renderer.render(v,view,k,size,ctrl)
                try:rendered=frame.rgba.numpy()
                finally:frame.close()
            finally:renderer.close()
            delta=abs(rendered-after)
            np.savez_compressed(out/'export-comparison.npz',training=after,runtime=rendered)
            receipt=dict(count=model.n,size=list(size),step_ms=elapsed,learning_rate=.0001,
                before_l1=float(abs(before[...,:3]-target).mean()),after_l1=float(abs(after[...,:3]-target).mean()),
                exported_runtime_l1=float(abs(rendered[...,:3]-target).mean()),export_rgba_max_error=float(delta.max()),
                export_rgba_mean_error=float(delta.mean()),allocated_peak_mib=model._gpu.peak_bytes/2**20,
                export_pixels_error_above_1e4=int((delta.max(-1)>1e-4).sum()),
                export_pixels_error_above_1e3=int((delta.max(-1)>1e-3).sum()),
                visual='appearance-warm.png',quality_accepted=False)
            sheet(out/'appearance-warm.png',[('Reference',target),('Imported fit',before[...,:3]),('Native GPU update',after[...,:3]),('Export/runtime',rendered[...,:3])])
            report['appearance_warm']=receipt;print(json.dumps({'appearance_warm':receipt}),flush=True)
        finally:model.close()


def cues(args,out,report):
    path=Path(args.cue_dataset);manifest=json.loads(path.with_suffix('.json').read_text())
    if manifest.get('real_media_used') is not False or sha256(path)!=manifest['dataset_sha256']:
        raise ValueError('verified synthetic-only cue data required')
    with np.load(path,allow_pickle=False) as z:
        rgb=z['rgb'].transpose(0,3,1,2).copy();truth=z['normals'].transpose(0,3,1,2).copy();prior=z['prior'].transpose(0,3,1,2).copy()
        mask=z['mask'].copy();groups=z['identity'].copy()
    heldout=np.unique(groups)[-max(1,len(np.unique(groups))//4):];test=np.isin(groups,heldout);fit=np.flatnonzero(~test)
    model=CueNet(rgb.shape[-1],device=args.device,resident=True,memory_mb=args.memory_mb)
    if args.cue_checkpoint:
        checkpoint=Path(args.cue_checkpoint);receipt=json.loads((checkpoint/'report.json').read_text())
        if receipt['dataset_sha256']!=manifest['dataset_sha256'] or receipt['weights_sha256']!=sha256(checkpoint/'normal_cue.safetensors'):
            raise ValueError('cue checkpoint receipt mismatch')
        model.load_state_dict(safetensors.load(checkpoint/'normal_cue.safetensors')[0])
    try:
        ids=np.flatnonzero(test)[:8];before=model(rgb[ids],prior[ids])[0];times=[]
        for step in range(args.steps):
            rows=fit[step*8:step*8+8]
            if not len(rows):rows=fit[:8]
            start=time.perf_counter();model.compute(rgb[rows],prior[rows],truth[rows],mask[rows],update=True,return_gradient=False,return_output=False)
            times.append((time.perf_counter()-start)*1000)
        after=model(rgb[ids],prior[ids])[0]
        receipt=dict(batch=min(8,len(fit)),side=rgb.shape[-1],step_ms=times,before=metrics(before,truth[ids],mask[ids]),
            after=metrics(after,truth[ids],mask[ids]),allocated_peak_mib=model._gpu.peak_bytes/2**20,
            visual='cue-heldout.png',real_geometry_gate_passed=False,default_enabled=False)
        receipt['synthetic_quality_improved']=receipt['after']['mean_degrees']<receipt['before']['mean_degrees']
        panels=[('Synthetic RGB',rgb[ids[0]].transpose(1,2,0))]
        for name,a in [('Prior',prior[ids[0]]),('Truth',truth[ids[0]]),('Before',before[0]),('After',after[0])]:
            panels.append((name,(a.transpose(1,2,0)+1)*.5))
        sheet(out/'cue-heldout.png',panels,display='raw');safetensors.save(out/'normal_cue.safetensors',model.state_dict())
        report['cues']=receipt;print(json.dumps({'cues':receipt}),flush=True)
    finally:model.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--avatar',type=Path);parser.add_argument('--cue-dataset',type=Path);parser.add_argument('--cue-checkpoint',type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--memory-mb',type=int,default=512)
    parser.add_argument('--count',type=int,default=50000);parser.add_argument('--side',type=int,default=256)
    parser.add_argument('--steps',type=int,default=2);args=parser.parse_args()
    if not 1<=args.steps<=3 or not 16<=args.side<=512 or not 1<=args.count<=200000:parser.error('brief budget: steps 1..3, side 16..512, count 1..200000')
    out=artifact_path(args.output)
    if out.exists():raise ValueError('choose a fresh output directory')
    out.mkdir(parents=True)
    report=dict(format='vhuman.native_gpu_training_validation.v1',framework_imports_blocked=True,
        library_sha256=sha256(ROOT/'cuda/vhuman/libvhuman_training_cuda.so'),gpu_before=inventory(),
        timings='synchronized wall time under contention; compilation excluded; not an isolated benchmark',
        limitations=['brief fit only; no convergence or expression quality claim','neutral corpus cannot validate expression-conditioned appearance',
                     'CUDA projection and raster backward use FP64 for CPU parity; host depth sort/tile bins remain',
                     'allocated_peak_mib excludes driver/module/context allocations'])
    appearance(args,out,report)
    if args.cue_dataset:cues(args,out,report)
    report['gpu_after']=inventory();(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
