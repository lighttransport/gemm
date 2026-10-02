"""Original compact native CPU/CUDA normal/mask experiment, synthetic data only.

Apache GNM geometry is rendered with our own BRDF and procedural albedo. Real
Multiface/Emily/SpeakingFaces media is never used for optimization or labels.
Held-out identities have independent coefficient/lighting seeds.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .observations import sha256
from .reference import Camera,rasterize,ggx,linear_to_srgb
from .artifacts import artifact_path
ARCHITECTURE='geometry_prior_residual_cnn_v3'
CODE_SHA256=sha256(Path(__file__))


def network(side=64,threads=4,device='cpu',resident=False,memory_mb=512):
    from .native_cue_training import CueNet
    return CueNet(side,threads=threads,device=device,resident=resident,memory_mb=memory_mb)


def synthesize(out,identities=8,views=4,side=64):
    from ..rig import face_models
    from ..rig.common import vertex_normals
    from scipy.spatial.transform import Rotation
    out=artifact_path(out)
    if out.exists():raise ValueError('synthetic dataset already exists')
    if not 4<=identities<=32 or not 2<=views<=16 or side not in (64,128):raise ValueError('bounded synthetic configuration required')
    cached=face_models.MODEL_CACHE/'gnm-v3/gnm_head.npz'
    if not cached.is_file():raise ValueError('cached GNM weights required; no implicit download')
    source=face_models.load('gnm_v3');images=[];normal_maps=[];masks=[];groups=[];records=[];priors=[]
    camera=Camera(side*1.6,side/2,side/2,np.array([0.,0.,.5]),np.eye(3))
    for identity in range(identities):
        rng=np.random.default_rng(3001+identity)
        coefficients=rng.normal(0,.65,8)
        geometry=source.vertices-source.eye_centers.mean(0)+np.einsum('i,ivc->vc',coefficients,source.identity_basis[:8])
        for view in range(views):
            seed=10000+identity*100+view;local=np.random.default_rng(seed)
            pose=Rotation.from_euler('xyz',[local.uniform(-.12,.12),local.uniform(-.35,.35),local.uniform(-.08,.08)]).as_matrix()
            p=geometry@pose.T
            n=vertex_normals(p,source.triangles)
            tid,bary,_=rasterize(p,source.triangles,camera,(side,side));mask=tid>=0
            yy,xx=np.nonzero(mask);tri=source.triangles[tid[yy,xx]]
            surface=(p[tri]*bary[yy,xx,:,None]).sum(1)
            normals=(n[tri]*bary[yy,xx,:,None]).sum(1);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-9)
            viewdir=camera.origin-surface;viewdir/=np.linalg.norm(viewdir,axis=1,keepdims=True)
            light=np.array([local.uniform(-.8,.8),local.uniform(-.5,.8),1.]);light/=np.linalg.norm(light)
            color=np.array([.55,.30,.20])*local.uniform(.6,1.4)
            procedural=.025*np.sin(surface[:,0]*230)*np.cos(surface[:,1]*170)
            albedo=np.clip(color+procedural[:,None],.05,.9)
            diffuse,specular=ggx(albedo,normals,viewdir,light,local.uniform(.25,.7),.028)
            rgb=diffuse*2+specular*2+albedo*.15
            image=np.full((side,side,3),.08,np.float32);image[yy,xx]=np.clip(linear_to_srgb(rgb),0,1)
            normal=np.zeros_like(image);normal[yy,xx]=normals
            # Pose-matched mean geometry is an inference-available prior, not target identity.
            prior_geometry=(source.vertices-source.eye_centers.mean(0))@pose.T
            pi,pb,_=rasterize(prior_geometry,source.triangles,camera,(side,side))
            py,px=np.nonzero(pi>=0);pn=vertex_normals(prior_geometry,source.triangles)
            prior=np.tile(np.array([0,0,1],np.float32),(side,side,1))
            sampled=(pn[source.triangles[pi[py,px]]]*pb[py,px,:,None]).sum(1)
            sampled/=np.maximum(np.linalg.norm(sampled,axis=1,keepdims=True),1e-9);prior[py,px]=sampled
            # Photometric randomization: no real media or target normals alter RGB.
            image=np.clip(image*local.uniform(.65,1.35,size=3),0,1)**local.uniform(.8,1.2)
            image=np.clip(image+local.normal(0,.008,image.shape),0,1).astype(np.float32)
            background=local.uniform(.02,.35,(1,1,3));image[~mask]=background
            images.append(image);normal_maps.append(normal);masks.append(mask);groups.append(identity);priors.append(prior)
            records.append(dict(identity=identity,seed=seed,coefficients=coefficients.tolist(),light=light.tolist()))
        print(f'Synthetic identity {identity+1}/{identities}',flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,rgb=np.asarray(images),normals=np.asarray(normal_maps),mask=np.asarray(masks),identity=np.asarray(groups),prior=np.asarray(priors))
    manifest=dict(format='vhuman.synthetic_cues.v1',dataset_sha256=sha256(out),source=source.provenance,code_sha256=CODE_SHA256,
                  renderer='original NumPy perspective rasterizer/GGX',albedo='original procedural colors',
                  records=records,real_media_used=False,units='H-frame camera normals, unit vectors',
                  limitations=['small synthetic pilot; no realistic skin/eye/hair distribution'])
    out.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def metrics(predicted,truth,mask):
    dot=np.clip(np.sum(predicted*truth,axis=1),-1,1)
    angles=np.rad2deg(np.arccos(dot))[mask]
    return dict(mean_degrees=float(angles.mean()),median_degrees=float(np.median(angles)),p95_degrees=float(np.quantile(angles,.95)))


def train(dataset,out,steps=400,device='cpu',threads=4,memory_mb=512):
    from ..rig import safetensors as st
    from ..native_gpu_training import device_index
    gpu=device_index(device) is not None
    rng=np.random.default_rng(1234)
    dataset,out=Path(dataset),artifact_path(out)
    if out.exists():raise ValueError('choose a fresh training directory')
    manifest=json.loads(dataset.with_suffix('.json').read_text())
    if manifest.get('format')!='vhuman.synthetic_cues.v1' or manifest.get('real_media_used') is not False or sha256(dataset)!=manifest['dataset_sha256']:
        raise ValueError('verified synthetic-only cue dataset required')
    if type(steps) is not int or not 50<=steps<=5000:raise ValueError('steps must be 50..5000')
    with np.load(dataset,allow_pickle=False) as z:rgb=z['rgb'].transpose(0,3,1,2);truth=z['normals'].transpose(0,3,1,2);mask=z['mask'];groups=z['identity'];priors=z['prior'].transpose(0,3,1,2)
    if (rgb.ndim!=4 or rgb.shape[1]!=3 or not 1<=len(rgb)<=512 or not 1<=rgb.shape[-1]<=128 or
            rgb.shape[-2]!=rgb.shape[-1] or truth.shape!=rgb.shape or priors.shape!=rgb.shape or
            mask.shape!=(len(rgb),*rgb.shape[2:]) or mask.dtype!=bool or groups.shape!=(len(rgb),) or
            groups.dtype.kind not in 'iu' or not all(np.isfinite(a).all() for a in (rgb,truth,priors)) or
            (rgb<0).any() or (rgb>1).any()):raise ValueError('invalid bounded synthetic cue tensors')
    identities=np.unique(groups);heldout=identities[-max(1,len(identities)//4):]
    fit=~np.isin(groups,heldout);test=~fit
    if fit.sum()<4 or test.sum()<2:raise ValueError('identity-separated fit/holdout required')
    model=network(rgb.shape[-1],threads,device,resident=gpu,memory_mb=memory_mb)
    mean=(truth[fit]*mask[fit,None]).sum(0)/np.maximum(mask[fit].sum(0)[None],1)
    mean/=np.maximum(np.linalg.norm(mean,axis=0,keepdims=True),1e-9)
    mean[:,mask[fit].sum(0)==0]=np.array([0,0,1])[:,None]
    model.prior[:]=mean[None]
    fit_rgb,fit_prior,fit_truth,fit_mask=rgb[fit],priors[fit],truth[fit],mask[fit]
    try:
        for step in range(steps):
            ids=rng.integers(fit.sum(),size=min(8,fit.sum()))
            model.compute(fit_rgb[ids],fit_prior[ids],fit_truth[ids],fit_mask[ids],update=True,
                          return_gradient=False,return_output=False)
        predictions=[];confidences=[]
        for start in range(0,test.sum(),8):
            normal,logits=model(rgb[test][start:start+8],priors[test][start:start+8])
            predictions.append(normal)
            confidences.append(1/(1+np.exp(-np.clip(logits[:,0],-80,80))))
        model.sync_parameters()
        peak_bytes=model._gpu.peak_bytes if gpu else 0
    finally:model.close()
    predicted=np.concatenate(predictions);confidence=np.concatenate(confidences)
    # Fixed reference masks score normals; predicted confidence cannot hide errors.
    # One cross-shaped binary erosion with zero border, using only NumPy.
    interior=np.zeros_like(mask[test])
    source=mask[test]
    interior[:,1:-1,1:-1]=(source[:,1:-1,1:-1]&source[:,:-2,1:-1]&source[:,2:,1:-1]&
                          source[:,1:-1,:-2]&source[:,1:-1,2:])
    learned=metrics(predicted,truth[test],interior)
    flat=np.zeros_like(predicted);flat[:,2]=1
    baseline=metrics(flat,truth[test],interior)
    template=metrics(priors[test],truth[test],interior)
    accepted=learned['mean_degrees']<template['mean_degrees']*.95
    out.mkdir(parents=True)
    st.save(out/'normal_cue.safetensors',model.state_dict())
    np.savez_compressed(out/'heldout.npz',rgb=rgb[test],normals=predicted,truth=truth[test],confidence=confidence,mask=mask[test])
    report=dict(format='vhuman.normal_cue.v1',architecture=ARCHITECTURE,code_sha256=CODE_SHA256,training_backend='repository_cuda_gemm' if gpu else 'repository_cpu_gemm',cuda_peak_bytes=peak_bytes,cuda_memory_budget_mb=memory_mb if gpu else None,weights_sha256=sha256(out/'normal_cue.safetensors'),dataset_sha256=manifest['dataset_sha256'],
                training_identities=identities[~np.isin(identities,heldout)].tolist(),heldout_identities=heldout.tolist(),
                steps=steps,device=device,thread_budget=threads,resolution=int(rgb.shape[-1]),parameters=int(model.parameters.size),
                heldout_normals=learned,flat_baseline=baseline,geometry_prior_baseline=template,synthetic_gate_passed=bool(accepted),
                real_geometry_gate_passed=False,default_enabled=False,
                limitations=['synthetic normal/mask pilot; no real-domain accuracy claim',
                             'mask probability is not calibrated normal uncertainty',
                             'do not use for geometry refinement until a separate real-data gate passes'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def infer(image,checkpoint,out,prior_file=None):
    from PIL import Image
    checkpoint,out=Path(checkpoint),artifact_path(out)
    report=json.loads((checkpoint/'report.json').read_text())
    if report.get('architecture')!=ARCHITECTURE:raise ValueError('unsupported cue architecture; train with the current implementation')
    if sha256(checkpoint/'normal_cue.safetensors')!=report['weights_sha256'] or not report['synthetic_gate_passed']:
        raise ValueError('verified checkpoint passing synthetic gate required')
    side=report.get('resolution',64)
    if prior_file is None:raise ValueError('pose/crop-matched geometry prior NPZ required for v3 inference')
    with np.load(prior_file,allow_pickle=False) as z:geometry_prior=z['normals']
    if geometry_prior.shape!=(side,side,3) or not np.isfinite(geometry_prior).all() or not np.allclose(np.linalg.norm(geometry_prior,axis=2),1,atol=.01):raise ValueError('unit geometry prior matching RGB crop required')
    rgb=np.asarray(Image.open(image).convert('RGB').resize((side,side)),np.float32)/255
    from ..native_models import run_image_model
    inputs=np.concatenate((rgb,geometry_prior),axis=2).transpose(2,0,1)
    result=run_image_model('cues',checkpoint/'normal_cue.safetensors',inputs)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,normals=result[:3].transpose(1,2,0),mask_probability=result[3])
    return dict(status='experimental inference',real_geometry_gate_passed=False,image_sha256=sha256(image),geometry_prior_sha256=sha256(prior_file))


def validate_real(checkpoint,prepared,candidate,out):
    """Real scan-normal ablation; evaluates frozen weights, never optimizes them."""
    from PIL import Image
    from scipy.ndimage import binary_erosion
    from ..native_models import run_image_model
    from ..rig.common import vertex_normals
    from . import observations
    checkpoint,prepared,candidate,out=map(Path,(checkpoint,prepared,candidate,out))
    out=artifact_path(out)
    report=json.loads((checkpoint/'report.json').read_text())
    if report.get('architecture')!=ARCHITECTURE:raise ValueError('unsupported cue architecture; train with the current implementation')
    if sha256(checkpoint/'normal_cue.safetensors')!=report['weights_sha256']:raise ValueError('cue checkpoint changed')
    doc=observations.load(prepared/'held-out.json')
    with np.load(prepared/'held-out_reference.npz',allow_pickle=False) as z:truth=z['positions'];tri=z['triangles']
    manifest=json.loads((candidate/'manifest.json').read_text())
    if sha256(candidate/'geometry.npz')!=manifest['geometry_sha256']:raise ValueError('candidate geometry changed')
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:baseline=z['neutral'];base_tri=z['triangles']
    crop_geometry=baseline
    correspondence=candidate.parent/'correspondence.npz'
    if correspondence.is_file():
        scan_report=json.loads((candidate.parent/'report.json').read_text())
        if sha256(correspondence)!=scan_report['correspondence_sha256']:raise ValueError('crop correspondence changed')
        with np.load(correspondence,allow_pickle=False) as z:roi=z['roi']
        if roi.shape!=(len(baseline),) or roi.dtype!=bool or roi.sum()<20:raise ValueError('invalid fitting-only crop ROI')
        crop_geometry=baseline[roi]
    rows=[]
    out.mkdir(parents=True,exist_ok=False)
    for i,view in enumerate(doc['views']):
        camera=Camera.from_dict(view['camera']);w,h=view['size']
        pixels,depth=camera.project(crop_geometry)
        front=pixels[depth>0]
        lo=np.maximum(np.floor(front.min(0)),[0,0]);hi=np.minimum(np.ceil(front.max(0)),[w,h])
        span=float(np.max(hi-lo)*1.12);centre=(hi+lo)/2
        left,top=centre-span/2
        if span<16:raise ValueError('fitted face crop too small')
        small=Camera(camera.focal*64/span,(camera.cx-left)*64/span,(camera.cy-top)*64/span,camera.origin,camera.rotation,
                     (camera.focal if camera.focal_y is None else camera.focal_y)*64/span,camera.skew*64/span)
        ids,bary,_=rasterize(truth[i],tri,small,(64,64));mask=binary_erosion(ids>=0,iterations=1)
        yy,xx=np.nonzero(mask);normal=vertex_normals(truth[i],tri)
        gt=np.zeros((64,64,3),np.float32)
        n=(normal[tri[ids[yy,xx]]]*bary[yy,xx,:,None]).sum(1)@camera.rotation.T
        n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);gt[yy,xx]=n
        base_ids,base_bary,_=rasterize(baseline,base_tri,small,(64,64))
        bn=vertex_normals(baseline,base_tri);by,bx=np.nonzero(base_ids>=0)
        base=np.tile(np.array([0.,0.,1.],np.float32),(64,64,1))
        n=(bn[base_tri[base_ids[by,bx]]]*base_bary[by,bx,:,None]).sum(1)@camera.rotation.T
        n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);base[by,bx]=n
        image=np.asarray(Image.open(view['image_path']).convert('RGB').transform((64,64),Image.Transform.AFFINE,(span/64,0,left,0,span/64,top),resample=Image.Resampling.BILINEAR),np.float32)/255
        inputs=np.concatenate((image,base),axis=2).transpose(2,0,1)
        prediction=run_image_model('cues',checkpoint/'normal_cue.safetensors',inputs)
        estimated=prediction[:3].transpose(1,2,0);probability=prediction[3]
        estimated_metrics=metrics(estimated.transpose(2,0,1)[None],gt.transpose(2,0,1)[None],mask[None])
        baseline_metrics=metrics(base.transpose(2,0,1)[None],gt.transpose(2,0,1)[None],mask[None])
        union=((probability>.5)|(ids>=0)).sum();iou=float(((probability>.5)&(ids>=0)).sum()/max(union,1))
        rows.append(dict(image_sha256=view['sha256'],learned=estimated_metrics,rendered_gnm=baseline_metrics,
                         mask_iou=iou,crop_xywh=[float(left),float(top),span,span],baseline_coverage=float((base_ids[mask]>=0).mean())))
        strip=np.concatenate((image,np.clip(gt*.5+.5,0,1),np.clip(base*.5+.5,0,1),np.clip(estimated*.5+.5,0,1)),axis=1)
        Image.fromarray(np.uint8(strip*255)).resize((1024,256)).save(out/f'view_{i}_normals.png')
    passed=all(r['learned']['mean_degrees']<r['rendered_gnm']['mean_degrees']*.95 and r['mask_iou']>.8 for r in rows)
    result=dict(format='vhuman.normal_cue_real_evaluation.v1',weights_sha256=report['weights_sha256'],views=rows,
                baseline_geometry_sha256=sha256(candidate/'geometry.npz'),
                pilot_gate_passed=bool(passed),real_geometry_gate_passed=False,default_enabled=False,
                training_performed=False,reference_sha256=sha256(prepared/'held-out_reference.npz'),
                limitations=['single subject/domain; production adoption requires additional identities',
                             'fixed scan mask scores all normals; predicted confidence cannot hide errors',
                             'both methods receive fitting-camera geometry prior; target scan only scores the frozen prediction'])
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    synth=sub.add_parser('synthesize');synth.add_argument('--out',type=Path,required=True)
    fit=sub.add_parser('train');fit.add_argument('--dataset',type=Path,required=True);fit.add_argument('--out',type=Path,required=True);fit.add_argument('--steps',type=int,default=400);fit.add_argument('--device',default='cpu',help='cpu, cuda, or cuda:N')
    fit.add_argument('--memory-mb',type=int,default=512)
    fit.add_argument('--threads',type=int,default=4)
    predict=sub.add_parser('infer');predict.add_argument('--image',type=Path,required=True);predict.add_argument('--prior',type=Path,required=True);predict.add_argument('--checkpoint',type=Path,required=True);predict.add_argument('--out',type=Path,required=True)
    validate=sub.add_parser('validate-real')
    for name in ('checkpoint','prepared','candidate','out'):validate.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();result=synthesize(a.out) if a.action=='synthesize' else train(a.dataset,a.out,a.steps,a.device,a.threads,a.memory_mb) if a.action=='train' else validate_real(a.checkpoint,a.prepared,a.candidate,a.out) if a.action=='validate-real' else infer(a.image,a.checkpoint,a.out,a.prior)
    print(json.dumps(result))


if __name__=='__main__':main()
