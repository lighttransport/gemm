"""Original compact PyTorch normal/mask experiment, synthetic data only.

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


def network(side=64):
    import torch
    from torch import nn
    class CueNet(nn.Module):
        def __init__(self):
            super().__init__()
            prior=torch.zeros((1,3,side,side));prior[:,2]=1
            self.register_buffer('prior',prior)
            self.encoder=nn.Sequential(nn.Conv2d(6,16,5,padding=2),nn.SiLU(),
                nn.Conv2d(16,24,3,stride=2,padding=1),nn.SiLU(),
                nn.Conv2d(24,32,3,stride=2,padding=1),nn.SiLU())
            self.decoder=nn.Sequential(nn.Conv2d(32,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,4,1))
        def forward(self,image,geometry_prior=None):
            prior=nn.functional.interpolate(self.prior if geometry_prior is None else geometry_prior,size=image.shape[-2:],mode='bilinear',align_corners=False).expand(len(image),-1,-1,-1)
            raw=self.decoder(self.encoder(torch.cat((image,prior),dim=1)))
            raw=nn.functional.interpolate(raw,size=image.shape[-2:],mode='bilinear',align_corners=False)
            normals=nn.functional.normalize(prior+torch.tanh(raw[:,:3])*.15,dim=1,eps=1e-6)
            return normals,raw[:,3:4]
    return CueNet()


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


def train(dataset,out,steps=400,device='cpu'):
    import torch
    from ..rig import safetensors as st
    from scipy.ndimage import binary_erosion
    torch.set_num_threads(4);torch.manual_seed(1234);np.random.seed(1234)
    dataset,out=Path(dataset),artifact_path(out)
    if out.exists():raise ValueError('choose a fresh training directory')
    manifest=json.loads(dataset.with_suffix('.json').read_text())
    if manifest.get('format')!='vhuman.synthetic_cues.v1' or manifest.get('real_media_used') is not False or sha256(dataset)!=manifest['dataset_sha256']:
        raise ValueError('verified synthetic-only cue dataset required')
    if not 50<=steps<=5000:raise ValueError('steps must be 50..5000')
    with np.load(dataset,allow_pickle=False) as z:rgb=z['rgb'].transpose(0,3,1,2);truth=z['normals'].transpose(0,3,1,2);mask=z['mask'];groups=z['identity'];priors=z['prior'].transpose(0,3,1,2)
    identities=np.unique(groups);heldout=identities[-max(1,len(identities)//4):]
    fit=~np.isin(groups,heldout);test=~fit
    if fit.sum()<4 or test.sum()<2:raise ValueError('identity-separated fit/holdout required')
    model=network(rgb.shape[-1]).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    mean=(truth[fit]*mask[fit,None]).sum(0)/np.maximum(mask[fit].sum(0)[None],1)
    mean/=np.maximum(np.linalg.norm(mean,axis=0,keepdims=True),1e-9)
    mean[:,mask[fit].sum(0)==0]=np.array([0,0,1])[:,None]
    model.prior.copy_(torch.tensor(mean[None],device=device))
    prior_tensor=torch.tensor(priors[fit],device=device)
    x=torch.tensor(rgb[fit],device=device);y=torch.tensor(truth[fit],device=device);m=torch.tensor(mask[fit,None],dtype=torch.float32,device=device)
    for step in range(steps):
        ids=torch.randint(len(x),(min(8,len(x)),),device=device);normal,logits=model(x[ids],prior_tensor[ids])
        cosine=(1-(normal*y[ids]).sum(1,keepdim=True))*m[ids]
        loss=cosine.sum()/m[ids].sum().clamp_min(1)+torch.nn.functional.binary_cross_entropy_with_logits(logits,m[ids])*.15
        loss+=((normal-prior_tensor[ids])**2*m[ids]).sum()/m[ids].sum().clamp_min(1)*.02
        optimizer.zero_grad();loss.backward();optimizer.step()
    model.eval()
    with torch.inference_mode():
        predicted,logits=model(torch.tensor(rgb[test],device=device),torch.tensor(priors[test],device=device))
    predicted=predicted.cpu().numpy();confidence=torch.sigmoid(logits).cpu().numpy()[:,0]
    # Fixed reference masks score normals; predicted confidence cannot hide errors.
    interior=np.array([binary_erosion(row,iterations=1) for row in mask[test]])
    learned=metrics(predicted,truth[test],interior)
    flat=np.zeros_like(predicted);flat[:,2]=1
    baseline=metrics(flat,truth[test],interior)
    template=metrics(priors[test],truth[test],interior)
    accepted=learned['mean_degrees']<template['mean_degrees']*.95
    out.mkdir(parents=True)
    st.save(out/'normal_cue.safetensors',{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    np.savez_compressed(out/'heldout.npz',rgb=rgb[test],normals=predicted,truth=truth[test],confidence=confidence,mask=mask[test])
    report=dict(format='vhuman.normal_cue.v1',architecture=ARCHITECTURE,code_sha256=CODE_SHA256,torch_version=torch.__version__,weights_sha256=sha256(out/'normal_cue.safetensors'),dataset_sha256=manifest['dataset_sha256'],
                training_identities=identities[~np.isin(identities,heldout)].tolist(),heldout_identities=heldout.tolist(),
                steps=steps,device=device,resolution=int(rgb.shape[-1]),parameters=sum(p.numel() for p in model.parameters()),
                heldout_normals=learned,flat_baseline=baseline,geometry_prior_baseline=template,synthetic_gate_passed=bool(accepted),
                real_geometry_gate_passed=False,default_enabled=False,
                limitations=['synthetic normal/mask pilot; no real-domain accuracy claim',
                             'mask probability is not calibrated normal uncertainty',
                             'do not use for geometry refinement until a separate real-data gate passes'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def infer(image,checkpoint,out,prior_file=None):
    import torch
    from PIL import Image
    from ..rig import safetensors as st
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
    weights,_=st.load(checkpoint/'normal_cue.safetensors')
    model=network(side);model.load_state_dict({k:torch.tensor(v) for k,v in weights.items()});model.eval()
    with torch.inference_mode():normal,logits=model(torch.tensor(rgb.transpose(2,0,1)[None]),torch.tensor(geometry_prior.transpose(2,0,1)[None],dtype=torch.float32))
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,normals=normal[0].permute(1,2,0).numpy(),mask_probability=torch.sigmoid(logits)[0,0].numpy())
    return dict(status='experimental inference',real_geometry_gate_passed=False,image_sha256=sha256(image),geometry_prior_sha256=sha256(prior_file))


def validate_real(checkpoint,prepared,candidate,out):
    """Real scan-normal ablation; evaluates frozen weights, never optimizes them."""
    import torch
    from PIL import Image
    from scipy.ndimage import binary_erosion
    from ..rig import safetensors as st
    from ..rig.common import vertex_normals
    from . import observations
    checkpoint,prepared,candidate,out=map(Path,(checkpoint,prepared,candidate,out))
    out=artifact_path(out)
    report=json.loads((checkpoint/'report.json').read_text())
    if report.get('architecture')!=ARCHITECTURE:raise ValueError('unsupported cue architecture; train with the current implementation')
    if sha256(checkpoint/'normal_cue.safetensors')!=report['weights_sha256']:raise ValueError('cue checkpoint changed')
    tensors,_=st.load(checkpoint/'normal_cue.safetensors');model=network(report.get('resolution',64))
    model.load_state_dict({k:torch.tensor(v) for k,v in tensors.items()});model.eval()
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
        with torch.inference_mode():prediction,logits=model(torch.tensor(image.transpose(2,0,1)[None]),torch.tensor(base.transpose(2,0,1)[None]))
        estimated=prediction[0].permute(1,2,0).numpy();probability=torch.sigmoid(logits)[0,0].numpy()
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
    fit=sub.add_parser('train');fit.add_argument('--dataset',type=Path,required=True);fit.add_argument('--out',type=Path,required=True);fit.add_argument('--steps',type=int,default=400);fit.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    predict=sub.add_parser('infer');predict.add_argument('--image',type=Path,required=True);predict.add_argument('--prior',type=Path,required=True);predict.add_argument('--checkpoint',type=Path,required=True);predict.add_argument('--out',type=Path,required=True)
    validate=sub.add_parser('validate-real')
    for name in ('checkpoint','prepared','candidate','out'):validate.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();result=synthesize(a.out) if a.action=='synthesize' else train(a.dataset,a.out,a.steps,a.device) if a.action=='train' else validate_real(a.checkpoint,a.prepared,a.candidate,a.out) if a.action=='validate-real' else infer(a.image,a.checkpoint,a.out,a.prior)
    print(json.dumps(result))


if __name__=='__main__':main()
