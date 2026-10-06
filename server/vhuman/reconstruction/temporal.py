"""Identity-frozen native GNM fitting to provenance-checked I2V catalogs.

Generated video is synthetic motion evidence, not a measured 3D scan. Dense
landmarks are fitted through the calibrated portrait camera and explicit crop.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps
from .. import gpu
from ..rig.gnm_model import GNMModel
from .dense_landmarks import attachments
from .observations import sha256
from .reference import Camera


def crop_camera(camera, source_size, target_size):
    """Exact centred ImageOps.fit calibration, preserving portrait extrinsics."""
    scale=max(target_size[0]/source_size[0],target_size[1]/source_size[1])
    offset=(np.asarray(source_size)*scale-target_size)/2
    c=camera.scaled(scale)
    c.cx-=offset[0];c.cy-=offset[1]
    return c


def fit_clip(candidate, folder, out, *, iterations=200, modes=64, device='cuda:0'):
    import torch
    candidate,folder,out=map(Path,(candidate,folder,out))
    if torch.version.hip is None and str(device).startswith('cuda'):
        raise RuntimeError('temporal fitting requires PyTorch ROCm')
    if not 20<=iterations<=2000 or not 8<=modes<=128:
        raise ValueError('invalid temporal iteration or mode count')
    if out.exists() and any(out.iterdir()):raise ValueError('temporal output must be empty')
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    clip=folder/'video/clip.mp4'
    observations=json.loads((folder/'observations.json').read_text())
    if observations['source_sha256']!=sha256(clip):raise ValueError('cached landmarks do not match clip')
    w,h=observations['width'],observations['height']
    # Refuse the former synthetic-face rig mismatch. Check actual reference pixels.
    original=Image.open(candidate/'portrait.png').convert('RGB')
    expected=ImageOps.fit(original,(w,h))
    reference=Image.open(folder.parent/'reference.png').convert('RGB')
    if expected.size!=reference.size or not np.array_equal(np.asarray(expected),np.asarray(reference)):
        raise ValueError('generated catalog reference belongs to a different portrait or crop')
    generation=json.loads((folder/'video/manifest.json').read_text())
    reference_hash=sha256(folder.parent/'reference.png')
    receipts=[r.get('sha256') for r in generation.get('references',[])]
    receipts.extend([generation.get('image_sha256'),generation.get('source_portrait_sha256')])
    adapter_receipt=folder/'video/adapter_receipt.json'
    if adapter_receipt.is_file():
        receipts.append(json.loads(adapter_receipt.read_text()).get('source_portrait_sha256'))
    if reference_hash not in receipts:raise ValueError('generated video reference hash does not match portrait crop')
    from .provenance import validate_candidate
    manifest=validate_candidate(candidate)
    camera=crop_camera(Camera.from_dict(manifest['geometry']['fitted_cameras'][0]),original.size,(w,h))
    with np.load(candidate/'geometry.npz',allow_pickle=False) as data:
        beta=data['gnm_identity'];reference_expression=data['gnm_expressions'][0]
        scale=float(data['scale']);rotation=data['rotation'];rest=data['full_neutral']
        triangles=data['full_triangles']
    model=GNMModel(device=device)
    native_rest,_=model.evaluate(beta)
    offset=np.median(rest-scale*native_rest.detach().cpu().numpy()@rotation.T,axis=0)
    residual=rest-(scale*native_rest.detach().cpu().numpy()@rotation.T+offset)
    bind_residual=residual@rotation/scale
    dense_ids,bary,confidence=attachments()
    exterior=np.flatnonzero(model.group('skin_exterior'))
    full_ids=exterior[dense_ids]
    pupil_ids=[]
    for side in ('right','left'):
        ids=np.flatnonzero(model.group('pupils')&model.group(side+'_eye'))
        if not len(ids):raise ValueError('GNM pupil anatomy missing')
        pupil_ids.append(ids)
    sampled=np.unique(np.concatenate((full_ids.ravel(),*pupil_ids)))
    remap=np.full(len(rest),-1,int);remap[sampled]=np.arange(len(sampled))
    frames=len(observations['observations'])
    if not 2<=frames<=129:raise ValueError('temporal clips require 2..129 frames')
    target=np.zeros((frames,470,2),np.float32);weights=np.zeros((frames,470),np.float32)
    with np.load(folder/'face_parsing.npz',allow_pickle=False) as data:
        labels=data['labels']
    if labels.shape!=(frames,h,w):raise ValueError('parsing dimensions mismatch')
    for i,row in enumerate(observations['observations']):
        if not row['visible'] or row.get('landmarks') is None:continue
        points=np.asarray(row['landmarks'],np.float32)
        if points.shape!=(478,3) or not np.isfinite(points).all():raise ValueError('invalid native landmarks')
        target[i,:468]=points[:468,:2]*[w,h]
        target[i,468:]=points[[468,473],:2]*[w,h]
        x=np.clip(target[i,:,0].astype(int),0,w-1);y=np.clip(target[i,:,1].astype(int),0,h-1)
        unoccluded=~np.isin(labels[i,y,x],[6,9,15,16,17,18])
        weights[i]=np.r_[confidence,np.ones(2)]*unoccluded
        ocular=np.array([33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,
            263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249])
        under_lens=labels[i,y[ocular],x[ocular]]==6
        weights[i,ocular[under_lens]]=confidence[ocular[under_lens]]*.15
        # Iris centres can be tracked through transparent lenses; retain a low
        # weight and report the uncertainty rather than treating frame pixels as skin.
        weights[i,468:]=.2 if np.any(labels[i,y[468:],x[468:]]==6) else .8
    if (weights.sum(1)<30).any():raise ValueError('insufficient unoccluded temporal landmarks')
    ipd=np.linalg.norm(target[:,468]-target[:,469],axis=-1)
    if (ipd<10).any():raise ValueError('invalid tracked inter-pupil distance')
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float32,device=device)
    # Observable coefficient combinations are selected on fixed anatomical
    # attachments. Eye and tongue modes unsupported by landmarks stay priors.
    basis=model.data['expression_basis'][:,full_ids]
    attached=(basis*bary[None,:,:,None]).sum(2)
    _,singular,vt=np.linalg.svd((attached*np.sqrt(confidence)[None,:,None]).reshape(383,-1).T,full_matrices=False)
    rank=min(modes,int((singular>singular[0]*1e-4).sum()))
    latent_basis=tensor(vt[:rank])
    baseline=tensor(reference_expression)
    latent=torch.zeros((frames,rank),device=device,requires_grad=True)
    head_rotation=torch.zeros((frames,3),device=device,requires_grad=True)
    eye_rotation=torch.zeros((frames,2,2),device=device,requires_grad=True)
    translation=torch.zeros((frames,3),device=device,requires_grad=True)
    optimizer=torch.optim.Adam([{'params':[latent],'lr':.04},
        {'params':[head_rotation,eye_rotation],'lr':.003},{'params':[translation],'lr':.0005}])
    evaluator=model.frame_evaluator(beta,sampled,bind_residual=bind_residual)
    rot=tensor(rotation);off=tensor(offset)
    target_t,weight_t,ipd_t=tensor(target),tensor(weights),tensor(ipd)
    attachment_ids=torch.as_tensor(remap[full_ids],device=device)
    bary_t=tensor(bary)
    camera_r=tensor(camera.rotation);camera_o=tensor(camera.origin)
    exterior_mask=model.group('skin_exterior')
    all_tri=model.data['triangles']
    skin_tri=all_tri[exterior_mask[all_tri].all(1)]
    skin_remap=np.full(len(rest),-1,int);skin_remap[exterior]=np.arange(len(exterior))
    guard_tri=torch.as_tensor(skin_remap[skin_tri],device=device)
    bind_guard=native_rest[exterior].detach()+tensor(bind_residual[exterior])
    guard_basis=model.tensors['expression_basis'][:,exterior].reshape(383,-1)
    base_faces=bind_guard[guard_tri]
    base_normal=torch.linalg.cross(base_faces[:,1]-base_faces[:,0],base_faces[:,2]-base_faces[:,0])
    base_area2=base_normal.square().sum(-1)
    active=base_area2>1e-24

    def jacobian_guard(coefficients):
        positions=bind_guard[None]+(coefficients@guard_basis).reshape(frames,len(exterior),3)
        faces=positions[:,guard_tri]
        normals=torch.linalg.cross(faces[:,:,1]-faces[:,:,0],faces[:,:,2]-faces[:,:,0])
        return (normals[:,active]*base_normal[None,active]).sum(-1)/base_area2[None,active]

    def forward(override=None):
        coefficients=torch.clamp(baseline+latent@latent_basis,-3,3) if override is None else override
        zero=torch.zeros((frames,1,3),device=device)
        eyes=torch.cat((eye_rotation,torch.zeros((frames,2,1),device=device)),-1)
        rotations=torch.cat((zero,head_rotation[:,None],eyes),1)
        vertices,joints=evaluator(coefficients,rotations,translation)
        points=scale*(vertices@rot.T)+off
        landmarks=(points[:,attachment_ids]*bary_t[None,:,:,None]).sum(2)
        pupils=torch.stack([points[:,remap[ids]].mean(1) for ids in pupil_ids],1)
        p=(torch.cat((landmarks,pupils),1)-camera_o)@camera_r.T
        depth=torch.clamp(-p[...,2],min=.02)
        projected=torch.stack(((camera.focal*p[...,0]-camera.skew*p[...,1])/depth+camera.cx,
            -(camera.focal_y or camera.focal)*p[...,1]/depth+camera.cy),-1)
        return coefficients,rotations,projected

    with torch.no_grad():before=forward()[2].detach()
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        coefficients,rotations,pixels=forward()
        error=(pixels-target_t)/ipd_t[:,None,None]
        robust=torch.sqrt(error.square().sum(-1)+.005**2)-.005
        loss=(robust*weight_t).sum()/weight_t.sum()+.0001*latent.square().mean()
        loss+=.02*translation.square().mean()+.0001*eye_rotation.square().mean()
        loss+=.004*(coefficients[1:]-coefficients[:-1]).square().mean()
        loss+=.01*(head_rotation[1:]-head_rotation[:-1]).square().mean()
        barrier=torch.relu(.15-jacobian_guard(coefficients)).square()
        loss+=10*barrier.topk(min(64,barrier.shape[1]),dim=1).values.mean()
        loss.backward();optimizer.step()
        with torch.no_grad():
            head_rotation.clamp_(-.8,.8);eye_rotation.clamp_(-.6,.6);translation.clamp_(-.03,.03)
        if iteration%50==0:print(f'{folder.name}: native temporal {iteration}/{iterations} loss={float(loss.detach()):.5f}',flush=True)
    with torch.no_grad():
        coefficients,rotations,pixels=forward()
        expression_steps=torch.ones(frames,device=device)
        # The differentiable penalty improves the solve; a hard final guard
        # guarantees that exported skin cannot retain a folded PCA triangle.
        for _ in range(12):
            unsafe=jacobian_guard(coefficients).min(1).values<.05
            if not bool(unsafe.any()):break
            coefficients[unsafe]*=.5;expression_steps[unsafe]*=.5
        coefficients,rotations,pixels=forward(coefficients)
        min_jacobian=jacobian_guard(coefficients).min(1).values
        if not bool((min_jacobian>=.05).all()):raise ValueError('native expression failed skin topology guard')
        full_evaluator=model.frame_evaluator(beta,bind_residual=bind_residual)
        vertices,joints=full_evaluator(coefficients,rotations,translation)
        # The shared identity residual participates in native LBS in bind space.
        vertices=scale*(vertices@rot.T)+off
        joints=scale*(joints@rot.T)+off
    numpy=lambda v:v.detach().cpu().numpy()
    normalized=np.linalg.norm(numpy(pixels)-target,axis=-1)/ipd[:,None]
    per_frame=(normalized*weights).sum(1)/weights.sum(1)
    before_error=np.linalg.norm(numpy(before)-target,axis=-1)/ipd[:,None]
    np.savez_compressed(out/'motion.npz',vertices=numpy(vertices),joints=numpy(joints),
        identity=beta,expression=numpy(coefficients),rotations=numpy(rotations),translation=numpy(translation),
        projected_landmarks=numpy(pixels),observed_landmarks=target,weights=weights,triangles=triangles)
    result=dict(schema='vhuman.native_gnm_motion.v1',candidate=str(candidate.resolve()),
        candidate_geometry_sha256=sha256(candidate/'geometry.npz'),clip=str(clip.resolve()),clip_sha256=sha256(clip),
        synthetic=True,identity_frozen=True,identity_sha256=hashlib.sha256(beta.tobytes()).hexdigest(),
        residual_transport='native_lbs_bind_space',
        frames=frames,fps=observations['fps'],size=[w,h],camera=camera.as_dict(),
        expression_rank=rank,expression_dim=383,iterations=iterations,
        landmark_error_before_ipd=float((before_error*weights).sum()/weights.sum()),
        landmark_error_after_ipd=float((normalized*weights).sum()/weights.sum()),
        per_frame_error_ipd=per_frame.tolist(),quality_gate=bool((per_frame<.05).all()),
        skin_min_oriented_area_ratio=numpy(min_jacobian).tolist(),expression_safe_steps=numpy(expression_steps).tolist(),
        peak_torch_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2 if str(device).startswith('cuda') else None,
        seconds=time.monotonic()-started,limitations=['I2V landmarks do not establish true 3D identity',
        'hidden anatomy remains a prior','camera intrinsics fixed from source; generated camera drift is nuisance motion',
        'bind-space residual uses native skinning; additional pose-dependent detail remains a prior',
        'dense canonical attachments beyond GNM68 are inferred correspondences'])
    result['ocular_observations']='native lid/iris tracks under glasses retain low authored weights; they do not become skin-color observations'
    result['motion_sha256']=sha256(out/'motion.npz')
    (out/'motion.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate');parser.add_argument('--catalog',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--expressions',default='neutral,happy,sad,angry,fear,surprise,disgust')
    parser.add_argument('--iterations',type=int,default=200);parser.add_argument('--modes',type=int,default=64)
    args=parser.parse_args();catalog=Path(args.catalog);out=Path(args.out)
    with gpu.execution('rocm',gpu.device_index()),gpu.device_session(2048):
        results={}
        for expression in args.expressions.split(','):
            if expression not in ('neutral','happy','sad','angry','fear','surprise','disgust'):
                raise ValueError('unknown catalog expression')
            results[expression]=fit_clip(args.candidate,catalog/expression,out/expression,iterations=args.iterations,modes=args.modes,
                                        device=f'cuda:{gpu.device_index()}')
        (out/'catalog.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
