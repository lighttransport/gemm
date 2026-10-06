"""Refine a portrait's native GNM expression with frozen identity and camera.

Uses fixed anatomical correspondences, a held-out landmark gate, and an
oriented-triangle barrier. The portrait is evidence for expression, not depth.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .provenance import validate_candidate
from .reference import Camera
from .observations import load,sha256

MOUTH=np.array([61,40,37,0,267,270,291,321,314,17,84,91,78,81,13,311,308,402,14,178])


def landmark_metrics(projected,target,weights,heldout):
    error=np.linalg.norm(projected-target,axis=-1)
    def mean(selection):
        w=weights*selection
        return float((error*w).sum()/max(w.sum(),1e-12))
    mouth=np.isin(np.arange(len(weights)),MOUTH)
    return dict(all_px=mean(np.ones(len(weights),bool)),heldout_px=mean(heldout),
                mouth_px=mean(mouth),heldout_mouth_px=mean(mouth&heldout),
                nonmouth_px=mean(~mouth))


def acceptance_gate(before,after,minimum_area_ratio):
    values=[minimum_area_ratio,*before.values(),*after.values()]
    return bool(np.isfinite(values).all() and minimum_area_ratio>=.05
        and after['heldout_px']<before['heldout_px']*.98
        and after['heldout_mouth_px']<before['heldout_mouth_px']*.98
        and after['mouth_px']<before['mouth_px']*.9
        and after['nonmouth_px']<=before['nonmouth_px']*1.1)


def refine(candidate,out, *, iterations=400,modes=64,device='cuda:0'):
    import torch
    from ..rig.gnm_model import GNMModel
    from .dense_landmarks import attachments
    from .materials import bake_portrait
    from ..face_parsing import FaceParser
    from PIL import Image
    if not 50<=iterations<=2000 or not 8<=modes<=128:raise ValueError('invalid fit budget')
    if str(device).startswith('cuda') and torch.version.hip is None:raise RuntimeError('PyTorch ROCm required')
    candidate,out=Path(candidate).resolve(),Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('output must be empty')
    manifest=validate_candidate(candidate);observations=load(candidate/'observations.json')
    if len(observations['views'])!=1:raise ValueError('native portrait refinement requires one source view')
    view=observations['views'][0];camera=Camera.from_dict(manifest['geometry']['fitted_cameras'][0])
    with np.load(candidate/'geometry.npz',allow_pickle=False) as data:geometry={k:data[k] for k in data.files}
    model=GNMModel();skin=np.flatnonzero(model.group('skin_exterior'))
    ids,bary,confidence=attachments();full_ids=skin[ids]
    target=np.zeros((468,2));weights=np.zeros(468)
    labels,certainty=FaceParser().predict(np.asarray(Image.open(candidate/'portrait.png').convert('RGB')))
    for i in range(468):
        anchor=view['anchors'].get(f'mp_{i:03d}')
        if anchor is None:continue
        target[i]=anchor['xy'];x,y=np.floor(target[i]).astype(int)
        if 0<=x<labels.shape[1] and 0<=y<labels.shape[0] and labels[y,x] not in (6,9,15,16,17,18):
            weights[i]=anchor['weight']
    if (weights>0).sum()<100:raise ValueError('insufficient visible fixed dense landmarks')
    heldout=np.arange(468)%10==0
    if ((weights>0)&heldout).sum()<20:raise ValueError('insufficient held-out landmarks')
    training=weights.copy();training[heldout]=0;training[MOUTH]*=3
    basis=np.asarray(float(geometry['scale'])*model.data['expression_basis']@geometry['rotation'].T,np.float32)
    attached=(basis[:,full_ids]*bary[None,:,:,None]).sum(2)
    # Select coefficient combinations observable on TRAINING attachments only.
    matrix=(attached*np.sqrt(training)[None,:,None]).reshape(383,-1).T
    _,singular,vt=np.linalg.svd(matrix,full_matrices=False)
    rank=min(modes,int((singular>singular[0]*1e-4).sum()))
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device=device)
    rest=tensor(geometry['full_neutral']);baseline=tensor(geometry['gnm_expressions'][0])
    latent_basis=tensor(vt[:rank]);latent=torch.zeros(rank,device=device,requires_grad=True)
    attached_basis=tensor(attached);attached_rest=(rest[full_ids]*tensor(bary)[:,:,None]).sum(1)
    exterior=geometry['full_neutral'][skin];triangles=geometry['triangles']
    guard_basis=tensor(basis[:,skin].reshape(383,-1));guard_rest=tensor(exterior)
    tri=torch.as_tensor(triangles,device=device)
    faces=guard_rest[tri];normals=torch.linalg.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0])
    area=normals.square().sum(-1);active=area>1e-24
    cr,co=tensor(camera.rotation),tensor(camera.origin)
    target_t,weight=tensor(target),tensor(training)
    ipd=float(np.linalg.norm(np.asarray(view['anchors']['eye_left']['xy'])-view['anchors']['eye_right']['xy']))
    def project(coeff):
        points=attached_rest+torch.einsum('i,ivc->vc',coeff,attached_basis)
        p=(points-co)@cr.T;z=torch.clamp(-p[:,2],min=.02)
        return torch.stack(((camera.focal*p[:,0]-camera.skew*p[:,1])/z+camera.cx,
            -(camera.focal_y or camera.focal)*p[:,1]/z+camera.cy),-1)
    def area_ratio(coeff):
        vertices=guard_rest+(coeff@guard_basis).reshape(-1,3);f=vertices[tri]
        n=torch.linalg.cross(f[:,1]-f[:,0],f[:,2]-f[:,0])
        return (n[active]*normals[active]).sum(-1)/area[active]
    numpy=lambda a:a.detach().cpu().numpy()
    before=landmark_metrics(numpy(project(baseline)),target,weights,heldout)
    optimizer=torch.optim.Adam([latent],lr=.025)
    started=time.monotonic()
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        coeff=torch.clamp(baseline+latent@latent_basis,-3,3)
        residual=(project(coeff)-target_t)/ipd
        robust=torch.sqrt(residual.square().sum(-1)+.005**2)-.005
        loss=(robust*weight).sum()/weight.sum()+.0002*latent.square().mean()
        barrier=torch.relu(.15-area_ratio(coeff)).square()
        loss+=10*barrier.topk(min(64,len(barrier))).values.mean()
        loss.backward();optimizer.step()
        if iteration%100==0:print(f'native portrait {iteration}/{iterations}: {float(loss.detach()):.6f}',flush=True)
    with torch.no_grad():
        coeff=torch.clamp(baseline+latent@latent_basis,-3,3)
        minimum=float(area_ratio(coeff).min());pixels=numpy(project(coeff));after=landmark_metrics(pixels,target,weights,heldout)
    accepted=acceptance_gate(before,after,minimum)
    report=dict(schema='vhuman.native_portrait_refinement.v1',accepted=bool(accepted),identity_frozen=True,camera_frozen=True,
        before=before,after=after,expression_rank=rank,expression_dim=383,iterations=iterations,
        skin_min_oriented_area_ratio=minimum,expression_safe_step=1.,training_landmarks=int((training>0).sum()),
        heldout_landmarks=int(((weights>0)&heldout).sum()),source_geometry_sha256=manifest['geometry_sha256'],
        peak_torch_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2 if str(device).startswith('cuda') else None,
        seconds=time.monotonic()-started,limitations=['held-out tracker consistency is not measured 3D accuracy',
            'identity, camera, teeth size and unseen anatomy remain source priors','dense attachments beyond GNM68 are inferred'])
    out.mkdir(parents=True,exist_ok=True)
    (out/'fit_refinement.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'landmark_diagnostic.npz',before=numpy(project(baseline)),after=pixels,target=target,weights=weights,heldout=heldout)
    if not accepted:return report
    for file in candidate.iterdir():
        if file.is_file() and file.name not in ('fit_refinement.json','landmark_diagnostic.npz','manifest.json'):
            shutil.copyfile(file,out/file.name)
    coefficients=numpy(coeff)
    geometry['gnm_expressions']=coefficients[None]
    geometry['full_captured']=(geometry['full_neutral']+np.einsum('i,ivc->vc',coefficients,basis))[None].astype(np.float32)
    geometry['captured']=geometry['full_captured'][:,skin]
    np.savez_compressed(out/'geometry.npz',**geometry)
    manifest['id']=out.name;manifest['geometry_sha256']=sha256(out/'geometry.npz')
    manifest['geometry']['native_expression_refinement']=report
    manifest['geometry']['expression_modes']=list(range(383));manifest['geometry']['expression_coefficients']=[coefficients.tolist()]
    manifest['geometry']['expression_steps']=[1.]
    manifest['geometry']['expression_labels']=model.data['expression_names'].tolist()
    manifest['deformation']='capture expression changed: rebuild rig and motion fits for this candidate'
    material=bake_portrait(geometry['captured'],geometry['triangles'],geometry['triangle_uvs'],[view],[camera],out,
        res=manifest['config']['texture_res'],roughness=manifest['material']['roughness']['value'],f0=manifest['material']['f0']['value'])
    if manifest['material'].get('authored_detail'):
        shutil.copyfile(candidate/'skin_normal.png',out/'skin_normal.png')
        material['authored_detail']=manifest['material']['authored_detail']
    manifest['material']=material
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));validate_candidate(out)
    return report


def main():
    from .. import gpu
    from contextlib import nullcontext
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate');p.add_argument('--out',required=True)
    p.add_argument('--iterations',type=int,default=400);p.add_argument('--modes',type=int,default=64)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    backend='rocm' if a.device.startswith('cuda') else 'cpu';index=int(a.device.split(':')[-1]) if ':' in a.device else 0
    with gpu.execution(backend,index):
        with gpu.device_session(2048) if backend=='rocm' else nullcontext():
            print(json.dumps(refine(**vars(a)),indent=2))


if __name__=='__main__':main()
