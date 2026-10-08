"""Refine a portrait's GNM fit with a fixed camera and anatomical attachments.

Optional native identity and bounded surface corrections complement expression
fitting. Held-out landmarks and oriented-triangle gates validate the result.
Single-view agreement does not establish true facial depth.
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


def surface_basis(vertices,training_points,count=192,width=.008):
    """Smooth metric correction field, with centres chosen from training only."""
    points=np.asarray(training_points)
    vertices=np.asarray(vertices)
    if (points.ndim!=2 or points.shape[1]!=3 or len(points)==0
            or vertices.ndim!=2 or vertices.shape[1]!=3 or count<1
            or not np.isfinite(width) or width<=0
            or not np.isfinite(points).all() or not np.isfinite(vertices).all()):
        raise ValueError('invalid metric surface field')
    selected=[int(np.argmin(np.linalg.norm(points-points.mean(0),axis=1)))]
    distance=np.full(len(points),np.inf)
    for _ in range(min(count,len(points))-1):
        distance=np.minimum(distance,np.square(points-points[selected[-1]]).sum(1))
        selected.append(int(distance.argmax()))
    centers=points[selected]
    kernel=np.exp(-np.square(np.asarray(vertices)[:,None]-centers).sum(-1)/(2*width**2))
    # Partition of unity near observed skin, fading to zero far from evidence.
    return (kernel/(kernel.sum(1,keepdims=True)+.01)).astype(np.float32)


def surface_displacement(basis,latent,camera_rotation,limit_mm):
    """Convex bounded control blend; zero camera-depth displacement."""
    import torch
    controls=latent/torch.sqrt(1+latent.square().sum(-1,keepdim=True))
    return (limit_mm*.001)*(basis@controls)@camera_rotation[:2]


def landmark_metrics(projected,target,weights,heldout):
    error=np.linalg.norm(projected-target,axis=-1)
    def mean(selection):
        w=weights*selection
        return float((error*w).sum()/max(w.sum(),1e-12))
    mouth=np.isin(np.arange(len(weights)),MOUTH)
    return dict(all_px=mean(np.ones(len(weights),bool)),heldout_px=mean(heldout),
                mouth_px=mean(mouth),heldout_mouth_px=mean(mouth&heldout),
                nonmouth_px=mean(~mouth))


def acceptance_gate(before,after,minimum_area_ratio,target_px=None):
    values=[minimum_area_ratio,*before.values(),*after.values()]
    return bool(np.isfinite(values).all() and minimum_area_ratio>=.05
        and after['heldout_px']<before['heldout_px']*.98
        and after['heldout_mouth_px']<before['heldout_mouth_px']*.98
        and after['mouth_px']<before['mouth_px']*.9
        and after['nonmouth_px']<=before['nonmouth_px']*1.1
        and (target_px is None or after['heldout_px']<target_px))


def refine(candidate,out, *, iterations=400,modes=64,identity_modes=0,surface_mm=0.,target_px=None,device='cuda:0',parsing_model=None):
    import torch
    from ..rig.gnm_model import GNMModel
    from .dense_landmarks import attachments
    from .materials import bake_portrait
    from ..face_parsing import FaceParser
    from PIL import Image
    if not 50<=iterations<=2000 or not 8<=modes<=128 or not 0<=identity_modes<=128:raise ValueError('invalid fit budget')
    if not 0<=surface_mm<=5:raise ValueError('surface correction must be 0..5 mm')
    if target_px is not None and (not np.isfinite(target_px) or target_px<=0):raise ValueError('target error must be finite and positive')
    if str(device).startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('PyTorch CUDA/ROCm device unavailable')
    candidate,out=Path(candidate).resolve(),Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('output must be empty')
    manifest=validate_candidate(candidate);observations=load(candidate/'observations.json')
    if len(observations['views'])!=1:raise ValueError('native portrait refinement requires one source view')
    view=observations['views'][0];camera=Camera.from_dict(manifest['geometry']['fitted_cameras'][0])
    with np.load(candidate/'geometry.npz',allow_pickle=False) as data:geometry={k:data[k] for k in data.files}
    model=GNMModel();skin=np.flatnonzero(model.group('skin_exterior'))
    ids,bary,_=attachments();full_ids=skin[ids]
    target=np.zeros((468,2));weights=np.zeros(468);lip_landmarks=np.zeros(468,bool)
    labels,_=FaceParser(model=parsing_model).predict(np.asarray(Image.open(candidate/'portrait.png').convert('RGB')))
    for i in range(468):
        anchor=view['anchors'].get(f'mp_{i:03d}')
        if anchor is None:continue
        target[i]=anchor['xy'];x,y=np.floor(target[i]).astype(int)
        if 0<=x<labels.shape[1] and 0<=y<labels.shape[0] and labels[y,x] not in (6,9,15,16,17,18):
            weights[i]=anchor['weight']
            lip_landmarks[i]=labels[y,x] in (12,13)
    if (weights>0).sum()<100:raise ValueError('insufficient visible fixed dense landmarks')
    heldout=np.arange(468)%10==0
    if ((weights>0)&heldout).sum()<20:raise ValueError('insufficient held-out landmarks')
    lip_landmarks[MOUTH]=True
    training=weights.copy();training[heldout]=0;training[lip_landmarks]*=3
    basis=np.asarray(float(geometry['scale'])*model.data['expression_basis']@geometry['rotation'].T,np.float32)
    attached=(basis[:,full_ids]*bary[None,:,:,None]).sum(2)
    # Select coefficient combinations observable on TRAINING attachments only.
    matrix=(attached*np.sqrt(training)[None,:,None]).reshape(383,-1).T
    _,singular,vt=np.linalg.svd(matrix,full_matrices=False)
    rank=min(modes,int((singular>singular[0]*1e-4).sum()))
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device=device)
    rest=tensor(geometry['full_neutral']);baseline=tensor(geometry['gnm_expressions'][0])
    latent_basis=tensor(vt[:rank]);latent=torch.zeros(rank,device=device,requires_grad=True)
    identity_basis=np.asarray(float(geometry['scale'])*model.data['vertex_identity_basis']@geometry['rotation'].T,np.float32)
    identity_attached=(identity_basis[:,full_ids]*bary[None,:,:,None]).sum(2)
    identity_matrix=(identity_attached*np.sqrt(training)[None,:,None]).reshape(model.identity_dim,-1).T
    _,identity_singular,identity_vt=np.linalg.svd(identity_matrix,full_matrices=False)
    identity_rank=min(identity_modes,int((identity_singular>identity_singular[0]*1e-4).sum()))
    identity_latent=torch.zeros(identity_rank,device=device,requires_grad=True)
    identity_latent_basis=tensor(identity_vt[:identity_rank])
    identity_baseline=tensor(geometry['gnm_identity'])
    identity_attached_t=tensor(identity_attached)
    attached_neutral=(geometry['full_neutral'][full_ids]*bary[:,:,None]).sum(1)
    correction_basis=tensor(surface_basis(geometry['full_neutral'],attached_neutral[training>0]))
    correction_latent=torch.zeros((correction_basis.shape[1],2),device=device,requires_grad=True)
    correction_rotation=tensor(camera.rotation)
    def surface_delta():
        # A convex blend of bounded control displacements preserves the metric
        # bound everywhere. No displacement along the camera depth axis.
        return surface_displacement(correction_basis,correction_latent,correction_rotation,surface_mm)
    attached_basis=tensor(attached);attached_rest=(rest[full_ids]*tensor(bary)[:,:,None]).sum(1)
    exterior=geometry['full_neutral'][skin];triangles=geometry['triangles']
    guard_basis=tensor(basis[:,skin].reshape(383,-1));guard_rest=tensor(exterior)
    identity_guard=tensor(identity_basis[:,skin].reshape(model.identity_dim,-1))
    tri=torch.as_tensor(triangles,device=device)
    faces=guard_rest[tri];normals=torch.linalg.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0])
    area=normals.square().sum(-1);active=area>1e-24
    cr,co=tensor(camera.rotation),tensor(camera.origin)
    target_t,weight=tensor(target),tensor(training)
    ipd=float(np.linalg.norm(np.asarray(view['anchors']['eye_left']['xy'])-view['anchors']['eye_right']['xy']))
    def identity_delta():
        if not identity_rank:return torch.zeros_like(identity_baseline)
        return torch.clamp(identity_baseline+identity_latent@identity_latent_basis,-3,3)-identity_baseline
    def project(coeff,delta=None,correction=None):
        points=attached_rest+torch.einsum('i,ivc->vc',coeff,attached_basis)
        if delta is not None:points=points+torch.einsum('i,ivc->vc',delta,identity_attached_t)
        if correction is not None:points=points+(correction[full_ids]*tensor(bary)[:,:,None]).sum(1)
        p=(points-co)@cr.T;z=torch.clamp(-p[:,2],min=.02)
        return torch.stack(((camera.focal*p[:,0]-camera.skew*p[:,1])/z+camera.cx,
            -(camera.focal_y or camera.focal)*p[:,1]/z+camera.cy),-1)
    def area_ratio(coeff,delta=None,correction=None):
        neutral=guard_rest if delta is None else guard_rest+(delta@identity_guard).reshape(-1,3)
        if correction is not None:neutral=neutral+correction[skin]
        vertices=neutral+(coeff@guard_basis).reshape(-1,3);f=vertices[tri]
        n=torch.linalg.cross(f[:,1]-f[:,0],f[:,2]-f[:,0])
        neutral_f=neutral[tri]
        neutral_n=torch.linalg.cross(neutral_f[:,1]-neutral_f[:,0],neutral_f[:,2]-neutral_f[:,0])
        return torch.cat(((neutral_n[active]*normals[active]).sum(-1)/area[active],
            (n[active]*neutral_n[active]).sum(-1)/neutral_n[active].square().sum(-1).clamp_min(1e-24)))
    numpy=lambda a:a.detach().cpu().numpy()
    before=landmark_metrics(numpy(project(baseline)),target,weights,heldout)
    optimizer=torch.optim.Adam([latent,identity_latent,correction_latent],lr=.025)
    started=time.monotonic()
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        coeff=torch.clamp(baseline+latent@latent_basis,-3,3)
        delta=identity_delta()
        correction=surface_delta()
        residual=(project(coeff,delta,correction)-target_t)/ipd
        robust=torch.sqrt(residual.square().sum(-1)+(2/ipd)**2)-2/ipd
        loss=(robust*weight).sum()/weight.sum()+.0002*latent.square().mean()
        if identity_rank:loss+=.0005*identity_latent.square().mean()
        if surface_mm:loss+=.0001*correction_latent.square().mean()
        barrier=torch.relu(.15-area_ratio(coeff,delta,correction)).square()
        loss+=10*barrier.topk(min(64,len(barrier))).values.mean()
        loss.backward();optimizer.step()
        if iteration%100==0:print(f'native portrait {iteration}/{iterations}: {float(loss.detach()):.6f}',flush=True)
    with torch.no_grad():
        coeff=torch.clamp(baseline+latent@latent_basis,-3,3)
        delta=identity_delta()
        correction=surface_delta()
        minimum=float(area_ratio(coeff,delta,correction).min());pixels=numpy(project(coeff,delta,correction));after=landmark_metrics(pixels,target,weights,heldout)
    accepted=acceptance_gate(before,after,minimum,target_px)
    report=dict(schema='vhuman.native_portrait_refinement.v2',accepted=bool(accepted),identity_frozen=not identity_rank and not surface_mm,camera_frozen=True,
        identity_coefficients_frozen=not identity_rank,target_px=target_px,
        target_met=None if target_px is None else after['heldout_px']<target_px,
        identity_rank=identity_rank,identity_delta_l2=float(delta.norm()),
        surface_limit_mm=surface_mm,surface_max_mm=float(correction.norm(dim=-1).max())*1000,
        cumulative_surface_max_mm=float(np.linalg.norm(geometry.get('portrait_surface_delta',np.zeros_like(numpy(correction)))+numpy(correction),axis=-1).max())*1000,
        surface_controls=int(correction_basis.shape[1]),surface_width_mm=8.,surface_camera_depth_preserved=True,
        robust_scale_px=2.,lip_training_weight=3.,
        before=before,after=after,expression_rank=rank,expression_dim=383,iterations=iterations,
        skin_min_oriented_area_ratio=minimum,expression_safe_step=1.,training_landmarks=int((training>0).sum()),
        heldout_landmarks=int(((weights>0)&heldout).sum()),source_geometry_sha256=manifest['geometry_sha256'],
        peak_torch_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2 if str(device).startswith('cuda') else None,
        seconds=time.monotonic()-started,limitations=['held-out tracker consistency is not measured 3D accuracy',
            'single-view identity depth and unseen anatomy remain prior-dependent','dense attachments beyond GNM68 are inferred',
            'surface correction is a neutral residual, not a native GNM coefficient; temporal residuals are not dynamically skinned'])
    out.mkdir(parents=True,exist_ok=True)
    (out/'fit_refinement.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'landmark_diagnostic.npz',before=numpy(project(baseline)),after=pixels,target=target,weights=weights,heldout=heldout)
    if not accepted:return report
    for file in candidate.iterdir():
        if file.is_file() and file.name not in ('fit_refinement.json','landmark_diagnostic.npz','manifest.json',
                'generated_skin.json','skin_generated_support.png','skin_projection_repair.png'):
            shutil.copyfile(file,out/file.name)
    coefficients=numpy(coeff)
    delta_np=numpy(delta)
    geometry['gnm_identity']=geometry['gnm_identity']+delta_np
    geometry['portrait_surface_delta']=geometry.get('portrait_surface_delta',np.zeros_like(numpy(correction)))+numpy(correction)
    geometry['full_neutral']=geometry['full_neutral']+np.einsum('i,ivc->vc',delta_np,identity_basis)+numpy(correction)
    geometry['neutral']=geometry['full_neutral'][skin]
    joint_basis=float(geometry['scale'])*model.data['joint_identity_basis']@geometry['rotation'].T
    geometry['gnm_joint_positions']=geometry['gnm_joint_positions']+np.einsum('i,ijc->jc',delta_np,joint_basis)
    geometry['gnm_expressions']=coefficients[None]
    geometry['full_captured']=(geometry['full_neutral']+np.einsum('i,ivc->vc',coefficients,basis))[None].astype(np.float32)
    geometry['captured']=geometry['full_captured'][:,skin]
    np.savez_compressed(out/'geometry.npz',**geometry)
    manifest['id']=out.name;manifest['geometry_sha256']=sha256(out/'geometry.npz')
    manifest['geometry']['native_expression_refinement']=report
    manifest['geometry']['expression_modes']=list(range(383));manifest['geometry']['expression_coefficients']=[coefficients.tolist()]
    manifest['geometry']['expression_steps']=[1.]
    manifest['geometry']['expression_labels']=model.data['expression_names'].tolist()
    if identity_rank:
        manifest['geometry']['identity_coefficients']=geometry['gnm_identity'].tolist()
    manifest['deformation']='portrait geometry changed: rebuild rig and motion fits for this candidate'
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
    p.add_argument('--identity-modes',type=int,default=0)
    p.add_argument('--surface-mm',type=float,default=0.)
    p.add_argument('--target-px',type=float,help='require held-out mean error below this pixel threshold')
    p.add_argument('--parsing-model',help='explicit checksum-pinned face parsing ONNX path')
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    import torch
    backend=('rocm' if torch.version.hip else 'cuda') if a.device.startswith('cuda') else 'cpu';index=int(a.device.split(':')[-1]) if ':' in a.device else 0
    with gpu.execution(backend,index):
        with gpu.device_session(2048) if backend in ('rocm','cuda') else nullcontext():
            print(json.dumps(refine(**vars(a)),indent=2))


if __name__=='__main__':main()
