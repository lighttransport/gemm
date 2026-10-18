"""Shared, explicitly CPU-only GNM portrait and performance fitting.

Cameras stay fixed. All returned surfaces include native object pose; identity
and bind residuals remain canonical. Hidden anatomy is never a fitted target.
"""
from contextlib import contextmanager
import copy
import hashlib
import resource
import time
import sys
import numpy as np

from .reference import Camera
from ..rig.gnm_model import GNMModel, rotation_matrices

SOLVER = 'gnm_full_cpu_v2'
MOUTH = set((61,40,37,0,267,270,291,321,314,17,84,91,78,81,13,311,308,402,14,178))
EYES = set((33,133,159,145,160,158,153,144,263,362,386,374,385,387,380,373,468,473))
SEMANTIC = dict(nose_tip=[4], upper_lip=[13], lower_lip=[14], menton=[152],
                mouth_right=[61], mouth_left=[291], eye_right=[33,133,159,145], eye_left=[263,362,386,374])


@contextmanager
def cpu_threads(threads=16):
    import torch
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError('GNM CPU fitter needs threadpoolctl in this interpreter; install requirements-rig.txt or rerun its environment setup') from exc
    if type(threads) is not int or not 1 <= threads <= 256:
        raise ValueError('threads must be 1..256')
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with threadpool_limits(limits=threads), torch.device('cpu'):
            yield
    finally:
        torch.set_num_threads(old)


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        from ..gpu import Cancelled
        raise Cancelled('GNM fitting cancelled')


def observation_arrays(model, views):
    """Explicit exterior ids or official anatomy; never projected nearest points."""
    from .dense_landmarks import attachments, MP68
    from .correspondence import occlusion_weight
    ids, bary, confidence = attachments()
    exterior = np.flatnonzero(model.group('skin_exterior'))
    names = sorted(set().union(*(v.get('anchors', {}) for v in views)))
    definitions, keep, canonical = [], [], []
    for name in names:
        examples = [v['anchors'][name] for v in views if name in v.get('anchors', {})]
        a = examples[0]
        if 'vertices' in a or 'vertex' in a:
            vi = np.asarray(a.get('vertices', [a.get('vertex')]), int)
            bw = np.asarray(a.get('barycentric', np.full(len(vi), 1/len(vi))), float)
            if a.get('vertex_space') != 'gnm_full':
                if (vi < 0).any() or (vi >= len(exterior)).any(): raise ValueError('GNM exterior attachment outside mesh')
                vi = exterior[vi]
        else:
            mp = [int(name[3:])] if name.startswith('mp_') and name[3:].isdigit() else SEMANTIC.get(name)
            if mp is None: continue
            if min(mp) < 0 or max(mp) >= len(ids): raise ValueError('unknown GNM landmark')
            vi, bw = exterior[ids[mp]].ravel(), (bary[mp]/len(mp)).ravel()
        if (vi.ndim != 1 or not len(vi) or len(vi) > 32 or (vi < 0).any() or
                (vi >= len(model.data['template_vertex_positions'])).any() or bw.shape != vi.shape or
                not np.isfinite(bw).all() or (bw < 0).any() or abs(bw.sum()-1) > 1e-5):
            raise ValueError('invalid GNM attachment')
        # A named attachment must mean the same anatomical point in every view.
        for other in examples[1:]:
            for key in ('vertices', 'vertex', 'barycentric', 'vertex_space'):
                if other.get(key) != a.get(key): raise ValueError('inconsistent multi-view attachment: '+name)
        index=int(name[3:]) if name.startswith('mp_') and name[3:].isdigit() else None
        inferred=(index is not None and index<468 and np.array_equal(vi,exterior[ids[index]]) and np.allclose(bw,bary[index]))
        canonical.append(inferred)
        definitions.append((vi, bw)); keep.append(name)
    if len(keep) < 4: raise ValueError('GNM fitting needs four anatomical attachments')
    sampled = np.unique(np.concatenate([d[0] for d in definitions]))
    width = max(len(d[0]) for d in definitions)
    ai = np.zeros((len(keep), width), int); aw = np.zeros_like(ai, float)
    for j, (vi, bw) in enumerate(definitions):
        ai[j,:len(vi)] = np.searchsorted(sampled, vi); aw[j,:len(vi)] = bw
    target = np.zeros((len(views),len(keep),2)); weight = np.zeros(target.shape[:2])
    region = []
    for name in keep:
        mp = int(name[3:]) if name.startswith('mp_') and name[3:].isdigit() else None
        region.append('mouth' if mp in MOUTH or name in ('upper_lip','lower_lip','mouth_left','mouth_right') else
                      'eyes' if mp in EYES or name.startswith('eye_') else 'face')
    for i, view in enumerate(views):
        for j, name in enumerate(keep):
            a = view.get('anchors', {}).get(name)
            if a is None: continue
            xy = np.asarray(a['xy'],float); w = float(a.get('weight',1))
            if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(w) or not 0 <= w <= 1:
                raise ValueError('invalid GNM observation')
            target[i,j] = xy
            weight[i,j] = w * occlusion_weight(view,xy)
            if canonical[j]:
                index=int(name[3:])
                # Wire-format explicit attachments already carry observation confidence.
                if 'vertices' not in a and 'vertex' not in a: weight[i,j] *= confidence[index]
                # A canonical RBF transfer is evidence with lower correspondence
                # certainty than the publisher's authored GNM68 anatomy.
                if index not in MP68: weight[i,j] *= .15
    if (weight > 0).sum() < 4: raise ValueError('insufficient visible GNM observations')
    return sampled, ai, aw, target, weight, keep, np.asarray(region)


def holdout_mask(names, region, positions):
    """Withhold canonical spatial patches; target pixels never select the split."""
    positions=np.asarray(positions,float)
    if positions.shape!=(len(names),3) or not np.isfinite(positions).all():
        raise ValueError('canonical landmark positions required for spatial validation')
    held = np.zeros(len(names), bool)
    if len(names) >= 30:
        for category in ('mouth', 'eyes', 'face'):
            indices = np.flatnonzero(region == category)
            if len(indices)<4: continue
            points=positions[indices];distance=np.linalg.norm(points-points[0],axis=1)
            # Two separated local patches per region. Nearby duplicate semantic
            # and dense attachments are withheld together rather than leaking
            # the same anatomical point into training under another name.
            count=max(1,len(indices)//12)
            for seed in (0,int(distance.argmax())):
                d=np.linalg.norm(points-points[seed],axis=1)
                cutoff=np.sort(d)[count-1]+1e-7
                held[indices[d<=cutoff]]=True
    return held


def landmark_metrics(pixels, target, weights, held, region):
    error = np.linalg.norm(np.asarray(pixels)-target,axis=-1)
    def mean(mask):
        w = weights * mask[None]
        return float((error*w).sum()/w.sum()) if w.sum() else None
    return dict(all_px=mean(np.ones(len(held),bool)), heldout_px=mean(held),
                mouth_px=mean(region=='mouth'), eyes_px=mean(region=='eyes'), nonmouth_px=mean(region!='mouth'))


def accepts(before, after, minimum_area_ratio, *, improvement=.02):
    if not np.isfinite(minimum_area_ratio) or minimum_area_ratio < .05: return False
    a,b = before.get('heldout_px'),after.get('heldout_px')
    if a is None or b is None or not np.isfinite([a,b]).all() or b >= a*(1-improvement): return False
    for key in ('mouth_px','eyes_px','nonmouth_px'):
        a,b = before.get(key),after.get(key)
        if a is not None and (b is None or not np.isfinite(b) or b > a*1.05+1e-4): return False
    return True


def windows(count, size=64, overlap=16):
    if count < 1 or not 0 <= overlap < size: raise ValueError('invalid GNM window')
    start=0
    while True:
        end=min(count,start+size)
        yield start,end
        if end==count: break
        start=end-overlap


def project_torch(points, cameras):
    import torch
    r = torch.as_tensor(np.stack([c.rotation for c in cameras]), dtype=points.dtype)
    o = torch.as_tensor(np.stack([c.origin for c in cameras]), dtype=points.dtype)
    p = torch.einsum('fac,fdc->fad',points-o[:,None],r)
    z = -p[...,2]; safe=z.clamp(min=.02)
    t=lambda values: torch.as_tensor(values,dtype=points.dtype)[:,None]
    x=(t([c.focal for c in cameras])*p[...,0]-t([c.skew for c in cameras])*p[...,1])/safe+t([c.cx for c in cameras])
    y=-t([c.focal_y or c.focal for c in cameras])*p[...,1]/safe+t([c.cy for c in cameras])
    return torch.stack((x,y),-1),z


def area_ratio(rest, posed, triangles):
    a,b=rest[triangles],posed[triangles]
    n0=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]); n=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    norm=(n0*n0).sum(-1); active=norm>1e-24
    return float(np.min((n*n0).sum(-1)[active]/norm[active])) if active.any() else 1.


def replay(model, identity, expression, rotations, translation, *, scale=1., alignment=None, offset=None, bind_residual=None, cancel=None):
    """Full CPU expansion in four-frame chunks; same evaluator as optimization."""
    import torch
    alignment=np.eye(3) if alignment is None else np.asarray(alignment)
    offset=np.zeros(3) if offset is None else np.asarray(offset)
    evaluate=model.frame_evaluator(identity,bind_residual=bind_residual)
    vertices,joints=[],[]
    with torch.no_grad():
        for lo in range(0,len(expression),4):
            check_cancel(cancel)
            args=[torch.as_tensor(x[lo:lo+4],dtype=model.dtype,device='cpu') for x in (expression,rotations,translation)]
            p,j=evaluate(*args)
            vertices.append(scale*p.numpy()@alignment.T+offset);joints.append(scale*j.numpy()@alignment.T+offset)
    return np.concatenate(vertices).astype(np.float32),np.concatenate(joints).astype(np.float32)


def _observable_basis(evaluate, beta, expression, rotations, translation, ai, aw, cameras, training, scale, alignment, limit):
    """SVD of projected native-expression derivatives, using training rows only."""
    import torch
    E=len(expression[0]); matrices=[]
    with torch.no_grad():
        # At most four representative views; cameras and visibility vary by view.
        for f in np.unique(np.linspace(0,len(cameras)-1,min(4,len(cameras))).astype(int)):
            base=torch.as_tensor(expression[f:f+1],dtype=beta.dtype)
            r=torch.as_tensor(rotations[f:f+1],dtype=beta.dtype); t=torch.as_tensor(translation[f:f+1],dtype=beta.dtype)
            p,_=evaluate(base,r,t,identity=beta)
            attached=(p[:,ai]*aw[None,:,:,None]).sum(2)[0]
            derivatives=[]
            for start in range(0,E,32):
                eye=torch.eye(E,dtype=beta.dtype)[start:start+32]
                q,_=evaluate(base+eye,r.expand(len(eye),-1,-1),t.expand(len(eye),-1),identity=beta)
                derivatives.append(((q[:,ai]*aw[None,:,:,None]).sum(2)-attached).numpy())
            delta=np.concatenate(derivatives)*scale@alignment.T
            c=cameras[f]
            local=(scale*attached.numpy()@alignment.T-c.origin)@c.rotation.T
            # offset is included by the caller's camera origin adjustment.
            z=np.maximum(-local[:,2],.02); d=delta@c.rotation.T
            dx=(c.focal*d[...,0]-c.skew*d[...,1])/z+(c.focal*local[:,0]-c.skew*local[:,1])*d[...,2]/z**2
            dy=-(c.focal_y or c.focal)*d[...,1]/z-(c.focal_y or c.focal)*local[:,1]*d[...,2]/z**2
            jac=np.stack((dx,dy),-1)*np.sqrt(training[f])[None,:,None]
            matrices.append(jac.reshape(E,-1).T)
    matrix=np.concatenate(matrices)
    _,singular,vt=np.linalg.svd(matrix,full_matrices=False)
    rank=min(limit,int((singular>max(singular[0]*1e-4,1e-8)).sum())) if len(singular) else 0
    if not rank: raise ValueError('GNM expression is unobservable in supplied views')
    return vt[:rank].astype(np.float32),singular[:rank].tolist()


def solve(views, *, identity=None, expression=None, rotations=None, translation=None,
          scale=1., alignment=None, offset=None, bind_residual=None, identity_modes=0,
          modes=64, iterations=400, threads=16, timestamps=None, groups=None,
          freeze_pose=False, fixed_pose_views=(), cancel=None, model=None):
    """Fit full GNM using CPU autograd; rejected stages restore the best state.

    ``groups`` ties simultaneous calibrated views to one physical pose/expression.
    Temporal windows operate on independent per-frame views; identity is frozen.
    """
    import torch
    if type(iterations) is not int or not 10<=iterations<=2000 or not 1<=modes<=128 or not 0<=identity_modes<=170:
        raise ValueError('invalid GNM solve budget')
    if not views: raise ValueError('GNM fitting needs observations')
    if timestamps is not None and identity_modes: raise ValueError('temporal identity must be frozen')
    with cpu_threads(threads):
        return _solve(views,identity,expression,rotations,translation,scale,alignment,offset,bind_residual,
                      identity_modes,modes,iterations,timestamps,groups,freeze_pose,fixed_pose_views,cancel,model)


def _solve(views,identity,expression,rotations,translation,scale,alignment,offset,bind_residual,
           identity_modes,modes,iterations,timestamps,groups,freeze_pose,fixed_pose_views,cancel,model):
    import torch
    started=time.monotonic();check_cancel(cancel)
    model=model or GNMModel(device='cpu')
    if str(model.device)!='cpu': raise ValueError('GNM quality solver requires a CPU model')
    F=len(views);E=model.expression_dim;I=model.identity_dim
    alignment=np.eye(3) if alignment is None else np.asarray(alignment,float)
    offset=np.zeros(3) if offset is None else np.asarray(offset,float)
    if (not np.isfinite(scale) or scale<=0 or alignment.shape!=(3,3) or offset.shape!=(3,) or
            not np.isfinite(alignment).all() or not np.isfinite(offset).all() or
            not np.allclose(alignment@alignment.T,np.eye(3),atol=1e-6) or np.linalg.det(alignment)<.999):
        raise ValueError('invalid GNM alignment')
    def initial(value,shape):
        a=np.zeros(shape,np.float32) if value is None else np.asarray(value,np.float32).copy()
        if a.shape!=shape or not np.isfinite(a).all(): raise ValueError('invalid GNM initial parameters')
        return a
    beta0=initial(identity,(I,)); exp0=initial(expression,(F,E)); rot0=initial(rotations,(F,4,3)); trans0=initial(translation,(F,3))
    if (np.abs(beta0)>3).any() or (np.abs(exp0)>3).any(): raise ValueError('GNM coefficients outside [-3,3]')
    groups=list(range(F)) if groups is None else list(groups)
    if len(groups)!=F: raise ValueError('GNM expression group count mismatch')
    unique=list(dict.fromkeys(groups)); gi=np.array([unique.index(g) for g in groups]);G=len(unique)
    group_consensus=0
    for g in range(G):
        indices=np.flatnonzero(gi==g)
        if any(not np.allclose(a[indices],a[indices[0]],atol=1e-6) for a in (rot0,trans0)):
            raise ValueError('simultaneous views need one shared initial pose')
        if not np.allclose(exp0[indices],exp0[indices[0]],atol=1e-6):
            # Older per-camera fits may disagree. Initialize a single physical
            # expression instead of retaining those disagreements as offsets.
            exp0[indices]=exp0[indices].mean(0);group_consensus+=1
    if timestamps is not None:
        times=np.asarray(timestamps,float)
        if times.shape!=(F,) or not np.isfinite(times).all() or (np.diff(times)<=0).any() or G!=F:
            raise ValueError('temporal timestamps must increase, one view per frame')
    else: times=None
    cams=[Camera.from_dict(v['camera']) for v in views]
    if identity_modes>64 and (F<3 or any(v.get('synthetic') or v.get('camera_kind')=='estimated' for v in views)
            or max(np.linalg.norm(c.rotation-cams[0].rotation) for c in cams)<.25):
        raise ValueError('more than 64 identity modes require independent calibrated multiview observations')
    sampled,ai,aw,target,weights,names,region=observation_arrays(model,views)
    canonical=(model.data['template_vertex_positions'][sampled][ai]*aw[...,None]).sum(1)
    held=holdout_mask(names,region,canonical);training=weights*(~held)[None]
    if training.sum()==0: raise ValueError('no training landmarks')
    tensor=lambda v:torch.as_tensor(v,dtype=model.dtype)
    ai=torch.as_tensor(ai);aw=tensor(aw)
    native_evaluate=model.frame_evaluator(beta0,sampled,bind_residual=bind_residual,fit_identity=bool(identity_modes))
    def evaluate(expression,rotations,translation,identity=None):
        return native_evaluate(expression,rotations,translation,**({'identity':identity} if identity_modes else {}))
    rot_align=tensor(alignment);off=tensor(offset)
    def points(beta,exp,rot,trans):
        p,j=evaluate(exp,rot,trans,identity=beta)
        return (scale*((p[:,ai]*aw[None,:,:,None]).sum(2)@rot_align.T)+off,
                scale*(j@rot_align.T)+off)
    def pixels(beta,exp,rot,trans,indices):
        return project_torch(points(beta,exp,rot,trans)[0],[cams[i] for i in indices])
    with torch.no_grad():
        baseline=pixels(tensor(beta0),tensor(exp0),tensor(rot0),tensor(trans0),range(F))[0].numpy()
    before=landmark_metrics(baseline,target,weights,held,region)
    pose_history=[]
    if times is not None and not freeze_pose:
        proposed_rot,proposed_t=initialize_rigid(model,evaluate,beta0,exp0,rot0,trans0,ai,aw,
                                               cams,target,training,region,scale,alignment,offset,cancel)
        with torch.no_grad():
            initialized=pixels(tensor(beta0),tensor(exp0),tensor(proposed_rot),tensor(proposed_t),range(F))[0].numpy()
        metrics=landmark_metrics(initialized,target,weights,held,region)
        accepted=accepts(before,metrics,1.)
        pose_history.append(dict(stage='rigid_initialization',identity_modes=0,expression_rank=0,before=before,after=metrics,
                                 accepted=accepted,minimum_area_ratio=1.,safe_steps=np.ones(F).tolist()))
        if accepted: rot0,trans0=proposed_rot,proposed_t
    # Camera origin shift accounts for alignment offset in the analytic Jacobian.
    adjusted=[copy.copy(c) for c in cams]
    for c in adjusted: c.origin=c.origin-offset
    basis,singular=_observable_basis(evaluate,tensor(beta0),exp0,rot0,trans0,ai,aw,adjusted,training,scale,alignment,modes)
    ipd=[]
    for i,v in enumerate(views):
        a=v.get('anchors',{});right=a.get('mp_468',a.get('eye_right'));left=a.get('mp_473',a.get('eye_left'))
        d=np.linalg.norm(np.array(right['xy'])-left['xy']) if right and left else max(v.get('size',[512,512]))*.15
        ipd.append(max(float(d),10))
    regional=np.ones_like(training)
    for category,factor in (('mouth',3.),('eyes',2.),('face',1.)):
        mask=region==category
        regional[:,mask]=factor/np.maximum(training[:,mask].sum(1),1)[:,None]
    contact = contact_constraints(model,names,scale)
    with torch.no_grad():
        rest_points,rest_joints=points(tensor(beta0),torch.zeros_like(tensor(exp0)),tensor(rot0),tensor(trans0))
        contact_baseline=contact_distances(rest_points,rest_joints,tensor(rot0),rot_align,contact)
    silhouettes=[]
    for view,c in zip(views,cams):
        silhouettes.append(None)
    silhouette_evaluate=None
    if any(v.get('silhouette_mask_path') for v in views):
        silhouette_evaluate=model.frame_evaluator(beta0,bind_residual=bind_residual,fit_identity=True)
    # Missing frames contribute no data; interpolate initial parameters only.
    valid=weights.sum(1)>0
    if times is not None:
        if not valid.any(): raise ValueError('no visible temporal frames')
        for a in (exp0,rot0,trans0):
            for j in np.ndindex(a.shape[1:]): a[(slice(None),*j)]=np.interp(times,times[valid],a[(valid,*j)])
    # Bind-space triangle barrier is sampled during optimization; every exterior
    # skin triangle is checked before accepting a stage.
    skin=model.group('skin_exterior');all_tri=model.data['triangles'];guard_tri=all_tri[skin[all_tri].all(1)]
    guard_tri=guard_tri[::max(1,len(guard_tri)//1024)]
    guard_ids=np.unique(guard_tri);guard_indices=torch.as_tensor(np.searchsorted(guard_ids,guard_tri))
    guard_template=model.tensors['template_vertex_positions'][guard_ids]
    if bind_residual is not None: guard_template=guard_template+tensor(bind_residual)[guard_ids]
    guard_identity=model.tensors['vertex_identity_basis'][:,guard_ids].reshape(I,-1)
    guard_expression=model.tensors['expression_basis'][:,guard_ids].reshape(E,-1)
    guard_correctives=model.tensors['pose_correctives_regressor'].reshape(36,-1,3)[:,guard_ids].reshape(36,-1)
    guard_rest=guard_template+(tensor(beta0)@guard_identity).reshape(-1,3)
    def normals(p):
        f=p[...,guard_indices,:]
        return torch.linalg.cross(f[...,1,:]-f[...,0,:],f[...,2,:]-f[...,0,:])
    guard_normal=normals(guard_rest).detach();guard_norm=guard_normal.square().sum(-1).clamp(min=1e-24)
    # Native head/neck relative motion is not identifiable from face-only views.
    # Preserve supplied local head pose; it remains fixed during face-only fitting.
    eye_observed=any(n in names for n in ('mp_468','mp_473','pupil_right','pupil_left'))
    fixed=set(fixed_pose_views)
    if any(type(i) is not int or not 0<=i<F for i in fixed): raise ValueError('invalid fixed pose view')
    beta_best=beta0.copy();exp_best=exp0.copy();rot_best=rot0.copy();trans_best=trans0.copy();history=pose_history;last_proposal=None
    stages=[(min(identity_modes,32),min(modes,32)),(min(identity_modes,64),min(modes,64)),(min(identity_modes,64),modes)]
    if identity_modes>64: stages.append((identity_modes,modes))
    stages=list(dict.fromkeys(stages))
    visibility_cache=None;visibility_refreshes=0
    for ni,ne in stages:
        check_cancel(cancel)
        ne=min(ne,len(basis));B=tensor(basis[:ne])
        # Refresh CPU visibility and contour attachments at each accepted stage.
        visibility=np.ones_like(weights) if visibility_cache is None else visibility_cache
        if visibility_cache is None and (ni or np.linalg.norm(rot_best[:,0],axis=1).max()>.15 or any(v.get('silhouette_mask_path') for v in views)):
            visibility_refreshes+=1
            surface,_=replay(model,beta_best,exp_best,rot_best,trans_best,scale=scale,alignment=alignment,
                             offset=offset,bind_residual=bind_residual,cancel=cancel)
            skin_mask=model.group('skin_exterior');skin_tri=model.data['triangles'];skin_tri=skin_tri[skin_mask[skin_tri].all(1)]
            from .reference import rasterize
            from .silhouette import prepare
            for i,view in enumerate(views):
                check_cancel(cancel)
                if not weights[i].any(): continue
                size=view.get('size',[512,512]);ratio=128/max(size)
                _,_,zbuffer=rasterize(surface[i],skin_tri,cams[i].scaled(ratio),[max(1,round(x*ratio)) for x in size])
                attached=(surface[i,sampled][ai.numpy()]*aw.numpy()[:,:,None]).sum(1)
                xy,z=cams[i].project(attached);uv=np.floor(xy*ratio).astype(int)
                inside=(uv[:,0]>=0)&(uv[:,0]<zbuffer.shape[1])&(uv[:,1]>=0)&(uv[:,1]<zbuffer.shape[0])
                depth=np.full(len(names),np.inf);depth[inside]=zbuffer[uv[inside,1],uv[inside,0]]
                visibility[i]=inside&(z>0)&((~np.isfinite(depth))|(z<=depth+.004))
                if view.get('silhouette_mask_path'):
                    silhouettes[i]=prepare(surface[i],model.data['triangles'],cams[i],view)
        visibility_cache=visibility
        stage_training=training*visibility
        if (stage_training>0).sum()<4: break
        beta=torch.nn.Parameter(tensor(beta_best[:ni]).clone())
        beta_tail=tensor(beta_best[ni:])
        exp_base=tensor(exp_best.copy())
        # Parameters shared by simultaneous multi-view observations.
        first=np.array([np.flatnonzero(gi==j)[0] for j in range(G)])
        latent=torch.nn.Parameter(torch.zeros((G,ne),dtype=model.dtype))
        root=torch.nn.Parameter(tensor(rot_best[first,0]).clone())
        eyes=torch.nn.Parameter(tensor(rot_best[first,2:,:2]).clone())
        trans=torch.nn.Parameter(tensor(trans_best[first]).clone())
        trainable=[latent]
        if ni: trainable.append(beta)
        if not freeze_pose: trainable.extend([root,trans])
        if eye_observed: trainable.append(eyes)
        def state(indices):
            g=torch.as_tensor(gi[indices]);n=len(indices)
            expr=torch.clamp(exp_base[indices]+latent[g]@B,-3,3)
            eye3=torch.cat((eyes[g],tensor(rot_best[indices,2:,2:3])),-1)
            r=torch.cat((root[g,None],tensor(rot0[indices,1:2]),eye3),1)
            return torch.cat((beta,beta_tail)),expr,r,trans[g]
        guard_latent=B@guard_expression
        guard_base=exp_base@guard_expression
        spans=list(windows(F)) if times is not None else [(0,F)]
        if times is not None:
            spans+= [(max(0,lo-8),min(F,lo+24)) for lo,_ in spans[1:]]
        for lo,hi in spans:
            indices=np.arange(lo,hi)
            weight=tensor(stage_training[indices]*regional[indices]); norm=tensor(np.asarray(ipd)[indices])[:,None]
            wanted=tensor(target[indices]);anchor_latent=latent.detach().clone()
            def loss_fn():
                check_cancel(cancel)
                b,e,r,t=state(indices)
                pts,jts=points(b,e,r,t)
                q,depth=project_torch(pts,[cams[i] for i in indices])
                error=(q-wanted)/norm[...,None]
                robust=torch.sqrt(error.square().sum(-1)+.005**2)-.005
                loss=(robust*weight).sum()/weight.sum().clamp(min=1)
                loss=loss+.0001*latent[torch.as_tensor(gi[indices])].square().mean()+.0001*beta.square().mean() if ni else loss+.0001*latent[torch.as_tensor(gi[indices])].square().mean()
                loss=loss+.02*(t-tensor(trans0[indices])).square().mean()
                loss=loss+10*torch.relu(.02-depth).square().mean()
                g=torch.as_tensor(gi[indices])
                bind=guard_template+(b@guard_identity).reshape(-1,3)
                if ni: loss=loss+.2*((scale*(bind-guard_rest))/.002).square().mean()
                # Projected observability alone does not constrain off-camera
                # expression depth. Keep a metric prior around the supplied
                # capture as well as the coefficient prior and contact guards.
                expression_delta=latent[g]@guard_latent
                loss=loss+.01*(scale*expression_delta/.002).square().mean()
                corrective=(rotation_matrices(r)-torch.eye(3,dtype=model.dtype)).reshape(len(indices),36)@guard_correctives
                posed=bind[None]+(guard_base[indices]+latent[g]@guard_latent+corrective).reshape(len(indices),-1,3)
                jac=(normals(posed)*guard_normal).sum(-1)/guard_norm
                loss=loss+10*torch.relu(.15-jac).square().topk(min(64,jac.shape[-1]),dim=-1).values.mean()
                distances=contact_distances(pts,jts,r,rot_align,contact)
                if distances.shape[1]:
                    limit=torch.minimum(contact_baseline[indices],torch.zeros_like(distances))
                    loss=loss+5*torch.relu(limit-distances).square().mean()/(.001**2)
                if silhouette_evaluate is not None:
                    full,_=silhouette_evaluate(e,r,t,identity=b)
                    full=scale*(full@rot_align.T)+off
                    for row,f in enumerate(indices):
                        if silhouettes[f] is not None:
                            loss=loss+.01*silhouette_loss(full[row],cams[f],silhouettes[f])
                if times is not None and len(indices)>1:
                    dt=tensor(np.diff(times[indices]));velocity=(e[1:]-e[:-1])/dt[:,None]
                    angular=(r[1:,0]-r[:-1,0])/dt[:,None]
                    loss=loss+1e-5*velocity.square().mean()+2e-5*angular.square().mean()
                    if len(indices)>2:
                        acceleration=2*(velocity[1:]-velocity[:-1])/(dt[1:]+dt[:-1])[:,None]
                        loss=loss+1e-8*acceleration.square().mean()
                    if lo:
                        loss=loss+.001*(latent[lo:lo+16]-anchor_latent[lo:lo+16]).square().mean()
                return loss
            parameter_groups=[{'params':[latent],'lr':.025}]
            if ni: parameter_groups.append({'params':[beta],'lr':.025})
            if not freeze_pose: parameter_groups.extend([{'params':[root],'lr':.006},{'params':[trans],'lr':.0005}])
            if eye_observed: parameter_groups.append({'params':[eyes],'lr':.006})
            optimizer=torch.optim.Adam(parameter_groups)
            # Pose starts before shape to avoid baking head motion into identity.
            for step in range(max(1,iterations*3//4)):
                if step%50==0: print(f'GNM CPU: identity={ni}, expression={ne}, frames={lo}:{hi}, step={step}/{iterations}',file=sys.stderr,flush=True)
                optimizer.zero_grad(set_to_none=True);loss=loss_fn()
                if not bool(torch.isfinite(loss)): break
                loss.backward();torch.nn.utils.clip_grad_norm_(trainable,10.);optimizer.step()
                with torch.no_grad():
                    beta.clamp_(-3,3)
                    if not freeze_pose:
                        root.copy_(torch.maximum(torch.minimum(root,tensor(rot0[first,0])+1.2),tensor(rot0[first,0])-1.2))
                    if eye_observed: eyes.clamp_(-.5,.5)
                    trans.copy_(torch.maximum(torch.minimum(trans,tensor(trans0[first])+.05),tensor(trans0[first])-.05))
                    for f in fixed:
                        root[gi[f]]=tensor(rot0[f,0]);trans[gi[f]]=tensor(trans0[f])
                    if step<min(20,iterations//5): latent.zero_();beta.copy_(tensor(beta_best[:ni]))
            optimizer=torch.optim.LBFGS(trainable,lr=.5,max_iter=max(1,iterations//4),max_eval=max(2,iterations//2),line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad(set_to_none=True);loss=loss_fn();loss.backward();return loss
            optimizer.step(closure)
            with torch.no_grad():
                beta.clamp_(-3,3)
                if not freeze_pose:
                    root.copy_(torch.maximum(torch.minimum(root,tensor(rot0[first,0])+1.2),tensor(rot0[first,0])-1.2))
                if eye_observed: eyes.clamp_(-.5,.5)
                trans.copy_(torch.maximum(torch.minimum(trans,tensor(trans0[first])+.05),tensor(trans0[first])-.05))
                for f in fixed: root[gi[f]]=tensor(rot0[f,0]);trans[gi[f]]=tensor(trans0[f])
        with torch.no_grad():
            b,e,r,t=state(np.arange(F));proposal=[x.detach().numpy().copy() for x in (b,e,r,t)]
            q=pixels(b,e,r,t,range(F))[0].numpy()
        if not all(np.isfinite(a).all() for a in proposal):
            history.append(dict(identity_modes=ni,expression_rank=ne,accepted=False,error='nonfinite optimizer proposal'))
            break
        # Hard full-surface guard at matching pose: large rigid turns are valid.
        steps=np.ones(F);ratios=[]
        with torch.no_grad():
            tri=model.data['triangles'];skin=model.group('skin_exterior');tri=tri[skin[tri].all(1)]
            previous_neutral,_=model.evaluate(beta_best,bind_residual=bind_residual)
            identity_step=1.
            for attempt in range(10):
                new_neutral,_=model.evaluate(proposal[0],bind_residual=bind_residual)
                neutral_ratio=area_ratio(previous_neutral.numpy(),new_neutral.numpy(),tri)
                if neutral_ratio>=.1: break
                proposal[0]=beta_best+.5*(proposal[0]-beta_best);identity_step*=.5
            full_eval=model.frame_evaluator(proposal[0],bind_residual=bind_residual)
            for f in range(F):
                check_cancel(cancel)
                reference_rotation=proposal[2][f:f+1].copy();reference_rotation[:,1:]=0
                rest,_=full_eval(torch.zeros((1,E),dtype=model.dtype),tensor(reference_rotation),tensor(proposal[3][f:f+1]))
                for attempt in range(9):
                    posed,_=full_eval(tensor(proposal[1][f:f+1]),tensor(proposal[2][f:f+1]),tensor(proposal[3][f:f+1]))
                    ratio=area_ratio(rest[0].numpy(),posed[0].numpy(),tri)
                    if ratio>=.05: break
                    proposal[1][f]=exp_best[f]+.5*(proposal[1][f]-exp_best[f])
                    proposal[2][f,1:]=rot_best[f,1:]+.5*(proposal[2][f,1:]-rot_best[f,1:]);steps[f]*=.5
                ratios.append(ratio)
            q=pixels(*[tensor(v) for v in proposal],range(F))[0].numpy()
        last_proposal={k:v.copy() for k,v in zip(('identity','expression','rotations','translation'),proposal)}
        metrics=landmark_metrics(q,target,weights,held,region)
        # Acceptance is against the previously accepted solution, never training loss.
        best_q=pixels(tensor(beta_best),tensor(exp_best),tensor(rot_best),tensor(trans_best),range(F))[0].detach().numpy()
        best_metrics=landmark_metrics(best_q,target,weights,held,region)
        with torch.no_grad():
            pts,jts=points(*[tensor(v) for v in proposal])
            distances=contact_distances(pts,jts,tensor(proposal[2]),rot_align,contact)
            limit=torch.minimum(contact_baseline,torch.zeros_like(distances))
            penetration=float(torch.relu(limit-distances).max()) if distances.numel() else 0.
        accepted=accepts(best_metrics,metrics,min(min(ratios),neutral_ratio)) and penetration<=.0001
        history.append(dict(identity_modes=ni,expression_rank=ne,before=best_metrics,after=metrics,
                            accepted=accepted,identity_safe_step=identity_step,neutral_minimum_area_ratio=neutral_ratio,contact_penetration_mm=penetration*1000,minimum_area_ratio=float(min(ratios)),safe_steps=steps.tolist()))
        if accepted:
            beta_best,exp_best,rot_best,trans_best=proposal
            visibility_cache=None
        # A low-rank expression may be unable to explain a regional motion.
        # Evaluate the next rank from the retained best state, never from the
        # rejected proposal; every rank still faces the same validation gates.
    if not identity_modes and identity is not None: beta_best=np.asarray(identity).copy()
    vertices,joints=replay(model,beta_best,exp_best,rot_best,trans_best,scale=scale,alignment=alignment,offset=offset,bind_residual=bind_residual,cancel=cancel)
    # Measure the final exported state, including a retained source or pose-only
    # result. Local joint motion must not mask a fold in its own reference.
    reference_rotations=rot_best.copy();reference_rotations[:,1:]=0
    reference,_=replay(model,beta_best,np.zeros_like(exp_best),reference_rotations,trans_best,
                       scale=scale,alignment=alignment,offset=offset,bind_residual=bind_residual,cancel=cancel)
    skin=model.group('skin_exterior');tri=model.data['triangles'];tri=tri[skin[tri].all(1)]
    final_ratios=[area_ratio(a,b,tri) for a,b in zip(reference,vertices)]
    if not np.isfinite(final_ratios).all() or min(final_ratios)<.05:
        raise ValueError('unsafe native GNM source or final deformation; rebuild the bind or neutralize its expression')
    neutral,_=model.evaluate(beta_best,bind_residual=bind_residual)
    with torch.no_grad(): projected=pixels(tensor(beta_best),tensor(exp_best),tensor(rot_best),tensor(trans_best),range(F))[0].numpy()
    after=landmark_metrics(projected,target,weights,held,region)
    report=dict(solver=SOLVER,coordinate_convention='fixed_camera_posed_surfaces_v2',backend='cpu',device='cpu',
                identity_frozen=not bool(identity_modes),identity_sha256=hashlib.sha256(beta_best.tobytes()).hexdigest(),
                before=before,after=after,stages=history,quality_gate=any(row['accepted'] for row in history),
                expression_rank=([r for r in history if r['accepted']][-1]['expression_rank'] if any(r['accepted'] for r in history) else 0),expression_dim=E,
                inferred_attachment_weight=.15,identity_trust_scale_mm=2.,identity_trust_weight=.2,
                expression_trust_scale_mm=2.,expression_trust_weight=.01,
                visibility_refreshes=visibility_refreshes,
                simultaneous_groups_initialized_by_consensus=group_consensus,
                heldout_landmarks=int((held[None]&(weights>0)).sum()),frames=F,iterations=iterations,
                threads=__import__('torch').get_num_threads(),seconds=time.monotonic()-started,
                peak_host_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                limitations=['Monocular identity/depth ambiguity; uncalibrated intrinsics/IPD retain supplied assumptions.',
                             'Dense non-GNM68 attachments are inferred; held-out tracker error is not independent 3D truth.',
                             'Hidden anatomy remains a prior; no image or video generation is performed.'])
    report['skin_min_oriented_area_ratio']=float(min(final_ratios))
    report['per_frame_min_oriented_area_ratio']=final_ratios
    report['expression_safe_steps']=([row for row in history if row['accepted']][-1]['safe_steps']
                                     if any(row['accepted'] for row in history) else np.ones(F).tolist())
    return dict(identity=beta_best,expression=exp_best,rotations=rot_best,translation=trans_best,
                vertices=vertices,joints=joints,neutral=(scale*neutral.detach().numpy()@alignment.T+offset).astype(np.float32),
                projected_landmarks=projected,baseline_landmarks=baseline,normalization_ipd=np.asarray(ipd),observed_landmarks=target,weights=weights,names=names,region=region,
                heldout=held,report=report,last_proposal=last_proposal)


def contact_constraints(model, names, scale=1.):
    """Sparse observed lip/lid constraints, relative to neutral anatomy."""
    pairs=[]
    for upper,lower in (('mp_13','mp_14'),('mp_013','mp_014'),('upper_lip','lower_lip'),
                        ('mp_081','mp_178'),('mp_311','mp_402')):
        if upper in names and lower in names: pairs.append((names.index(upper),names.index(lower)))
    eyes=[]
    for side,joint,landmarks in (('left',2,(263,362,386,374,385,387,380,373)),
                                 ('right',3,(33,133,159,145,160,158,153,144))):
        globe=model.data['template_vertex_positions'][model.group(side+'_eye')]
        radius=float(np.quantile(np.linalg.norm(globe-model.data['template_joint_positions'][joint],axis=1),.8))
        for landmark in landmarks:
            name=f'mp_{landmark:03d}'
            if name in names: eyes.append((names.index(name),joint,radius*scale))
    return pairs,eyes


def contact_distances(points,joints,rotations,alignment,spec):
    import torch
    pairs,eyes=spec;distances=[]
    up=rotation_matrices(rotations[:,0])[...,1]@alignment.T
    for u,l in pairs: distances.append(((points[:,u]-points[:,l])*up).sum(-1))
    for lid,joint,radius in eyes:
        distances.append(torch.linalg.vector_norm(points[:,lid]-joints[:,joint],dim=-1)-radius)
    return torch.stack(distances,1) if distances else points[:,:0,0]


def silhouette_loss(vertices,camera,prepared):
    import torch
    ids=torch.as_tensor(prepared['triangles']);bary=torch.as_tensor(prepared['barycentric'],dtype=vertices.dtype)
    points=(vertices[ids]*bary[...,None]).sum(1)
    xy,depth=project_torch(points[None],[camera]);xy=xy[0]
    h,w=prepared['sdf'].shape
    uv=xy*prepared['scale']-.5
    grid=2*uv/torch.as_tensor([max(w-1,1),max(h-1,1)],dtype=vertices.dtype)-1
    sdf=torch.as_tensor(prepared['sdf'],dtype=vertices.dtype)[None,None]
    distance=torch.nn.functional.grid_sample(sdf,grid[None,None],align_corners=True,padding_mode='border')[0,0,0]
    allowed=torch.as_tensor(prepared['allowed'],dtype=vertices.dtype)[None,None]
    visible=torch.nn.functional.grid_sample(allowed,grid[None,None],align_corners=True,padding_mode='zeros')[0,0,0].detach()
    target=torch.as_tensor(prepared['target'],dtype=vertices.dtype)
    reverse=torch.cdist(target,xy).amin(1)
    return ((distance.abs()*visible).sum()/visible.sum().clamp(min=1)+reverse.mean())/max(w,h)


def initialize_rigid(model,evaluate,beta,expression,rotations,translation,ai,aw,cameras,target,training,region,scale,alignment,offset,cancel):
    """Training-only perspective SE(3) initialization; no per-landmark offsets."""
    import torch
    from scipy.optimize import least_squares
    roots=rotations.copy();translations=translation.copy()
    tensor=lambda x:torch.as_tensor(x,dtype=model.dtype)
    joints=model.data['template_joint_positions']+np.einsum('i,ijc->jc',beta,model.data['joint_identity_basis'])
    with torch.no_grad():
        local_rot=rotations.copy();local_rot[:,0]=0
        positions,_=evaluate(tensor(expression),tensor(local_rot),torch.zeros((len(expression),3),dtype=model.dtype),identity=tensor(beta))
        attachments=(positions[:,ai]*aw[None,:,:,None]).sum(2).numpy()
    for i,c in enumerate(cameras):
        check_cancel(cancel)
        w=training[i]*np.where(region=='mouth',.1,1.)
        if (w>0).sum()<6: continue
        active=w>0
        def objective(x):
            root=rotation_matrices(np.asarray(x[:3]))
            p=(attachments[i]-joints[0])@root.T+joints[0]+x[3:]
            xy,z=c.project(scale*p@alignment.T+offset)
            return np.r_[((xy-target[i])*np.sqrt(w)[:,None])[active].ravel(),np.minimum(z[active]-.02,0)*1000]
        x=np.r_[roots[i,0],translations[i]]
        fit=least_squares(objective,x,loss='soft_l1',f_scale=2,max_nfev=100,
                          bounds=(x-np.array([1.4]*3+[2.]*3),x+np.array([1.4]*3+[2.]*3)))
        if np.isfinite(fit.x).all() and np.square(objective(fit.x)).mean()<np.square(objective(x)).mean():
            roots[i,0]=fit.x[:3];translations[i]=fit.x[3:]
    return roots,translations
