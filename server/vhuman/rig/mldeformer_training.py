"""Framework-free corrective training: native ARAP/contact solve, PCA and MLP.

Model math and analytic gradients use cpu/vhuman with repository GEMM. NumPy
supplies arrays, small skin transforms/linear solves and bounded QR/SVD. The
optional Torch implementation lives only in ref/vhuman/corrective_torch_reference.py.
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
from ..native_training import AdamW, matmul, randomized_basis, set_threads
from . import native_corrective as native
from .common import edges, vertex_normals
from .contact_setup import EYE_MARGIN_MM, TOOTH_MARGIN_MM, export_contacts, sample_controls, region_mask
from .mlruntime import MLDeformer
from .native_corrective import MLP2, NativeRig
from . import safetensors as st


class Contacts:
    PAIR_RINGS = (1,0,-1,-2)

    def __init__(self, tmpl, feat, skel, teeth, device, rest, tongue=None, dtype=None):
        if device != 'cpu': raise ValueError('native corrective training currently supports CPU')
        if dtype is not None and dtype != np.float32: raise ValueError('native contacts use float32')
        self._export=export_contacts(tmpl,feat,skel,teeth,rest,tongue)
        rest=native.f32(rest);self.eye=[];self.spheres={}
        for e in self._export['eye']:
            ids=native.i32(e['ids']);center=native.f32([*e['center'],1])
            threshold=np.minimum(e['radius_mm']+EYE_MARGIN_MM,np.linalg.norm(rest[ids]-center[:3],axis=1)*1000)
            self.eye.append((ids,e['joint'],center,native.f32(threshold[:,None])))
        self.lip_ids=native.i32(self._export['lip_ids'])
        for name,e in self._export['spheres'].items():
            center=native.f32(e['centers']);threshold=np.minimum(np.asarray(e['radius_mm'])[None]+TOOTH_MARGIN_MM,
                        np.linalg.norm(rest[self.lip_ids,None]-center[None],axis=-1)*1000)
            self.spheres[name]=(native.f32(np.column_stack((center,np.ones(len(center))))),native.f32(threshold),
                                native.i32(e['joints']),native.f32(e['weights']))
        self.pair_u=native.i32(self._export['pairs_upper']);self.pair_l=native.i32(self._export['pairs_lower'])
        self.sep0=native.f32(np.minimum((rest[self.pair_u]-rest[self.pair_l])[:,1]*1000,0))
        self.head,self.jaw=self._export['head'],self._export['jaw']

    def export(self): return self._export

    def up(self, skin):
        up=skin[:,self.head,:3,1]+skin[:,self.jaw,:3,1]
        return native.f32(up/np.maximum(np.linalg.norm(up,axis=-1,keepdims=True),1e-12))

    def posed(self, skin):
        skin=native.f32(skin)
        if skin.ndim!=4 or skin.shape[2:]!=(4,4) or not np.isfinite(skin).all():raise ValueError('invalid contact skin transforms')
        groups=[]
        for ids,j,center,threshold in self.eye:
            groups.append(('eye',ids,native.f32((skin[:,j]@center)[:,:3,None].transpose(0,2,1)*1000),threshold))
        for name,(center,threshold,joints,weights) in self.spheres.items():
            blended=(weights[None,:,:,None,None]*skin[:,joints]).sum(2)
            transformed=(blended@center[None,:,:,None])[...,:3,0]*1000
            groups.append((name,self.lip_ids,native.f32(transformed),threshold))
        return groups,self.up(skin)

    def compute(self, x, posed, per_vertex=False):
        x=native.f32(x);groups,up=posed
        energy=np.zeros(len(x),np.float32);gradient=np.zeros_like(x);stats={}
        depth=np.zeros(x.shape[:2],np.float32) if per_vertex else None
        for name,ids,centers,threshold in groups:
            e,g,d=native.spheres(x,ids,centers,threshold);energy+=e;gradient+=g
            stats[name]=stats.get(name,np.zeros(len(x),np.int64))+(d>.05).sum(1)
            if per_vertex:
                for s in range(len(x)):np.maximum.at(depth[s],ids,d[s])
        e,g,d=native.pairs(x,self.pair_u,self.pair_l,up,self.sep0);energy+=e;gradient+=g
        stats['lips']=(d>.05).sum(1)
        if per_vertex:
            for s in range(len(x)):
                np.maximum.at(depth[s],self.pair_u,d[s]);np.maximum.at(depth[s],self.pair_l,d[s])
        return energy,stats,gradient,depth

    def energy(self, x, skin, per_vertex=False):
        energy,stats,_,depth=self.compute(x,self.posed(skin),per_vertex)
        return (energy,stats,depth) if per_vertex else (energy,stats)


class Solver:
    def __init__(self, tr, tmpl, contacts, w_arap=.4, w_contact=400., iters=80):
        if type(iters) is not int or iters<1 or not all(np.isfinite(v) and v>=0 for v in (w_arap,w_contact)):
            raise ValueError('invalid corrective solver settings')
        self.tr,self.c=tr,contacts;self.edges=native.i32(edges(tmpl.tris));self.faces=native.i32(tmpl.tris)
        self.rest_mm=native.f32(tr.rest*1000);self.e0=self.rest_mm[self.edges[:,1]]-self.rest_mm[self.edges[:,0]]
        self.n0=native.f32(vertex_normals(self.rest_mm,self.faces));acc=np.zeros(len(tr.rest),np.float32)
        length2=(self.e0**2).sum(-1)
        np.add.at(acc,self.edges[:,0],length2);np.add.at(acc,self.edges[:,1],length2)
        degree=np.bincount(self.edges.ravel(),minlength=len(tr.rest))
        self.nscale=native.f32(acc/np.maximum(degree,1))
        self.w_arap,self.w_contact,self.iters=w_arap,w_contact,iters

    def rotations(self, x):return native.rotations(x,self.e0,self.n0,self.nscale,self.edges,self.faces)

    def solve(self, controls):
        linear=self.tr(controls);x_linear=native.f32(linear['pos']*1000);posed=self.c.posed(linear['skin'])
        offset=np.zeros_like(x_linear);optimizer=AdamW(offset,lr=.1,weight_decay=0)
        _,before,_,_=self.c.compute(x_linear,posed)
        for iteration in range(self.iters):
            x=x_linear+offset
            if iteration%10==0:
                rotation=self.rotations(x);target=native.f32((rotation[:,self.edges[:,0]]@self.e0[None,:,:,None])[...,0])
            _,gradient=native.arap(x,x_linear,self.edges,target,self.w_arap)
            _,_,contact_gradient,_=self.c.compute(x,posed)
            gradient+=self.w_contact/len(self.tr.rest)*contact_gradient
            optimizer.step(gradient)
        _,after,_,_=self.c.compute(x_linear+offset,posed)
        # NumPy small 3x3 solves, bounded by the caller's ground-truth batch.
        residual=native.f32(np.linalg.solve(linear['blend'].astype(np.float64),offset.astype(np.float64)[...,None])[...,0])
        return dict(residual_mm=residual,before=before,after=after,offset_mm=offset)


def objective(pred, target, cs, sample_weights, indices, lips, lip_weight):
    """Weighted coefficient MSE and analytic posed lip-separation gradient."""
    prediction=native.f32(pred);maximum=float(cs.max());scale=cs/maximum
    difference=prediction-target
    gradient=native.f32(2*difference*scale**2*sample_weights/prediction.size)
    loss=float(np.mean(difference**2*scale**2*sample_weights,dtype=np.float64))
    if lip_weight:
        basis_u,basis_l,mean_u,mean_l,au,al,sep_linear,floor,cm=lips
        n,k=prediction.shape;p=len(floor)
        if p:
            coefficients=prediction*cs+cm
            du=matmul(coefficients,basis_u.reshape(k,-1)).reshape(n,p,3)+mean_u
            dl=matmul(coefficients,basis_l.reshape(k,-1)).reshape(n,p,3)+mean_l
            sep=sep_linear[indices]+(au[indices]*du).sum(-1)-(al[indices]*dl).sum(-1)
            penetration=np.maximum(floor-sep,0)
            factor=lip_weight/maximum**2
            loss+=factor*float(np.mean(penetration**2,dtype=np.float64))
            dsep=native.f32(-2*factor*penetration/penetration.size)
            dc=matmul((dsep[:,:,None]*au[indices]).reshape(n,-1),basis_u.reshape(k,-1),transpose_b=True)
            dc-=matmul((dsep[:,:,None]*al[indices]).reshape(n,-1),basis_l.reshape(k,-1),transpose_b=True)
            gradient+=dc*cs
    return loss,gradient


def train(tr, tmpl, contacts, out_dir, samples=4096, k=48, k_mouth=48, hidden=256, batch=128,
          iters=100, epochs=2500, seed=0, log=print, progress=None, w_contact=1500., tongue_weight=4.,
          tongue_share=.3, weight_decay=1e-3, solve_cache=None, lip_weight=10., threads=4):
    settings=((samples,65536),(k,4096),(k_mouth,4096),(hidden,4096),(batch,1024),(iters,1000000),(epochs,1000000))
    if any(type(v) is not int or not 1<=v<=limit for v,limit in settings) or samples<2:
        raise ValueError('invalid bounded corrective training dimensions')
    if not all(np.isfinite(v) and v>=0 for v in (w_contact,tongue_weight,weight_decay,lip_weight)):
        raise ValueError('invalid corrective loss/optimizer weights')
    if not 0<=tongue_share<=1:raise ValueError('invalid tongue sampling share')
    set_threads(threads);t0=time.perf_counter();cache=Path(solve_cache) if solve_cache else None
    X=sample_controls(tr.controls,samples,seed,tongue_share=tongue_share)
    keys=['eye',*contacts.spheres,'lips']
    if cache is not None and cache.exists():
        with np.load(cache,allow_pickle=False) as blob:
            if str(blob['format'])!='vhuman.corrective_solve.v1':raise ValueError('unsupported solve cache; legacy Torch caches require explicit conversion')
            if not np.array_equal(blob['X'],X):raise ValueError('solve cache controls differ')
            Y=native.f32(blob['residual_mm']);before={key:blob['before.'+key].copy() for key in keys};after={key:blob['after.'+key].copy() for key in keys}
            expected=json.dumps(dict(iters=iters,w_contact=w_contact,geometry_sha256=_rig_hash(tr,tmpl,contacts)),sort_keys=True)
            if str(blob['settings'])!=expected:raise ValueError('solve cache rig/settings differ')
    else:
        solver=Solver(tr,tmpl,contacts,iters=iters,w_contact=w_contact);residuals=[];befores=[];afters=[]
        for start in range(0,samples,batch):
            solved=solver.solve(X[start:start+batch]);residuals.append(solved['residual_mm']);befores.append(solved['before']);afters.append(solved['after'])
            if progress:progress(start/samples,f'deformer ground truth {start}/{samples}')
        Y=np.concatenate(residuals);before={key:np.concatenate([s[key] for s in befores]) for key in keys};after={key:np.concatenate([s[key] for s in afters]) for key in keys}
        if cache is not None:
            cache.parent.mkdir(parents=True,exist_ok=True);partial=cache.with_name(cache.name+'.partial')
            with partial.open('wb') as stream:
                np.savez(stream,format='vhuman.corrective_solve.v1',X=X,residual_mm=Y,
                    settings=json.dumps(dict(iters=iters,w_contact=w_contact,geometry_sha256=_rig_hash(tr,tmpl,contacts)),sort_keys=True),
                    **{'before.'+key:before[key] for key in keys},**{'after.'+key:after[key] for key in keys})
            partial.replace(cache)
    if Y.shape!=(samples,len(tr.rest),3) or not np.isfinite(Y).all():raise ValueError('invalid solved residuals')
    if any(v.shape!=(samples,) or v.dtype.kind not in 'iu' or (v<0).any() for v in [*before.values(),*after.values()]):raise ValueError('invalid solve contact counts')
    n,vertices=Y.shape[:2];Y=Y.reshape(n,-1);t_solve=time.perf_counter()-t0
    contact={key:dict(linear=int(before[key].sum()),solved=int(after[key].sum())) for key in keys}
    mean=Y.mean(0);mouth=np.repeat(region_mask(tmpl,tr.rest),3);bases=[];explained=[]
    for mask,components in ((mouth,k_mouth),(~mouth,k)):
        ids=np.flatnonzero(mask);z=native.f32((Y-mean)[:,ids]);rank=min(components,n,len(ids))
        if not rank:explained.append(1.);continue
        compact=randomized_basis(z,rank,seed=seed);basis=np.zeros((rank,Y.shape[1]),np.float32);basis[:,ids]=compact
        norm=float(np.sum(z.astype(np.float64)**2));coeff=matmul(z,compact,transpose_b=True)
        explained.append(min(1.,float(np.sum(coeff.astype(np.float64)**2))/norm) if norm else 1.);bases.append(basis)
    basis=np.concatenate(bases);k=len(basis);coeff=matmul(Y-mean,basis,transpose_b=True)
    xt=tr.input_vector(X);xm=xt.mean(0);std=xt.std(0,ddof=1);xs=np.ones_like(std);np.divide(1,std,out=xs,where=std>1e-6)
    cm=coeff.mean(0);cs=np.maximum(coeff.std(0,ddof=1),1e-6)
    perm=np.random.default_rng(seed).permutation(n);n_val=max(1,n//8);va,trn=perm[:n_val],perm[n_val:]
    net=MLP2(xt.shape[1],hidden,k,seed=seed,weight_decay=weight_decay);xin=native.f32((xt-xm)*xs);target=native.f32((coeff-cm)/cs)
    sep_linear,au,al=[],[],[];pu,pl=contacts.pair_u,contacts.pair_l
    for start in range(0,n,batch):
        linear=tr(X[start:start+batch]);up=contacts.up(linear['skin']);x=linear['pos']*1000
        sep_linear.append(((x[:,pu]-x[:,pl])*up[:,None]).sum(-1))
        au.append((up[:,None,None,:]@linear['blend'][:,pu])[...,0,:]);al.append((up[:,None,None,:]@linear['blend'][:,pl])[...,0,:])
    lips=(basis.reshape(k,vertices,3)[:,pu],basis.reshape(k,vertices,3)[:,pl],mean.reshape(vertices,3)[pu],mean.reshape(vertices,3)[pl],
          np.concatenate(au),np.concatenate(al),np.concatenate(sep_linear),contacts.sep0,cm)
    rare=before.get('tongue',np.zeros(n));sample_weights=native.f32((1+tongue_weight*(rare>0))[:,None])
    def errors(ids):
        result=[]
        for start in range(0,len(ids),batch):
            index=ids[start:start+batch];pred=net(xin[index])*cs+cm
            residual=matmul(pred,basis)+mean
            result.append(np.linalg.norm((residual-Y[index]).reshape(len(index),vertices,3),axis=-1))
        return np.concatenate(result)
    for epoch in range(epochs):
        pred=net(xin[trn]);loss,gradient=objective(pred,target[trn],cs,sample_weights[trn],trn,lips,lip_weight)
        _,parameter_gradient=net.compute(xin[trn],gradient)
        net.optimizer.step(parameter_gradient,lr=.003*.5*(1+math.cos(math.pi*epoch/epochs)))
        if log and epoch%500==0:log(f'deformer mlp epoch {epoch}: loss {loss:.5f}, val mean err {errors(va).mean():.4f} mm')
    ev=errors(va);magnitude=np.linalg.norm(Y[va].reshape(len(va),vertices,3),axis=-1)
    def contact_counts(ids):
        totals={key:dict(linear=0,ml=0) for key in keys}
        for start in range(0,len(ids),batch):
            index=ids[start:start+batch];pre=(matmul(net(xin[index])*cs+cm,basis)+mean).reshape(len(index),vertices,3)/1000
            linear=tr(X[index]);ml=tr(X[index],pre=pre)
            _,cl=contacts.energy(linear['pos']*1000,linear['skin']);_,corrected=contacts.energy(ml['pos']*1000,ml['skin'])
            for key in keys:totals[key]['linear']+=int(cl[key].sum());totals[key]['ml']+=int(corrected[key].sum())
        return totals
    held_out=contact_counts(va);train_contacts=contact_counts(trn[:len(va)])
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    lrm=net.state_dict();lrm.update({'input.mean':xm,'input.scale':xs,'output.mean':cm/1000,'output.scale':cs/1000})
    st.save(out_dir/'deformer.lrm',lrm,dict(format='vhuman-ml-deformer',version='2',inputs=json.dumps(tr.inputs),
              outputs='pca coefficients (metres) of pre-skinning corrective offsets',kind='face-corrective'))
    st.save(out_dir/'deformer_basis.safetensors',dict(basis=basis.reshape(k,vertices,3),mean=(mean/1000).reshape(vertices,3)),
            dict(vertices=vertices,components=k,units='basis is unit-norm; offsets = mean + sum_k c_k basis_k'))
    stats=dict(samples=n,components=k,hidden=hidden,backend='repository_cpu_gemm',training_device='cpu',thread_budget=threads,
               pca='bounded regional randomized PCA, repository GEMM and NumPy thin QR/SVD',
               explained_variance=dict(mouth=round(explained[0],4),rest=round(explained[1],4)),
               residual_mean_mm=round(float(magnitude.mean()),4),residual_p99_mm=round(float(np.quantile(magnitude,.99)),4),
               val_error_mean_mm=round(float(ev.mean()),4),val_error_p99_mm=round(float(np.quantile(ev,.99)),4),
               contact_vertices=contact,held_out_contacts=held_out,train_contacts=train_contacts,
               solve_seconds=round(t_solve,1),seconds=round(time.perf_counter()-t0,1))
    (out_dir/'deformer.json').write_text(json.dumps(stats,indent=1,allow_nan=False));return stats


def _rig_hash(tr, tmpl, contacts):
    import hashlib
    digest=hashlib.sha256(json.dumps(tr.d,sort_keys=True).encode())
    digest.update(json.dumps(contacts.export(),sort_keys=True).encode())
    for array in (tr.rest,tr.jn,tr.w,tr.D,np.asarray(tmpl.tris)):
        digest.update(str(array.shape).encode());digest.update(str(array.dtype).encode());digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
