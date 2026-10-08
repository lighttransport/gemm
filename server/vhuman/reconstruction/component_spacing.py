"""Bounded rigid spacing of disconnected mesh components.

Separating planes resolve transverse inter-component crossings while retaining
each component's shape. This is a geometric prior, not measured tooth placement.
Coplanar overlap and containment are outside the crossing detector's scope.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from .mesh_crossings import crossing_pairs


def separate_components(vertices, triangles, *, clearance_mm=.02, maximum_shift_mm=.5,
                        iterations=8, projection_sweeps=10000):
    """Return proposed vertex offsets and an acceptance report; never mutate input.

    Callers must check accepted and validate other triangles sharing vertices,
    other anatomy, visibility and motion before applying the proposal.
    """
    vertices,triangles=np.asarray(vertices,float),np.asarray(triangles)
    if (not np.isfinite([clearance_mm,maximum_shift_mm]).all() or clearance_mm<=0
            or maximum_shift_mm<=0 or iterations<1 or projection_sweeps<1):
        raise ValueError('invalid component spacing limits')
    pairs=crossing_pairs(vertices,triangles)
    initial=len(pairs)
    ids=np.unique(triangles)
    if not len(ids):
        return np.zeros_like(vertices),dict(accepted=True,components=0,before_crossings=0,after_crossings=0,maximum_shift_mm=0.)
    remap=np.full(len(vertices),-1,int);remap[ids]=np.arange(len(ids))
    edges=remap[triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)]
    adjacency=coo_matrix((np.ones(len(edges)*2),
        (np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(ids),len(ids)))
    count,labels=connected_components(adjacency)
    triangle_labels=labels[remap[triangles[:,0]]]
    groups=[ids[labels==i] for i in range(count)]
    centers=np.array([vertices[group].mean(0) for group in groups])
    translation=np.zeros((count,3));constraints=[];known=set();converged=True
    reason=None
    for step in range(iterations):
        contacts={tuple(sorted((int(triangle_labels[a]),int(triangle_labels[b])))) for a,b in pairs}
        if not contacts:break
        for a,b in sorted(contacts):
            if a==b:
                reason='within-component crossing cannot be repaired rigidly';break
            if (a,b) in known:continue
            axis=centers[b]+translation[b]-centers[a]-translation[a]
            length=np.linalg.norm(axis)
            if length<1e-12:
                reason='coincident component centres';break
            axis/=length
            gap=(vertices[groups[a]]@axis).max()-(vertices[groups[b]]@axis).min()+clearance_mm*.001
            constraints.append((a,b,axis,gap));known.add((a,b))
        if reason:break
        translation[:]=0
        multipliers=np.zeros(len(constraints));converged=False
        for sweep in range(projection_sweeps):
            previous=translation.copy()
            for i,(a,b,axis,gap) in enumerate(constraints):
                first=translation[a]+multipliers[i]*axis
                second=translation[b]-multipliers[i]*axis
                multiplier=max(0.,(gap-(second-first)@axis)/2)
                translation[a]=first-multiplier*axis;translation[b]=second+multiplier*axis
                multipliers[i]=multiplier
            residual=max((gap-(translation[b]-translation[a])@axis for a,b,axis,gap in constraints),default=0)
            if residual<=1e-10 and np.abs(translation-previous).max()<1e-10:
                converged=True;break
        proposed=vertices.copy();proposed[ids]+=translation[labels]
        pairs=crossing_pairs(proposed,triangles)
        if not converged:
            reason='separating-plane projection did not converge';break
        if np.linalg.norm(translation,axis=1).max()*1000>maximum_shift_mm:
            reason='translation limit exceeded';break
    maximum=float(np.linalg.norm(translation,axis=1).max()*1000)
    delta=np.zeros_like(vertices);delta[ids]=translation[labels]
    accepted=not len(pairs) and converged and maximum<=maximum_shift_mm and reason is None
    report=dict(accepted=bool(accepted),components=count,before_crossings=initial,after_crossings=len(pairs),
                maximum_shift_mm=maximum,maximum_allowed_mm=maximum_shift_mm,clearance_mm=clearance_mm,
                constraints=len(constraints),iterations=step+1,reason=reason,
                translations_mm=(translation*1000).tolist(),rigid_within_selected_components=True)
    return delta,report


def extend_offsets(vertices, triangles, fixed_ids, offsets, *, stiffness=.1):
    """Screened harmonic extension into surrounding vertices of this mesh.

    Fixed offsets remain exact; vertices outside triangles remain unchanged.
    This smooths transitions but is not an orientation or collision guarantee.
    """
    vertices,triangles=np.asarray(vertices,float),np.asarray(triangles)
    offsets=np.asarray(offsets,float);fixed_ids=np.asarray(fixed_ids)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or offsets.shape!=vertices.shape
            or not np.isfinite(vertices).all() or not np.isfinite(offsets).all()
            or triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles)
            or triangles.dtype.kind not in 'iu' or triangles.min()<0 or triangles.max()>=len(vertices)
            or fixed_ids.ndim!=1 or fixed_ids.dtype.kind not in 'iu' or not len(fixed_ids)
            or not np.isfinite(stiffness) or stiffness<=0):
        raise ValueError('invalid offset extension inputs')
    ids=np.unique(triangles)
    if not np.isin(fixed_ids,ids).all() or len(np.unique(fixed_ids))!=len(fixed_ids):
        raise ValueError('fixed vertices must be unique mesh vertices')
    free=np.setdiff1d(ids,fixed_ids)
    result=offsets.copy()
    if not len(free):return result
    edges=np.unique(np.sort(triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
    adjacency=coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],
                         np.r_[edges[:,1],edges[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
    laplacian=diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    result[free]=spsolve(laplacian[free][:,free]+diags(np.full(len(free),stiffness)),
                        -laplacian[free][:,fixed_ids]@offsets[fixed_ids])
    return result


def repair_free_areas(frames, triangles, offsets, free_vertices, *, minimum_ratio=.15,
                      maximum_repair_mm=.02, iterations=2000):
    """Bounded local area repair with fixed vertices held exact across poses.

    Uses sequential area-gradient steps and checks the complete supplied pose
    set after each step. This nonlinear repair may fail; accepted=False must
    reject the proposal. It does not establish collision freedom.
    """
    frames,triangles=np.asarray(frames,float),np.asarray(triangles)
    offsets=np.asarray(offsets,float);free=np.asarray(free_vertices)
    if (frames.ndim!=3 or frames.shape[-1]!=3 or not len(frames)
            or offsets.shape!=frames.shape[1:] or not np.isfinite(frames).all()
            or not np.isfinite(offsets).all() or free.shape!=(frames.shape[1],) or free.dtype!=bool
            or triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles)
            or triangles.dtype.kind not in 'iu' or triangles.min()<0 or triangles.max()>=frames.shape[1]
            or not 0<minimum_ratio<1 or not np.isfinite(maximum_repair_mm) or maximum_repair_mm<=0
            or iterations<1):
        raise ValueError('invalid area repair inputs')
    original=frames[:,triangles]
    normal=np.cross(original[:,:,1]-original[:,:,0],original[:,:,2]-original[:,:,0])
    area=np.square(normal).sum(-1);valid=area>1e-24
    if not valid.any():raise ValueError('no nondegenerate reference triangles')
    result=offsets.copy();reason=None;accepted=False
    for step in range(iterations):
        posed=original+result[triangles]
        changed=np.cross(posed[:,:,1]-posed[:,:,0],posed[:,:,2]-posed[:,:,0])
        ratios=np.where(valid,(changed*normal).sum(-1)/np.maximum(area,1e-30),1.)
        minimum=float(ratios.min())
        if minimum>=minimum_ratio-1e-9:
            accepted=True;break
        pose,face=np.unravel_index(ratios.argmin(),ratios.shape)
        ids=triangles[face];points=posed[pose,face];n=normal[pose,face]/area[pose,face]
        gb=np.cross(points[2]-points[0],n);gc=np.cross(n,points[1]-points[0])
        gradient=np.array([-gb-gc,gb,gc]);gradient[~free[ids]]=0
        norm=np.square(gradient).sum()
        if norm<1e-20:
            reason='violating triangle has no usable free direction';break
        update=(minimum_ratio-minimum+1e-8)*gradient/norm
        proposed=result[ids]+update
        if np.linalg.norm(proposed-offsets[ids],axis=1).max()>maximum_repair_mm*.001:
            reason='area repair displacement limit exceeded';break
        # Leave the final iteration for checking, never return an unchecked step.
        if step==iterations-1:
            reason='area repair iteration limit';break
        result[ids]=proposed
    report=dict(accepted=accepted,iterations=step+1,minimum_area_ratio=minimum,
                minimum_required=minimum_ratio,maximum_repair_mm=float(np.linalg.norm(result-offsets,axis=1).max()*1000),
                maximum_allowed_mm=maximum_repair_mm,fixed_offsets_unchanged=bool(np.array_equal(result[~free],offsets[~free])),reason=reason)
    return result,report
