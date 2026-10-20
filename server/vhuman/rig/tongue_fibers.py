"""Authored tongue compartments and CPU harmonic fiber directions.

The Laplace-gradient construction follows the geometric approach described by
Gomez et al., Laplace-based modeling of fiber orientation in the tongue (2018),
https://pmc.ncbi.nlm.nih.gov/articles/PMC6050131/ . Boundary patches and support
bands here are authored animation priors, not MRI segmentations or physiology.
"""
import itertools

import numpy as np


def _cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise RuntimeError('tongue fiber fitting cancelled')


def _laplace_system(points,tets):
    from scipy.sparse import coo_matrix
    p=points[tets]
    edges=np.stack((p[:,1]-p[:,0],p[:,2]-p[:,0],p[:,3]-p[:,0]),axis=2)
    determinants=np.linalg.det(edges)
    if not np.isfinite(determinants).all() or (abs(determinants)<1e-24).any():
        raise ValueError('degenerate harmonic tongue tetrahedron')
    inverse=np.linalg.inv(edges)
    gradients=np.concatenate((-inverse.sum(1)[:,None],inverse),axis=1)
    volume=abs(determinants)/6
    element=volume[:,None,None]*np.einsum('nik,njk->nij',gradients,gradients)
    rows=np.broadcast_to(tets[:,:,None],element.shape).reshape(-1)
    columns=np.broadcast_to(tets[:,None,:],element.shape).reshape(-1)
    matrix=coo_matrix((element.reshape(-1),(rows,columns)),shape=(len(points),len(points))).tocsr()
    return matrix,gradients,volume


def _potential(matrix,fixed,values,*,cancel=None):
    from scipy.sparse import diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import cg
    _cancelled(cancel)
    fixed=np.asarray(fixed,bool);values=np.asarray(values,np.float64)
    if fixed.shape!=(matrix.shape[0],) or values.shape!=fixed.shape or not fixed.any() or not np.isfinite(values).all():
        raise ValueError('invalid harmonic tongue boundary')
    _,component=connected_components(matrix,directed=False)
    if set(component[fixed])!=set(component):raise ValueError('unanchored harmonic tongue component')
    free=np.flatnonzero(~fixed);boundary=np.flatnonzero(fixed)
    potential=values.copy();iterations=0
    if not len(free):return potential,dict(iterations=0,relative_residual=0.)
    reduced=matrix[free][:,free];diagonal=reduced.diagonal()
    if not np.isfinite(diagonal).all() or (diagonal<=0).any():raise ValueError('invalid harmonic tongue stiffness')
    scale=float(diagonal.mean());reduced=reduced/scale
    rhs=-(matrix[free][:,boundary]@values[boundary])/scale
    def update(_):
        nonlocal iterations
        iterations+=1;_cancelled(cancel)
    solution,status=cg(reduced,rhs,M=diags(scale/diagonal),rtol=1e-10,atol=1e-12,maxiter=4000,callback=update)
    residual=float(np.linalg.norm(reduced@solution-rhs)/max(np.linalg.norm(rhs),1e-12))
    if status or not np.isfinite(solution).all() or residual>1e-7:
        raise ValueError('harmonic tongue solve did not converge')
    potential[free]=solution
    return potential,dict(iterations=iterations,relative_residual=residual)


def fit_fields(arrays,meta,*,cancel=None):
    """Fit intrinsic longitudinal/transverse/vertical and genioglossus fields.

    Other tongue families retain their authored directions. Skin, jaw paths,
    surface geometry, tetrahedra and attachments remain exactly unchanged.
    """
    _cancelled(cancel)
    regions=np.asarray(arrays['regions']);ids=np.flatnonzero(regions==1)
    if not len(ids):raise ValueError('missing tongue volume for harmonic fibers')
    global_tets=np.asarray(arrays['tets'])[ids]
    nodes,inverse=np.unique(global_tets,return_inverse=True)
    tets=inverse.reshape(-1,4);points=np.asarray(arrays['rest'],np.float64)[nodes]
    if not np.isfinite(points).all():raise ValueError('invalid harmonic tongue volume')
    matrix,gradients,volume=_laplace_system(points,tets)
    faces=np.concatenate([tets[:,face] for face in itertools.combinations(range(4),3)])
    faces,count=np.unique(np.sort(faces,axis=1),axis=0,return_counts=True)
    boundary=np.unique(faces[count==1]);surface=points[boundary]
    if not len(surface):raise ValueError('missing harmonic tongue boundary')
    potentials=[];directions=[];reports={}
    for axis,name in enumerate(('transverse','vertical','longitudinal')):
        low,high=np.quantile(surface[:,axis],[.05,.95])
        if high-low<1e-6:raise ValueError('collapsed harmonic tongue coordinate')
        lower=boundary[surface[:,axis]<=low];upper=boundary[surface[:,axis]>=high]
        fixed=np.zeros(len(points),bool);values=np.zeros(len(points))
        fixed[np.r_[lower,upper]]=True;values[upper]=1
        potential,receipt=_potential(matrix,fixed,values,cancel=cancel)
        potentials.append(potential[tets].mean(1))
        directions.append(np.einsum('ni,nij->nj',potential[tets],gradients))
        reports[name]=dict(receipt,source_nodes=len(lower),target_nodes=len(upper))
    low,high=np.quantile(surface,[.05,.95],axis=0);extent=np.maximum(high-low,1e-6)
    origin=np.array([(low[0]+high[0])/2,low[1],low[2]+.8*extent[2]])
    distance=np.linalg.norm((surface-origin)/extent,axis=1)
    source=boundary[np.argsort(distance,kind='stable')[:max(1,len(boundary)//50)]]
    target=boundary[(surface[:,1]>=np.quantile(surface[:,1],.65))&~np.isin(boundary,source)]
    if not len(target):raise ValueError('missing authored dorsal tongue patch')
    fixed=np.zeros(len(points),bool);values=np.zeros(len(points))
    fixed[np.r_[source,target]]=True;values[target]=1
    fan,receipt=_potential(matrix,fixed,values,cancel=cancel)
    directions.append(np.einsum('ni,nij->nj',fan[tets],gradients))
    reports['genioglossus_fan']=dict(receipt,source_nodes=len(source),target_nodes=len(target),
                                   source='authored ventral-anterior patch',target='authored dorsal patch')
    coordinates=np.column_stack(potentials)
    fibers=np.asarray(arrays['fibers'],np.float64).copy();retained=np.ones(len(fibers),bool)
    lookup=np.full(len(arrays['tets']),-1,np.int32);lookup[ids]=np.arange(len(ids))
    changed={}
    for index,name in enumerate(meta['muscles']):
        family=name.rsplit('_',1)[0]
        field=None;weight=None
        if family in ('superior_longitudinal','inferior_longitudinal'):
            field=directions[2];center=.9 if family.startswith('superior') else .1
            weight=np.exp(-.5*((coordinates[:,1]-center)/.2)**2)
        elif family in ('transverse','vertical'):
            field=directions[0 if family=='transverse' else 1]
            weight=np.exp(-.5*((coordinates[:,1]-.5)/.4)**2)
        elif family.startswith('genioglossus_'):
            segment=family.split('_',1)[1]
            center={'anterior':.8,'medial':.5,'posterior':.2}.get(segment)
            if center is not None:
                field=directions[3];weight=np.exp(-.5*((coordinates[:,2]-center)/.22)**2)
        if field is None:continue
        rows=np.flatnonzero(fibers[:,1]==index)
        if not len(rows):raise ValueError('missing tongue muscle '+name)
        local=lookup[fibers[rows,0].astype(np.int32)]
        if (local<0).any():raise ValueError('tongue muscle has non-tongue support')
        length=np.linalg.norm(field[local],axis=1)
        valid=(length>1e-10)&(weight[local]>.05)
        if not valid.any():raise ValueError('harmonic fitting removed all fibers from '+name)
        old=fibers[rows,5].copy();stress=old*weight[local]*valid
        # Redistribute each authored stress-volume integral before the existing
        # per-tetrahedron occupancy cap. No measured force is inferred.
        stress*=float(old@volume[local])/max(float(stress@volume[local]),1e-30)
        fibers[rows[valid],2:5]=field[local[valid]]/length[valid,None]
        fibers[rows,5]=stress;retained[rows[~valid]]=False
        changed[name]=dict(original_fibers=len(rows),retained_fibers=int(valid.sum()),
                           mean_harmonic_height=float(np.average(coordinates[local,1],weights=stress)),
                           mean_harmonic_front=float(np.average(coordinates[local,2],weights=stress)))
    fibers=fibers[retained]
    tongue=lookup[fibers[:,0].astype(np.int32)]>=0
    occupancy=np.bincount(fibers[tongue,0].astype(np.int32),weights=fibers[tongue,5],minlength=len(arrays['tets']))
    fibers[tongue,5]/=np.maximum(1,occupancy[fibers[tongue,0].astype(np.int32)]/3500.)
    _cancelled(cancel)
    return fibers,dict(model='vhuman.harmonic_tongue_fibers.v1',nodes=len(nodes),tetrahedra=len(ids),
        boundary='authored extreme caps and ventral-anterior/dorsal patches on inferred volume',
        direction_method='linear tetrahedral Laplace gradients; CPU preconditioned conjugate gradients',
        compartments=changed,solves=reports,original_fibers=len(arrays['fibers']),retained_fibers=len(fibers),
        stress_redistribution='preserve authored stress-volume integrals then cap summed tongue stress at 3500 Pa',
        anatomy_source='authored geometric priors, not measured muscle compartments or forces',
        reference='https://pmc.ncbi.nlm.nih.gov/articles/PMC6050131/')
