"""Bounded surface-neighborhood harmonic color completion, original implementation."""
import numpy as np


def harmonic(points,normals,colors,measured,max_distance=.02):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix,diags
    from scipy.sparse.linalg import spsolve
    from .materials import complete_surface
    points,normals,colors=np.asarray(points,float),np.asarray(normals,float),np.asarray(colors,float)
    measured=np.asarray(measured,bool)
    if points.shape!=normals.shape or colors.shape!=points.shape or measured.shape!=(len(points),) or not np.isfinite(np.r_[points.ravel(),normals.ravel(),colors.ravel()]).all():raise ValueError('invalid surface completion arrays')
    output=complete_surface(points,normals,colors,measured,max_distance)
    visible=np.flatnonzero(measured)
    nearest,local=cKDTree(points[visible]).query(points,k=min(8,len(visible)))
    nearest,local=nearest.reshape(len(points),-1),local.reshape(len(points),-1)
    agreement=(normals[:,None]*normals[visible[local]]).sum(-1)
    distance=np.min(np.where(agreement>.7,nearest,np.inf),axis=1)
    confidence=np.where(measured,1.,np.clip(1-distance/max_distance,0,1)*.5)
    missing=np.flatnonzero(~measured&(distance<max_distance))
    if len(missing):
        d,j=cKDTree(points).query(points,k=min(9,len(points)))
        d,j=d[:,1:],j[:,1:]
        row=np.repeat(np.arange(len(points)),j.shape[1]);col=j.ravel()
        agreement=(normals[row]*normals[col]).sum(1)
        # Local edges cannot bridge opposite skin sheets or distant UV islands.
        keep=(agreement>.7)&(d.ravel()<.004)
        row,col,d=row[keep],col[keep],d.ravel()[keep]
        weight=np.maximum(agreement[keep],0)**4/np.maximum(d,.00025)**2
        adjacency=coo_matrix((weight,(row,col)),shape=(len(points),len(points))).tocsr()
        adjacency=(adjacency+adjacency.T)*.5
        degree=np.asarray(adjacency.sum(1)).ravel();lap=diags(degree)-adjacency
        # Weak completed prior makes isolated components well posed; measured anchors stay exact.
        ridge=np.maximum(degree[missing]*.01,1.)
        fixed=np.flatnonzero(measured|(distance>=max_distance))
        system=lap[missing][:,missing]+diags(ridge)
        rhs=adjacency[missing][:,fixed]@output[fixed]+ridge[:,None]*output[missing]
        output[missing]=np.clip(spsolve(system,rhs),0,1)
    output[measured]=colors[measured]
    return output,confidence,dict(method='surface-neighborhood harmonic completion; exact measured anchors',
        bounded_distance_m=max_distance,max_edge_m=.004,normal_dot_min=.7,harmonically_completed_samples=len(missing),
        measured_samples=int(measured.sum()),completion_is_observation=False,
        limitations=['nearby surface graph approximates geodesics; no recovered hidden detail','far skin retains bounded median prior'])


def feather(points,normals,colors,confidence,radius=.004):
    """Blend uncertain transfer edges in metric space, across UV islands.

    High-confidence samples retain source detail. Opposing surface sheets and
    distant regions cannot exchange color. This is seam cleanup, not recovery.
    """
    from scipy.spatial import cKDTree
    points,normals,colors,confidence=map(np.asarray,(points,normals,colors,confidence))
    if points.shape!=normals.shape or colors.shape!=points.shape or confidence.shape!=(len(points),) or not len(points):
        raise ValueError('invalid seam arrays')
    if not np.isfinite(radius) or radius<=0 or not all(np.isfinite(a).all() for a in (points,normals,colors,confidence)) or ((confidence<0)|(confidence>1)).any():
        raise ValueError('invalid seam samples or radius')
    tree=cKDTree(points);out=colors.copy()
    uncertain=np.flatnonzero(confidence<.6)
    for start in range(0,len(uncertain),8192):
        ids=uncertain[start:start+8192]
        distance,near=tree.query(points[ids],k=min(32,len(points)))
        distance,near=distance.reshape(len(ids),-1),near.reshape(len(ids),-1)
        agreement=(normals[ids,None]*normals[near]).sum(-1)
        weights=(distance<radius)*(agreement>.7)*np.exp(-.5*(distance/(radius/2))**2)
        smooth=(colors[near]*weights[...,None]).sum(1)/np.maximum(weights.sum(1)[:,None],1e-12)
        t=np.clip(confidence[ids]/.6,0,1)
        strength=.85*(1-t*t*(3-2*t))
        out[ids]=colors[ids]*(1-strength[:,None])+smooth*strength[:,None]
    before=after=0.;edge_count=0
    for start in range(0,len(uncertain),8192):
        ids=uncertain[start:start+8192]
        distance,near=tree.query(points[ids],k=min(8,len(points)))
        distance,near=distance.reshape(len(ids),-1),near.reshape(len(ids),-1)
        agreement=(normals[ids,None]*normals[near]).sum(-1)
        boundary=((confidence[ids,None]>.1)!=(confidence[near]>.1))&(distance<radius)&(agreement>.7)
        before+=float(((colors[ids,None]-colors[near])**2)[boundary].sum())
        after+=float(((out[ids,None]-out[near])**2)[boundary].sum())
        edge_count+=int(boundary.sum())
    return out,dict(method='confidence-feathered metric surface color',radius_m=radius,
                    boundary_edges=edge_count,boundary_rms_before=float(np.sqrt(before/max(3*edge_count,1))),
                    boundary_rms_after=float(np.sqrt(after/max(3*edge_count,1))),
                    normal_dot_min=.7,protected_confidence=.6,filtered_samples=len(uncertain),
                    rms_linear_change=float(np.sqrt(np.mean((out-colors)**2))))
