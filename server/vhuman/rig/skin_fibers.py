"""Geometric transport of authored facial fibers into the exterior skin shell."""
import math
import numpy as np


def _ocular_directions(arrays,meta,points,muscles,directions):
    """Authored elliptical palpebral bands, before surface-tangent transport."""
    result=directions.copy();reports={}
    surface=arrays['surface_rest']
    for side,part_name in (('Left','eye_left'),('Right','eye_right')):
        name='orbicularis_oculi_'+side
        if name not in meta['muscles']:continue
        part=next((p for p in meta['parts'] if p['name']==part_name),None)
        if part is None:continue
        globe=surface[part['surface_start']:part['surface_start']+part['surface_count']]
        low,high=globe.min(0),globe.max(0);center=(low+high)/2;radius=(high-low)/2
        radii=radius[:2]*[1.1,.45];source='authored ellipse from globe bounds'
        if 'fit_eye_features' in arrays:
            lid=surface[arrays['fit_eye_features'].astype(bool)]
            lid=lid[(abs(lid[:,0]-center[0])<=1.5*radius[0])&(abs(lid[:,1]-center[1])<=1.5*radius[1])]
            if len(lid)>=4:
                radii=np.maximum(np.quantile(abs(lid[:,:2]-center[:2]),.95,axis=0),.2*radius[:2])
                source='95% bounds of exterior eyelid features around globe center'
        if not np.isfinite(radii).all() or (radii<=0).any():raise ValueError('invalid ocular fiber ellipse')
        selected=np.flatnonzero(muscles==meta['muscles'].index(name))
        local=points[selected,:2]-center[:2]
        tangent=np.column_stack((-radii[0]*local[:,1]/radii[1],radii[1]*local[:,0]/radii[0],np.zeros(len(local))))
        result[selected]=tangent/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-12)
        reports[name]=dict(center=center.tolist(),ellipse_radii=radii.tolist(),source=source,fibers=len(selected))
    return result,reports

def tangent_fields(arrays,meta,exterior,scale,*,depth=.005):
    """Project facial directions onto nearby exterior-surface tangent planes.

    Deep oral lining is not an exterior facial muscle layer. Tongue fibers
    remain volumetric and unchanged. This fits authored field geometry, not
    measured physiological directions or strengths.
    """
    from scipy.spatial import cKDTree
    from ..reconstruction.contact_runtime import signed_contact
    if not math.isfinite(scale) or scale<=0 or not math.isfinite(depth) or depth<=0:
        raise ValueError('invalid facial fiber scale/depth')
    exterior=np.asarray(exterior,bool)
    if exterior.shape!=(len(arrays['surface_rest']),):
        raise ValueError('invalid exterior facial fiber mask')
    part=next(p for p in meta['parts'] if p['name']=='skin')
    begin,count=part['surface_start'],part['surface_count']
    p=arrays['surface_rest'][begin:begin+count]
    f=arrays['faces'][part['face_start']:part['face_start']+part['face_count']]-begin
    visible=exterior[begin:begin+count]
    f=f[visible[f].all(1)]
    if not len(f):raise ValueError('missing exterior facial fiber support')
    triangles=p[f]
    valid=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)>1e-16
    triangles=triangles[valid]
    if not len(triangles):raise ValueError('degenerate exterior facial fiber support')
    fibers=np.asarray(arrays['fibers'],np.float64).copy()
    skin=arrays['regions'][fibers[:,0].astype(np.int32)]==0
    ids=np.flatnonzero(skin)
    tets,inverse=np.unique(fibers[ids,0].astype(np.int32),return_inverse=True)
    centers=arrays['rest'][arrays['tets'][tets]].mean(1)
    tree=cKDTree(triangles.mean(1));distances=np.empty(len(centers));normals=np.empty_like(centers)
    for begin in range(0,len(centers),2048):
        points=centers[begin:begin+2048]
        _,near=tree.query(points,k=min(16,len(triangles)))
        near=np.asarray(near).reshape(len(points),-1)
        tri=triangles[near]
        closest,normal,_=signed_contact(points,tri[:,:,0],tri[:,:,1],tri[:,:,2])
        distances[begin:begin+len(points)]=np.linalg.norm(points-closest,axis=1)
        normals[begin:begin+len(points)]=normal
    normal=normals[inverse]
    directions=fibers[ids,2:5]
    directions,ocular=_ocular_directions(arrays,meta,centers[inverse],fibers[ids,1].astype(np.int32),directions)
    tangent=directions-np.einsum('ij,ij->i',directions,normal)[:,None]*normal
    length=np.linalg.norm(tangent,axis=1)
    keep=(distances[inverse]<=depth*scale)&(length>.1)&(np.linalg.norm(normal,axis=1)>.5)
    fibers[ids[keep],2:5]=tangent[keep]/length[keep,None]
    selected=np.ones(len(fibers),bool);selected[ids[~keep]]=False
    result=fibers[selected]
    before=np.bincount(fibers[:,1].astype(np.int32),minlength=len(meta['muscles']))
    after=np.bincount(result[:,1].astype(np.int32),minlength=len(meta['muscles']))
    silent=[name for name,a,b in zip(meta['muscles'],before,after) if a and not b]
    if silent:raise ValueError('surface fitting removed all fibers from '+', '.join(silent))
    dot=np.einsum('ij,ij->i',fibers[ids[keep],2:5],normal[keep])
    return result,dict(model='vhuman.exterior_tangent_fibers.v2_ocular',maximum_support_distance_mm=depth*scale*1000,ocular_bands=ocular,
        support_query='exact point-to-triangle distances on 16 nearest centroid candidates',
        original_skin_fibers=len(ids),retained_skin_fibers=int(keep.sum()),
        max_normal_component=float(abs(dot).max()) if len(dot) else 0.,
        fibers_per_muscle={name:int(n) for name,n in zip(meta['muscles'],after)},
        anatomy_source='authored directions projected onto fitted exterior; tongue fields unchanged')
