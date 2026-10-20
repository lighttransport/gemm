"""CPU source-mouth, dentition and tongue geometry conditioning.

The hidden lining uses an authored convex oral envelope, not measured anatomy.
Visible lip separation is bounded; topology and identity coefficients stay fixed.
"""
from __future__ import annotations

import ctypes as ct
import math

import numpy as np

from .biomechanics_native import P, ContactStats, check, ptr
from .geometry_fit import CrossingCounter, _closest_barycentric, _surface_inputs


def _preserve_oriented_area(result, faces, reference, moving, *, minimum=.15):
    squared = (reference**2).sum(1)
    for _ in range(12):
        p = result[faces]
        normal = np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        ratio = (normal*reference).sum(1)/np.maximum(squared,1e-30)
        active = ratio<minimum
        if not active.any():
            break
        p,n,ids = p[active],reference[active],faces[active]
        g1,g2 = np.cross(p[:,2]-p[:,0],n),np.cross(n,p[:,1]-p[:,0])
        gradient = np.stack((-g1-g2,g1,g2),axis=1)*moving[ids][...,None]
        denominator = (gradient**2).sum((1,2))
        delta = ((minimum-ratio[active])*squared[active]/np.maximum(denominator,1e-30))[:,None,None]*gradient
        change,count = np.zeros_like(result),np.zeros(len(result))
        np.add.at(change,ids.ravel(),delta.reshape(-1,3))
        np.add.at(count,ids.ravel(),moving[ids].ravel())
        result[moving] += change[moving]/np.maximum(count[moving,None],1)


def fit_tongue(model, vertices, *, limit=.001, iterations=100, cancel=None, progress=None):
    """Untangle local source tongue folds with bounded, area-preserving fairing.

    Only two edge rings around existing crossings may move. The frontmost
    fifth of the tongue is fixed, preserving its visible tip. This conditions
    an authored anatomical prior; it does not estimate a subject's anatomy.
    """
    from scipy.sparse import coo_matrix
    from .anatomy import part_masks
    vertices,triangles = _surface_inputs(vertices,model.data['triangles'])
    tongue = np.asarray(model.group('tongue'),bool)
    if (tongue.shape!=(len(vertices),) or not tongue.any() or not math.isfinite(limit) or limit<=0
            or type(iterations) is not int or not 1<=iterations<=300):
        raise ValueError('invalid tongue fitting inputs')
    faces = triangles[tongue[triangles].all(1)]
    selected = np.zeros(len(triangles),bool)
    parts = []
    for name,mask,_ in part_masks(model):
        use = mask[triangles].all(1)
        selected |= use
        if not (mask&tongue).any() and use.any():
            parts.append((name,triangles[use]))
    result = vertices.copy()
    with CrossingCounter(vertices,triangles[selected]) as all_counter, CrossingCounter(vertices,faces) as counter:
        _cancel(cancel)
        before,before_self = all_counter.count(vertices),counter.count(vertices)
        records = counter.intersections(vertices)
        moving = np.zeros(len(vertices),bool)
        moving[np.unique(records)] = True
        edges = np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
        rows,cols = np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]]
        degree = np.bincount(rows,minlength=len(vertices))
        adjacency = coo_matrix((1/np.maximum(degree[rows],1),(rows,cols)),shape=(len(vertices),len(vertices))).tocsr()
        editable = tongue&(vertices[:,2]<=np.quantile(vertices[tongue,2],.8))
        if (moving&~editable).any():
            raise ValueError('tongue tip crossings require a separate anatomical fit')
        for _ in range(2):
            moving |= ((adjacency@moving.astype(float))>0)&editable
        p = vertices[faces]
        reference = np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        for iteration in range(iterations+1):
            _cancel(cancel)
            remaining = counter.count(result)
            if progress:
                progress(iteration/max(1,iterations),f'tongue fit: {remaining} self-crossings remain')
            if not remaining:
                break
            if iteration==iterations:
                raise ValueError('bounded tongue fairing did not converge')
            average = adjacency@result
            result[moving] += .1*(average[moving]-result[moving])
            _preserve_oriented_area(result,faces,reference,moving)
            if np.linalg.norm(result-vertices,axis=1).max()>limit:
                raise ValueError('tongue fitting exceeded its displacement bound')
        after = all_counter.count(result)
    interfaces = {}
    for name,part_faces in parts:
        with CrossingCounter(vertices,part_faces) as part_counter, CrossingCounter(vertices,np.concatenate((faces,part_faces))) as pair_counter:
            fixed = part_counter.count(vertices)
            old,new = pair_counter.count(vertices)-fixed-before_self,pair_counter.count(result)-fixed
        if new>old:
            raise ValueError(f'tongue fitting introduced crossings against {name}')
        interfaces[name] = dict(before_crossings=old,after_crossings=new)
    p = result[faces]
    normal = np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    ratio = (normal*reference).sum(1)/np.maximum((reference**2).sum(1),1e-30)
    if (ratio.min()<.1 or after>before-before_self
            or not np.array_equal(result[~moving],vertices[~moving])):
        raise ValueError('tongue fitting failed its independent geometry gates')
    return result,dict(schema='vhuman.tongue_fit.v1',accepted=True,applied=bool(np.any(result!=vertices)),
        geometry_only=True,before_crossings=before,after_crossings=after,
        before_tongue_self_crossings=before_self,after_tongue_self_crossings=0,
        max_displacement_mm=float(np.linalg.norm(result-vertices,axis=1).max()*1000),
        min_oriented_area_ratio=float(ratio.min()),iterations=iteration,editable_vertices=int(moving.sum()),
        interfaces=interfaces,topology_preserved=True,anterior_tip_fixed=True,
        identity_coefficients_frozen=True,anatomy='authored geometric prior; not measured anatomy')


def fit_tongue_root(model, vertices, *, limit=.009, scale=1., iterations=480, cancel=None, progress=None):
    """Separate an inferred ventral tongue base from the fixed lower arch.

    Radial continuation first protects the anterior rim, then releases its
    ventral portion. The upper 60% of source tongue vertices are restored
    exactly and independently checked before returning. No other part moves.
    This conditions authored hidden geometry, not measured tongue anatomy.
    """
    from scipy.sparse import coo_matrix
    from .anatomy import part_masks
    vertices,triangles=_surface_inputs(vertices,model.data['triangles'])
    tongue=np.asarray(model.group('tongue'),bool)
    arch=np.asarray(model.group('lower_teeth_and_gums'),bool)
    if (tongue.shape!=(len(vertices),) or arch.shape!=tongue.shape or not tongue.any() or not arch.any()
            or not math.isfinite(limit) or limit<=0 or not math.isfinite(scale) or scale<=0
            or type(iterations) is not int or not 2<=iterations<=1200):
        raise ValueError('invalid tongue-root fitting inputs')
    faces=triangles[tongue[triangles].all(1)]
    selected=np.zeros(len(triangles),bool);parts=[]
    for name,mask,_ in part_masks(model):
        use=mask[triangles].all(1);selected|=use
        if not (mask&tongue).any() and use.any():parts.append((name,triangles[use]))
    edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    rows,cols=np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]]
    degree=np.bincount(rows,minlength=len(vertices))
    adjacency=coo_matrix((1/np.maximum(degree[rows],1),(rows,cols)),shape=(len(vertices),len(vertices))).tocsr()
    p=vertices[faces];reference=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    dorsal=tongue&(vertices[:,1]>=np.quantile(vertices[tongue,1],.4))
    result=vertices.copy();best=result.copy();history=[];best_score=None
    def root_records(records):return records[tongue[records].any(1)&arch[records].any(1)]
    with CrossingCounter(vertices,triangles[selected]) as counter,CrossingCounter(vertices,faces) as own:
        _cancel(cancel)
        if own.count(vertices):raise ValueError('tongue-root fitting requires an untangled tongue; use --fit-tongue')
        records=counter.intersections(vertices);before=len(records);before_root=len(root_records(records))
        budgets=(iterations//2,iterations-iterations//2)
        for stage,(quantile,budget) in enumerate(zip((.65,.6),budgets)):
            stage_limit=min(limit,.007*scale) if stage==0 else limit
            center=np.array([np.median(vertices[tongue,0]),0,np.quantile(vertices[tongue,2],quantile)])
            editable=(tongue&(vertices[:,2]<=np.quantile(vertices[tongue,2],.95)) if stage==0 else
                      tongue&~((vertices[:,2]>=np.quantile(vertices[tongue,2],.9))&
                               (vertices[:,1]>=np.quantile(vertices[tongue,1],.7))))
            if stage:result=best.copy()
            for iteration in range(budget+1):
                _cancel(cancel)
                records=counter.intersections(result);active=root_records(records)
                movement=float(np.linalg.norm(result-vertices,axis=1).max())
                if movement>stage_limit:break
                score=(len(records),movement)
                if best_score is None or score<best_score:
                    best_score=score;best=result.copy()
                if progress:progress((sum(budgets[:stage])+iteration)/iterations,
                    f'tongue root fit: {len(active)} lower-arch crossings remain')
                if iteration%10==0 or not len(active):
                    history.append(dict(stage=stage+1,iteration=iteration,root_crossings=len(active),
                                        total_crossings=len(records),max_displacement_mm=movement*1000))
                if not len(active) or iteration==budget:break
                point,weights=crossing_weights(result,active);weights*=editable[active]
                direction=center-point;direction[:,1]=0
                direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-20)
                delta=.0001*scale*weights[...,None]*direction[:,None]/np.maximum((weights**2).sum(1)[:,None,None],1e-20)
                delta*=np.minimum(1,.00015*scale/np.maximum(np.linalg.norm(delta,axis=2,keepdims=True),1e-20))
                change,count=np.zeros_like(result),np.zeros(len(result))
                np.add.at(change,active.ravel(),delta.reshape(-1,3));np.add.at(count,active.ravel(),(weights!=0).ravel())
                moving=count>0;result[moving]+=.8*change[moving]/count[moving,None]
                displacement=result-vertices
                result[editable]+=.08*max(0,1-iteration/150)*((adjacency@displacement)[editable]-displacement[editable])
                _preserve_oriented_area(result,faces,reference,editable,minimum=.12)
            result=best.copy();result[dorsal]=vertices[dorsal]
            if not len(root_records(counter.intersections(result))):break
        result[dorsal]=vertices[dorsal]
        after=counter.count(result);after_root=len(root_records(counter.intersections(result)));after_self=own.count(result)
    interfaces={}
    for name,fixed_faces in parts:
        with CrossingCounter(vertices,fixed_faces) as fixed,CrossingCounter(vertices,np.concatenate((faces,fixed_faces))) as pair:
            fixed_count=fixed.count(vertices)
            old,new=pair.count(vertices)-fixed_count,pair.count(result)-fixed_count-after_self
        if new>old:raise ValueError(f'tongue-root fitting introduced crossings against {name}')
        interfaces[name]=dict(before_crossings=old,after_crossings=new)
    p=result[faces];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    ratio=(normal*reference).sum(1)/np.maximum((reference**2).sum(1),1e-30)
    movement=float(np.linalg.norm(result-vertices,axis=1).max())
    if (after_root or after_self or after>before-before_root or movement>limit or ratio.min()<.1
            or not np.array_equal(result[~tongue|dorsal],vertices[~tongue|dorsal])):
        raise ValueError('bounded tongue-root fitting did not pass its independent geometry gates')
    return result,dict(schema='vhuman.tongue_root_fit.v1',accepted=True,applied=bool(np.any(result!=vertices)),
        before_crossings=before,after_crossings=after,before_root_crossings=before_root,after_root_crossings=after_root,
        max_displacement_mm=movement*1000,displacement_limit_mm=limit*1000,min_oriented_area_ratio=float(ratio.min()),
        dorsal_vertices_fixed=int(dorsal.sum()),non_tongue_vertices_fixed=True,topology_preserved=True,
        geometry_only=True,interfaces=interfaces,history=history,anatomy_source='authored ventral tongue base; not measured anatomy')


def fit_dentition(model, vertices, *, limit=.001, iterations=160, cancel=None, progress=None):
    """Separate disconnected crowns by bounded rigid translations.

    Crown topology defines individual teeth. Shared gingival seam vertices
    follow their crown; other vertices are fixed. A crossing within a crown
    is rejected. This is resting
    geometry conditioning, not a change to the simulated mandible transform.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from .anatomy import part_masks
    vertices,triangles = _surface_inputs(vertices,model.data['triangles'])
    teeth = np.asarray(model.group('teeth'),bool)
    if (teeth.shape!=(len(vertices),) or not teeth.any() or not math.isfinite(limit) or limit<=0
            or type(iterations) is not int or not 1<=iterations<=300):
        raise ValueError('invalid dentition fitting inputs')
    faces = triangles[teeth[triangles].all(1)]
    edges = np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    rows,cols = np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]]
    graph = coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(vertices),len(vertices))).tocsr()
    _,labels = connected_components(graph,directed=False)
    components = np.unique(labels[teeth])
    crown = np.full(len(vertices),-1,np.int32)
    ids = [np.flatnonzero(teeth&(labels==label)) for label in components]
    for index,indices in enumerate(ids):
        crown[indices] = index
    centers = np.array([vertices[index].mean(0) for index in ids])
    offsets = np.zeros_like(centers)
    selected = np.zeros(len(triangles),bool)
    other_faces = []
    for name,mask,_ in part_masks(model):
        part_faces = triangles[mask[triangles].all(1)]
        selected |= mask[triangles].all(1)
        part_faces = part_faces[~teeth[part_faces].all(1)]
        if len(part_faces):
            other_faces.append((name,part_faces))
    result = vertices.copy()
    with CrossingCounter(vertices,triangles[selected]) as all_counter, CrossingCounter(vertices,faces) as counter:
        before = all_counter.count(vertices)
        before_teeth = counter.count(vertices)
        for iteration in range(iterations+1):
            _cancel(cancel)
            records = counter.intersections(result)
            if progress:
                progress(iteration/max(1,iterations),f'dentition fit: {len(records)} crossings remain')
            if not len(records):
                break
            if iteration==iterations:
                raise ValueError('rigid crown separation did not converge')
            pairs = np.unique(np.sort(np.column_stack((crown[records[:,0]],crown[records[:,2]])),axis=1),axis=0)
            if (pairs[:,0]==pairs[:,1]).any():
                raise ValueError('a tooth crown intersects itself; rigid fitting cannot repair it')
            step,count = np.zeros_like(offsets),np.zeros(len(ids))
            for a,b in pairs:
                direction = centers[b]+offsets[b]-centers[a]-offsets[a]
                distance = np.linalg.norm(direction)
                if distance<1e-12:
                    raise ValueError('coincident crown centers have no authored separation direction')
                direction /= distance
                step[a] -= .000025*direction
                step[b] += .000025*direction
                count[[a,b]] += 1
            offsets += step/np.maximum(count[:,None],1)
            if np.linalg.norm(offsets,axis=1).max()>limit:
                raise ValueError('dentition fit exceeded its rigid displacement bound')
            for indices,offset in zip(ids,offsets):
                result[indices] = vertices[indices]+offset
        after = all_counter.count(result)
    # Count each dental/other interface independently: removal of crown
    # overlap must not hide newly introduced gum, tongue or skin crossings.
    interfaces = {}
    for name,part_faces in other_faces:
        affected = teeth[part_faces].any(1)
        minimum = 1.
        if affected.any():
            old_points,new_points = vertices[part_faces[affected]],result[part_faces[affected]]
            old_normal = np.cross(old_points[:,1]-old_points[:,0],old_points[:,2]-old_points[:,0])
            new_normal = np.cross(new_points[:,1]-new_points[:,0],new_points[:,2]-new_points[:,0])
            minimum = float(((new_normal*old_normal).sum(1)/np.maximum((old_normal**2).sum(1),1e-30)).min())
            if minimum<.1:
                raise ValueError(f'dentition fitting violated {name} triangle orientation')
        with CrossingCounter(vertices,part_faces) as part_counter, CrossingCounter(vertices,np.concatenate((faces,part_faces))) as pair_counter:
            before_self,after_self = part_counter.count(vertices),part_counter.count(result)
            old = pair_counter.count(vertices)-before_self-before_teeth
            new = pair_counter.count(result)-after_self
        if new>old or after_self>before_self:
            raise ValueError(f'dentition fitting introduced crossings against {name}')
        interfaces[name] = dict(before_crossings=old,after_crossings=new,
                               before_self_crossings=before_self,after_self_crossings=after_self,
                               min_oriented_area_ratio=minimum)
    if after>before or (before_teeth and after>=before) or not np.array_equal(result[~teeth],vertices[~teeth]):
        raise ValueError('dentition fit failed its independent geometry gates')
    return result,dict(schema='vhuman.dentition_fit.v1',accepted=True,applied=bool(np.any(offsets)),
        geometry_only=True,before_crossings=before,after_crossings=after,
        before_teeth_crossings=before_teeth,after_teeth_crossings=0,crowns=len(ids),iterations=iteration,
        max_displacement_mm=float(np.linalg.norm(offsets,axis=1).max()*1000),
        crown_translations_m=offsets.tolist(),interfaces=interfaces,topology_preserved=True,
        crown_shapes_preserved=True,non_dental_vertices_fixed=True,identity_coefficients_frozen=True,
        gingival_seams='shared crown vertices follow their rigid tooth')


def crossing_weights(vertices, records):
    """Intersection points and positive coordinates on both primitives."""
    a,b,c,d,e = vertices[records].transpose(1,0,2)
    normal = np.cross(d-c,e-c)
    denominator = np.einsum('ij,ij->i',b-a,normal)
    if (np.abs(denominator)<1e-20).any():
        raise ValueError('degenerate crossing witness')
    t = np.einsum('ij,ij->i',c-a,normal)/denominator
    point = a+t[:,None]*(b-a)
    bary = _closest_barycentric(point,np.stack((c,d,e),axis=1))
    return point,np.column_stack((1-t,t,bary))


def _cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise RuntimeError('oral fitting cancelled')


def separate_lips(vertices, faces, upper, lower, *, limit=.0015, iterations=100, cancel=None):
    """Resolve retained upper/lower lip crossings along the authored up axis."""
    vertices,faces = _surface_inputs(vertices,faces)
    upper,lower = np.asarray(upper,bool),np.asarray(lower,bool)
    if (upper.shape!=(len(vertices),) or lower.shape!=upper.shape or (upper&lower).any()
            or not math.isfinite(limit) or limit<=0 or type(iterations) is not int or not 1<=iterations<=200):
        raise ValueError('invalid lip fitting inputs')
    kind = upper.astype(np.int32)-lower
    result = vertices.copy()
    before = None
    with CrossingCounter(vertices,faces) as counter:
        for iteration in range(iterations+1):
            _cancel(cancel)
            records = counter.intersections(result)
            active = (kind[records]>0).any(1)&(kind[records]<0).any(1)
            records = records[active]
            if before is None:
                before = len(records)
            if not len(records):
                break
            if iteration==iterations:
                raise ValueError('lip separation did not converge')
            _,weights = crossing_weights(result,records)
            weights *= kind[records]!=0
            step = .0001*weights*kind[records]/np.maximum((weights**2).sum(1,keepdims=True),1e-20)
            change,count = np.zeros(len(vertices)),np.zeros(len(vertices))
            np.add.at(change,records.ravel(),step.ravel())
            np.add.at(count,records.ravel(),(weights>0).ravel())
            moving = count>0
            result[moving,1] += .8*change[moving]/count[moving]
            if np.abs(result[:,1]-vertices[:,1]).max()>limit:
                raise ValueError('lip separation exceeded its visible displacement bound')
    return result,dict(before_crossings=before,after_crossings=0,iterations=iteration,
                       max_displacement_mm=float(np.linalg.norm(result-vertices,axis=1).max()*1000))


class _SkinContact:
    def __init__(self, vertices, faces, moving, lib):
        self.lib,self.handle = lib,P()
        mass = np.ascontiguousarray(moving,np.float64)
        faces = np.ascontiguousarray(faces,np.int32)
        check(lib.lightphysics_surface_contact_create(ptr(vertices),ptr(mass),len(vertices),ptr(faces),len(faces),ct.byref(self.handle)))

    def project(self, previous, result):
        stats = ContactStats()
        check(self.lib.lightphysics_surface_contact_project(self.handle,ptr(previous),ptr(result),1.,.00005,3,None,ct.byref(stats)))
        return int(stats.contacts)

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.lib.lightphysics_surface_contact_destroy(self.handle)
        self.handle = P()


def fit_mouth(model, vertices, *, visible_limit=.0015, hidden_limit=.03, clearance=.0002,
              iterations=160, fairing=0., cancel=None, progress=None):
    """Fit lips and inferred lining; independently reject geometry regressions."""
    from scipy.spatial import ConvexHull
    from .anatomy import part_masks
    vertices,triangles = _surface_inputs(vertices,model.data['triangles'])
    if (not all(math.isfinite(x) and x>0 for x in (visible_limit,hidden_limit,clearance))
            or not math.isfinite(fairing) or not 0<=fairing<=.5
            or type(iterations) is not int or not 1<=iterations<=300):
        raise ValueError('invalid oral fitting bounds')
    skin,hidden,exterior = model.group('skin'),model.group('mouth_sock'),model.group('skin_exterior')
    upper,lower = model.group('upper_lip'),model.group('lower_lip')
    oral = model.group('tongue')|model.group('teeth')|model.group('gums')
    if (not hidden.any() or (hidden&~skin).any() or (hidden&exterior).any()
            or (hidden&(upper|lower)).any() or (oral&skin).any()):
        raise ValueError('invalid oral semantic groups')
    selected = np.zeros(len(triangles),bool)
    for _,mask,_ in part_masks(model):
        selected |= mask[triangles].all(1)
    faces = triangles[selected]
    skin_faces = triangles[skin[triangles].all(1)]
    with CrossingCounter(vertices,faces) as counter, CrossingCounter(vertices,skin_faces) as self_counter:
        before,before_self = counter.count(vertices),self_counter.count(vertices)
        result,lip_report = separate_lips(vertices,skin_faces,upper,lower,limit=visible_limit,cancel=cancel)
        records = counter.intersections(result)
        active = hidden[records].any(1)&oral[records].any(1)
        before_oral = int(active.sum())
        contacts,iteration = 0,0
        if before_oral:
            from scipy.sparse import coo_matrix
            edges = np.unique(np.sort(np.concatenate((skin_faces[:,[0,1]],skin_faces[:,[1,2]],skin_faces[:,[2,0]])),axis=1),axis=0)
            rows,cols = np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]]
            degree = np.bincount(rows,minlength=len(vertices))
            adjacency = coo_matrix((1/np.maximum(degree[rows],1),(rows,cols)),shape=(len(vertices),len(vertices))).tocsr()
            lining_reference = result.copy()
            hull = ConvexHull(vertices[oral])
            planes = hull.equations
            ids = np.flatnonzero(hidden)
            # Conservative crop of faces that can meet a bounded hidden move.
            lo,hi = vertices[hidden].min(0)-hidden_limit,vertices[hidden].max(0)+hidden_limit
            corners = vertices[skin_faces]
            crop = ((corners.max(1)>=lo)&(corners.min(1)<=hi)).all(1)
            with _SkinContact(result,skin_faces[crop],hidden,counter.lib) as contact:
                for iteration in range(iterations+1):
                    _cancel(cancel)
                    if progress and iteration%10==0:
                        progress(iteration/max(1,iterations),f'oral lining fit: {int(active.sum())} crossings remain')
                    if not active.any():
                        break
                    if iteration==iterations:
                        raise ValueError('oral lining fit did not converge')
                    previous = result.copy()
                    w = records[active]
                    point,weights = crossing_weights(result,w)
                    value = point@planes[:,:3].T+planes[:,3]
                    plane = value.argmax(1)
                    gap = clearance-value[np.arange(len(w)),plane]
                    weights *= hidden[w]
                    step = gap[:,None,None]*weights[:,:,None]*planes[plane,None,:3]/np.maximum((weights**2).sum(1)[:,None,None],1e-20)
                    step *= np.minimum(1,.0005/np.maximum(np.linalg.norm(step,axis=2,keepdims=True),1e-20))
                    change,count = np.zeros_like(result),np.zeros(len(result))
                    np.add.at(change,w.ravel(),step.reshape(-1,3))
                    np.add.at(count,w.ravel(),(weights>0).ravel())
                    moving = count>0
                    result[moving] += .8*change[moving]/count[moving,None]
                    value = result[ids]@planes[:,:3].T+planes[:,3]
                    plane = value.argmax(1)
                    gap = np.maximum(0,clearance-value[np.arange(len(ids)),plane])
                    result[ids] += np.minimum(gap,.0005)[:,None]*planes[plane,:3]
                    # Fair only displacement, keeping the source's shape and
                    # fixed lip boundary. Decay the regularizer so clearance
                    # can converge without fighting an enduring shrink force.
                    displacement = result-lining_reference
                    alpha = fairing*max(0,1-iteration/100)
                    result[hidden] += alpha*((adjacency@displacement)[hidden]-displacement[hidden])
                    contacts += contact.project(previous,result)
                    if np.linalg.norm(result[hidden]-vertices[hidden],axis=1).max()>hidden_limit:
                        raise ValueError('oral lining exceeded its hidden displacement bound')
                    records = counter.intersections(result)
                    active = hidden[records].any(1)&oral[records].any(1)
        after,after_self = counter.count(result),self_counter.count(result)
    moving = hidden|upper|lower
    if not np.array_equal(result[~moving],vertices[~moving]) or after_self>before_self:
        raise ValueError('oral fit changed fixed anatomy or increased skin self-crossings')
    external = triangles[exterior[triangles].all(1)]
    old,new = vertices[external],result[external]
    reference = np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0])
    normal = np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0])
    ratio = (normal*reference).sum(1)/np.maximum((reference**2).sum(1),1e-30)
    visible_change = float(np.linalg.norm(result[exterior]-vertices[exterior],axis=1).max(initial=0))
    hidden_change = float(np.linalg.norm(result[hidden]-vertices[hidden],axis=1).max())
    applied = bool(np.any(result!=vertices))
    if (ratio.min()<.1 or visible_change>visible_limit or hidden_change>hidden_limit
            or after>before or (applied and after==before)):
        raise ValueError('oral fit failed its independent geometry gates')
    return result,dict(schema='vhuman.mouth_fit.v1',accepted=True,applied=applied,geometry_only=True,
        before_crossings=before,after_crossings=after,before_skin_self_crossings=before_self,
        after_skin_self_crossings=after_self,before_lining_oral_crossings=before_oral,after_lining_oral_crossings=0,
        lips=lip_report,lining_iterations=iteration,lining_fairing=fairing,skin_ccd_contacts=contacts,
        visible_max_mm=visible_change*1000,hidden_max_mm=hidden_change*1000,
        visible_min_oriented_area_ratio=float(ratio.min()),topology_preserved=True,identity_coefficients_frozen=True,
        teeth_tongue_eyes_fixed=True,hidden_anatomy='authored convex oral envelope; not measured anatomy')
