"""Authored external tongue-muscle segments with distributed tissue insertions.

Origins are inferred head-local supports, not measured styloid, hyoid or soft
palate anatomy. The existing internal active-stress fields remain unchanged.
"""
import math
import numpy as np

MODEL='vhuman.tongue_extrinsic_cables.v1'
REFERENCE='https://pmc.ncbi.nlm.nih.gov/articles/PMC3687025/'
ROW_WIDTH=40
MAX_VERTICES=16


def validate_rows(values,nvertices,nmuscles,*,inverse_mass=None,rest=None):
    rows=np.asarray(values)
    if rows.ndim!=2 or rows.shape[1]!=ROW_WIDTH or rows.dtype.kind!='f' or not np.isfinite(rows).all():
        raise ValueError('invalid external muscle cable array')
    for row in rows:
        if row[0]!=int(row[0]) or not 0<=row[0]<nmuscles or row[1]!=int(row[1]) or not 1<=row[1]<=MAX_VERTICES:
            raise ValueError('invalid external muscle cable indices/count')
        count=int(row[1]);indices=row[8:8+count];weights=row[24:24+count]
        if ((indices!=np.floor(indices)).any() or (indices<0).any() or
                (indices>=min(nvertices,np.iinfo(np.int32).max)).any() or len(np.unique(indices))!=count):
            raise ValueError('invalid external muscle cable insertion vertices')
        if (weights<=0).any() or abs(float(weights.sum())-1)>1e-8:
            raise ValueError('invalid external muscle cable insertion weights')
        if (row[8+count:24]!=0).any() or (row[24+count:40]!=0).any():
            raise ValueError('external muscle cable padding must be zero')
        force,length,shortening=map(float,row[5:8])
        product=length*shortening
        if force<0 or length<=0 or not 0<shortening<1 or product<=0 or not math.isfinite(product) or not math.isfinite(force/product):
            raise ValueError('invalid external muscle cable force/length')
        indices=indices.astype(np.int32)
        if inverse_mass is not None and (np.asarray(inverse_mass)[indices]<=0).any():
            raise ValueError('external muscle cable insertions require free tissue')
        if rest is not None:
            center=np.average(np.asarray(rest)[indices],axis=0,weights=weights)
            actual=float(np.linalg.norm(center-row[2:5]))
            if actual<1e-8 or abs(actual-length)>1e-8+1e-6*length:
                raise ValueError('external muscle cable rest length differs from its geometry')


def dental_path_checks(arrays,metadata,*,positions=None,surface=None,activation=None):
    """Check straight force segments against rigid dental triangles only.

    This is a proper-intersection diagnostic, not a muscle wrapping model.
    Supplied positions/surface use the same head-local coordinate frame.
    """
    parts=[part for part in metadata.get('parts',[]) if part['name'].startswith('teeth_') and part.get('face_count',0)]
    if not parts or 'faces' not in arrays:
        return dict(evaluated=False,passed=True,proper_crossings=0,scope='no rigid dental surfaces supplied')
    volume=arrays['rest'] if positions is None else np.asarray(positions)
    if surface is None:
        surface=(arrays['surface_rest'] if positions is None else
                 (volume[arrays['embed_ids']]*arrays['embed_weights'][...,None]).sum(1))
    faces=np.concatenate([arrays['faces'][part['face_start']:part['face_start']+part['face_count']] for part in parts])
    triangles=np.asarray(surface)[faces];edge1=triangles[:,1]-triangles[:,0];edge2=triangles[:,2]-triangles[:,0]
    edge_scale=np.linalg.norm(edge1,axis=1)*np.linalg.norm(edge2,axis=1);records=[]
    for row in arrays.get('tongue_cables',[]):
        muscle=int(row[0])
        if activation is not None and activation[muscle]<=1e-8:continue
        count=int(row[1]);nodes=row[8:8+count].astype(np.int32);weights=row[24:24+count]
        endpoint=np.average(volume[nodes],axis=0,weights=weights);origin=row[2:5];direction=endpoint-origin
        crossed=np.cross(direction,edge2);determinant=np.einsum('ij,ij->i',edge1,crossed)
        valid=abs(determinant)>128*np.finfo(float).eps*edge_scale*np.linalg.norm(direction)
        inverse=np.zeros(len(determinant));inverse[valid]=1/determinant[valid]
        offset=origin-triangles[:,0];u=np.einsum('ij,ij->i',offset,crossed)*inverse
        q=np.cross(offset,edge1);v=(q@direction)*inverse;t=np.einsum('ij,ij->i',edge2,q)*inverse
        hit=valid&(u>1e-9)&(v>1e-9)&(u+v<1-1e-9)&(t>1e-9)&(t<1-1e-9)
        records.append(dict(muscle=metadata['muscles'][muscle],proper_tooth_crossings=int(hit.sum())))
    crossings=sum(record['proper_tooth_crossings'] for record in records)
    return dict(evaluated=True,passed=crossings==0,proper_crossings=crossings,paths=records,
                scope='head-local straight segments vs rigid dental surfaces; proper intersections only')


def _quantile(values,weights,fractions):
    order=np.argsort(values,kind='stable');values=values[order];weights=weights[order]
    cumulative=np.cumsum(weights)-.5*weights
    return np.interp(np.asarray(fractions)*weights.sum(),cumulative,values)


def _patch(points,nodes,center,radii,side,midline):
    distance=np.square((points-center)/radii).sum(1)
    eligible=side*(points[:,0]-midline)>.1*radii[0]
    for expansion in (1.,1.5,2.,3.):
        candidates=np.flatnonzero(eligible&(distance<=expansion**2))
        if len(candidates)>=4:break
    if len(candidates)<4:
        raise ValueError('missing distributed extrinsic tongue insertion patch')
    local=(points[candidates]-center)/radii
    count=min(MAX_VERTICES,len(candidates));selected=[];nearest=np.full(len(candidates),np.inf)
    current=int(np.argmin(distance[candidates]))
    for _ in range(count):
        selected.append(current)
        nearest=np.minimum(nearest,np.square(local-local[current]).sum(1))
        nearest[selected]=-1
        current=int(np.argmax(nearest))
    chosen=candidates[selected];weights=np.exp(-.5*distance[chosen])
    weights/=weights.sum()
    return nodes[chosen],weights


def fit(arrays,metadata,*,scale=1.,cancel=None):
    """Add six external segments without changing any tissue geometry.

    Force scales come from the existing authored stress-volume integrals
    divided by their internal bundle extent. They are animation priors.
    """
    if cancel is not None and cancel.is_set():raise RuntimeError('external tongue muscle fitting cancelled')
    if not math.isfinite(scale) or scale<=0:raise ValueError('invalid external tongue muscle scale')
    rest=np.asarray(arrays['rest'],np.float64);tets=np.asarray(arrays['tets'],np.int32)
    regions=np.asarray(arrays['regions']);mass=np.asarray(arrays['inverse_mass'])
    nodes=np.unique(tets[regions==1]);nodes=nodes[mass[nodes]>0]
    if len(nodes)<8:raise ValueError('insufficient free tongue tissue for external muscles')
    points=rest[nodes];low,high=np.quantile(points,[.02,.98],axis=0);extent=high-low
    if (extent<.001*scale).any():raise ValueError('degenerate external tongue muscle geometry')
    center=(low+high)/2;half_width=.5*extent[0]
    positions=rest[tets];edges=positions[:,1:]-positions[:,:1]
    volume=np.abs(np.einsum('ij,ij->i',np.cross(edges[:,0],edges[:,1]),edges[:,2]))/6
    centers=positions.mean(1)
    fibers=np.asarray(arrays['fibers'],np.float64);names=metadata['muscles'];rows=[];reports=[]
    families={
        'styloglossus':(.72,.30,.62,'styloid',(.010,.020,-.012)),
        'hyoglossus':(.66,.35,.52,'hyoid',(0.,-.020,0.)),
        'palatoglossus':(.62,.70,.38,'soft_palate',(0.,.010,0.)),
    }
    radii=np.maximum(extent*np.array([.30,.25,.22]),.001*scale)
    for family,(lateral,height,front,support,offset) in families.items():
        for side,sign in (('Left',1),('Right',-1)):
            if cancel is not None and cancel.is_set():raise RuntimeError('external tongue muscle fitting cancelled')
            name=family+'_'+side
            if name not in names:raise ValueError('missing external tongue muscle '+name)
            muscle=names.index(name);field=fibers[fibers[:,1]==muscle]
            if not len(field):raise ValueError('missing internal fiber field for '+name)
            insertion=np.array([center[0]+sign*lateral*half_width,low[1]+height*extent[1],low[2]+front*extent[2]])
            if support=='styloid':
                origin=np.array([center[0]+sign*(half_width+offset[0]*scale),high[1]+offset[1]*scale,low[2]+offset[2]*scale])
            elif support=='hyoid':
                origin=np.array([center[0]+sign*.45*half_width,low[1]+offset[1]*scale,low[2]+.15*extent[2]])
            else:
                origin=np.array([center[0]+sign*.55*half_width,high[1]+offset[1]*scale,low[2]+.30*extent[2]])
            selected,weights=_patch(points,nodes,insertion,radii,sign,center[0])
            insertion=np.average(rest[selected],axis=0,weights=weights)
            length=float(np.linalg.norm(origin-insertion))
            if length<.001*scale:raise ValueError('short external tongue muscle path for '+name)
            direction=(origin-insertion)/length;tet_ids=field[:,0].astype(np.int32)
            stress_volume=field[:,5]*volume[tet_ids];integral=float(stress_volume.sum())
            if integral<=0 or not math.isfinite(integral):raise ValueError('missing positive internal muscle stress for '+name)
            limits=_quantile(centers[tet_ids]@direction,stress_volume,[.05,.95])
            internal_length=max(.005*scale,float(limits[1]-limits[0]));force=integral/internal_length
            row=np.zeros(ROW_WIDTH,np.float64);row[:8]=[muscle,len(selected),*origin,force,length,.15]
            row[8:8+len(selected)]=selected;row[24:24+len(selected)]=weights;rows.append(row)
            reports.append(dict(muscle=name,support=support+'_authored_fixed_head_local',insertion_nodes=len(selected),
                origin_m=origin.tolist(),insertion_centroid_m=insertion.tolist(),rest_length_m=length,
                max_force_n=force,stress_volume_integral_nm=integral,internal_bundle_length_m=internal_length))
    result=np.asarray(rows,np.float64).reshape(-1,ROW_WIDTH)
    validate_rows(result,len(rest),len(names),inverse_mass=mass,rest=rest)
    dental=dental_path_checks(dict(arrays,tongue_cables=result),metadata)
    if not dental['passed']:raise ValueError('authored external tongue muscle paths cross rigid teeth')
    return result,dict(model=MODEL,reference=REFERENCE,cables=reports,
        origin_geometry='inferred head-local supports; not measured bone, hyoid or palate attachments',
        force_scale='existing authored internal muscle stress-volume integral / weighted 5–95% bundle extent',
        actuation='tension-only 15% unloaded shortening with a Gaussian length-tension envelope',
        geometry_preserved=True,internal_fibers_preserved=True,rest_dental_paths=dental,
        hyoid_motion='fixed authored support; not a tracked hyoid')
