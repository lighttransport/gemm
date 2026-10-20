"""Offline phoneme-to-articulator goals, independent of rendered appearance.

Channels describe intended geometry. They are inverse-control targets, never
an extra deformation applied on top of the biomechanical simulation.
"""
from __future__ import annotations

import math
import numpy as np

CHANNELS = ('opening', 'rounding', 'spread', 'seal', 'labiodental',
            'tip_up', 'tip_forward', 'dorsum_up', 'tongue_retract')
BILABIAL = frozenset(('p','py','pʰ','pʲ','b','by','bʰ','m','my','mʲ'))
BILABIAL_FRICATIVES = frozenset(('ɸ','β'))
ALVEOLAR = frozenset(('t','ty','t̪','d','dy','d̪','n','ny','N','n̩','l','l̩','ɾ','r','ry','ɹ'))
DENTAL = frozenset(('θ','ð','th','dh'))
LABIODENTAL = frozenset(('f','v','F','V'))
VELAR = frozenset(('k','ky','g','gy','ŋ','ɡ'))
SIBILANT = frozenset(('s','z','ɕ','ʃ','ʒ','ts','tɕ','tʃ','dʒ','ʑ','ch','sh','hy'))
VOWELS = {
    'a':(.85,0,.1), 'aa':(.85,0,.1), 'ɑ':(.9,.1,0), 'æ':(.75,0,.6), 'ʌ':(.6,0,.15),
    'ə':(.35,0,0), 'e':(.45,0,.7), 'ɛ':(.6,0,.5), 'E':(.45,0,.7),
    'i':(.2,0,.85), 'ih':(.25,0,.65), 'ɪ':(.3,0,.6), 'ɨ':(.2,0,.3),
    'o':(.6,.8,0), 'oh':(.6,.8,0), 'ɔ':(.7,.65,0), 'u':(.25,.95,0),
    'ou':(.3,.95,0), 'ʊ':(.35,.7,0), 'ɯ':(.25,.35,0), 'ɯᵝ':(.25,.45,0),
}
DIPHTHONGS={'aʊ':('a','u'),'aɪ':('a','i'),'eɪ':('e','i'),'oʊ':('o','u'),'ɔɪ':('o','i')}
GEOMETRIC_POLICY='vhuman.lip_surface_constraints.v5'
ARTICULATOR_POLICY='vhuman.phoneme_drives.v2'


def normalized(phone):
    if not isinstance(phone,str):
        raise ValueError('phoneme symbols must be text')
    return phone.replace('ː','').replace('ˑ','').replace('ˈ','').replace('ˌ','')


def _alignment_symbol(phone,format):
    phone=normalized(phone)
    return {'j':'dʒ','f':'ɸ'}.get(phone,phone) if format=='ja_align.v1' else phone


def _pose(phone, next_phone=''):
    phone,next_phone=normalized(phone),normalized(next_phone)
    # Japanese moraic nasals and geminate closures inherit the following
    # consonant's place. A velar closure must not request an alveolar tip pose.
    if phone=='cl' and next_phone in BILABIAL|BILABIAL_FRICATIVES|VELAR|ALVEOLAR|DENTAL|LABIODENTAL|SIBILANT:
        return _pose(next_phone)
    if phone=='N':
        if next_phone in BILABIAL|BILABIAL_FRICATIVES:return _pose('m')
        if next_phone in VELAR:return _pose('ŋ')
        if next_phone in ALVEOLAR or next_phone in SIBILANT:return _pose('n')
        v=np.zeros(len(CHANNELS));v[[0,7,8]]=[.25,.7,.4]
        return v
    phone={'A':'a','I':'i','U':'ɯ','E':'e','O':'o','ɐ':'ʌ','ä':'a','ɚ':'ə','ɜ':'ə','j':'i','y':'i','w':'u'}.get(phone,phone)
    v=np.zeros(len(CHANNELS))
    if phone in VOWELS:
        v[:3]=VOWELS[phone]
        v[7]=.65 if phone in ('i','ɪ','u','ɯ','ɯᵝ','ʊ') else .15
    elif phone in BILABIAL:
        # A closure by itself does not request a jaw opening. Neighboring
        # vowels still contribute their anticipatory opening in build().
        v[3]=1
    elif phone in BILABIAL_FRICATIVES:
        v[0]=.12;v[1]=.25;v[3]=.4
    elif phone in LABIODENTAL:
        v[0]=.2;v[4]=1
    elif phone in DENTAL:
        v[0]=.25;v[5]=.5;v[6]=.9
    elif phone in ALVEOLAR:
        v[0]=.25;v[5]=.9
        if phone in ('r','ɹ'):v[8]=.8
    elif phone in VELAR:
        v[0]=.3;v[7]=1;v[8]=.35
    elif phone in SIBILANT:
        v[0]=.2;v[2]=.25;v[5]=.5;v[7]=.4
    elif phone not in ('sil','pau','sp','','[SIL]'):
        v[0]=.2
    return v


def build(aux, times, *, strength=1.):
    times=np.asarray(times,dtype=np.float64)
    duration=float(aux.get('duration',0))
    if not math.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('speech strength must be in [0,1]')
    if times.ndim!=1 or not len(times) or not np.isfinite(times).all() or (np.diff(times)<=0).any() or times[0]<0 or not math.isfinite(duration) or duration<=0 or times[-1]>duration+1e-4:
        raise ValueError('invalid articulation timestamps')
    intervals=aux.get('intervals') or []
    output=np.zeros((len(times),len(CHANNELS)))
    denominator=np.zeros_like(output)
    receipts=[];forced_seals=[]
    if not intervals:
        # Imported alignments sometimes contain visemes only. Mark the loss of
        # tongue observability explicitly rather than inventing phone labels.
        vis=aux.get('visemes',{});names=vis.get('names',[])
        values=np.asarray(vis.get('frames',[]),float)
        fps=float(vis.get('fps',0))
        if (values.ndim!=2 or not len(values) or values.shape[1]!=len(names) or not math.isfinite(fps) or fps<=0
                or not np.isfinite(values).all() or (values<0).any()):
            raise ValueError('articulation needs phoneme intervals or valid viseme curves')
        mapping={'PP':'m','FF':'f','TH':'θ','DD':'t','kk':'k','CH':'tʃ','SS':'s','nn':'n','RR':'ɹ','sil':'sil'}
        for i,name in enumerate(names):
            value=np.interp(times,np.arange(len(values))/fps,values[:,i])
            output+=value[:,None]*_pose(mapping.get(name,name))[None]
        return dict(schema='vhuman.articulation.v1',geometry_policy=GEOMETRIC_POLICY,articulator_policy=ARTICULATOR_POLICY,
                    channels=list(CHANNELS),values=np.clip(output,0,1)*strength,
                    times=times,intervals=[],speech_strength=strength,source='viseme_only; tongue targets are priors')
    previous_end=0.
    for i,item in enumerate(intervals):
        start,end=float(item['start']),float(item['end'])
        phone=item.get('s',item.get('phone',''))
        if not isinstance(phone,str) or not math.isfinite(start+end) or not 0<=start<end<=duration+1e-4 or start<previous_end-1e-4:
            raise ValueError('invalid/overlapping phoneme interval')
        previous_end=end
        next_phone=intervals[i+1].get('s',intervals[i+1].get('phone','')) if i+1<len(intervals) else ''
        next_phone=_alignment_symbol(next_phone,aux.get('format'))
        symbol=_alignment_symbol(phone,aux.get('format'))
        if normalized(symbol) in DIPHTHONGS:
            first,last=DIPHTHONGS[normalized(phone)]
            fraction=np.clip((times-start)/max(end-start,1e-6),0,1)
            fraction=fraction*fraction*(3-2*fraction)
            pose=(1-fraction[:,None])*_pose(first)+fraction[:,None]*_pose(last)
        else:
            pose=np.tile(_pose(symbol,next_phone),(len(times),1))
        # Rounding anticipates the vowel; seals have a short, dominant window.
        attack=np.array([.045,.085,.055,.025,.035,.045,.045,.060,.060])
        release=np.array([.060,.075,.055,.025,.035,.045,.045,.060,.060])
        before=np.clip((times[:,None]-(start-attack))/attack,0,1)
        after=np.clip(((end+release)-times[:,None])/release,0,1)
        envelope=np.minimum(before*before*(3-2*before),after*after*(3-2*after))
        dominance=np.ones(len(CHANNELS))
        dominance[3:5]=4
        active=pose>0
        output+=envelope*pose*dominance
        denominator+=envelope*active*dominance
        # Preserve sub-frame closures at the nearest audio-timeline sample.
        if pose[:,3].max()>=1:
            frame=int(np.argmin(abs(times-(start+end)/2)))
            forced_seals.append(frame)
        receipts.append(dict(phone=phone,articulator_phone=symbol,start=start,end=end))
    output/=np.maximum(denominator,1)
    output[forced_seals,3]=1
    output[:,0]*=1-.7*output[:,3]
    output[:,1]*=1-.6*output[:,3]
    return dict(schema='vhuman.articulation.v1',geometry_policy=GEOMETRIC_POLICY,articulator_policy=ARTICULATOR_POLICY,
                channels=list(CHANNELS),values=np.clip(output,0,1)*strength,
                times=times,intervals=receipts,speech_strength=strength,source='phoneme_targets; hidden tongue anatomy inferred')


def compose_controls(base, values, *, emotions=None, emotion_strength=.6):
    from .speech import EMOTION_POSES
    controls={name:value for name,value in base.items() if not name.startswith(('mouth','jaw','tongue'))}
    controls.update(goal_controls(values))
    for emotion,weight in (emotions or {}).items():
        for name,value in EMOTION_POSES[emotion].items():
            if name.startswith(('mouth','jaw','tongue')):
                controls[name]=np.clip(controls.get(name,0)+.35*emotion_strength*weight*value,0,1)
    return controls


def goal_controls(values):
    v=dict(zip(CHANNELS,values))
    return dict(jawOpen=.65*v['opening'],mouthPucker=.8*v['rounding'],mouthFunnel=.5*v['rounding'],
                mouthStretchLeft=.65*v['spread'],mouthStretchRight=.65*v['spread'],
                mouthClose=v['seal'],tongueUp=v['tip_up'],tongueOut=.4*v['tip_forward'],
                tongueCurlUp=.4*v['tongue_retract'])


def _lip_columns(rest,upper,lower):
    """Rest-selected opposing seam pairs at spatial columns along the arch.

    Only seam pairs define the displacement. All vertices in a column follow
    that displacement, retaining the thickness of both semantic lip patches.
    """
    left=max(rest[upper,0].min(),rest[lower,0].min())
    right=min(rest[upper,0].max(),rest[lower,0].max())
    if right<left:raise ValueError('lip patches need an overlapping arch span')
    grid=np.linspace(left,right,33)
    radius=max((right-left)/20,.0005)
    pairs=[]
    for x in grid:
        u=upper[np.abs(rest[upper,0]-x)<=radius]
        l=lower[np.abs(rest[lower,0]-x)<=radius]
        if not len(u):u=upper[np.argsort(abs(rest[upper,0]-x))[:1]]
        if not len(l):l=lower[np.argsort(abs(rest[lower,0]-x))[:1]]
        distance=np.square(rest[u,None]-rest[l]).sum(2)
        distance+=np.square((rest[u,None,0]+rest[l,0])/2-x)
        a,b=np.unravel_index(np.argmin(distance),distance.shape)
        pairs.append((u[a],l[b]))
    return grid,np.asarray(pairs,np.int32)


def _smooth_columns(values):
    padded=np.pad(values,[(1,1)]+[(0,0)]*(values.ndim-1),mode='edge')
    return .25*padded[:-2]+.5*padded[1:-1]+.25*padded[2:]


def _feather_lip_displacements(rest,goal,faces,seeds,*,width=.004):
    """Extend lip displacement through a bounded geodesic surface collar."""
    from scipy.sparse import coo_matrix,diags
    from scipy.sparse.csgraph import dijkstra
    from scipy.sparse.linalg import spsolve
    delta=goal-rest
    if not np.any(delta):return goal
    edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    length=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
    rows,cols=np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]]
    lengths=np.tile(np.maximum(length,1e-8),2)
    graph=coo_matrix((lengths,(rows,cols)),shape=(len(rest),len(rest))).tocsr()
    ids=np.flatnonzero(seeds)
    distance=dijkstra(graph,directed=False,indices=ids,min_only=True,limit=width)
    collar=np.flatnonzero((distance>0)&np.isfinite(distance))
    if not len(collar):return goal
    weights=graph.copy();weights.data=1/weights.data
    laplacian=diags(np.asarray(weights.sum(1)).ravel())-weights
    rhs=weights[collar][:,ids]@delta[ids]
    extension=spsolve(laplacian.tocsr()[collar][:,collar],rhs)
    if not np.isfinite(extension).all():raise ValueError('lip target collar has no bounded displacement solution')
    result=goal.copy();result[collar]=rest[collar]+extension
    return result


def _guard_oral_area(surface,goal,faces,minimum=.15,*,reference=None):
    """Bound lip displacement before any incident face folds along its path."""
    delta=goal-surface
    changed=np.any(delta!=0,axis=1)
    faces=faces[changed[faces].any(1)]
    if not len(faces):return goal
    p=surface[faces];d=delta[faces]
    e1,e2=p[:,1]-p[:,0],p[:,2]-p[:,0]
    d1,d2=d[:,1]-d[:,0],d[:,2]-d[:,0]
    base_normal=np.cross(e1,e2)
    r=surface[faces] if reference is None else reference[faces]
    n=np.cross(r[:,1]-r[:,0],r[:,2]-r[:,0]);denominator=np.einsum('ij,ij->i',n,n)
    valid=denominator>1e-28
    constant=np.einsum('ij,ij->i',base_normal,n)[valid]/denominator[valid]
    linear=np.einsum('ij,ij->i',np.cross(d1,e2)+np.cross(e1,d2),n)[valid]/denominator[valid]
    quadratic=np.einsum('ij,ij->i',np.cross(d1,d2),n)[valid]/denominator[valid]
    fraction=1.
    for _ in range(40):
        # Signed area is quadratic, so a safe endpoint alone is insufficient.
        critical=np.zeros_like(linear)
        np.divide(-linear,2*quadratic,out=critical,where=quadratic>0)
        critical=np.clip(critical,0,fraction)
        lowest=np.minimum(constant+linear*fraction+quadratic*fraction**2,
                          constant+linear*critical+quadratic*critical**2)
        if (lowest>=minimum).all():return surface+fraction*delta
        fraction*=.5
    return surface.copy()


def _closure_surface_targets(surface,rest,faces,upper,lower,strength,source_ids):
    """Fit medial lip gaps with balanced surface constraints and safe steps."""
    from ..reconstruction.contact_runtime import closest_on_triangles
    from .geometry_fit import CrossingCounter,_closest_barycentric
    result=surface.copy();samples=[]
    lip=np.zeros(len(rest),bool);lip[upper]=True;lip[lower]=True
    center=float(np.median(rest[lip,0]));width=max(.001,float(np.ptp(rest[lip,0])))
    corners=lip&(abs(rest[:,0]-center)>=.4*width)
    for source,target in ((upper,lower),(lower,upper)):
        mask=np.zeros(len(rest),bool);mask[target]=True
        triangles=faces[mask[faces].all(1)];p=rest[triangles]
        valid=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)>1e-16
        triangles=triangles[valid];p=p[valid]
        if not len(p):raise ValueError('missing opposing lip surface for closure target')
        ids=source[abs(rest[source,0]-center)<=.3*width+1e-12]
        if not len(ids):raise ValueError('missing medial lip target samples')
        _,_,distance=closest_on_triangles(rest[ids],p[None,:,0],p[None,:,1],p[None,:,2])
        ids=ids[np.argsort(distance.min(1),kind='stable')[:min(32,len(ids))]]
        p=surface[triangles]
        _,_,distance=closest_on_triangles(surface[ids],p[None,:,0],p[None,:,1],p[None,:,2])
        desired=(1-strength)*distance.min(1)+strength*.0001
        samples.append((ids,triangles,desired))
    _,first,inverse=np.unique(source_ids,return_index=True,return_inverse=True)
    canonical=first[inverse][faces]
    with CrossingCounter(surface,canonical) as counter:
        original_crossings=counter.count(surface)
        for _ in range(24):
            change,count=np.zeros_like(result),np.zeros(len(result));remaining=[]
            for ids,triangles,desired in samples:
                p=result[triangles]
                point,_,distance=closest_on_triangles(result[ids],p[None,:,0],p[None,:,1],p[None,:,2])
                nearest=distance.argmin(1);rows=np.arange(len(ids));gap=distance[rows,nearest]
                remaining.append(float(np.median(np.maximum(0,gap-desired))))
                bary=_closest_barycentric(point[rows,nearest],p[nearest])
                involved=np.column_stack((ids,triangles[nearest]))
                coefficients=np.column_stack((np.ones(len(ids)),-bary))
                coefficients*=~corners[involved]
                normal=(result[ids]-point[rows,nearest])/np.maximum(gap[:,None],1e-20)
                step=np.minimum(.0001,np.maximum(0,gap-desired))/np.maximum((coefficients**2).sum(1),1e-20)
                correction=-step[:,None,None]*coefficients[...,None]*normal[:,None]
                np.add.at(change,involved.ravel(),correction.reshape(-1,3))
                np.add.at(count,involved.ravel(),(coefficients!=0).ravel())
            if max(remaining)<=.00001:break
            seeds=(count>0)|corners
            proposal=result.copy();moving=count>0
            proposal[moving]+=.8*change[moving]/count[moving,None]
            proposal=_feather_lip_displacements(result,proposal,faces,seeds)
            proposal=_guard_oral_area(result,proposal,faces,reference=surface)
            accepted=False
            for fraction in (1.,.5,.25,.125,.0625):
                candidate=result+fraction*(proposal-result)
                if counter.count(candidate)<=original_crossings:
                    result=candidate;accepted=True;break
            if not accepted:break
    return result


def geometric_targets(surface, arrays, model, profile, frame):
    """Subject-matched oral geometry goals; articulation wins at closure sites."""
    source=arrays['source_ids']
    v=dict(zip(CHANNELS,profile['values'][frame]))
    groups={name:model.group(name)[source] for name in ('upper_lip','lower_lip','tongue','teeth','upper_teeth_and_gums')}
    upper=np.flatnonzero(groups['upper_lip']);lower=np.flatnonzero(groups['lower_lip'])
    result=surface.copy()
    rest=arrays['surface_rest']
    columns=_lip_columns(rest,upper,lower) if len(upper) and len(lower) else None
    if columns is not None and v['seal']:
        if 'faces' in arrays:
            result=_closure_surface_targets(surface,rest,arrays['faces'],upper,lower,v['seal'],source)
        else:
            result=_column_closure_targets(surface,rest,upper,lower,columns,v['seal'])
    if v['labiodental'] and len(lower):
        teeth=np.flatnonzero(groups['teeth']&groups['upper_teeth_and_gums'])
        if len(teeth) and columns is not None:
            grid,pairs=columns
            front=teeth[rest[teeth,2]>=np.quantile(rest[teeth,2],.7)]
            dental_triangles=None
            if 'faces' in arrays:
                from .biomechanics_contacts import labiodental_samples
                from ..reconstruction.contact_runtime import closest_on_triangles
                inner,dental_triangles=labiodental_samples(rest,arrays['faces'],groups['lower_lip'],
                                                          groups['teeth']&groups['upper_teeth_and_gums'])
            shifts=[]
            for x,(_,lip) in zip(grid,pairs):
                if dental_triangles is not None:
                    lip=inner[np.argmin(abs(rest[inner,0]-x))]
                    p=surface[dental_triangles]
                    nearest,_,distance=closest_on_triangles(result[lip:lip+1],p[None,:,0],p[None,:,1],p[None,:,2])
                    point=nearest[0,distance[0].argmin()]
                    away=result[lip]-point;length=np.linalg.norm(away)
                    desired=point+away*(.00025/max(length,1e-20))
                    delta=desired[1:]-result[lip,1:]
                else:
                    local=front[abs(rest[front,0]-x)<=.002]
                    if not len(local):local=front[np.argsort(abs(rest[front,0]-x))[:1]]
                    tooth=local[np.argmin(surface[local,1])]
                    delta=surface[tooth,1:]+[-.00015,.0002]-result[lip,1:]
                delta*=min(1,.008/max(np.linalg.norm(delta),1e-20))
                shifts.append(delta)
            shifts=_smooth_columns(np.asarray(shifts))
            center=(grid[0]+grid[-1])/2;width=max(.001,(grid[-1]-grid[0])*.24)
            strength=np.exp(-np.power((rest[lower,0]-center)/width,4))*v['labiodental']
            for axis in range(2):
                result[lower,axis+1]+=strength*np.interp(rest[lower,0],grid,shifts[:,axis])
    if 'faces' in arrays and v['labiodental']:
        seeds=groups['upper_lip']|groups['lower_lip']
        result=_feather_lip_displacements(surface,result,arrays['faces'],seeds)
        result=_guard_oral_area(surface,result,arrays['faces'])
        from .geometry_fit import CrossingCounter
        _,first,inverse=np.unique(source,return_index=True,return_inverse=True)
        canonical=first[inverse][arrays['faces']]
        with CrossingCounter(surface,canonical) as counter:
            original=counter.count(surface)
            for fraction in (1.,.5,.25,.125,.0625,0.):
                candidate=surface+fraction*(result-surface)
                if counter.count(candidate)<=original:
                    result=candidate;break
    tongue=np.flatnonzero(groups['tongue'])
    if len(tongue):
        rest=arrays['surface_rest'][tongue]
        front=np.quantile(rest[:,2],.9)
        tip=np.clip((rest[:,2]-np.quantile(rest[:,2],.3))/max(.001,front-np.quantile(rest[:,2],.3)),0,1)
        dorsal=1-abs(tip-.5)*2
        result[tongue,1]+=.008*v['tip_up']*tip+.006*v['dorsum_up']*dorsal
        result[tongue,2]+=.006*v['tip_forward']*tip-.004*v['tongue_retract']*tip
    return result


def _column_closure_targets(surface,rest,upper,lower,columns,strength):
    result=surface.copy()
    grid,pairs=columns
    a,b=surface[pairs[:,0]],surface[pairs[:,1]]
    gap=a-b;distance=np.linalg.norm(gap,axis=1)
    shift=_smooth_columns(.5*gap*np.maximum(0,1-.0001/np.maximum(distance,1e-20))[:,None])*strength
    center=(grid[0]+grid[-1])/2;half=max(.0001,(grid[-1]-grid[0])/2)
    u=np.clip((abs(grid-center)/half-.6)/.3,0,1)
    shift*= (1-u*u*(3-2*u))[:,None]
    for axis in range(3):
        result[upper,axis]-=np.interp(rest[upper,0],grid,shift[:,axis])
        result[lower,axis]+=np.interp(rest[lower,0],grid,shift[:,axis])
    return result
