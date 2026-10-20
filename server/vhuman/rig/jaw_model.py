"""CPU geometry calibration of the jaw's opening-dependent condylar track.

The reference is the subject's native rig, not measured mandibular motion.
Only rigid geometry is fitted; this never applies a post-simulation warp.
"""
import numpy as np

DEFAULT_TRACK=(0.,-.003333333333333333,0.)
POLICY='vhuman.rigid_condylar_track.v1'


def points(rest,hinge,q,track=DEFAULT_TRACK):
    rest=np.asarray(rest,np.float64);hinge=np.asarray(hinge,np.float64)
    q=np.asarray(q,np.float64);track=np.asarray(track,np.float64)
    if (rest.ndim!=2 or rest.shape[1]!=3 or hinge.shape!=(3,) or track.shape!=(3,) or
            q.ndim!=2 or q.shape[1]!=3 or not all(np.isfinite(v).all() for v in (rest,hinge,q,track))):
        raise ValueError('invalid rigid jaw geometry')
    relative=rest-hinge
    cosine=np.cos(q[:,0,None]);sine=np.sin(q[:,0,None])
    result=np.stack((np.broadcast_to(relative[None,:,0],(len(q),len(rest))),
        cosine*relative[:,1]-sine*relative[:,2],sine*relative[:,1]+cosine*relative[:,2]),axis=-1)
    result+=hinge+q[:,:1,None]*track
    result[:,:,0]+=q[:,1,None];result[:,:,2]+=q[:,2,None]
    return result


def fit_track(definition,hinge,rest):
    """Fit a linear track to alternate native jaw openings and hold out poses.

    The angle is measured from each native rigid rotation, so changing the
    track cannot absorb angular or dental shape errors. The hinge stays fixed.
    """
    from .rigdef import Rig
    rest=np.asarray(rest,np.float64);hinge=np.asarray(hinge,np.float64)
    if rest.ndim!=2 or rest.shape[1]!=3 or len(rest)<4 or hinge.shape!=(3,) or not np.isfinite(rest).all() or not np.isfinite(hinge).all():
        raise ValueError('invalid dental geometry for jaw track fitting')
    rig=Rig(definition);index=rig.jidx['teeth_lower']
    openings=np.linspace(0,1,9)
    matrices=np.array([rig.evaluate({'jawOpen':float(value)})['skin'][index] for value in openings])
    rotations=matrices[:,:3,:3]
    angle=np.arctan2(rotations[:,2,1]-rotations[:,1,2],rotations[:,1,1]+rotations[:,2,2])
    training=np.arange(len(openings))%2==0;testing=~training
    if not np.isfinite(matrices).all():
        raise ValueError('nonfinite native jaw transforms')
    angle[np.abs(angle)<1e-12]=0
    if (angle<0).any() or (angle>.48).any() or np.max(angle)<.01:
        # Calibration is optional. A rig outside this reduced model's range
        # retains the legacy track rather than failing anatomy generation.
        return np.array(DEFAULT_TRACK),dict(model=POLICY,
            source='subject native rigid dental transforms; authored animation reference',
            accepted=False,reason='unsupported_native_rotation_range',hinge_fixed=True,
            baseline_track_m_rad=list(DEFAULT_TRACK),minimum_angle_rad=float(angle.min()),
            maximum_angle_rad=float(angle.max()),training_openings=openings[training].tolist(),
            withheld_openings=openings[testing].tolist())
    displacement=matrices[:,:3,3]-hinge+np.einsum('nij,j->ni',rotations,hinge)
    track=(angle[training,None]*displacement[training]).sum(0)/np.square(angle[training]).sum()
    q=np.zeros((len(openings),3));q[:,0]=angle
    target=np.einsum('nij,pj->npi',rotations,rest)+matrices[:,:3,3][:,None,:]
    baseline=points(rest,hinge,q);candidate=points(rest,hinge,q,track)
    def rms(value,mask):return float(np.sqrt(np.mean(np.square(value[mask]-target[mask])))*1000)
    before,after=rms(baseline,testing),rms(candidate,testing)
    accepted=bool(np.isfinite(track).all() and np.linalg.norm(track)<=.05 and after<=.02 and after<=before*.9)
    receipt=dict(model=POLICY,source='subject native rigid dental transforms; authored animation reference',
        accepted=accepted,training_openings=openings[training].tolist(),withheld_openings=openings[testing].tolist(),
        candidate_track_m_rad=track.tolist(),baseline_track_m_rad=list(DEFAULT_TRACK),
        training_rms_mm=rms(candidate,training),withheld_before_mm=before,withheld_after_mm=after,
        maximum_angle_rad=float(angle.max()),hinge_fixed=True)
    return track if accepted else np.array(DEFAULT_TRACK),receipt
