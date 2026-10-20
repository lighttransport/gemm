"""Authored skull-normal support and tangential tethers for facial skin."""
import math
import numpy as np
from .common import vertex_normals


def orbital_supports(arrays,meta,scale):
    """Keep canthal/back-orbit support; release inferred anterior lid pins.

    A thin eyelid's inner surface is not a skull attachment. The globe contact
    and connected tissue support this authored anterior band instead. Bounds
    are geometric priors, not a measured orbital or ligament segmentation.
    """
    if not math.isfinite(scale) or scale<=0:raise ValueError('invalid orbital support scale')
    anchors=arrays['anchor_ids'];points=arrays['rest'][anchors]
    skin=np.zeros(len(arrays['rest']),bool)
    skin[np.unique(arrays['tets'][arrays['regions']==0])]=True
    eligible=skin[anchors]&(arrays['anchor_bone']==0)
    keep=np.ones(len(anchors),bool);eyes={}
    for part in meta['parts']:
        if part['name'] not in ('eye_left','eye_right'):continue
        p=arrays['surface_rest'][part['surface_start']:part['surface_start']+part['surface_count']]
        low,high=p.min(0),p.max(0);center=(low+high)/2;radius=(high-low)/2
        if (radius<1e-6).any():raise ValueError('collapsed globe for orbital support fitting')
        release=(eligible&(abs(points[:,0]-center[0])<.75*radius[0])
                 &(abs(points[:,1]-center[1])<1.1*radius[1])
                 &(points[:,2]>center[2])&(points[:,2]<high[2]+.003*scale))
        eyes[part['name']]=dict(released_anchors=int(release.sum()),globe_center=center.tolist(),
                               globe_radii=radius.tolist())
        keep[release]=False
    return keep,dict(model='vhuman.orbital_support_release.v1',original_anchors=len(anchors),
        retained_anchors=int(keep.sum()),released_anchors=int((~keep).sum()),eyes=eyes,
        released_vertices=anchors[~keep].tolist(),canthal_width_fraction=.75,vertical_radius_factor=1.1,
        anterior_depth_margin_mm=3*scale,
        anatomy_source='authored anterior eyelid support prior; posterior orbit and canthi retained')

def frames(arrays,meta,exterior,scale,*,tangent_density=10000.):
    """Return static normal frames for frontal, skull-supported skin anchors.

    The upper scalp and lateral boundaries retain isotropic attachments.
    Tangential spring stiffness scales with the lumped nodal area, keeping
    the authored tether density independent of local tetrahedral refinement.
    Mandibular, oral-lining, tongue and gingival attachments remain isotropic.
    This is an animation prior, not measured connective-tissue stiffness.
    """
    from scipy.spatial import cKDTree
    if not math.isfinite(scale) or scale<=0 or not math.isfinite(tangent_density) or tangent_density<=0:
        raise ValueError('invalid skin attachment scale/stiffness')
    exterior=np.asarray(exterior,bool)
    if exterior.shape!=(len(arrays['surface_rest']),):
        raise ValueError('invalid exterior skin attachment mask')
    skin=next(p for p in meta['parts'] if p['name']=='skin')
    start,count=skin['surface_start'],skin['surface_count']
    rest=arrays['surface_rest'][start:start+count]
    triangles=arrays['faces'][skin['face_start']:skin['face_start']+skin['face_count']]-start
    normal=vertex_normals(rest,triangles)
    ids=arrays['anchor_ids'];position=arrays['rest'][ids]
    _,nearest=cKDTree(rest).query(position)
    # Require a skin-volume anchor rather than a neighboring oral domain.
    skin_nodes=np.unique(arrays['tets'][arrays['regions']==0])
    allowed=(np.isin(ids,skin_nodes)&(arrays['anchor_bone']==0)&exterior[start+nearest]
             &(arrays['inverse_mass'][ids]>0)&(np.linalg.norm(normal[nearest],axis=1)>.5))
    mouth=np.asarray(meta['landmarks']['mouth']);eye_y=float(meta['landmarks']['eye_y'])
    def smooth(x):
        x=np.clip(x,0,1)
        return x*x*(3-2*x)
    weight=1-smooth((abs(position[:,0]-mouth[0])-.035*scale)/(.020*scale))
    weight*=1-smooth((position[:,1]-eye_y-.025*scale)/(.025*scale))
    weight*=smooth((position[:,1]-mouth[1]+.008*scale)/(.015*scale))
    weight*=smooth((position[:,2]-mouth[2]+.045*scale)/(.015*scale))
    allowed &= weight>1e-4
    result=np.zeros((len(ids),3),np.float64)
    result[allowed]=normal[nearest[allowed]]
    tangent=np.zeros(len(ids),np.float64)
    mass=1/arrays['inverse_mass'][ids[allowed]]
    area=mass/(1000*.003*scale)
    tangent[allowed]=np.minimum(1e4,weight[allowed]/(tangent_density*area))
    return result,tangent,dict(model='vhuman.skull_normal_tethers.v1',sliding_anchors=int(allowed.sum()),
        tangent_stiffness_density_n_m3=tangent_density,normal_compliance_m_n=1e-7,
        scope='frontal exterior skin on skull; upper scalp/lateral boundaries retain isotropic support',
        material_calibration='authored animation prior')
