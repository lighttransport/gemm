"""Fixed anatomical MediaPipe-to-GNM attachments, never fitted to a photograph.

Official GNM68 barycentric landmarks seed a smooth canonical registration.
Additional attachments are inferred correspondences, not authored GNM labels.
"""
import hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np

DATA = Path(__file__).parent/'data'
MP68 = np.array([234,93,132,58,172,136,150,149,152,378,379,365,397,288,361,323,454,
    70,63,105,66,107,336,296,334,293,300,168,6,197,4,98,97,2,326,327,
    33,160,158,133,153,144,362,385,387,263,373,380,
    61,40,37,0,267,270,291,321,314,17,84,91,78,81,13,311,308,402,14,178])


def checked(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('canonical anatomical asset hash mismatch')


@lru_cache(maxsize=1)
def attachments():
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import cKDTree
    from ..rig.face_models import load
    from ..rig.gnm_model import GNMModel
    checked(DATA/'gnm_landmarks68.txt','d8b6066a87ca37c48bcf4d0834542db841709b65cb983e873fa1e441a22219d0')
    checked(DATA/'mediapipe_canonical.txt','8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618')
    canonical = np.array([[float(v) for v in line.split()[1:4]] for line in
        (DATA/'mediapipe_canonical.txt').read_text().splitlines() if line.startswith('v ')])*.01
    model, source = GNMModel(), load('gnm_v3')
    rows = np.loadtxt(DATA/'gnm_landmarks68.txt')
    official_ids, weights = rows[:,::2].astype(int), rows[:,1::2]
    weights = weights/weights.sum(1,keepdims=True)
    target = (model.data['template_vertex_positions'][official_ids]*weights[:,:,None]).sum(1)
    a,b = canonical[MP68],target
    ac,bc = a.mean(0),b.mean(0)
    u,s,vt = np.linalg.svd((a-ac).T@(b-bc))
    correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
    rotation=u@correction@vt
    scale=(s*np.diag(correction)).sum()/np.square(a-ac).sum()
    registered=(canonical-ac)@rotation*scale+bc
    deformation=RBFInterpolator(registered[MP68],target-registered[MP68],
                                kernel='thin_plate_spline',smoothing=1e-7)
    query=registered+deformation(registered)
    # Only the face-facing skin can receive canonical face points.
    tri=source.triangles
    centers=source.vertices[tri].mean(1)
    eligible=np.flatnonzero((centers[:,2]>target[:,2].min()-.01)&
        (centers[:,1]>target[:,1].min()-.015)&(centers[:,1]<target[:,1].max()+.04))
    _,nearest=cKDTree(centers[eligible]).query(query,k=8)
    ids=tri[eligible[nearest]]
    points=source.vertices[ids]
    first=points[:,:,0]; e=points[:,:,1]-first; f=points[:,:,2]-first; delta=query[:,None]-first
    ee=(e*e).sum(-1);ff=(f*f).sum(-1);ef=(e*f).sum(-1)
    de=(delta*e).sum(-1);df=(delta*f).sum(-1)
    determinant=np.maximum(ee*ff-ef*ef,1e-20)
    b1=(de*ff-df*ef)/determinant;b2=(df*ee-de*ef)/determinant
    candidates=[np.stack((1-b1-b2,b1,b2),-1)]
    for i,j in ((0,1),(1,2),(2,0)):
        edge=points[:,:,j]-points[:,:,i]
        t=np.clip(((query[:,None]-points[:,:,i])*edge).sum(-1)/np.maximum((edge*edge).sum(-1),1e-20),0,1)
        bary=np.zeros((*t.shape,3));bary[:,:,i]=1-t;bary[:,:,j]=t;candidates.append(bary)
    bary=np.stack(candidates,2)
    projected=(points[:,:,None]*bary[:,:,:,:,None]).sum(-2)
    distance=np.square(projected-query[:,None,None]).sum(-1)
    distance[:,:,0]=np.where((bary[:,:,0]>=0).all(-1),distance[:,:,0],np.inf)
    best=distance.reshape(len(query),-1).argmin(1);tidx,kind=best//4,best%4
    row=np.arange(len(query)); selected_ids=ids[row,tidx];selected_weights=bary[row,tidx,kind]
    confidence=np.exp(-distance[row,tidx,kind]/.005**2)
    remap=np.full(len(model.data['template_vertex_positions']),-1,int)
    remap[np.flatnonzero(model.group('skin_exterior'))]=np.arange(len(source.vertices))
    for n,mp in enumerate(MP68):
        selected_ids[mp]=remap[official_ids[n]];selected_weights[mp]=weights[n];confidence[mp]=1
    if (selected_ids<0).any():
        raise ValueError('official GNM landmark outside exterior skin')
    return selected_ids,selected_weights,confidence


def observe_points(points):
    ids,weights,confidence=attachments()
    return {f'mp_{i:03d}':dict(xy=np.asarray(points[i]).tolist(),vertices=ids[i].tolist(),
                barycentric=weights[i].tolist(),weight=float(.8*confidence[i])) for i in range(len(ids))}
