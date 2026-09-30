"""Original triangle-bound static RGB radiance prototype (not relightable PBR).

Covariance is pushed forward by the triangle's two edges and unit normal.
The caller must supply the FINAL posed mesh, including contacts/residuals.
"""
import json
import numpy as np
from .fitting import topology_hash
from .reference import srgb_to_linear

PRESETS = {'desktop': 20000, 'mobile': 8000, 'low': 2000}


def frames(vertices, triangles):
    p = np.asarray(vertices)[triangles]
    a, b = p[:,1]-p[:,0], p[:,2]-p[:,0]
    n = np.cross(a,b)
    length = np.linalg.norm(n,axis=1)
    n /= np.maximum(length[:,None],1e-12)
    return np.stack((a,b,n),-1), length > 1e-10


def bind(vertices, triangles, *, count=2000, colors=None, triangle_mask=None, seed=7):
    if not 1 <= count <= 20000:
        raise ValueError('Gaussian count must be 1..20000')
    f, valid = frames(vertices, triangles)
    if triangle_mask is not None:
        valid &= triangle_mask
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise ValueError('no valid Gaussian attachment triangles')
    area = np.linalg.norm(np.cross(f[ids,:,0],f[ids,:,1]),axis=1)
    rng = np.random.default_rng(seed)
    selected = rng.choice(ids,count,p=area/area.sum())
    r = rng.random((count,2))
    root = np.sqrt(r[:,0])
    bary = np.stack((1-root,root*(1-r[:,1]),root*r[:,1]),-1)
    # Basis-coordinate variances; normal thickness 0.15 mm in metric H frame.
    cov = np.tile(np.diag([.035**2,.035**2,.00015**2]),(count,1,1))
    rgb = np.full((count,3),.4,np.float32) if colors is None else np.asarray(colors,np.float32)
    if rgb.shape != (count,3) or not np.isfinite(rgb).all():
        raise ValueError('RGB radiance shape mismatch')
    return dict(format='vhuman.gaussian_binding.v1',topology_sha256=topology_hash(triangles),
                units='metres',radiance='static linear RGB; not relightable', triangle=selected,
                barycentric=bary.astype(np.float32),normal_offset=np.zeros(count,np.float32),
                covariance_local=cov.astype(np.float32),opacity=np.full(count,.25,np.float32),rgb=rgb)


def deform(binding, vertices, triangles):
    if binding['topology_sha256'] != topology_hash(triangles):
        raise ValueError('Gaussian topology mismatch')
    f, valid = frames(vertices,triangles)
    ids = np.asarray(binding['triangle'],int)
    if (ids<0).any() or (ids>=len(triangles)).any():
        raise ValueError('invalid Gaussian triangle')
    basis = f[ids]
    center = (vertices[triangles[ids]]*binding['barycentric'][...,None]).sum(1)
    center += basis[:,:,2]*binding['normal_offset'][:,None]
    cov = basis @ binding['covariance_local'] @ basis.transpose(0,2,1)
    eig, rot = np.linalg.eigh(cov)
    eig = np.clip(eig,1e-10,.01**2)
    cov = (rot*eig[:,None,:]) @ rot.transpose(0,2,1)
    return center,cov, np.asarray(binding['opacity'])*valid[ids]


def save(binding, path):
    doc = {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in binding.items()}
    path.write_text(json.dumps(doc,separators=(',',':')))


def load(path):
    doc = json.loads(path.read_text())
    if doc.get('format') != 'vhuman.gaussian_binding.v1':
        raise ValueError('unsupported Gaussian binding')
    n = len(doc.get('triangle',[]))
    if not 1<=n<=20000:
        raise ValueError('invalid Gaussian count')
    shapes = {'triangle':(n,), 'barycentric':(n,3), 'normal_offset':(n,),
              'covariance_local':(n,3,3),'opacity':(n,), 'rgb':(n,3)}
    for key, shape in shapes.items():
        doc[key] = np.asarray(doc[key],int if key=='triangle' else float)
        if doc[key].shape != shape or not np.isfinite(doc[key]).all():
            raise ValueError('invalid Gaussian array '+key)
    if (doc['barycentric']<0).any() or not np.allclose(doc['barycentric'].sum(1),1,atol=1e-5):
        raise ValueError('invalid barycentrics')
    c = doc['covariance_local']
    if not np.allclose(c,c.transpose(0,2,1)) or (np.linalg.eigvalsh(c)<=0).any():
        raise ValueError('covariance must be symmetric positive definite')
    if (doc['rgb']<0).any():
        raise ValueError('radiance must be nonnegative')
    if (abs(doc['normal_offset'])>.005).any() or (doc['opacity']<0).any() or (doc['opacity']>1).any():
        raise ValueError('invalid opacity/normal offset')
    return doc


def fit_radiance(binding, surfaces, triangles, views, cameras):
    """Weighted static RGB least-squares fit from visible permitted views.

This prototype estimates radiance only. Covariance/offset remain geometric
priors, not one-shot learned inference. Unobserved splats are transparent.
"""
    from PIL import Image
    from ..rig import bake
    from ..rig.common import normalize
    from .reference import rasterize
    count = len(binding['triangle'])
    color = np.zeros((count,3))
    weight = np.zeros(count)
    for vertices,view,cam in zip(surfaces,views,cameras):
        positions,_,_ = deform(binding,vertices,triangles)
        pixels,z = cam.project(positions)
        image = np.asarray(Image.open(view['image_path']).convert('RGBA'))
        h,w = image.shape[:2]
        _,_,metric = rasterize(vertices,triangles,cam,(w,h))
        ix,iy = np.floor(pixels[:,0]).astype(int),np.floor(pixels[:,1]).astype(int)
        valid = (ix>=0)&(ix<w)&(iy>=0)&(iy<h)&(z>0)
        ix,iy = np.clip(ix,0,w-1),np.clip(iy,0,h-1)
        f,_ = frames(vertices,triangles)
        n = f[binding['triangle'],:,2]
        confidence = valid*(abs(metric[iy,ix]-z)<.002)*np.maximum((n*normalize(cam.origin-positions)).sum(1),0)
        confidence *= image[iy,ix,3]/255.
        if view.get('exclusion_mask_path'):
            mask = np.asarray(Image.open(view['exclusion_mask_path']).convert('L'))
            confidence *= 1-mask[iy,ix]/255.
        rgb = bake._sample(srgb_to_linear(image[:,:,:3]/255.),pixels/np.array([w,h]))
        color += confidence[:,None]*rgb
        weight += confidence
    binding['rgb'] = (color/np.maximum(weight[:,None],1e-9)).astype(np.float32)
    binding['opacity'] = (np.clip(weight,0,1)*.35).astype(np.float32)
    binding['fit'] = dict(method='weighted visible RGB least squares',views=len(views),observed=int((weight>.1).sum()),
                          geometry='triangle attachment and covariance priors, not learned',unobserved='opacity zero')
    return binding
