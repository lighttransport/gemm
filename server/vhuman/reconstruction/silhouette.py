"""Original bounded, concave outline distance-field objective.

Contour barycentric attachments are fixed at initial pose. Excluded occlusion
regions carry no outline evidence; masks are supplied, not inferred segmentation.
"""
import numpy as np
from PIL import Image
from .reference import Camera, rasterize


def prepare(vertices, triangles, camera, view):
    if not view.get('silhouette_mask_path'):
        return None
    from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt
    w,h = view['size']
    scale = min(1.,512/max(w,h))
    size = (max(1,round(w*scale)),max(1,round(h*scale)))
    with Image.open(view['silhouette_mask_path']) as image:
        mask = np.asarray(image.convert('L').resize(size,Image.Resampling.NEAREST))>127
    mask = binary_fill_holes(mask)
    if mask.sum()<16 or mask.all():
        raise ValueError('silhouette mask needs foreground and background')
    allowed = np.ones(mask.shape,bool)
    if view.get('exclusion_mask_path'):
        with Image.open(view['exclusion_mask_path']) as image:
            allowed = np.asarray(image.convert('L').resize(size,Image.Resampling.NEAREST))<128
    small = camera.scaled(scale)
    tid,bary,_ = rasterize(vertices,triangles,small,size)
    foreground = binary_fill_holes(tid>=0)
    boundary = foreground & ~binary_erosion(foreground) & (tid>=0) & allowed
    yy,xx = np.nonzero(boundary)
    selected = np.linspace(0,len(xx)-1,min(128,len(xx))).astype(int)
    yy,xx = yy[selected],xx[selected]
    target_y,target_x = np.nonzero(mask & ~binary_erosion(mask) & allowed)
    chosen = np.linspace(0,len(target_x)-1,min(128,len(target_x))).astype(int)
    if len(xx)<8 or len(chosen)<8:
        raise ValueError('silhouette mask has insufficient unoccluded outline')
    return dict(triangles=triangles[tid[yy,xx]],barycentric=bary[yy,xx],
                sdf=(distance_transform_edt(~mask)-distance_transform_edt(mask))/scale,
                allowed=allowed,scale=scale,
                target=np.column_stack((target_x[chosen]+.5,target_y[chosen]+.5))/scale)


def residual(vertices, camera, prepared):
    from scipy.ndimage import map_coordinates
    p = (vertices[prepared['triangles']]*prepared['barycentric'][...,None]).sum(1)
    xy,z = camera.project(p)
    uv = xy*prepared['scale']-.5
    h,w = prepared['sdf'].shape
    clipped = np.clip(uv,[0,0],[w-1,h-1])
    distance = map_coordinates(prepared['sdf'],[clipped[:,1],clipped[:,0]],order=1)
    distance += np.linalg.norm(uv-clipped,axis=1)/prepared['scale']
    allowed = map_coordinates(prepared['allowed'].astype(float),[clipped[:,1],clipped[:,0]],order=0)
    # Reverse distances prevent fitting only a small subset of the outline.
    target = prepared['target']
    distances = np.linalg.norm(target[:,None]-xy[None],axis=2)
    distances[:,allowed<.5] = 1e4
    reverse = distances.min(1)
    return np.concatenate((distance*allowed,reverse,np.minimum(z-.02,0)*1000))*.04
