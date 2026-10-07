"""Geometry conditioning and surface projection for multiview texture generators.

Renders the fitted GNM head with MV-Adapter's six orthographic cameras
(front, right, back, left, top, bottom) so any multiview backend receives
position/normal maps, the partial photographed texture and a known-texel mask
in one shared frame. Generated views are projected back onto atlas texels with
the same cameras, so fusion never depends on the generator's own camera model.

Orthography is emulated by a distant perspective camera through the reference
rasterizer; at 200 extents the depth-induced scale error is below 0.3%.
"""
from dataclasses import dataclass
import numpy as np

from .reference import Camera, rasterize, srgb_to_linear, linear_to_srgb
from ..rig.common import vertex_normals, normalize
from ..mobile.baking import sample

# MV-Adapter ig2mv defaults: azimuth x-90 for x in (0,90,180,270,180,180).
VIEWS=(('front',0.,-90.),('right',0.,0.),('back',0.,90.),('left',0.,180.),
       ('top',89.99,90.),('bottom',-89.99,90.))
HALF=.55       # orthographic half-extent of MV-Adapter cameras
DISTANCE=200.  # far camera emulating orthography (scene units after rescale)


@dataclass
class Frame:
    """GNM (+Y up, +Z front) -> MV-Adapter std frame (+Z up, front at -Y), rescaled."""
    centre: np.ndarray
    scale: float

    @classmethod
    def fit(cls, points):
        centre=(points.min(0)+points.max(0))/2
        return cls(centre,.5/float(np.abs(points-centre).max()))

    def __call__(self, points):
        p=(np.asarray(points,float)-self.centre)*self.scale
        return np.stack((p[...,0],-p[...,2],p[...,1]),-1)

    def direction(self, vectors):
        v=np.asarray(vectors,float)
        return np.stack((v[...,0],-v[...,2],v[...,1]),-1)


def ortho_camera(elevation, azimuth, resolution):
    e,a=np.deg2rad(elevation),np.deg2rad(azimuth)
    position=DISTANCE*np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
    look=-position/np.linalg.norm(position)
    right=normalize(np.cross(look,[0.,0.,1.]));up=normalize(np.cross(right,look))
    focal=resolution/(2*HALF)*DISTANCE
    return Camera(focal,resolution/2,resolution/2,position,np.stack((right,up,-look)))


def render_conditions(geometry, atlas, known, resolution=768, only=None):
    """Return per-view dict of pos/normal maps (std-frame, [0,1]), RGB, known mask, depth.

    atlas: linear RGB UV texture; known: UV weight in [0,1] of photographed support.
    """
    points=geometry['captured'][0].astype(float);tri=geometry['triangles']
    frame=Frame.fit(points);p=frame(points)
    n=frame.direction(vertex_normals(points,tri))
    views=[]
    for name,elevation,azimuth in VIEWS:
        if only is not None and name not in only:continue
        camera=ortho_camera(elevation,azimuth,resolution)
        ids,bary,depth=rasterize(p,tri,camera,(resolution,resolution));valid=ids>=0
        faces=tri[ids[valid]];w=bary[valid].astype(float)
        pos=np.full((resolution,resolution,3),.5);nrm=np.full((resolution,resolution,3),.5)
        pos[valid]=np.clip((p[faces]*w[...,None]).sum(1)+.5,0,1)
        nrm[valid]=np.clip(normalize((n[faces]*w[...,None]).sum(1))/2+.5,0,1)
        uv=(geometry['triangle_uvs'][ids[valid]]*w[...,None]).sum(1)
        rgb=np.full((resolution,resolution,3),.5**2.2);rgb[valid]=sample(atlas,uv)
        mask=np.zeros((resolution,resolution));mask[valid]=sample(np.asarray(known,float),uv)
        views.append(dict(name=name,elevation=elevation,azimuth=azimuth,camera=camera,valid=valid,
            position=pos,normal=nrm,rgb=np.uint8(np.clip(linear_to_srgb(rgb)*255+.5,0,255)),
            known=mask,depth=depth))
    return frame,views


def project_views(frame, views, images, points, normals, *, min_facing=.25, depth_tolerance=.004):
    """Sample generated images at atlas surface points.

    Returns colors [V,N,3] (linear) and weights [V,N] = visibility * facing^2.
    depth_tolerance is in rescaled scene units (head half-extent = 0.5).
    """
    p=frame(points);n=normalize(frame.direction(normals))
    colors=[];weights=[]
    for view,image in zip(views,images):
        camera=view['camera'];res=view['depth'].shape[0]
        xy,z=camera.project(p);uv=xy/res
        ix=np.clip(xy[:,0].astype(int),0,res-1);iy=np.clip(xy[:,1].astype(int),0,res-1)
        facing=np.maximum((n*camera.rotation[2]).sum(1),0)
        seen=(uv.min(1)>=0)&(uv.max(1)<1)&(abs(view['depth'][iy,ix]-z)<depth_tolerance)&(facing>min_facing)
        # Stay off the silhouette, where generators blend into the background.
        from scipy.ndimage import binary_erosion
        interior=binary_erosion(view['valid'],iterations=2)
        seen&=interior[iy,ix]
        rgb=srgb_to_linear(np.asarray(image,float)/255)
        colors.append(sample(rgb,uv));weights.append(seen*facing**2)
    return np.stack(colors),np.stack(weights)
