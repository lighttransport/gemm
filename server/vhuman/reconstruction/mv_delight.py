"""Remove generator-baked lighting from geometry-conditioned multiview views.

Generated views carry directional shading, scalp sheen and dark blobs that must
not become albedo. Per view: (1) fit second-order spherical-harmonic shading of
log-luminance on the GNM normals (robust IRLS) and divide it out; (2) flatten
remaining broad luminance blobs with a masked low-pass, keeping chroma and
fine detail. Both steps only rescale luminance, so hue is the generator's, unless
chroma=True (used for hybrid structure views) also flattens broad hue blotches.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from .reference import srgb_to_linear, linear_to_srgb

LUMA=np.array([.2126,.7152,.0722])


def sh9(n):
    x,y,z=n[...,0],n[...,1],n[...,2]
    return np.stack((np.ones_like(x),x,y,z,x*y,x*z,y*z,x*x-y*y,3*z*z-1),-1)


def masked_blur(values, mask, sigma):
    m=mask.astype(float)
    return gaussian_filter(values*m,sigma)/np.maximum(gaussian_filter(m,sigma),1e-6)


def delight(image, normal_map, valid, *, blob_sigma=None, blob_strength=1., chroma=False):
    """image uint8 sRGB view; normal_map [0,1] world normals; valid bool mask.

    chroma=True additionally flattens each channel's broad log variation toward the channel median, removing
    low-frequency hue blotches (e.g. MV-Adapter's pink back of head); fine detail and the mean colour remain.
    """
    rgb=srgb_to_linear(np.asarray(image,float)/255)
    interior=valid&(rgb.mean(-1)>.01)
    luma=np.maximum(rgb@LUMA,1e-4);log=np.log(luma)
    n=np.asarray(normal_map,float)*2-1;n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-6)
    A=sh9(n[interior]);b=log[interior];w=np.ones(len(b))
    for _ in range(4):
        # IRLS (Huber-like) so dark eyebrows/stubble do not bend the shading fit.
        coef=np.linalg.lstsq(A*w[:,None],b*w,rcond=None)[0]
        r=b-A@coef;s=1.4826*np.median(abs(r))+1e-6;w=np.minimum(1,1.5*s/np.maximum(abs(r),1e-9))
    shading=sh9(n)@coef;shading-=np.median(shading[interior])
    log2=log-shading
    sigma=blob_sigma or rgb.shape[0]/24
    low=masked_blur(log2,interior,sigma)
    log2-=blob_strength*(low-np.median(low[interior]))
    scale=np.where(valid,np.exp(log2-log),1.)[...,None]
    out=np.clip(rgb*scale,0,1)
    if chroma:
        # per-channel broad variation relative to luminance (hue blotches), flattened to the median hue
        lo=np.log(np.maximum(out,1e-4))-np.log(np.maximum(out@LUMA,1e-4))[...,None]
        for c in range(3):
            band=masked_blur(lo[...,c],interior,sigma)
            out[...,c]*=np.where(valid,np.exp(-(band-np.median(band[interior]))),1.)
        out=np.clip(out,0,1)
    return np.uint8(np.clip(linear_to_srgb(out)*255+.5,0,255)),dict(sh=coef.tolist(),
        shading_range=float(np.ptp(shading[interior])),blob_range=float(np.ptp(low[interior])))
