"""Ear appearance synthesis/completion for refined surface-bound ears.

Sources, in priority order: (1) the single source portrait, sampled only
through an eroded ear-core mask with visibility and facing weights and a
partial illumination correction; (2) optional generated side views (an
appearance prior), contributing only high-pass log-luminance detail and
bounded chroma; (3) exemplar residual detail from photographed facial skin and
a capped anatomical tint prior. Low-frequency tone of unobserved ear texels is
completed harmonically from photographed ear skin. Every texel outside the
declared edit region stays byte-identical. Nothing here is measured
reflectance or newly observed anatomy, except samples labelled photographed.
"""
import numpy as np
from scipy.spatial import cKDTree

from ..rig.bake import rasterize_uv
from ..rig.common import vertex_normals, normalize
from .reference import srgb_to_linear, linear_to_srgb, rasterize


def atlas_surface(points, triangles, triangle_uvs, resolution, normals=None):
    """Rasterize per-corner UV triangles into the atlas.

    Returns (triangle id map (R,R), texel mask, texel 3D points, unit normals).
    """
    points, triangles = np.asarray(points, float), np.asarray(triangles)
    uv = np.asarray(triangle_uvs, float)
    ids, bary = rasterize_uv(uv.reshape(-1, 2), np.arange(uv.size//2).reshape(-1, 3), resolution)
    valid = ids >= 0
    faces = triangles[ids[valid]]
    weights = bary[valid].astype(float)
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
    if normals is None:
        normals = vertex_normals(points, triangles)
    texel_points = (points[faces]*weights[..., None]).sum(1)
    texel_normals = normalize((normals[faces]*weights[..., None]).sum(1))
    return ids, valid, texel_points, texel_normals, bary[valid].astype(float)


def surface_lowpass(points, values, radius, *, normals=None, weights=None, normal_dot=.5, k=48):
    """Gaussian surface-neighbourhood average (k nearest, normal-compatible)."""
    points, values = np.asarray(points, float), np.asarray(values, float)
    w0 = np.ones(len(points)) if weights is None else np.asarray(weights, float)
    tree = cKDTree(points)
    d, j = tree.query(points, k=min(k, len(points)))
    g = np.exp(-(d/radius)**2)*w0[j]
    if normals is not None:
        g *= ((normals[:, None]*normals[j]).sum(-1) > normal_dot)
    g /= np.maximum(g.sum(1, keepdims=True), 1e-12)
    return np.einsum('nk,nk...->n...', g, values[j])


def portrait_samples(image, camera, points, normals, scene_points, scene_triangles, allowed_mask,
                     *, depth_tolerance=.002):
    """Visibility/facing-weighted linear RGB samples of atlas points in the portrait.

    ``allowed_mask`` (H,W bool) marks source pixels that may contribute.
    Returns (linear RGB (N,3), confidence (N,)).
    """
    from .materials import sample_portrait
    h, w = allowed_mask.shape
    pixels, z = camera.project(points)
    tid, _, depth = rasterize(scene_points, scene_triangles, camera, (w, h))
    ix = np.clip(np.floor(pixels[:, 0]).astype(int), 0, w-1)
    iy = np.clip(np.floor(pixels[:, 1]).astype(int), 0, h-1)
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    visible = inside & (np.abs(depth[iy, ix]-z) < depth_tolerance)
    view = camera.origin-points
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
    facing = np.maximum((normals*view).sum(1), 0)
    rgba = np.dstack((image[..., :3], np.full((h, w), 255, np.uint8)))
    exclusion = np.where(allowed_mask, 0, 255).astype(np.uint8)
    sampled, support = sample_portrait(rgba, pixels, exclusion)
    confidence = visible*facing*support
    return srgb_to_linear(sampled/255.), confidence


def half_illumination(normals, lighting):
    """Square root of the estimator's clipped low-order irradiance (partial correction)."""
    illum = np.exp(np.clip(normals@np.asarray(lighting['log_direction'], float), -.5, .5))
    illum = np.clip(illum/lighting['irradiance_median'], .5, 2.)
    return np.sqrt(illum)


def exemplar_residual(target_points, target_normals, source_points, source_log, *, radius=.0012, seed=0,
                      patch=.006, clip=2.):
    """Transfer high-pass log-RGB residual from photographed source skin.

    Target texels are grouped into ~``patch``-sized surface cells; each cell
    copies the residual of a randomly chosen source cell (rigidly offset), so
    detail statistics follow real skin without a repeating tile.
    """
    rng = np.random.default_rng(seed)
    source_points = np.asarray(source_points, float)
    low = surface_lowpass(source_points, source_log, radius)
    residual = source_log-low
    residual -= np.median(residual, 0)
    # Pore-scale only: clip creases/wrinkle lines (heavy tails) at 2 robust sigma.
    sigma = 1.4826*np.median(np.abs(residual), 0)
    residual = np.clip(residual, -clip*sigma, clip*sigma)
    tree = cKDTree(source_points)
    cells = np.floor(np.asarray(target_points)/patch).astype(np.int64)
    _, cell_id = np.unique(cells, axis=0, return_inverse=True)
    out = np.zeros((len(target_points), 3))
    anchors = source_points[rng.integers(0, len(source_points), cell_id.max()+1)]
    centres = np.zeros((cell_id.max()+1, 3))
    np.add.at(centres, cell_id, target_points)
    centres /= np.maximum(np.bincount(cell_id)[:, None], 1)
    query = anchors[cell_id]+(np.asarray(target_points)-centres[cell_id])
    d, j = tree.query(query)
    out = residual[j]*(d < patch)[:, None]
    # Blend cell borders by a light surface low-pass of the transferred field.
    smooth = surface_lowpass(np.asarray(target_points), out, patch*.18, normals=target_normals, k=16)
    return out*.6+smooth*.4


def anatomical_tint(xy, outline, params=None):
    """Capped log-RGB tint prior: slight vascular flush at helix rim and lobule.

    Returns (N,3) log offsets; a generic prior, not a measured pigmentation map.
    """
    p = dict(rim=.06, lobule=.05, width=.04, red=(1., .35, .25))
    p.update(params or {})
    red = np.asarray(p['red'], float)
    rim = np.exp(-(np.asarray(outline)/p['width'])**2)*(np.asarray(xy)[:, 1] > .2)
    lobe = np.clip((.22-np.asarray(xy)[:, 1])/.12, 0, 1)
    return (p['rim']*rim+p['lobule']*lobe)[:, None]*red[None]


def to_srgb8(linear):
    return np.uint8(np.clip(linear_to_srgb(np.clip(linear, 0, 1))*255+.5, 0, 255))
