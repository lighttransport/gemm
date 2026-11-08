"""Metric distance fields for continuous eyelid surface shading (NumPy only)."""
import numpy as np


def distance_to_segments(points, path):
    points = np.asarray(points, dtype=np.float64)
    path = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError('expected Nx3 points and a path with at least two points')
    if not np.isfinite(points).all() or not np.isfinite(path).all():
        raise ValueError('nonfinite contact geometry')
    distance = np.full(len(points), np.inf)
    for a, b in zip(path[:-1], path[1:]):
        edge = b - a
        t = np.clip(np.einsum("ij,j->i", points - a, edge) / max(float(edge @ edge), 1e-24), 0., 1.)
        distance = np.minimum(distance, np.linalg.norm(points - a - t[:, None] * edge, axis=1))
    return distance.astype(np.float32)


def build_contact_atlas(base, res=2048):
    """Bake metric lid distances at texel centres, independent of mesh tessellation.

    UVs use the staged skin convention (image rows run from v=0 to v=1).
    The Cycles skin mesh flips v, so float_image's usual top-row convention
    samples this atlas just like skin_detail and skin_templates.
    """
    import hashlib
    import json
    from pathlib import Path
    from scipy.ndimage import distance_transform_edt
    from ..rig.bake import rasterize_uv
    base = Path(base)
    asset = base / 'scene_assets.npz'
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    stamp = dict(version=1, assets_sha256=digest, resolution=int(res))
    dest = base / 'peri_contact_atlas.npz'
    meta = dest.with_suffix('.json')
    if dest.is_file() and meta.is_file() and json.loads(meta.read_text()) == stamp:
        return dest
    with np.load(asset, allow_pickle=False) as data:
        positions = data['skin_positions']
        triangles = data['skin_triangles']
        uvs = data['skin_uvs']
        paths = {}
        for margin, suffix in (('lower', 'tearline'), ('upper', 'tearline_upper')):
            paths[margin] = [data[f'{side}_{suffix}_positions'].reshape(-1, 8, 3).mean(1)
                             for side in ('left', 'right') if f'{side}_{suffix}_positions' in data]
        if not all(paths.values()):
            raise ValueError('contact atlas requires staged upper and lower lid paths')
    # Only orbital triangles can enter the distance band. Keep a generous 20 mm
    # surround; all other texels have an explicitly distant value.
    vertex_distance = np.minimum.reduce([distance_to_segments(positions, p)
                                         for group in paths.values() for p in group])
    selected = np.flatnonzero(vertex_distance[triangles].min(1) < .02)
    selected_uv = uvs[selected]
    tid, bary = rasterize_uv(selected_uv.reshape(-1, 2), np.arange(len(selected) * 3).reshape(-1, 3), res)
    yy, xx = np.nonzero(tid >= 0)
    points = (positions[triangles[selected[tid[yy, xx]]]] * bary[yy, xx, :, None]).sum(1)
    maps = {}
    valid = tid >= 0
    # Dilate only four texels for bilinear filtering at UV chart edges.
    delta, nearest = distance_transform_edt(~valid, return_indices=True)
    padding = (~valid) & (delta <= 4)
    for name, group in paths.items():
        values = np.full((res, res), .1, np.float32)
        values[yy, xx] = np.minimum.reduce([distance_to_segments(points, p) for p in group])
        values[padding] = values[nearest[0][padding], nearest[1][padding]]
        maps[name + '_distance_m'] = values
    np.savez_compressed(dest, **maps)
    meta.write_text(json.dumps(stamp, indent=2) + '\n')
    return dest


def native_preservation_profile(data, excluded_native, surface_positions, skin_mask):
    """Interpolate an anatomical face-treatment mask on a static captured surface.

    excluded_native uses full GNM vertex IDs (e.g. ears and lips). Four nearest
    staged skin vertices provide a smooth transition; refined bound vertices
    inherit the same exclusion weights. Animated callers need posed skin points.
    """
    from scipy.spatial import cKDTree
    points = np.asarray(data['skin_positions'], np.float64)
    ids = np.asarray(data['skin_native_ids'], np.int64)
    excluded = np.asarray(excluded_native, np.float64)
    weights = 1. - excluded[ids]
    if len(points) != len(weights):
        if 'skin_bound_ids' not in data:
            raise ValueError('refined skin requires anatomical binding weights')
        bound = 1. - (excluded[data['skin_bound_ids']] * data['skin_bound_weights']).sum(-1)
        weights = np.concatenate((weights, bound))
    surface_positions, mask = np.asarray(surface_positions), np.asarray(skin_mask, np.float32)
    if len(weights) != len(points) or surface_positions.shape != mask.shape + (3,):
        raise ValueError('matching skin geometry, surface positions and coverage required')
    if not all(np.isfinite(x).all() for x in (points, weights, surface_positions, mask)):
        raise ValueError('nonfinite anatomical profile input')
    result = np.zeros(mask.shape, np.float32)
    valid = mask > 0
    if not valid.any():
        return result
    distance, neighbours = cKDTree(points).query(surface_positions[valid], k=min(4,len(points)), workers=1)
    if distance.ndim == 1:
        distance, neighbours = distance[:,None], neighbours[:,None]
    inverse = 1. / np.maximum(distance, 1e-7) ** 2
    value = (weights[neighbours] * inverse).sum(-1) / inverse.sum(-1)
    result[valid] = np.clip(value,0,1) * np.clip(mask[valid],0,1)
    return result
