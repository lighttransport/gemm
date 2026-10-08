"""Diagnostic transverse crossings of nonadjacent triangle surfaces.

This excludes shared-vertex pairs, degenerate facets, coplanar overlaps and boundary-only touches.
It detects strict crossings, not penetration depth, containment, or a complete
collision-free guarantee. Coordinates and distance_tolerance are in metres.
"""
import numpy as np
from scipy.spatial import cKDTree


def crossing_points(first, second, distance_tolerance=1e-7):
    """Return six edge/plane points per pair and their strict-intersection mask.

    Only masked points are intersections. Noncoplanar intersection segments
    generally have two valid endpoints; exact boundary cases are excluded.
    """
    first, second = np.asarray(first, float), np.asarray(second, float)
    if (first.shape != second.shape or first.ndim != 3 or first.shape[1:] != (3,3)
            or not np.isfinite(first).all() or not np.isfinite(second).all()
            or not np.isfinite(distance_tolerance) or distance_tolerance <= 0):
        raise ValueError('invalid triangle crossing inputs')
    points = np.zeros((len(first),6,3))
    mask = np.zeros((len(first),6),bool)
    valid_pair = np.ones(len(first), bool)
    for triangle in (first, second):
        valid_pair &= np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1) > 1e-15
    for side, (source, target) in enumerate(((first, second), (second, first))):
        e1, e2 = target[:,1]-target[:,0], target[:,2]-target[:,0]
        normal = np.cross(e1, e2)
        normal_length = np.linalg.norm(normal, axis=1)
        distance = ((source-target[:,0,None])*normal[:,None]).sum(-1)/np.maximum(normal_length[:,None],1e-30)
        aa, bb, ab = (e1*e1).sum(1), (e2*e2).sum(1), (e1*e2).sum(1)
        determinant = aa*bb-ab*ab
        valid = valid_pair & (normal_length > 1e-15) & (determinant > 1e-30)
        for edge, (start, end) in enumerate(((0,1), (1,2), (2,0))):
            d0, d1 = distance[:,start], distance[:,end]
            straddles = ((d0 > distance_tolerance)&(d1 < -distance_tolerance)
                         | (d1 > distance_tolerance)&(d0 < -distance_tolerance)) & valid
            fraction = np.divide(d0,d0-d1,out=np.zeros_like(d0),where=straddles)
            point = source[:,start]+fraction[:,None]*(source[:,end]-source[:,start])-target[:,0]
            p1, p2 = (point*e1).sum(1), (point*e2).sum(1)
            u = (p1*bb-p2*ab)/np.maximum(determinant,1e-30)
            v = (p2*aa-p1*ab)/np.maximum(determinant,1e-30)
            index=side*3+edge
            points[:,index]=point+target[:,0]
            mask[:,index]=straddles & (u > 1e-8) & (v > 1e-8) & (u+v < 1-1e-8)
    return points,mask


def strict_crossings(first, second, distance_tolerance=1e-7):
    """Return a boolean per paired triangle, checking edges in both directions."""
    return crossing_points(first,second,distance_tolerance)[1].any(1)


def crossing_pairs(vertices, triangles, *, distance_tolerance=1e-7, batch_size=128):
    """Return local triangle-index pairs with strict nonadjacent crossings."""
    vertices, triangles = np.asarray(vertices,float), np.asarray(triangles)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3
            or triangles.dtype.kind not in 'iu' or batch_size < 1
            or not np.isfinite(distance_tolerance) or distance_tolerance <= 0
            or (len(triangles) and (triangles.min()<0 or triangles.max()>=len(vertices)))):
        raise ValueError('invalid crossing mesh')
    if not len(triangles):
        return np.empty((0,2),int)
    points = vertices[triangles]
    centers = points.mean(1)
    radii = np.linalg.norm(points-centers[:,None],axis=-1).max(1)
    low, high = points.min(1), points.max(1)
    tree = cKDTree(centers)
    found = []
    for start in range(0,len(triangles),batch_size):
        end = min(start+batch_size,len(triangles))
        neighbors = tree.query_ball_point(centers[start:end],radii[start:end]+radii.max()+distance_tolerance)
        pairs = [(i,j) for i,near in enumerate(neighbors,start) for j in near if j>i]
        if not pairs:
            continue
        pairs = np.asarray(pairs,int)
        a,b = pairs.T
        possible = ((low[a] <= high[b]+distance_tolerance)&(low[b] <= high[a]+distance_tolerance)).all(1)
        possible &= ~(triangles[a,:,None] == triangles[b,None,:]).any(axis=(1,2))
        pairs = pairs[possible]
        if not len(pairs):
            continue
        a,b = pairs.T
        hits = strict_crossings(points[a],points[b],distance_tolerance)
        if hits.any():
            found.append(pairs[hits])
    return np.concatenate(found) if found else np.empty((0,2),int)
