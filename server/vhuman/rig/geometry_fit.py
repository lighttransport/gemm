"""CPU contact fitting of inferred ocular cavities, without topology changes.

GNM identity coefficients and eye geometry stay fixed. The hidden skin cups
are fitted around the eye hull; visible lid displacement is explicitly bounded.
This is an anatomical geometry prior, not a measurement of a subject's orbit.
"""
from __future__ import annotations

import ctypes as ct
import math

import numpy as np

from .biomechanics_native import P, check, library, ptr


def _surface_inputs(vertices, faces):
    vertices, faces = np.asarray(vertices, np.float64), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in 'iu' or (faces < 0).any() or (faces >= len(vertices)).any()):
        raise ValueError('invalid socket surface geometry')
    return vertices, faces.astype(np.int32, copy=False)


class CrossingCounter:
    """Reusable native BVH, with the same adjacency rules as the bake gate."""
    def __init__(self, vertices, faces):
        vertices, faces = _surface_inputs(vertices, faces)
        self.lib = library()
        self.handle = P()
        self.vertices = np.ascontiguousarray(vertices, np.float64)
        self.faces = np.ascontiguousarray(faces, np.int32)
        mass = np.zeros(len(self.vertices))
        check(self.lib.lightphysics_surface_contact_create(
            ptr(self.vertices), ptr(mass), len(mass), ptr(self.faces), len(self.faces), ct.byref(self.handle)))

    def count(self, vertices):
        vertices = np.ascontiguousarray(vertices, np.float64)
        if vertices.shape != self.vertices.shape or not np.isfinite(vertices).all():
            raise ValueError('invalid surface crossing query')
        count = ct.c_uint64()
        check(self.lib.lightphysics_surface_contact_count_intersections(self.handle, ptr(vertices), ct.byref(count)))
        return int(count.value)

    def __enter__(self):
        return self

    def intersections(self, vertices, *, limit=1000000):
        """Return bounded edge/triangle witnesses, using the bake predicate."""
        if type(limit) is not int or not 1 <= limit <= 1000000:
            raise ValueError('invalid surface crossing witness limit')
        find = getattr(self.lib, 'lightphysics_surface_contact_find_intersections', None)
        if find is None:
            raise RuntimeError('rebuild the CPU LightGeom library for crossing witnesses')
        vertices = np.ascontiguousarray(vertices, np.float64)
        count = self.count(vertices)
        if count > limit:
            raise ValueError('surface crossing witness limit exceeded')
        records = np.empty((count, 5), np.int32)
        total = ct.c_uint64()
        check(find(self.handle, ptr(vertices), ptr(records), count, ct.byref(total)))
        if total.value != count:
            raise RuntimeError('surface crossing count changed during read-only query')
        return records[np.lexsort(records.T[::-1])] if count else records

    def __exit__(self, *_):
        if self.handle:
            self.lib.lightphysics_surface_contact_destroy(self.handle)
            self.handle = P()


def disk_chart(vertices, faces, hidden):
    """Tutte coordinates of a manifold triangulated disk with a fixed rim."""
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    from .anatomy import boundary_edges
    vertices, faces = _surface_inputs(vertices, faces)
    hidden = np.asarray(hidden, bool)
    if hidden.shape != (len(vertices),):
        raise ValueError('invalid socket interior mask')
    edges = boundary_edges(faces)
    if not len(edges):
        raise ValueError('socket patch needs a single boundary')
    successor = {int(a): int(b) for a, b in edges}
    if len(successor) != len(edges) or set(successor) != set(map(int, edges[:, 1])):
        raise ValueError('socket rim branches')
    loop = [min(successor)]
    while successor[loop[-1]] != loop[0]:
        following = successor[loop[-1]]
        if following in loop:
            raise ValueError('socket patch has multiple boundary loops')
        loop.append(following)
    if len(loop) != len(edges):
        raise ValueError('socket patch has multiple boundary loops')
    loop = np.asarray(loop, np.int32)
    ids = np.unique(faces[hidden[faces]])
    if hidden[loop].any() or len(ids) + len(loop) != len(np.unique(faces)):
        raise ValueError('socket labels do not form a disk interior and rim')
    adjacency = {int(i): set() for i in np.unique(faces)}
    for face in faces:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            adjacency[int(face[a])].add(int(face[b]))
            adjacency[int(face[b])].add(int(face[a]))
    edge_count = sum(map(len, adjacency.values())) // 2
    if len(adjacency) - edge_count + len(faces) != 1:
        raise ValueError('socket patch is not a topological disk')
    reached = set(map(int, loop))
    queue = list(reached)
    for i in queue:
        for j in adjacency[i] - reached:
            reached.add(j)
            queue.append(j)
    if len(reached) != len(adjacency):
        raise ValueError('socket patch is disconnected')
    lengths = np.linalg.norm(vertices[loop] - vertices[np.roll(loop, -1)], axis=1)
    if (lengths <= 1e-9).any():
        raise ValueError('collapsed socket rim')
    angles = np.r_[0., np.cumsum(lengths)[:-1]] / lengths.sum() * 2 * np.pi
    uv_boundary = np.column_stack((np.cos(angles), np.sin(angles)))
    boundary = dict(zip(map(int, loop), uv_boundary))
    lookup = {int(i): row for row, i in enumerate(ids)}
    matrix = lil_matrix((len(ids), len(ids)))
    rhs = np.zeros((len(ids), 2))
    for row, i in enumerate(ids):
        matrix[row, row] = len(adjacency[int(i)])
        for j in adjacency[int(i)]:
            if j in lookup:
                matrix[row, lookup[j]] = -1
            else:
                rhs[row] += boundary[j]
    uv = spsolve(matrix.tocsr(), rhs)
    if not np.isfinite(uv).all() or (np.linalg.norm(uv, axis=1) >= 1).any():
        raise ValueError('invalid socket disk chart')
    return ids, loop, angles, uv


def _closest_barycentric(points, triangles):
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    relative = points - triangles[:, 0]
    aa, ab, bb = (a*a).sum(1), (a*b).sum(1), (b*b).sum(1)
    denominator = aa*bb - ab*ab
    if (denominator <= 1e-26).any():
        raise ValueError('socket fitting collapsed a triangle')
    ar, br = (a*relative).sum(1), (b*relative).sum(1)
    w1 = (bb*ar - ab*br) / denominator
    w2 = (aa*br - ab*ar) / denominator
    weights = np.maximum(0, np.column_stack((1-w1-w2, w1, w2)))
    return weights / weights.sum(1, keepdims=True)


def _preserve_visible_faces(original, result, skin_faces, visible, center):
    """Resolve projection-induced folds with bounded tangential displacements."""
    faces = skin_faces[visible[skin_faces].all(1)]
    corners = original[faces]
    reference = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
    area = np.linalg.norm(reference, axis=1)
    normal = reference / np.maximum(area[:, None], 1e-30)
    for _ in range(100):
        corners = result[faces]
        current = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
        signed = (current*normal).sum(1)
        bad = signed < .149*area
        if not bad.any():
            return
        triangles = corners[bad]
        ref = normal[bad]
        gb = np.cross(triangles[:, 2]-triangles[:, 0], ref)
        gc = np.cross(ref, triangles[:, 1]-triangles[:, 0])
        gradient = np.stack((-gb-gc, gb, gc), axis=1)
        radial = triangles-center
        radial /= np.linalg.norm(radial, axis=2, keepdims=True)
        gradient -= (gradient*radial).sum(2, keepdims=True)*radial
        denominator = np.square(gradient).sum((1, 2))
        if (denominator < 1e-24).any():
            raise ValueError('visible socket projection has an unresolved fold')
        step = (.15*area[bad]-signed[bad])[:, None, None]*gradient/denominator[:, None, None]
        step *= np.minimum(1, .00005/np.maximum(np.linalg.norm(step, axis=2, keepdims=True), 1e-20))
        change = np.zeros_like(result)
        count = np.zeros(len(result))
        np.add.at(change, faces[bad].ravel(), step.reshape(-1, 3))
        np.add.at(count, faces[bad].ravel(), 1)
        moving = count > 0
        result[moving] += change[moving]/count[moving, None]
    raise ValueError('visible socket projection did not preserve face orientation')


def fit_socket(vertices, skin_faces, hidden, visible, eye, *, visible_limit=.0015,
               hidden_limit=.03, clearance=.0001, iterations=100):
    """Wrap one hidden cup around a convex eye envelope, retaining its faces."""
    from scipy.spatial import ConvexHull, cKDTree
    from ..reconstruction.contact_runtime import closest_on_triangles
    vertices, skin_faces = _surface_inputs(vertices, skin_faces)
    eye = np.asarray(eye, np.float64)
    hidden, visible = np.asarray(hidden, bool), np.asarray(visible, bool)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or hidden.shape != (len(vertices),)
            or visible.shape != hidden.shape or (hidden & visible).any()
            or not hidden.any() or eye.ndim != 2 or eye.shape[1] != 3 or len(eye) < 16
            or not np.isfinite(eye).all() or type(iterations) is not int or not 1 <= iterations <= 200):
        raise ValueError('invalid socket fitting inputs')
    if not all(math.isfinite(v) and v > 0 for v in (visible_limit, hidden_limit, clearance)):
        raise ValueError('invalid socket fitting bounds')
    patch = skin_faces[hidden[skin_faces].any(1)]
    ids, loop, angles, uv = disk_chart(vertices, patch, hidden)
    if not visible[loop].all():
        raise ValueError('socket rim must belong to the visible lid mask')
    hull = ConvexHull(eye)
    # Fit the posterior globe, excluding the forward corneal bulge. The hull
    # supplies actual directional radii; no spherical cornea assumption follows.
    posterior = eye[eye[:, 2] < np.quantile(eye[:, 2], .75)]
    sphere = np.linalg.lstsq(np.column_stack((2*posterior, np.ones(len(posterior)))),
                             np.square(posterior).sum(1), rcond=None)[0]
    center = sphere[:3]
    heights = -(center @ hull.equations[:, :3].T + hull.equations[:, 3])
    if heights.min() <= 1e-5:
        raise ValueError('eye envelope has no stable interior center')

    def radial(directions):
        denominator = directions @ hull.equations[:, :3].T
        return np.min(np.where(denominator > 1e-12,
                      heights / np.maximum(denominator, 1e-12), np.inf), axis=1)

    result = vertices.copy()
    candidates = np.flatnonzero(visible)
    inside = (result[candidates] @ hull.equations[:, :3].T + hull.equations[:, 3] <= clearance).all(1)
    candidates = candidates[inside]
    triangles = eye[hull.simplices]
    if len(candidates):
        nearest = cKDTree(triangles.mean(1)).query(vertices[candidates], k=min(64, len(triangles)))[1]
        selected = triangles[nearest]
        q, _, distance = closest_on_triangles(vertices[candidates], selected[:, :, 0], selected[:, :, 1], selected[:, :, 2])
        choice = distance.argmin(1)
        q = q[np.arange(len(q)), choice]
        normal = hull.equations[nearest[np.arange(len(q)), choice], :3]
        result[candidates] = q + 2*clearance*normal
    _preserve_visible_faces(vertices, result, skin_faces, visible, center)
    direction = result[loop] - center
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    azimuth = np.unwrap(np.arctan2(direction[:, 1], direction[:, 0]))
    period = 2*np.pi*np.sign(azimuth[-1]-azimuth[0])
    if not period:
        raise ValueError('socket rim has no angular winding')
    boundary_polar = np.arccos(np.clip(direction[:, 2], -1, 1))
    radius = np.linalg.norm(uv, axis=1)
    parameter = np.mod(np.arctan2(uv[:, 1], uv[:, 0]), 2*np.pi)
    theta = np.r_[angles, 2*np.pi]
    phi = np.interp(parameter, theta, np.r_[azimuth, azimuth[0]+period])
    polar = np.pi - (np.pi - np.interp(parameter, theta, np.r_[boundary_polar, boundary_polar[0]]))*radius
    unit = np.column_stack((np.sin(polar)*np.cos(phi), np.sin(polar)*np.sin(phi), np.cos(polar)))
    result[ids] = center + (radial(unit) + 6*clearance)[:, None]*unit
    maximum_radius = 1.5*np.linalg.norm(eye-center, axis=1).max()
    for iteration in range(iterations+1):
        triangles = result[patch]
        closest, _, _ = closest_on_triangles(center[None], triangles[None, :, 0], triangles[None, :, 1], triangles[None, :, 2])
        closest = closest[0]
        direction = closest-center
        distance = np.linalg.norm(direction, axis=1)
        direction /= np.maximum(distance[:, None], 1e-20)
        gap = radial(direction) + clearance-distance
        bad = gap > 1e-8
        if not bad.any():
            break
        if iteration == iterations:
            raise ValueError('socket clearance fitting did not converge')
        faces = patch[bad]
        corners = result[faces]
        barycentric = _closest_barycentric(closest[bad], corners)
        radial_direction = corners-center
        radial_direction /= np.linalg.norm(radial_direction, axis=2, keepdims=True)
        derivative = barycentric*hidden[faces]*np.einsum('fkc,fc->fk', radial_direction, direction[bad])
        step = np.minimum(.0005, .8*gap[bad, None]*derivative / np.maximum((derivative**2).sum(1, keepdims=True), 1e-10))
        increments = np.zeros(len(vertices))
        np.maximum.at(increments, faces.ravel(), step.ravel())
        moving = np.flatnonzero(increments)
        direction = result[moving]-center
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        result[moving] += increments[moving, None]*direction
        if np.linalg.norm(result[ids]-center, axis=1).max() > maximum_radius:
            raise ValueError('socket clearance constraints exceed the radial bound')
    visible_change = np.linalg.norm(result[visible]-vertices[visible], axis=1).max(initial=0)
    hidden_change = np.linalg.norm(result[hidden]-vertices[hidden], axis=1).max()
    if visible_change > visible_limit or hidden_change > hidden_limit:
        raise ValueError('socket fitting exceeded visible/hidden displacement bounds')
    new = result[patch]
    normal = np.cross(new[:, 1]-new[:, 0], new[:, 2]-new[:, 0])
    if (np.einsum('ij,ij->i', normal, new.mean(1)-center) >= 0).any():
        raise ValueError('socket fitting reversed cavity winding')
    return result, dict(hidden_vertices=len(ids), lid_vertices=len(candidates), iterations=iteration,
                        visible_max_mm=visible_change*1000, hidden_max_mm=hidden_change*1000,
                        clearance_mm=clearance*1000, radial_support_check=True)


def fit_sockets(model, vertices):
    """Fit both cups and accept only independently checked improvements."""
    from .anatomy import part_masks
    vertices = np.asarray(vertices, np.float64)
    triangles = model.data['triangles']
    selected = np.zeros(len(triangles), bool)
    for _, mask, _ in part_masks(model):
        selected |= mask[triangles].all(1)
    faces = triangles[selected]
    skin = model.group('skin')
    skin_faces = triangles[skin[triangles].all(1)]
    exterior = model.group('skin_exterior')
    with CrossingCounter(vertices, faces) as all_crossings, CrossingCounter(vertices, skin_faces) as self_crossings:
        before = all_crossings.count(vertices)
        before_self = self_crossings.count(vertices)
        result = vertices.copy()
        reports = {}
        applied = False
        for side in ('left', 'right'):
            eye_mask = model.group(side+'_eye') & model.group('eye_exteriors')
            eye_faces = triangles[eye_mask[triangles].all(1)]
            hidden = skin & model.group('eye_sockets') & ~exterior & model.group(side)
            with CrossingCounter(vertices, np.concatenate((skin_faces, eye_faces))) as paired:
                prior = paired.count(result) - self_crossings.count(result)
                if prior == 0:
                    reports[side] = dict(applied=False, before_eye_crossings=0, after_eye_crossings=0,
                                         reason='ocular geometry already clear')
                    continue
                candidate, report = fit_socket(result, skin_faces, hidden, skin & exterior & model.group(side), result[eye_mask])
                after = paired.count(candidate) - self_crossings.count(candidate)
            if after != 0 or self_crossings.count(candidate) > before_self:
                raise ValueError(f'{side} socket fit failed its independent crossing check')
            report.update(applied=True, before_eye_crossings=prior, after_eye_crossings=after)
            reports[side] = report
            result = candidate
            applied = True
        after = all_crossings.count(result)
        after_self = self_crossings.count(result)
    external_faces = triangles[exterior[triangles].all(1)]
    original = np.cross(vertices[external_faces[:, 1]]-vertices[external_faces[:, 0]], vertices[external_faces[:, 2]]-vertices[external_faces[:, 0]])
    current = np.cross(result[external_faces[:, 1]]-result[external_faces[:, 0]], result[external_faces[:, 2]]-result[external_faces[:, 0]])
    oriented = np.einsum('ij,ij->i', original, current) / np.maximum((original**2).sum(1), 1e-30)
    if oriented.min() < .05 or after > before or (applied and after == before):
        raise ValueError('socket fit did not preserve visible faces and improve source geometry')
    return result, dict(schema='vhuman.socket_fit.v1', accepted=True, applied=applied, geometry_only=True,
                        before_crossings=before, after_crossings=after,
                        before_skin_self_crossings=before_self, after_skin_self_crossings=after_self,
                        visible_min_oriented_area_ratio=float(oriented.min()),
                        identity_coefficients_frozen=True, topology_preserved=True, eyes_fixed=True,
                        hidden_anatomy='authored convex ocular envelope and disk parameterization', sides=reports)
