"""Boundary tracing for experimental local head reconstruction.

Trace the retained mesh, including old holes intersected by a new cut.
Tracing only newly clipped edges loses those connections. UV seams are
welded geometrically for this calculation; source vertices stay unchanged.
"""
from __future__ import annotations

import numpy as np


def boundary_loops(positions, triangles, required_vertices=(), tolerance=1e-7,
                   split_vertex_fans=False, diagnostics=False):
    """Return closed boundary loops as representative source vertex IDs.

    If required_vertices is nonempty, keep only loops touching those vertices
    (normally new cut points). Open/branched boundary components are reported
    and omitted; they must not be silently connected by an invented edge.
    This checks connectivity, not geometric self-intersection or orientation.
    With split_vertex_fans, boundary branches may be separated only when
    incident triangle connectivity proves distinct fans with two ends each.
    Nonmanifold edges remain rejected; no boundary edge is invented.
    With diagnostics, info also contains rejection reasons and representative
    source vertex IDs. A rejected walk is evidence for repair, not a valid loop.
    rejected_walks includes all walks of components rejected during traversal,
    including simple walks omitted because another walk in that component fails.
    """
    p = np.asarray(positions, dtype=np.float64)
    t = np.asarray(triangles)
    required = np.asarray(required_vertices, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("positions must be finite N x 3")
    if t.ndim != 2 or t.shape[1] != 3 or not np.issubdtype(t.dtype, np.integer):
        raise ValueError("triangles must be integer M x 3")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if (t < 0).any() or (t >= len(p)).any() or (required < 0).any() or (required >= len(p)).any():
        raise ValueError("vertex index out of range")
    _, first, group = np.unique(np.rint(p / tolerance), axis=0,
                                return_index=True, return_inverse=True)
    welded = group[t]
    collapsed = ((welded[:, 0] == welded[:, 1]) | (welded[:, 1] == welded[:, 2]) |
                 (welded[:, 2] == welded[:, 0]))
    if collapsed.any():
        raise ValueError("welding collapses a triangle; reduce tolerance or repair the mesh")
    edges = np.concatenate([welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]])
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    neighbors = {}
    for a, b in edges[counts == 1]:
        neighbors.setdefault(int(a), set()).add(int(b))
        neighbors.setdefault(int(b), set()).add(int(a))
    # A component touching any nonmanifold edge is not a safe join candidate.
    unsafe = set(edges[counts > 2].ravel())
    required_groups = set(group[required])
    seen, loops = set(), []
    rejected = 0
    rejections = []
    rejected_walks = []
    for start in sorted(neighbors):
        if start in seen:
            continue
        pending, component = [start], set()
        while pending:
            vertex = pending.pop()
            if vertex in component:
                continue
            component.add(vertex)
            pending.extend(neighbors[vertex] - component)
        seen.update(component)
        if required_groups and not component.intersection(required_groups):
            continue
        if component.intersection(unsafe):
            rejected += 1
            if diagnostics:
                rejections.append({"reason": "nonmanifold_touch",
                                   "vertices": first[sorted(component.intersection(unsafe))].tolist()})
            continue
        pairs = {}
        for vertex in component:
            adjacent = neighbors[vertex]
            if len(adjacent) == 2:
                a, b = sorted(adjacent)
                pairs[vertex, a], pairs[vertex, b] = b, a
            elif split_vertex_fans:
                # The link of a vertex contains one edge per incident
                # triangle. Disconnected link components are separate fans.
                link = {}
                for triangle in welded[np.any(welded == vertex, axis=1)]:
                    a, b = triangle[triangle != vertex]
                    link.setdefault(int(a), set()).add(int(b))
                    link.setdefault(int(b), set()).add(int(a))
                linked = set()
                for seed in sorted(link):
                    if seed in linked:
                        continue
                    pending, fan = [seed], set()
                    while pending:
                        v = pending.pop()
                        if v in fan:
                            continue
                        fan.add(v)
                        pending.extend(link[v] - fan)
                    linked.update(fan)
                    ends = fan.intersection(adjacent)
                    if len(ends) == 2 and all(len(link[v]) <= 2 for v in fan):
                        a, b = sorted(ends)
                        pairs[vertex, a], pairs[vertex, b] = b, a
        if any((v, n) not in pairs for v in component for n in neighbors[v]):
            rejected += 1
            if diagnostics:
                unresolved = [v for v in sorted(component)
                              if any((v, n) not in pairs for n in neighbors[v])]
                rejections.append({"reason": "unresolved_vertex_fan",
                                   "vertices": first[unresolved].tolist()})
            continue
        traversed, candidates = set(), []
        component_walks, invalid = [], False
        for a in sorted(component):
            for b in sorted(neighbors[a]):
                if (a, b) in traversed:
                    continue
                start, previous, current = (a, b), a, b
                order, visited = [], set()
                while (previous, current) not in visited:
                    visited.add((previous, current))
                    traversed.update(((previous, current), (current, previous)))
                    order.append(previous)
                    previous, current = current, pairs[current, previous]
                if diagnostics:
                    component_walks.append(first[order].tolist())
                if (previous, current) != start or len(set(order)) != len(order):
                    invalid = True
                    if diagnostics:
                        rejections.append({"reason": "open_walk" if (previous, current) != start
                                           else "self_touching_walk",
                                           "vertices": first[order].tolist()})
                        continue
                    candidates = None
                    break
                if not required_groups or required_groups.intersection(order):
                    candidates.append(first[order])
            if candidates is None:
                break
        if invalid:
            rejected += 1
            if diagnostics:
                rejected_walks.extend(component_walks)
        else:
            loops.extend(candidates)
    info = {"rejected_components": rejected,
            "nonmanifold_edges": int((counts > 2).sum())}
    if diagnostics:
        info["rejections"] = rejections
        info["rejected_walks"] = rejected_walks
    return loops, info


def projected_winding(pixels, center):
    """Signed turns around center; reject undefined passage through center.

    A small source hole near an eye has winding zero even if it is closer to
    the camera than the surrounding cut. Winding one does not prove that a
    projected loop is simple, so folds still require 3-D validation.
    """
    delta = np.asarray(pixels, dtype=np.float64) - np.asarray(center)
    if delta.ndim != 2 or delta.shape[1] != 2 or len(delta) < 3 or not np.isfinite(delta).all():
        raise ValueError("pixels must be finite N x 2, N >= 3")
    if (np.linalg.norm(delta, axis=1) < 1e-12).any():
        raise ValueError("loop passes through center")
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    step = (np.roll(angle, -1) - angle + np.pi) % (2 * np.pi) - np.pi
    if (np.abs(np.abs(step) - np.pi) < 1e-12).any():
        raise ValueError("loop edge passes through center")
    return float(step.sum() / (2 * np.pi))
