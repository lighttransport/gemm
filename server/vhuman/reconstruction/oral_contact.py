"""Refined mouth-cavity lining resting on the dental arch.

The GNM mouth "sock" is a coarse disk from the lip rim to the throat that runs
straight from the lips into the palate/floor; incisors pierce it and nothing
lines the inside of the lips against the teeth, leaving a dark pit. Here the
sock is subdivided once (surface-bound to its native cage, like the refined
ears; the native lip rim stays fixed) and given a static bind-space contact
shape solved over many poses: the vestibular lining behind each lip rests on
the labial teeth/gum surface with a small clearance (a stand-in for lip muscle
tone pressing mucosa onto the arch), smoothly feathered from the rim, without
moving visible lining into the arch. It is a geometric approximation, not a
soft-tissue simulation.
"""
import numpy as np

from . import ear_mesh as em


def sock_patch(quads, triangles, sock_mask, component_ids):
    """Quads whose four corners are mouth-sock vertices and whose triangles belong to skin."""
    quads, triangles = np.asarray(quads), np.asarray(triangles)
    pair = em.quad_triangles(quads, triangles)
    skin_tri = np.asarray(component_ids) == 0
    inside = np.asarray(sock_mask, bool)[quads].all(1) & skin_tri[pair].all(1)
    return np.flatnonzero(inside), pair


def build_sock_topology(full, triangles, triangle_uvs, quads, sock_mask, component_ids):
    """One Catmull-Clark level over the sock disk with the native lip rim as fixed boundary."""
    full = np.asarray(full, float)
    native = len(full)
    patch, pair = sock_patch(quads, triangles, sock_mask, component_ids)
    sub = em.subdivide(np.asarray(quads)[patch], em.quad_corner_uvs(np.asarray(quads)[patch], triangles, triangle_uvs,
                                                                       pair[patch]), native)
    base = sub['weights']@full
    ids, weights, offsets = em.bind(full, triangles, pair, patch, base, sub['origin_quad'])
    return dict(base=base, triangles=sub['triangles'], triangle_uvs=sub['triangle_uvs'], quad_pairs=sub['quad_pairs'],
                removed=np.sort(pair[patch].ravel()), boundary=sub['boundary'], operator=sub['weights'],
                bind_ids=ids, bind_weights=weights, bind_offsets=offsets, native_count=native,
                source_native=sub['source_native'], kind=sub['kind'])


def posed_frames(full, ids):
    """Attachment frames (...,V,3,3) of bound vertices for native positions full (...,N,3)."""
    return em.attachment_frames(np.asarray(full, float)[..., ids, :])


def candidate_arrays(full_neutral, full_captured, topo, offsets, spec=None, edges=None):
    """``geometry.npz`` fields for the refined sock (``sock_*``), mirroring the ear fields.

    ``spec``/``edges`` (dense-local indices) add the per-pose contact deformer (``sock_contact_*``).
    """
    neutral = em.evaluate(full_neutral, topo['bind_ids'], topo['bind_weights'], offsets)
    captured = em.evaluate(np.asarray(full_captured)[0], topo['bind_ids'], topo['bind_weights'], offsets)
    return dict(sock_dense_neutral=neutral.astype(np.float32), sock_dense_captured=captured.astype(np.float32),
                sock_dense_triangles=topo['triangles'].astype(np.int32),
                sock_dense_triangle_uvs=topo['triangle_uvs'].astype(np.float32),
                sock_dense_quad_pairs=topo['quad_pairs'].astype(np.int32),
                sock_bind_ids=topo['bind_ids'].astype(np.int32), sock_bind_weights=topo['bind_weights'].astype(np.float64),
                sock_bind_offsets=np.asarray(offsets, np.float64), sock_removed_triangles=topo['removed'].astype(np.int32),
                **({} if spec is None else dict(
                    sock_contact_active=np.asarray(spec['active'], np.int32), sock_contact_weight=np.asarray(spec['weight'], np.float64),
                    sock_contact_candidates=np.asarray(spec['candidates'], np.int32),
                    sock_contact_params=np.array([spec['clearance'], spec['max_move'], spec['smoothing'][0], spec['smoothing'][1]], np.float64),
                    sock_contact_edges=np.asarray(edges, np.int32))))


from .contact_runtime import closest_on_triangles, contact_deform  # noqa: E402,F401


def contact_spec(rest_positions, rest_native, arch_triangles, depth, *, k=24, labial_min=.25, forward=(0, 0, 1),
                 ramp=(.0015, .005, .018, .026), clearance=5e-4, max_move=.008, smoothing=(.5, 2), neighbours=None):
    """Precompute the per-pose contact deformer for lining vertices (positions in part order).

    Candidates are the k nearest labial-facing arch triangles at rest (normal . forward > labial_min).
    Weights ramp in from the lip rim and out toward the palate/floor by surface depth (metres).
    """
    rest_positions, rest_native = np.asarray(rest_positions, float), np.asarray(rest_native, float)
    tri = np.asarray(arch_triangles)
    n = np.cross(rest_native[tri[:, 1]]-rest_native[tri[:, 0]], rest_native[tri[:, 2]]-rest_native[tri[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    labial = tri[n@np.asarray(forward, float) > labial_min]
    a, b, c, d = ramp
    f = lambda x, lo, hi: (lambda t: t*t*(3-2*t))(np.clip((x-lo)/(hi-lo), 0, 1))
    weight = f(np.asarray(depth), a, b)*(1-f(np.asarray(depth), c, d))
    active = np.flatnonzero(weight > 0)
    from scipy.spatial import cKDTree
    _, near = cKDTree(rest_native[labial].mean(1)).query(rest_positions[active], k=min(k, len(labial)))
    return dict(active=active, weight=weight[active], candidates=labial[np.atleast_2d(near)], clearance=float(clearance),
                max_move=float(max_move), smoothing=list(smoothing), neighbours=neighbours)
