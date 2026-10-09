"""Locally refined, surface-bound GNM ear geometry.

GNM v3 ear vertices carry no expression or pose-corrective deltas and are
skinned almost entirely to the head joint. A denser ear is therefore stored as
new vertices bound to the hidden native ear triangles: barycentric anchors plus
offsets in the attachment frame shared by Blender and the mobile runtime
(``offline_assets.attachment_frames``). Root-boundary vertices remain native,
so the refined ear meets unchanged skin without T-junctions. GNM, its bases and
the stored motion stay authoritative; nothing here edits ``full_*`` arrays.

Exactness of a bound vertex holds when its anchor triangle moves rigidly. It is
not a collision, containment or anatomical-accuracy guarantee.
"""
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix

SIDES = (('left', -1), ('right', 1))


def attachment_frames(points):
    """Same frame as ``offline_assets.attachment_frames`` (columns x, y, z)."""
    x = points[..., 1, :]-points[..., 0, :]
    x = x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    z = np.cross(x, points[..., 2, :]-points[..., 0, :])
    z = z/np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
    return np.stack((x, np.cross(z, x), z), -1)


def quad_triangles(quads, triangles):
    """Return (Q,2) triangle ids per quad: (q0,q1,q2) and (q0,q2,q3) or the other diagonal."""
    quads, triangles = np.asarray(quads), np.asarray(triangles)
    if len(triangles) != 2*len(quads):
        raise ValueError('triangles are not a two-per-quad split')
    lookup = {frozenset(map(int, t)): i for i, t in enumerate(triangles)}
    pair = np.zeros((len(quads), 2), np.int64)
    for i, q in enumerate(quads):
        q = list(map(int, q))
        for a, b in (((0, 1, 2), (0, 2, 3)), ((0, 1, 3), (1, 2, 3))):
            ka, kb = frozenset(q[j] for j in a), frozenset(q[j] for j in b)
            if ka in lookup and kb in lookup:
                pair[i] = lookup[ka], lookup[kb]
                break
        else:
            raise ValueError('quad without its two triangles')
    if len(np.unique(pair)) != len(triangles):
        raise ValueError('triangles are not a two-per-quad split')
    return pair


def ear_patches(ear_mask, quads, positions):
    """Per-side quad ids whose four corners are in the GNM ``ears`` group."""
    ear_mask, quads, positions = np.asarray(ear_mask, bool), np.asarray(quads), np.asarray(positions, float)
    inside = np.flatnonzero(ear_mask[quads].all(1))
    side = np.sign(positions[quads[inside]].mean(1)[:, 0])
    patches = {}
    for name, sign in SIDES:
        selected = inside[side == sign]
        if not len(selected):
            raise ValueError('missing ear patch: '+name)
        patches[name] = selected
    return patches


def _edges(quads):
    count = defaultdict(list)
    for f, q in enumerate(quads):
        for k in range(4):
            count[tuple(sorted((int(q[k]), int(q[(k+1) % 4]))))].append(f)
    return count


def boundary_loop(quads):
    """Ordered root-boundary vertex loop of a disk-like quad patch."""
    edges = [e for e, faces in _edges(quads).items() if len(faces) == 1]
    nxt = defaultdict(list)
    for a, b in edges:
        nxt[a].append(b)
        nxt[b].append(a)
    if any(len(v) != 2 for v in nxt.values()):
        raise ValueError('ear patch boundary is not a simple loop')
    start = edges[0][0]
    loop, previous = [start], None
    while True:
        a, b = nxt[loop[-1]]
        step = a if a != previous else b
        if step == start:
            break
        previous = loop[-1]
        loop.append(step)
    if len(loop) != len(edges):
        raise ValueError('ear patch boundary has several loops')
    return np.asarray(loop)


def subdivide(quads, quad_uv, native_count):
    """One Catmull-Clark level inside a disk patch with an unsplit outer boundary.

    Boundary vertices keep their native ids and boundary edges are not split;
    boundary-adjacent faces are fan-triangulated around their face point.
    Positions are linear in the native vertices: ``weights`` is a CSR matrix
    (new x native). UVs are interpolated linearly per coarse face corner
    (``quad_uv`` (Q,4,2)), so UV seams inside the patch are preserved and new
    corners stay inside their original quad. New vertex k has extended index
    ``native_count + k``. Triangles keep the quad winding.
    """
    quads, quad_uv = np.asarray(quads), np.asarray(quad_uv, float)
    edges = _edges(quads)
    boundary = {e for e, faces in edges.items() if len(faces) == 1}
    boundary_vertices = {v for e in boundary for v in e}
    patch_vertices = np.unique(quads)
    rows, origin, kind, source = [], [], [], []

    def add(row, face, k, native=-1):
        rows.append(row)
        origin.append(face)
        kind.append(k)
        source.append(native)
        return native_count+len(rows)-1

    face_point = {f: add({int(v): .25 for v in q}, f, 2) for f, q in enumerate(quads)}
    edge_point = {}
    for e, faces in edges.items():
        if e in boundary:
            continue
        row = defaultdict(float)
        for v in e:
            row[v] += .25
        for f in faces:
            for v in quads[f]:
                row[int(v)] += .25/4
        edge_point[e] = add(dict(row), faces[0], 1)
    vertex_faces = defaultdict(list)
    vertex_edges = defaultdict(list)
    for f, q in enumerate(quads):
        for v in q:
            vertex_faces[int(v)].append(f)
    for e in edges:
        for v in e:
            vertex_edges[v].append(e)
    vertex_point = {}
    for v in patch_vertices:
        v = int(v)
        if v in boundary_vertices:
            vertex_point[v] = v
            continue
        faces, ring = vertex_faces[v], vertex_edges[v]
        n = len(ring)
        if len(faces) != n:
            raise ValueError('interior patch vertex is not manifold')
        row = defaultdict(float)
        for f in faces:
            for u in quads[f]:
                row[int(u)] += 1/(4*len(faces))/n
        for e in ring:
            for u in e:
                row[u] += 1/(len(ring)*n)
        row[v] += (n-3)/n
        vertex_point[v] = add(dict(row), faces[0], 0, v)
    triangles, triangle_uvs, triangle_face, quad_pairs = [], [], [], []
    rows_kind = lambda r: ('vertex', 'edge', 'face')[kind[r-native_count]]
    for f, q in enumerate(quads):
        ring, ring_uv = [], []
        for k in range(4):
            a, b = int(q[k]), int(q[(k+1) % 4])
            ring.append(vertex_point[a])
            ring_uv.append(quad_uv[f, k])
            e = tuple(sorted((a, b)))
            if e not in boundary:
                ring.append(edge_point[e])
                ring_uv.append((quad_uv[f, k]+quad_uv[f, (k+1) % 4])/2)
        c, cuv = face_point[f], quad_uv[f].mean(0)
        start = len(triangles)
        for i in range(len(ring)):
            j = (i+1) % len(ring)
            triangles.append((ring[i], ring[j], c))
            triangle_uvs.append((ring_uv[i], ring_uv[j], cuv))
            triangle_face.append(f)
        # Pair fan triangles (e_prev, v, c) + (v, e_next, c) into Catmull-Clark quads
        # wherever both neighbours of a corner are edge points.
        kinds = [(r >= native_count) and rows_kind(r) for r in ring]
        for i in range(len(ring)):
            if kinds[i] == 'vertex' or (ring[i] < native_count):
                a, b = (i-1) % len(ring), (i+1) % len(ring)
                if kinds[a] == 'edge' and kinds[b] == 'edge':
                    quad_pairs.append((start+a, start+i))
    data, indices, indptr = [], [], [0]
    for row in rows:
        keys = sorted(row)
        indices.extend(keys)
        data.extend(row[k] for k in keys)
        indptr.append(len(indices))
    weights = csr_matrix((data, indices, indptr), shape=(len(rows), native_count))
    return dict(weights=weights, triangles=np.asarray(triangles, np.int64),
                triangle_uvs=np.asarray(triangle_uvs, float), triangle_face=np.asarray(triangle_face),
                quad_pairs=np.asarray(quad_pairs, np.int64).reshape(-1, 2),
                origin_quad=np.asarray(origin), kind=np.asarray(kind, np.int8), source_native=np.asarray(source),
                boundary=np.asarray(sorted(boundary_vertices)),
                interior_native=np.asarray(sorted(set(map(int, patch_vertices))-boundary_vertices)))


def closest_barycentric(points, triangles):
    """Clamped barycentrics of the closest point on each triangle (N,3,3)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b-a, c-a, points-a
    d00, d01, d11 = (ab*ab).sum(1), (ab*ac).sum(1), (ac*ac).sum(1)
    d20, d21 = (ap*ab).sum(1), (ap*ac).sum(1)
    det = np.maximum(d00*d11-d01*d01, 1e-30)
    v = (d11*d20-d01*d21)/det
    w = (d00*d21-d01*d20)/det
    bary = np.stack((1-v-w, v, w), 1)
    bary = np.clip(bary, 0, None)
    return bary/bary.sum(1, keepdims=True)


def bind(full, triangles, quad_pair, patch_quads, positions, origin_quad):
    """Anchor each new vertex to the nearer native triangle of its origin quad."""
    full, positions = np.asarray(full, float), np.asarray(positions, float)
    candidates = quad_pair[patch_quads[origin_quad]]
    best = None
    for column in range(2):
        tid = candidates[:, column]
        corners = full[triangles[tid]]
        bary = closest_barycentric(positions, corners)
        distance = np.linalg.norm((corners*bary[:, :, None]).sum(1)-positions, axis=1)
        if best is None:
            best = [tid, bary, distance]
        else:
            take = distance < best[2]
            best[0] = np.where(take, tid, best[0])
            best[1] = np.where(take[:, None], bary, best[1])
            best[2] = np.where(take, distance, best[2])
    ids = triangles[best[0]]
    weights = best[1]
    corners = full[ids]
    roots = (corners*weights[:, :, None]).sum(1)
    offsets = np.einsum('vji,vj->vi', attachment_frames(corners), positions-roots)
    return ids.astype(np.int64), weights, offsets


def evaluate(full, ids, weights, offsets):
    """Bound positions for native vertices ``full`` of shape (...,N,3)."""
    full = np.asarray(full, float)
    corners = full[..., ids, :]
    roots = (corners*weights[..., None]).sum(-2)
    return roots+np.einsum('...vij,vj->...vi', attachment_frames(corners), offsets)


def quad_corner_uvs(quads, triangles, triangle_uvs, pair):
    """Per-corner UVs of each quad taken from its own two triangles."""
    out = np.zeros((len(quads), 4, 2))
    for i, q in enumerate(quads):
        lookup = {}
        for t in pair[i]:
            for v, u in zip(triangles[t], triangle_uvs[t]):
                lookup[int(v)] = u
        out[i] = [lookup[int(v)] for v in q]
    return out


def band_patches(ear_mask, quads, positions, triangles, allowed, band_m, region=None):
    """Per-side quads: the ear disk plus surrounding quads within ``band_m`` (graph geodesic)
    of the ear root loop whose vertices are all ``allowed``. Holes inside ``region``
    (vertex mask of the surface component, default all) are filled so the patch
    stays a disk."""
    from scipy.sparse.csgraph import dijkstra, connected_components
    positions = np.asarray(positions, float)
    ear = ear_patches(ear_mask, quads, positions)
    if band_m <= 0:
        return ear, {k: boundary_loop(quads[v]) for k, v in ear.items()}
    e = mesh_edges(triangles)
    length = np.linalg.norm(positions[e[:, 0]]-positions[e[:, 1]], axis=1)
    n = len(positions)
    graph = csr_matrix((np.r_[length, length], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    out, loops = {}, {}
    for name, patch in ear.items():
        loop = boundary_loop(quads[patch])
        distance = dijkstra(graph, indices=loop, min_only=True)
        near = (distance <= band_m) & np.asarray(allowed, bool)
        candidate = np.flatnonzero(near[quads].all(1))
        side = np.sign(positions[quads[candidate]].mean(1)[:, 0])
        candidate = np.union1d(candidate[side == np.sign(positions[loop, 0].mean())], patch)
        # Keep the face-connected component containing the ear.
        index = {int(q): i for i, q in enumerate(candidate)}
        rows, cols = [], []
        for f, faces in _edges(quads[candidate]).items():
            if len(faces) == 2:
                rows.append(faces[0])
                cols.append(faces[1])
        adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(candidate),)*2)
        _, label = connected_components(adjacency, directed=False)
        candidate = candidate[label == label[index[int(patch[0])]]]
        # Fill holes: add outside quads enclosed by the patch (complement components
        # not touching the global complement's largest component).
        in_region = np.ones(len(quads), bool) if region is None else np.asarray(region, bool)[quads].all(1)
        others = np.setdiff1d(np.flatnonzero(in_region), candidate)
        rows, cols = [], []
        sub = quads[others]
        for f, faces in _edges(sub).items():
            if len(faces) == 2:
                rows.append(faces[0])
                cols.append(faces[1])
        adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(others),)*2)
        count, label = connected_components(adjacency, directed=False)
        if count > 1:
            sizes = np.bincount(label)
            enclosed = others[label != np.argmax(sizes)]
            near_ear = np.linalg.norm(positions[quads[enclosed]].mean(1)-positions[loop].mean(0), axis=1) < .08
            candidate = np.union1d(candidate, enclosed[near_ear])
        out[name] = candidate
        loops[name] = loop
    return out, loops


def build_topology(full, triangles, triangle_uvs, quads, ear_mask, *, band_m=0., allowed=None, region=None):
    """Refined ear topology for both sides on the native neutral surface.

    The patch is the GNM ear disk, optionally widened by a band of near-rigid
    native skin (``allowed`` vertices within ``band_m`` of the ear root loop).
    Returns new-vertex base positions (subdivided, before shape changes),
    per-corner UVs, extended-index triangles, removed native triangle ids, the
    CSR position operator, per-side bookkeeping and the binding of the base.
    """
    full, triangles, triangle_uvs = np.asarray(full, float), np.asarray(triangles), np.asarray(triangle_uvs, float)
    native = len(full)
    pair = quad_triangles(quads, triangles)
    if allowed is None:
        allowed = np.ones(native, bool)
    patches, ear_loops = band_patches(ear_mask, quads, full, triangles, allowed, band_m, region)
    weights, tris, tuvs, origin, kind, side_of, removed, sides, source = [], [], [], [], [], [], [], {}, []
    side_removed = {}
    pairs, tri_offset = [], 0
    offset = 0
    ear_quads = set(map(int, np.flatnonzero(np.asarray(ear_mask, bool)[quads].all(1))))
    for s, (name, _) in enumerate(SIDES):
        patch = patches[name]
        sub = subdivide(quads[patch], quad_corner_uvs(quads[patch], triangles, triangle_uvs, pair[patch]), native)
        n = sub['weights'].shape[0]
        t = sub['triangles'].copy()
        t[t >= native] += offset
        weights.append(sub['weights'])
        tris.append(t)
        tuvs.append(sub['triangle_uvs'])
        pairs.append(sub['quad_pairs']+tri_offset)
        tri_offset += len(t)
        origin.append(sub['origin_quad'])
        kind.append(sub['kind'])
        source.append(sub['source_native'])
        side_of.append(np.full(n, s, np.int8))
        removed.append(pair[patch].ravel())
        side_removed[name] = pair[patch].ravel()
        in_ear = np.array([int(q) in ear_quads for q in patch])
        sides[name] = dict(patch_quads=patch, boundary=sub['boundary'], interior_native=sub['interior_native'],
                           new=np.arange(offset, offset+n), boundary_loop=boundary_loop(quads[patch]),
                           ear_loop=ear_loops[name], new_in_band=~in_ear[sub['origin_quad']],
                           triangle_in_band=~in_ear[sub['triangle_face']],
                           ear_loop_new=offset+np.flatnonzero(np.isin(sub['source_native'], ear_loops[name])),
                           ear_loop_native=np.intersect1d(ear_loops[name], sub['boundary']))
        offset += n
    from scipy.sparse import vstack
    operator = vstack(weights).tocsr()
    base = operator@full
    ids = np.zeros((offset, 3), np.int64)
    bary = np.zeros((offset, 3))
    off = np.zeros((offset, 3))
    origin = np.concatenate(origin)
    for name, info in sides.items():
        k = info['new']
        ids[k], bary[k], off[k] = bind(full, triangles, pair, info['patch_quads'], base[k], origin[k])
    return dict(operator=operator, base=base, triangles=np.concatenate(tris), triangle_uvs=np.concatenate(tuvs),
                removed=np.sort(np.concatenate(removed)), origin_quad=origin, kind=np.concatenate(kind),
                side=np.concatenate(side_of), source_native=np.concatenate(source), sides=sides, bind_ids=ids, bind_weights=bary, bind_offsets=off,
                native_count=native, band_m=float(band_m), side_removed=side_removed,
                quad_pairs=np.concatenate(pairs))


def extended(full, new_positions):
    """Concatenate native and new vertices along the vertex axis."""
    full, new_positions = np.asarray(full), np.asarray(new_positions)
    return np.concatenate((full, new_positions), axis=-2)


def surface_triangles(triangles, removed, ear_triangles):
    """Full triangle list with native ear faces replaced by refined faces."""
    keep = np.ones(len(triangles), bool)
    keep[removed] = False
    return np.concatenate((np.asarray(triangles)[keep], ear_triangles)), keep


def oriented_area_ratio(reference, current, triangles):
    """Signed area of current relative to reference along reference normals."""
    r = reference[triangles]
    c = current[triangles]
    nr = np.cross(r[:, 1]-r[:, 0], r[:, 2]-r[:, 0])
    nc = np.cross(c[:, 1]-c[:, 0], c[:, 2]-c[:, 0])
    ar = np.linalg.norm(nr, axis=1)
    return (nc*nr).sum(1)/np.maximum(ar*ar, 1e-30)


def edge_ratio(reference, current, triangles):
    """Maximum edge-length ratio per triangle (current / reference)."""
    out = np.zeros(len(triangles))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        lr = np.linalg.norm(reference[triangles[:, a]]-reference[triangles[:, b]], axis=1)
        lc = np.linalg.norm(current[triangles[:, a]]-current[triangles[:, b]], axis=1)
        out = np.maximum(out, lc/np.maximum(lr, 1e-12))
    return out


def mesh_edges(triangles):
    e = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    return np.unique(np.sort(e, 1), axis=0)


def root_distance(positions, triangles, boundary, native_count):
    """Graph-geodesic distance (metres) of each new vertex from the native root loop."""
    from scipy.sparse.csgraph import dijkstra
    positions = np.asarray(positions, float)
    e = mesh_edges(triangles)
    used = np.unique(e)
    index = -np.ones(positions.shape[0], np.int64)
    index[used] = np.arange(len(used))
    length = np.linalg.norm(positions[e[:, 0]]-positions[e[:, 1]], axis=1)
    graph = csr_matrix((np.concatenate((length, length)), (index[np.concatenate((e[:, 0], e[:, 1]))],
                        index[np.concatenate((e[:, 1], e[:, 0]))])), shape=(len(used), len(used)))
    sources = index[np.asarray(boundary)]
    distance = dijkstra(graph, indices=sources[sources >= 0], min_only=True)
    out = np.full(positions.shape[0], np.inf)
    out[used] = distance
    return out[native_count:]


def ear_frame(full, loop, ear_points, sign):
    """Ear-local frame: origin at the root-loop centroid; columns (anterior, up, lateral normal)."""
    full, ear_points = np.asarray(full, float), np.asarray(ear_points, float)
    root = full[loop]
    origin = root.mean(0)
    centred = ear_points-ear_points.mean(0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][2]
    if normal[0]*sign < 0:
        normal = -normal
    up = np.array([0., 1, 0])-normal*normal[1]
    up /= np.linalg.norm(up)
    anterior = np.cross(up, normal)
    if anterior[2] < 0:
        anterior = -anterior
    # Hinge: principal direction of the root loop, oriented upward.
    hinge = np.linalg.svd(root-origin, full_matrices=False)[2][0]
    if hinge[1] < 0:
        hinge = -hinge
    return dict(origin=origin, axes=np.stack((anterior, up, normal), 1), hinge=hinge)


def rotation(axis, angle):
    axis = np.asarray(axis, float)/np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)


def global_deform(points, frame, feather, *, length=1., width=1., protrusion=0., flare=0., shift=(0, 0, 0),
                  pivot=None):
    """Feathered ear placement about the lobe: length/width stretch, hinge rotation, flare, shift.

    In the ear frame (anterior, up, lateral normal) the free ear is stretched
    from ``pivot`` (default: lowest ear point, i.e. the lobule) by ``length``
    along up and ``width`` along anterior; the lateral coordinate is kept, so
    concha depth is unchanged. It is then rotated about the root hinge by
    ``protrusion + flare*(t-0.5)`` radians, ``t`` being normalised height, so
    positive values move the free rim laterally away from the head and a
    positive flare tilts the upper ear outward. ``shift`` is in ear-frame metres.
    ``feather`` is 0 at the native root loop and 1 on the free ear.
    """
    points = np.asarray(points, float)
    origin, axes = frame['origin'], frame['axes']
    local = (points-origin)@axes
    if pivot is None:
        pivot = np.array([local[:, 0].mean(), local[:, 1].min(), 0.])
    span = max(np.ptp(local[:, 1]), 1e-9)
    t = (local[:, 1]-local[:, 1].min())/span
    stretched = local.copy()
    stretched[:, 0] = pivot[0]+(local[:, 0]-pivot[0])*width
    stretched[:, 1] = pivot[1]+(local[:, 1]-pivot[1])*length
    moved = stretched@axes.T+origin
    hinge = frame['hinge']
    side = np.sign(np.cross(hinge, axes[:, 0])@axes[:, 2]) or 1.
    angle = protrusion+flare*(t-.5)
    theta = (-side*angle)[:, None]
    v = moved-origin
    k = hinge/np.linalg.norm(hinge)
    turned = (v*np.cos(theta)+np.cross(k, v)*np.sin(theta)
              + k*(v@k)[:, None]*(1-np.cos(theta)))+origin
    turned = turned+axes@np.asarray(shift, float)
    w = np.asarray(feather, float)[:, None]
    return points*(1-w)+turned*w


def cotangent_weights(points, triangles):
    """Symmetric cotangent edge weights as a CSR matrix (clamped non-negative)."""
    n = len(points)
    rows, cols, vals = [], [], []
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        a, b, c = points[triangles[:, i]], points[triangles[:, j]], points[triangles[:, k]]
        u, v = a-c, b-c
        cot = (u*v).sum(1)/np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-30)
        rows += [triangles[:, i], triangles[:, j]]
        cols += [triangles[:, j], triangles[:, i]]
        vals += [cot/2, cot/2]
    w = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    w.data = np.maximum(w.data, 1e-4*np.abs(w.data).mean())
    return w


def arap(rest, triangles, fixed, targets, initial=None, iterations=20, project=None):
    """As-rigid-as-possible deformation of a local mesh with fixed vertices.

    ``rest`` (n,3) local rest positions, ``fixed`` boolean (n,), ``targets``
    (n,3) used for fixed vertices. Returns deformed positions. Free vertices
    are solved with a cotangent-weighted local/global iteration. ``project``
    (optional) maps positions after each global step, e.g. to keep a subset on
    a reference surface (alternating projection, not an exact constraint).
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import factorized
    rest = np.asarray(rest, float)
    n = len(rest)
    w = cotangent_weights(rest, triangles).tocoo()
    i, j, wij = w.row, w.col, w.data
    lap = (diags(np.asarray(csr_matrix((wij, (i, j)), shape=(n, n)).sum(1)).ravel())
           - csr_matrix((wij, (i, j)), shape=(n, n))).tocsr()
    free = ~np.asarray(fixed, bool)
    solve = factorized(lap[free][:, free].tocsc())
    cross = lap[free][:, ~free]
    x = np.array(initial if initial is not None else rest, float)
    x[~free] = targets[~free]
    erest = rest[i]-rest[j]
    for _ in range(iterations):
        ecur = x[i]-x[j]
        cov = np.zeros((n, 3, 3))
        np.add.at(cov, i, wij[:, None, None]*erest[:, :, None]*ecur[:, None, :])
        u, _, vt = np.linalg.svd(cov)
        r = np.einsum('nji,nkj->nik', vt, u)
        bad = np.linalg.det(r) < 0
        if bad.any():
            u2 = u[bad].copy()
            u2[:, :, -1] *= -1
            r[bad] = np.einsum('nji,nkj->nik', vt[bad], u2)
        rhs_e = .5*wij[:, None]*np.einsum('nij,nj->ni', r[i]+r[j], erest)
        b = np.zeros((n, 3))
        np.add.at(b, i, rhs_e)
        rhs = b[free]-cross@x[~free]
        x[free] = np.stack([solve(rhs[:, c]) for c in range(3)], 1)
        if project is not None:
            x = project(x)
    return x


def arap_place(full, topo, side, transformed, handle, iterations=20):
    """Solve one side's refined ear: root loop native, ``handle`` new vertices at ``transformed``."""
    native = topo['native_count']
    info = topo['sides'][side]
    k = info['new']
    side_tri = topo['triangles'][np.isin(topo['triangles']-native, k).any(1)]
    ext_ids = np.unique(side_tri)
    remap = -np.ones(native+len(topo['base']), np.int64)
    remap[ext_ids] = np.arange(len(ext_ids))
    rest = np.concatenate((full, topo['base']))[ext_ids]
    target = rest.copy()
    fixed = ext_ids < native
    is_new = ext_ids >= native
    local_new = ext_ids[is_new]-native
    sel = np.zeros(len(ext_ids), bool)
    position = {int(v): i for i, v in enumerate(k)}
    idx = np.array([position[int(v)] for v in local_new])
    h = np.asarray(handle)[k[idx]]
    sel[np.flatnonzero(is_new)[h]] = True
    target[np.flatnonzero(is_new)[h]] = np.asarray(transformed)[k[idx]][h]
    fixed = fixed | sel
    initial = rest.copy()
    initial[np.flatnonzero(is_new)] = np.asarray(transformed)[k[idx]]
    solved = arap(rest, remap[side_tri], fixed, target, initial, iterations)
    out = np.asarray(transformed).copy()
    out[k[idx]] = solved[np.flatnonzero(is_new)]
    return out


class SurfaceProjector:
    """Closest-point projection onto a fixed triangle surface (k-nearest candidates)."""

    def __init__(self, vertices, triangles, k=12):
        from scipy.spatial import cKDTree
        self.corners = np.asarray(vertices, float)[np.asarray(triangles)]
        self.tree = cKDTree(self.corners.mean(1))
        self.k = min(k, len(self.corners))

    def __call__(self, points):
        points = np.asarray(points, float)
        _, near = self.tree.query(points, self.k)
        near = np.atleast_2d(near)
        best = np.full(len(points), np.inf)
        out = points.copy()
        for column in range(near.shape[1]):
            corners = self.corners[near[:, column]]
            bary = closest_barycentric(points, corners)
            q = (corners*bary[:, :, None]).sum(1)
            d = np.linalg.norm(q-points, axis=1)
            take = d < best
            best[take] = d[take]
            out[take] = q[take]
        return out


def place_side(full, topo, side, transformed, handle, surface, head_triangles, *, iterations=30, rest_new=None,
               surface_vertices=None):
    """ARAP placement of one side of a widened patch.

    Native outer-boundary vertices stay fixed; ``handle`` new vertices follow
    ``transformed``; new vertices flagged in ``surface`` (band skin and the old
    ear-root loop) slide on the native neutral head surface ``head_triangles``
    (alternating projection; ``surface_vertices`` overrides the vertex array the
    triangles index, e.g. the smooth subdivided base); all others are free. ``rest_new`` replaces the
    subdivided base as the ARAP rest shape of new vertices (e.g. with relief).
    """
    native = topo['native_count']
    k = topo['sides'][side]['new']
    side_tri = topo['triangles'][np.isin(topo['triangles']-native, k).any(1)]
    ext_ids = np.unique(side_tri)
    remap = -np.ones(native+len(topo['base']), np.int64)
    remap[ext_ids] = np.arange(len(ext_ids))
    rest = np.concatenate((full, topo['base'] if rest_new is None else rest_new))[ext_ids]
    is_new = ext_ids >= native
    new_global = ext_ids[is_new]-native
    transformed = np.asarray(transformed)
    local_new = np.flatnonzero(is_new)
    hn = np.asarray(handle)[new_global]
    fixed = ~is_new
    fixed[local_new[hn]] = True
    target = rest.copy()
    target[local_new[hn]] = transformed[new_global[hn]]
    project_ids = local_new[np.asarray(surface)[new_global] & ~hn]
    projector = SurfaceProjector(full if surface_vertices is None else surface_vertices, head_triangles)

    def project(x):
        x = x.copy()
        x[project_ids] = projector(x[project_ids])
        return x
    initial = rest.copy()
    initial[local_new] = transformed[new_global]
    solved = arap(rest, remap[side_tri], fixed, target, project(initial), iterations, project)
    out = transformed.copy()
    out[new_global] = solved[local_new]
    return out


# Authored anatomical relief in normalised lateral-view ear coordinates:
# x = 0 posterior rim .. 1 anterior root, y = 0 lobule bottom .. 1 helix top.
# Heights are millimetres along the surface normal of laterally visible ear
# skin (positive outward). These are generic anatomical priors, not measured
# subject geometry.
RELIEF_DEFAULTS = dict(
    antihelix=1.6, antihelix_width=.045,
    superior_crus=1.3, inferior_crus=1.2, crus_width=.035,
    fossa=-1.2, fossa_radius=.07,
    scapha=-1.3, scapha_width=.04,
    helix_rim=.9, helix_width=.035,
    helix_crus=1.0, helix_crus_width=.03,
    tragus=1.8, tragus_radius=.06,
    antitragus=1.5, antitragus_radius=.055,
    concha=-1.5, concha_radius=(.17, .14),
    intertragic=-.8, intertragic_radius=.045,
    lobule=.0,
)
RELIEF_CURVES = dict(
    antihelix=((.42, .29), (.37, .40), (.355, .52), (.38, .635)),
    superior_crus=((.38, .635), (.40, .75), (.45, .86)),
    inferior_crus=((.38, .635), (.52, .675), (.67, .685)),
    scapha=((.22, .30), (.20, .44), (.22, .60), (.28, .75), (.37, .875)),
    helix_crus=((.97, .58), (.84, .565), (.72, .55)),
)
RELIEF_POINTS = dict(fossa=(.50, .765), tragus=(.885, .43), antitragus=(.44, .29), concha=(.63, .46),
                     intertragic=(.68, .31))


def _polyline_distance(xy, curve):
    curve = np.asarray(curve, float)
    best = np.full(len(xy), np.inf)
    for a, b in zip(curve[:-1], curve[1:]):
        ab = b-a
        t = np.clip(((xy-a)@ab)/max(ab@ab, 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(xy-(a+t[:, None]*ab), axis=1))
    return best


def relief_height(xy, params=None):
    """Relief height (mm) at normalised ear coordinates ``xy`` (N,2).

    Ridges are combined as a union (maximum) and grooves as an intersection
    (minimum), so meeting curves such as the antihelix fork do not add up.
    """
    p = dict(RELIEF_DEFAULTS, **(params or {}))
    xy = np.asarray(xy, float)
    gauss = lambda d, w: np.exp(-(d/w)**2)
    terms = [p['antihelix']*gauss(_polyline_distance(xy, RELIEF_CURVES['antihelix']), p['antihelix_width']),
             p['superior_crus']*gauss(_polyline_distance(xy, RELIEF_CURVES['superior_crus']), p['crus_width']),
             p['inferior_crus']*gauss(_polyline_distance(xy, RELIEF_CURVES['inferior_crus']), p['crus_width']),
             p['scapha']*gauss(_polyline_distance(xy, RELIEF_CURVES['scapha']), p['scapha_width']),
             p['helix_crus']*gauss(_polyline_distance(xy, RELIEF_CURVES['helix_crus']), p['helix_crus_width'])]
    for name in ('fossa', 'tragus', 'antitragus', 'intertragic'):
        terms.append(p[name]*gauss(np.linalg.norm(xy-RELIEF_POINTS[name], axis=1), p[name+'_radius']))
    radius = np.asarray(p['concha_radius'], float)
    terms.append(p['concha']*gauss(np.linalg.norm((xy-RELIEF_POINTS['concha'])/radius, axis=1), 1.))
    terms = np.stack(terms)
    return np.clip(terms, 0, None).max(0)+np.clip(terms, None, 0).min(0)


def lateral_coordinates(points, normals, frame, triangles_local, *, resolution=4e-4, tolerance=6e-4):
    """Normalised lateral-view coordinates and visibility for one ear.

    ``points``/``normals`` are the side's ear vertices; ``triangles_local``
    index them. Returns (xy (N,2), visible weight (N,), outline distance (N,)
    in normalised width units). Visibility is an orthographic z-buffer test
    from the lateral normal direction, softened by facing.
    """
    from scipy.ndimage import distance_transform_edt, binary_fill_holes
    local = (np.asarray(points, float)-frame['origin'])@frame['axes']
    a, u, n = local.T
    lo = np.array([a.min(), u.min()])
    size = np.ceil((np.array([a.max(), u.max()])-lo)/resolution).astype(int)+3
    zbuf = np.full(size[::-1], -np.inf)
    px = (local[:, :2]-lo)/resolution+1
    for t in np.asarray(triangles_local):
        q = px[t]
        x0, y0 = np.floor(q.min(0)).astype(int)
        x1, y1 = np.ceil(q.max(0)).astype(int)
        A, B, C = q
        det = (B[0]-A[0])*(C[1]-A[1])-(B[1]-A[1])*(C[0]-A[0])
        if abs(det) < 1e-12:
            continue
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        p = np.stack((xx+.5, yy+.5), -1)-A
        v = (p[..., 0]*(C[1]-A[1])-p[..., 1]*(C[0]-A[0]))/det
        w = ((B[0]-A[0])*p[..., 1]-(B[1]-A[1])*p[..., 0])/det
        inside = (v >= -1e-6) & (w >= -1e-6) & (v+w <= 1+1e-6)
        z = n[t[0]]*(1-v-w)+n[t[1]]*v+n[t[2]]*w
        region = zbuf[y0:y1+1, x0:x1+1]
        take = inside & (z > region)
        region[take] = z[take]
    ix = np.clip(px.astype(int), 0, np.array(size)-1)
    top = zbuf[ix[:, 1], ix[:, 0]]
    visible = (n >= top-tolerance).astype(float)
    facing = np.clip((np.asarray(normals)@frame['axes'][:, 2]-.05)/.35, 0, 1)
    mask = binary_fill_holes(np.isfinite(zbuf))
    outline = distance_transform_edt(mask)*resolution
    width = max(np.ptp(a), 1e-9)
    xy = np.stack(((a-a.min())/width, (u-u.min())/max(np.ptp(u), 1e-9)), 1)
    return xy, visible*facing, outline[ix[:, 1], ix[:, 0]]/width


def smooth_scalar(values, triangles, count, iterations=3):
    """Uniform Laplacian smoothing of a per-vertex scalar on a local mesh."""
    e = mesh_edges(np.asarray(triangles))
    adjacency = csr_matrix((np.ones(2*len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(count, count))
    degree = np.maximum(np.asarray(adjacency.sum(1)).ravel(), 1)
    v = np.asarray(values, float).copy()
    for _ in range(iterations):
        v = .5*v+.5*(adjacency@v)/degree
    return v


def rebind(full, triangles, topo, positions):
    """Re-anchor new vertices to the nearest replaced native triangle of their side.

    Keeps offsets small after large placement changes; exact for rigid anchor
    motion like ``bind``.
    """
    full, positions = np.asarray(full, float), np.asarray(positions, float)
    ids = np.zeros((len(positions), 3), np.int64)
    weights = np.zeros((len(positions), 3))
    offsets = np.zeros((len(positions), 3))
    pair_side = {}
    for name, info in topo['sides'].items():
        k = info['new']
        cage = np.asarray(triangles)[topo['removed'][np.isin(topo['removed'], topo['side_removed'][name])]]
        from scipy.spatial import cKDTree
        corners = full[cage]
        tree = cKDTree(corners.mean(1))
        _, near = tree.query(positions[k], min(16, len(cage)))
        best = np.full(len(k), np.inf)
        bt = np.zeros(len(k), np.int64)
        bb = np.zeros((len(k), 3))
        for column in range(near.shape[1]):
            c = corners[near[:, column]]
            bary = closest_barycentric(positions[k], c)
            dist = np.linalg.norm((c*bary[:, :, None]).sum(1)-positions[k], axis=1)
            take = dist < best
            best[take], bt[take], bb[take] = dist[take], near[take, column], bary[take]
        ids[k] = cage[bt]
        weights[k] = bb
        c = full[ids[k]]
        offsets[k] = np.einsum('vji,vj->vi', attachment_frames(c), positions[k]-(c*bb[:, :, None]).sum(1))
    return ids, weights, offsets


def apply_relief(full, triangles, topo, positions, band, root_distance_m, frames, params=None, *,
                 root_feather_m=.004):
    """Displace laterally visible ear skin by the authored anatomical relief.

    Displacement follows smoothed vertex normals and is feathered to zero at the
    old ear root; band skin is unchanged. Returns (positions, heights_mm).
    """
    from ..rig.common import vertex_normals
    native = topo['native_count']
    p = dict(RELIEF_DEFAULTS, **(params or {}))
    ext = extended(full, positions)
    tri, _ = surface_triangles(triangles, topo['removed'], topo['triangles'])
    normals = vertex_normals(ext, tri)[native:]
    out = np.asarray(positions, float).copy()
    heights = np.zeros(len(positions))
    for name, _ in SIDES:
        k = topo['sides'][name]['new']
        ek = k[~np.asarray(band)[k]]
        t = topo['triangles']
        local = t[np.isin(t-native, ek).all(1)]-native
        remap = -np.ones(len(positions), np.int64)
        remap[ek] = np.arange(len(ek))
        lt = remap[local]
        xy, visible, outline = lateral_coordinates(out[ek], normals[ek], frames[name], lt)
        h = relief_height(xy, p)
        rim = smoothstep((.80-xy[:, 0])/.25)*smoothstep((xy[:, 1]-.18)/.12)
        h += p['helix_rim']*np.exp(-((outline-p['helix_width'])/(p['helix_width']*.7))**2)*rim
        weight = visible*smoothstep(np.asarray(root_distance_m)[ek]/root_feather_m)
        delta = smooth_scalar(h*weight, lt, len(ek), 4)
        smooth = np.stack([smooth_scalar(normals[ek][:, c], lt, len(ek), 8) for c in range(3)], 1)
        smooth /= np.linalg.norm(smooth, axis=1, keepdims=True)
        out[ek] = out[ek]+smooth*delta[:, None]*1e-3
        heights[ek] = delta
    return out, heights


def candidate_arrays(full_neutral, full_captured, topo, ids, weights, offsets, band, triangles=None,
                     triangle_uvs=None):
    """``geometry.npz`` fields describing refined ears without altering native arrays.

    ``ear_dense_triangles`` index the extended vertex list (native, then new);
    ``ear_removed_triangles`` index ``full_triangles``. Dense positions are
    re-evaluated from the binding on both native rest and captured surfaces.
    """
    neutral = evaluate(full_neutral, ids, weights, offsets)
    captured = evaluate(np.asarray(full_captured)[0], ids, weights, offsets)
    op = topo['operator']
    return dict(ear_dense_neutral=neutral.astype(np.float32), ear_dense_captured=captured.astype(np.float32),
                ear_dense_triangles=topo['triangles'].astype(np.int32),
                ear_dense_triangle_uvs=topo['triangle_uvs'].astype(np.float32),
                ear_bind_ids=ids.astype(np.int32), ear_bind_weights=weights.astype(np.float64),
                ear_bind_offsets=offsets.astype(np.float64),
                ear_removed_triangles=topo['removed'].astype(np.int32),
                ear_dense_quad_pairs=topo['quad_pairs'].astype(np.int32),
                ear_dense_side=topo['side'].astype(np.int8), ear_dense_band=np.asarray(band, bool),
                ear_subdivision_data=op.data.astype(np.float64), ear_subdivision_indices=op.indices.astype(np.int32),
                ear_subdivision_indptr=op.indptr.astype(np.int64),
                **({} if triangles is None else dict(zip(('ear_lod_triangles', 'ear_lod_triangle_uvs'),
                    (a.astype(t) for a, t in zip(coarse_lod(topo, triangles, triangle_uvs), (np.int32, np.float32)))))))


def coarse_lod(topo, triangles, triangle_uvs):
    """Level-0 refined ear: native replaced triangles re-indexed to the shaped vertex points.

    Same triangle count and per-corner UVs as the native faces; interior native
    vertices are replaced by their (bound) Catmull-Clark vertex points, so the
    fitted placement and coarse relief survive at native resolution.
    """
    native = topo['native_count']
    lookup = {int(s): native+i for i, s in enumerate(topo['source_native']) if s >= 0}
    boundary = set(int(v) for info in topo['sides'].values() for v in info['boundary'])
    removed = np.asarray(topo['removed'])
    out = np.asarray(triangles)[removed].copy()
    for idx, v in np.ndenumerate(out):
        v = int(v)
        out[idx] = v if v in boundary else lookup[v]
    return out, np.asarray(triangle_uvs)[removed]
