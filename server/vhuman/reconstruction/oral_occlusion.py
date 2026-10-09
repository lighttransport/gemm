"""Dynamic mouth-interior occlusion from the lip opening.

Light reaches teeth, gums, tongue and the oral cavity almost only through the
opening between the lips. The shipped runtime model (browser shader) is
"morphable" ambient occlusion: per-vertex visibility affine in the opening's
area and height (``aperture_frame``), fitted to GPU ray-traced cosine-weighted
escape visibility, plus a soft analytic aperture shadow for directional lights
(``aperture_shadow``) gated by the vertex visibility. The point-to-polygon form
factor (Lambert contour integral) is kept as a reference feature; on this
anatomy it did not predict visibility because the lip rim lies inside the lips
and teeth protrude past it. These are approximations of escape visibility, not
path-traced lighting.
"""
import numpy as np

# MediaPipe inner-lip loop (upper 78..308, lower 324..95), closed.
INNER_LIP = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)


def contour(vertices, ids, weights):
    """Inner-lip contour points from barycentric attachments; vertices (...,N,3)."""
    return (np.asarray(vertices)[..., ids, :]*np.asarray(weights)[..., None]).sum(-2)


def polygon_form_factor(points, normals, polygon, *, eps=1e-9):
    """Cosine-weighted visibility of a closed polygon from points (fan triangulation).

    Fan triangles from the polygon centroid are clipped against each point's
    tangent plane by scaling with the fraction of their corners above it (a
    conservative horizon treatment). Returns values in [0, 1].
    """
    p = np.asarray(points, float)[:, None, :]
    n = np.asarray(normals, float)
    poly = np.asarray(polygon, float)
    c = poly.mean(0)
    a, b = poly, np.roll(poly, -1, axis=0)
    tri = np.stack((np.broadcast_to(c, a.shape), a, b), 1)            # (E,3,3)
    r = tri[None]-p[:, :, None, :]                                       # (P,E,3,3)
    dist = np.linalg.norm(r, axis=-1, keepdims=True)
    u = r/np.maximum(dist, eps)
    above = ((r*n[:, None, None, :]).sum(-1) > 0).mean(-1)               # (P,E)
    ff = np.zeros(above.shape)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        cross = np.cross(u[:, :, i], u[:, :, j])
        length = np.linalg.norm(cross, axis=-1)
        theta = np.arccos(np.clip((u[:, :, i]*u[:, :, j]).sum(-1), -1, 1))
        ff += theta*(cross*n[:, None, :]).sum(-1)/np.maximum(length, eps)
    total = np.abs(ff)/(2*np.pi)*above
    return np.clip(total.sum(1), 0, 1)


def fit_static_scale(visibility, form_factor, *, floor=.02):
    """Per-vertex least-squares scale s with visibility ~ s * F over training poses."""
    v, f = np.asarray(visibility, float), np.asarray(form_factor, float)
    s = (v*f).sum(0)/np.maximum((f*f).sum(0), 1e-9)
    weak = (f > floor).sum(0) < 2
    s[weak] = np.clip(v.mean(0)[weak]/max(float(f.mean()), 1e-6), 0, 1)
    return np.clip(s, 0, 1)


def fit_affine(visibility, features, *, ridge=1e-3):
    """Per-vertex ridge regression visibility ~ [1, features] @ w (shared design)."""
    x = np.column_stack((np.ones(len(features)), np.asarray(features, float)))
    gram = x.T@x+ridge*np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T@np.asarray(visibility, float))   # (K+1, V)


def predict_affine(weights, features):
    x = np.column_stack((np.ones(len(features)), np.asarray(features, float)))
    return np.clip(x@weights, 0, 1)


def aperture_frame(polygon):
    """Best-fit plane of the rim (centroid, in-plane axes, normal facing +Z), 2D outline, area, height."""
    poly = np.asarray(polygon, float)
    c = poly.mean(0)
    _, _, vt = np.linalg.svd(poly-c, full_matrices=False)
    n = vt[2] if vt[2][2] >= 0 else -vt[2]
    ref = np.array([1., 0, 0]) if abs(n[0]) < .9 else np.array([0, 1., 0])
    ux = np.cross(n, ref)
    ux /= np.linalg.norm(ux)
    uy = np.cross(n, ux)
    q = np.stack(((poly-c)@ux, (poly-c)@uy), 1)
    area = .5*abs(np.dot(q[:, 0], np.roll(q[:, 1], -1))-np.dot(q[:, 1], np.roll(q[:, 0], -1)))
    qc = q-q.mean(0)
    minor = np.linalg.svd(qc, full_matrices=False)[2][1]
    return dict(c=c, n=n, ux=ux, uy=uy, outline=q, area=area, height=float(np.ptp(qc@minor)))


def _signed_distance(points, outline):
    a, b = outline, np.roll(outline, 1, axis=0)
    x, y = points[:, :1], points[:, 1:]
    cross = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < (b[:, 0]-a[:, 0])*(y-a[:, 1])/(b[:, 1]-a[:, 1]+1e-12)+a[:, 0])
    inside = cross.sum(1) % 2 == 1
    e = b-a
    t = np.clip(((points[:, None]-a)*e).sum(-1)/np.maximum((e*e).sum(-1), 1e-12), 0, 1)
    d = np.linalg.norm(points[:, None]-a-t[..., None]*e, axis=-1).min(1)
    return np.where(inside, d, -d)


def aperture_shadow(points, light, frame, *, angular_radius=.12, min_penumbra=.0008):
    """Soft visibility of a directional light through the rim aperture (matches the browser shader)."""
    p = np.asarray(points, float)
    l = np.asarray(light, float)/np.linalg.norm(light)
    denom = float(l@frame['n'])
    if denom <= 1e-4:
        return np.zeros(len(p))
    t = ((frame['c']-p)@frame['n'])/denom
    h = p+t[:, None]*l-frame['c']
    sd = _signed_distance(np.stack((h@frame['ux'], h@frame['uy']), 1), frame['outline'])
    w = min_penumbra+t*np.tan(angular_radius)
    s = np.clip((sd+w)/(2*w), 0, 1)
    return np.where(t <= 0, 1., s*s*(3-2*s))
