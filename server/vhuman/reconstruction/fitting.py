"""Robust staged PCA portrait fit with neutral identity and per-view expression.

Original least-squares objective; camera translation is nuisance, not metric
scale recovery. Single-view identity/expression ambiguity is reported explicitly.
"""
import hashlib
from pathlib import Path
import numpy as np
from .reference import Camera


def topology_hash(triangles):
    import hashlib
    return hashlib.sha256(np.asarray(triangles, '<i4').tobytes()).hexdigest()


def safe_geometry(base, candidate, triangles):
    a, b = base[triangles], candidate[triangles]
    n0 = np.cross(a[:, 1]-a[:, 0], a[:, 2]-a[:, 0])
    n1 = np.cross(b[:, 1]-b[:, 0], b[:, 2]-b[:, 0])
    area0, area1 = np.linalg.norm(n0, axis=1), np.linalg.norm(n1, axis=1)
    active = area0 > 1e-12
    return bool(np.isfinite(candidate).all() and (area1[active] > area0[active]*.1).all()
                and (np.sum(n0[active]*n1[active], axis=1) > 0).all())


def anchor_indices(vertices, camera, anchors, anatomical=None):
    """Explicit ids override fixed anatomy; unknown names use a provisional fallback.

    Known names never select correspondence from the portrait's projected pixels.
    A centroid attachment may span several vertices (for example eyelid centres).
    """
    xy, z = camera.project(vertices)
    out = []
    for name, a in anchors.items():
        if 'vertex' in a or 'vertices' in a:
            i = np.asarray(a.get('vertices', [a.get('vertex')]),int)
            if i.ndim!=1 or not 1<=len(i)<=32 or (i<0).any() or (i>=len(vertices)).any():
                raise ValueError('anchor vertex outside model')
        elif anatomical and name in anatomical:
            i = np.asarray(anatomical[name],int)
        else:
            d = np.linalg.norm(xy - a['xy'], axis=1)
            valid = np.flatnonzero((d < d.min()+2) & (z > 0))
            i = np.array([int(valid[np.argmin(z[valid])]) if len(valid) else int(np.argmin(d))])
        out.append((name, i, np.asarray(a['xy'], float), float(a.get('weight', 1))))
    return out


def initialize(source, subject):
    """Align model anatomy to analytic eyes before weak surface fitting.

Eye positions anchor the metric frame; generated hair/neck never determine
head height or move the model's sockets. This is a neutral initialization.
"""
    if source.eye_centers is None:
        raise ValueError('face source lacks anatomical eye centres')
    source_eyes = np.asarray(source.eye_centers)
    target = np.array([e['center'] for e in subject.eyes])
    def basis(eyes):
        x = eyes[1]-eyes[0]
        x /= np.linalg.norm(x)
        z = np.cross(x,[0,1,0]);z /= np.linalg.norm(z)
        y = np.cross(z,x)
        return np.column_stack((x,y,z))
    rotation = basis(target) @ basis(source_eyes).T
    scale = np.linalg.norm(target[1]-target[0])/np.linalg.norm(source_eyes[1]-source_eyes[0])
    p = scale*(source.vertices-source_eyes.mean(0)) @ rotation.T+target.mean(0)
    return p, float(scale), rotation, np.zeros(0)


def fit(source, initial, views, *, scale=1., rotation=None, max_modes=24, iterations=80, surface_prior=None):
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from scipy.spatial import cKDTree
    rotation = np.eye(3) if rotation is None else rotation
    identity = source.identity_basis
    expr = source.expression_basis
    expression_names = getattr(source,'expression_names',None)
    authored = False
    if expr is None and getattr(source,'authored_shapes',{}):
        expression_names = sorted(source.authored_shapes)
        expr = np.stack([source.authored_shapes[name] for name in expression_names])
        authored = True
    # Compact optimization basis; full export topology remains unchanged.
    ib = np.zeros((0, len(initial), 3)) if identity is None else scale*identity[:max_modes] @ rotation.T
    expression_modes = np.zeros(0, int) if expr is None else np.argsort(-np.mean(expr**2, axis=(1,2)))[:12]
    eb = np.zeros((0, len(initial), 3)) if expr is None else scale*expr[expression_modes] @ rotation.T
    ni, ne, nv = len(ib), len(eb), len(views)
    cams = [Camera.from_dict(v['camera']) for v in views]
    from .correspondence import attachments, occlusion_weight
    anatomical = attachments(source)
    anchors = [anchor_indices(initial, c, v.get('anchors', {}), anatomical) for c, v in zip(cams, views)]
    anchors = [[(name,ids,xy,w*occlusion_weight(v,xy)) for name,ids,xy,w in rows] for v,rows in zip(views,anchors)]
    if sum(sum(a[3]>0 for a in rows) for rows in anchors) < 4:
        raise ValueError('geometry fitting needs at least four weighted anchors')
    prior_ids = np.arange(0,len(initial),max(1,len(initial)//500))
    prior_target, prior_normal, prior_weight = None, None, None
    if surface_prior is not None:
        from ..rig.common import vertex_normals
        surface, normals = surface_prior
        dist, nearest = cKDTree(surface).query(initial[prior_ids])
        prior_target, prior_normal = surface[nearest], normals[nearest]
        agreement = (vertex_normals(initial,source.triangles)[prior_ids]*prior_normal).sum(1)
        prior_weight = (agreement>.75)*(dist<.01)*.2
    directions = np.column_stack((np.cos(np.linspace(0,2*np.pi,32,endpoint=False)),
                                  np.sin(np.linspace(0,2*np.pi,32,endpoint=False))))
    from .silhouette import prepare
    silhouettes = [prepare(initial,source.triangles,c,v) for c,v in zip(cams,views)]
    eye_guard = np.linalg.norm(initial,axis=1)<0  # no implicit anatomical guess
    if source.eye_centers is not None:
        # Initial model eye centres have already been transformed to H's origin.
        eye_h = scale*(source.eye_centers-source.eye_centers.mean(0))@rotation.T
        eye_guard = np.min(np.linalg.norm(initial[:,None]-eye_h[None],axis=2),axis=1)<.022
    # Pose translation and expression are independent per view; shared identity.
    count = ni + nv*(6+ne)
    x = np.zeros(count)
    def split(x, j):
        off = ni+j*(6+ne)
        return x[:ni], x[off:off+3], x[off+6:off+6+ne], x[off+3:off+6]
    def neutral(x):
        return initial + np.einsum('i,ivc->vc', x[:ni], ib)
    def objective(x):
        p = neutral(x)
        residual = []
        for j, c in enumerate(cams):
            _, trans, expression, pose_rotation = split(x, j)
            attachments_p = np.array([p[a[1]].mean(0) for a in anchors[j]])
            attachments_e = np.stack([eb[:,a[1]].mean(1) for a in anchors[j]],axis=1)
            moving = (attachments_p + np.einsum('i,ivc->vc', expression, attachments_e)) @ Rotation.from_rotvec(pose_rotation).as_matrix().T + trans
            pixels, depth = c.project(moving)
            target = np.array([a[2] for a in anchors[j]])
            weights = np.sqrt([a[3] for a in anchors[j]])[:, None]
            residual.extend(((pixels-target)*weights/3).reshape(-1))
            residual.extend(np.minimum(depth-.02, 0)*1000)
            residual.extend(expression*3)  # strong nuisance expression prior
            residual.extend(trans*100)
            residual.extend(pose_rotation*.5)
            if silhouettes[j] is not None:
                from .silhouette import residual as silhouette_residual
                posed = (p + np.einsum('i,ivc->vc',expression,eb)) @ Rotation.from_rotvec(pose_rotation).as_matrix().T + trans
                residual.extend(silhouette_residual(posed,c,silhouettes[j]))
            elif view_silhouette := views[j].get('silhouette'):
                posed = (p + np.einsum('i,ivc->vc',expression,eb)) @ Rotation.from_rotvec(pose_rotation).as_matrix().T + trans
                projected,depth_all = c.project(posed)
                support = (projected[depth_all>.02] @ directions.T).max(0)
                target_support = (np.asarray(view_silhouette) @ directions.T).max(0)
                residual.extend((support-target_support)*.05)
        residual.extend(x[:ni]*.6)
        if prior_target is not None:
            plane = ((p[prior_ids]-prior_target)*prior_normal).sum(1)
            residual.extend(np.clip(plane,-.005,.005)*prior_weight*100)
        residual.extend(((p[eye_guard]-initial[eye_guard])*200).reshape(-1))
        # Weak surface prior: initialization is not a hard projection target.
        sample = np.arange(0, len(initial), max(1, len(initial)//500))
        residual.extend(((p[sample]-initial[sample])*50).reshape(-1))
        return np.asarray(residual)
    lo, hi = np.full(count, -2.), np.full(count, 2.)
    for j in range(nv):
        off = ni+j*(6+ne)
        lo[off:off+3], hi[off:off+3] = -.03, .03
        lo[off+3:off+6], hi[off+3:off+6] = -.8, .8
        if authored:
            lo[off+6:off+6+ne],hi[off+6:off+6+ne] = 0,1
    # Camera/pose first, then shared identity and expression.
    pose_ids = np.array([ni+j*(6+ne)+k for j in range(nv) for k in range(6)], int)
    def pose_fun(y):
        xx = x.copy()
        xx[pose_ids] = y
        return objective(xx)
    pose = least_squares(pose_fun, x[pose_ids], bounds=(lo[pose_ids], hi[pose_ids]),
                         loss='soft_l1', max_nfev=iterations)
    x[pose_ids] = pose.x
    before = float(np.mean(objective(np.zeros(count))**2))
    solve = least_squares(objective, x, bounds=(lo, hi), loss='soft_l1', max_nfev=iterations)
    x = solve.x
    candidate = neutral(x)
    factor = 1.
    while not safe_geometry(initial, candidate, source.triangles) and factor > 1/128:
        factor *= .5
        x[:ni] *= .5
        candidate = neutral(x)
    if not safe_geometry(initial, candidate, source.triangles):
        raise ValueError('unsafe fitted topology')
    captured = []
    expression_steps = []
    for j in range(nv):
        _,trans,expression,pose_rotation = split(x,j)
        step = 1.
        moving = candidate+np.einsum('i,ivc->vc',expression,eb)
        while not safe_geometry(candidate,moving,source.triangles) and step>1/128:
            step *= .5
            expression *= .5
            moving = candidate+np.einsum('i,ivc->vc',expression,eb)
        if not safe_geometry(candidate,moving,source.triangles):
            expression[:] = 0
            moving = candidate.copy()
            step = 0
        captured.append(moving)
        expression_steps.append(step)
    after = float(np.mean(objective(x)**2))
    for j in range(nv):
        _,trans,expression,pose_rotation = split(x,j)
        rot = Rotation.from_rotvec(pose_rotation).as_matrix()
        cams[j].origin = rot.T @ (cams[j].origin-trans)
        cams[j].rotation = cams[j].rotation @ rot
    report = dict(objective_before=before, objective_after=after,
                  anatomical_anchors=anatomical,
                  anatomical_anchors_sha256=hashlib.sha256((Path(__file__).parent/'data/anatomical_anchors.json').read_bytes()).hexdigest() if anatomical else None,
                  identity_coefficients=x[:ni].tolist(), expression_coefficients=[split(x,j)[2].tolist() for j in range(nv)],
                  expression_modes=expression_modes.tolist(), expression_steps=expression_steps,
                  expression_labels=[expression_names[i] for i in expression_modes] if expression_names else None,
                  pose_rotations=[split(x,j)[3].tolist() for j in range(nv)],
                  anchors=[{a[0]: a[1].tolist() for a in v} for v in anchors], safe_step=factor,
                  correspondence='topology-pinned anatomy with explicit annotation overrides' if anatomical else 'provisional projected nearest vertex',
                  silhouettes=['mask distance field' if s is not None else 'convex support or absent' for s in silhouettes],
                  converged=bool(solve.success), evaluations=solve.nfev,
                  limitations=['PCA detail only; no inferred pores in geometry',
                               'single-view identity/expression ambiguity',
                               'GNM anatomical mapping is a neutral transfer from ICT; artist review recommended',
                               'camera intrinsics and metric scale held fixed; pose is a nuisance fit',
                               'mask silhouette contour attachments fixed at initialization; large pose changes need new initialization'])
    return candidate.astype(np.float32), np.asarray(captured, np.float32), cams, report
