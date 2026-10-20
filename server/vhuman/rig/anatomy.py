"""Geometry-only subject anatomy preparation for native CPU biomechanics.

GNM is an exterior/interior geometry prior, not a muscle atlas or dental scan.
All muscle fields and attachment rules below are repository-authored priors.
The fTetWild wheel is an offline meshing dependency, never a playback runtime.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import numpy as np

from .biomechanics_native import ROOT, volumes
from .common import vertex_normals

SCHEMA = 'vhuman.anatomy.v1'
MUSCLE_ATLAS = 'vhuman.authored_surface_bands.v3'
MESH_VERSION = '0.4.1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(value, indent=2, allow_nan=False))
    partial.replace(path)


def boundary_edges(faces):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keys, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    if (counts > 2).any():
        raise ValueError('nonmanifold anatomy surface')
    boundary = {tuple(e) for e in keys[counts == 1]}
    return np.asarray([e for e in edges if tuple(sorted(e)) in boundary], np.int32).reshape(-1, 2)


def close_root(vertices, faces):
    """Cap each open boundary separately; retain the original visible surface."""
    vertices, faces = np.asarray(vertices).copy(), np.asarray(faces).copy()
    edges = boundary_edges(faces)
    if not len(edges):
        return vertices, faces
    successor = {}
    for a, b in edges:
        if int(a) in successor:
            raise ValueError('anatomy boundary branches')
        successor[int(a)] = int(b)
    caps, points = [], []
    while successor:
        start = min(successor)
        ring, at = [], start
        while at in successor:
            ring.append(at)
            at = successor.pop(at)
        if at != start or len(ring) < 3:
            raise ValueError('open anatomy boundary chain')
        center = len(vertices) + len(points)
        points.append(vertices[ring].mean(0))
        caps.extend((ring[(i + 1) % len(ring)], ring[i], center) for i in range(len(ring)))
    return np.concatenate((vertices, points)), np.concatenate((faces, np.asarray(caps, np.int32)))


def tongue_domain(vertices, margin=.00025):
    """Authored convex volume envelope; retain folded GNM surface separately.

    The source's small self-overlapping folds are unsuitable as a winding
    boundary. A bounded outward support margin makes every original point an
    interior point of the solid used for mechanics, without editing its
    visible geometry. This is a geometric prior, not measured tongue anatomy.
    """
    from scipy.spatial import ConvexHull
    offsets=np.concatenate((np.eye(3),-np.eye(3)))*margin
    expanded=np.concatenate([vertices+offset for offset in offsets])
    hull=ConvexHull(expanded)
    used=np.unique(hull.simplices)
    remap=np.full(len(expanded),-1,np.int32);remap[used]=np.arange(len(used))
    faces=hull.simplices.copy()
    normals=np.cross(expanded[faces[:,1]]-expanded[faces[:,0]],expanded[faces[:,2]]-expanded[faces[:,0]])
    flip=(normals*hull.equations[:,:3]).sum(1)<0
    faces[flip]=faces[flip][:,::-1]
    return expanded[used],remap[faces]


def shell(vertices, faces, thickness):
    """Closed skin/gingiva tissue domain: outer + inner + open-boundary rim.

    fTetWild resolves the shell as a volume, rather than extruding simulation
    tetrahedra and dropping difficult regions during animation.
    """
    n = len(vertices)
    normals=vertex_normals(vertices,faces)
    from scipy.spatial import cKDTree
    distances,neighbors=cKDTree(vertices).query(vertices,k=min(48,n))
    opposing=(normals[:,None]*normals[neighbors]).sum(2)<-.5
    clearance=np.min(np.where(opposing,distances,np.inf),axis=1)
    # Lips and gingiva can be much thinner than the cheek. Avoid creating
    # intersecting inner sheets simply by applying one global shell thickness.
    local=np.minimum(thickness,np.maximum(.00015,.35*clearance))
    inner = vertices - local[:,None] * normals
    edges = boundary_edges(faces)
    rims = [(int(b), int(a), int(a + n)) for a, b in edges]
    rims += [(int(b), int(a + n), int(b + n)) for a, b in edges]
    closed = np.concatenate((faces, faces[:, ::-1] + n, np.asarray(rims, np.int32).reshape(-1, 3)))
    return np.concatenate((vertices, inner)), closed, inner


def mesh_worker(source, destination, *, edge=.004, threads=4, preserve_surface=False):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    # fTetWild emits a tracked-surface debug file in its working directory.
    # Keep it with the cancellable job, never in the repository root.
    os.chdir(destination.parent)
    import wildmeshing as wm
    if importlib.metadata.version('wildmeshing') != MESH_VERSION:
        raise ValueError(f'wildmeshing=={MESH_VERSION} is required')
    with np.load(source, allow_pickle=False) as z:
        v, f = z['vertices'], z['faces'].astype(np.int32)
    diagonal = float(np.linalg.norm(np.ptp(v, axis=0)))
    if not 1e-5 < diagonal < 2:
        raise ValueError('anatomy mesher requires metre-space geometry')
    # A 25 micrometre envelope bounds visible-surface embedding error. Keep
    # thin lips and gingiva even when the coarse regional edge target is 4 mm.
    mesher = wm.Tetrahedralizer(max_its=30, max_threads=threads, epsilon=.000025 / diagonal,
                               edge_length_r=edge / diagonal, skip_simplify=preserve_surface, coarsen=not preserve_surface)
    mesher.set_log_level(4)
    mesher.set_mesh(np.asarray(v, np.float64), f)
    mesher.tetrahedralize()
    points, tetrahedra, *_ = mesher.get_tet_mesh(correct_surface_orientation=True)
    tetrahedra = np.asarray(tetrahedra, np.int32)
    used,compact=np.unique(tetrahedra,return_inverse=True)
    points=np.asarray(points)[used]
    tetrahedra=compact.reshape(-1,4).astype(np.int32)
    signed = volumes(points, tetrahedra)
    flip = signed < 0
    tetrahedra[flip] = tetrahedra[flip][:, [0, 2, 1, 3]]
    np.savez_compressed(destination, rest=points, tets=tetrahedra)
    stats=dict(vertices=len(points),tetrahedra=len(tetrahedra),min_volume=float(np.min(np.abs(signed))) if len(signed) else None)
    write_json(Path(destination).with_suffix('.json'),stats)
    if not len(tetrahedra) or len(tetrahedra)>2000000 or not np.isfinite(points).all() or np.min(np.abs(signed))<1e-18:
        raise ValueError(f'tetrahedralization failed anatomy quality/size bounds: {stats}')


MESH_WORKER_SHA256=hashlib.sha256(inspect.getsource(mesh_worker).encode()).hexdigest()


def mesh_cache_recipe(vertices,faces,edge,threads,preserve_surface):
    """Exact canonical input bytes and all mesher settings identify a volume."""
    vertices,faces=np.asarray(vertices),np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1:]!=(3,) or not len(vertices) or not np.isfinite(vertices).all()
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or not len(faces) or faces.dtype.kind not in 'iu'
            or (faces<0).any() or (faces>=len(vertices)).any() or (faces>np.iinfo(np.int32).max).any()
            or not math.isfinite(edge) or not 0<edge<1 or type(threads) is not int or not 1<=threads<=256
            or type(preserve_surface) is not bool):raise ValueError('invalid tetrahedral cache inputs')
    h=hashlib.sha256()
    for value,dtype in ((vertices,'<f8'),(faces,'<i4')):
        value=np.ascontiguousarray(value,dtype=dtype)
        h.update(str(value.shape).encode());h.update(value.tobytes())
    recipe=dict(schema='vhuman.tetrahedral_cache.v1',input_sha256=h.hexdigest(),
                wildmeshing=MESH_VERSION,worker_sha256=MESH_WORKER_SHA256,
                edge=float(edge),threads=int(threads),preserve_surface=bool(preserve_surface))
    key=hashlib.sha256(json.dumps(recipe,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return key,recipe


def read_volume(path):
    with np.load(path,allow_pickle=False) as z:points,tets=z['rest'],z['tets']
    if (points.ndim!=2 or points.shape[1:]!=(3,) or not len(points) or len(points)>8000000
            or points.dtype.kind!='f' or not np.isfinite(points).all()
            or tets.ndim!=2 or tets.shape[1:]!=(4,) or not len(tets) or len(tets)>2000000
            or tets.dtype.kind not in 'iu' or (tets<0).any() or (tets>=len(points)).any()):
        raise ValueError('invalid cached tetrahedral volume')
    signed=volumes(points,tets)
    if not np.isfinite(signed).all() or (signed<1e-18).any():
        raise ValueError('invalid cached tetrahedral volume orientation/quality')
    return points,tets


def tetrahedralize(vertices, faces, scratch, name, *, edge, threads, cancel=None, preserve_surface=False,timeout=3600,cache=None):
    from .meshing_cache import MeshCache
    if cancel is not None and cancel.is_set():raise RuntimeError('anatomy fitting cancelled')
    key,recipe=mesh_cache_recipe(vertices,faces,edge,threads,preserve_surface)
    store=MeshCache(ROOT/'tmp/vhuman-tetrahedra' if cache is None else cache)
    cached=store.read(key,recipe)
    if cached is not None:
        result=read_volume(cached)
        write_json(scratch/f'{name}-meshing-cache.json',dict(key=key,hit=True,recipe=recipe,volume_sha256=digest(cached)))
        (scratch/f'{name}-meshing.log').write_text('Reused verified exact-input tetrahedral volume '+key+'\n')
        return result
    source, output = scratch / f'{name}-input.npz', scratch / f'{name}-volume.npz'
    np.savez_compressed(source, vertices=vertices, faces=faces)
    log = scratch / f'{name}-meshing.log'
    command = [sys.executable, '-B', '-m', 'server.vhuman.rig.anatomy', 'mesh', str(source), str(output),
               '--edge', str(edge), '--threads', str(threads)]
    if preserve_surface:
        command.append('--preserve-surface')
    from ..process import run
    with log.open('w') as stream:
        def record(line):
            stream.write(line+'\n')
            stream.flush()
        try:
            status = run(command, cancel=cancel or threading.Event(), cwd=ROOT, env=dict(os.environ, TMPDIR=str(scratch.resolve())),
                on_line=record,timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'{name} tetrahedralization exceeded {timeout} seconds; increase --mesh-timeout (maximum 7200); inspect {log}') from exc
    if status:
        raise RuntimeError(f'{name} tetrahedralization failed; inspect {log}')
    read_volume(output)
    if cancel is not None and cancel.is_set():raise RuntimeError('anatomy fitting cancelled')
    cached=store.publish(key,recipe,output)
    write_json(scratch/f'{name}-meshing-cache.json',dict(key=key,hit=False,recipe=recipe,volume_sha256=digest(cached)))
    return read_volume(cached)


def embed(points, rest, tets):
    """Barycentric transport, including small, bounded meshing-envelope offsets."""
    from scipy.spatial import cKDTree
    centers = rest[tets].mean(1)
    tree = cKDTree(centers)
    ids_out, weights_out = [], []
    inverses = np.linalg.inv((rest[tets[:, 1:]] - rest[tets[:, :1]]).transpose(0, 2, 1))
    def choose(p,candidates):
        local = np.einsum('pkij,pkj->pki', inverses[candidates], p[:,None]-rest[tets[candidates,0]])
        weights = np.concatenate((1-local.sum(2,keepdims=True),local),2)
        violation = np.maximum(-weights,0).sum(2)
        choice = violation.argmin(1)
        selected = candidates[np.arange(len(p)),choice]
        w = weights[np.arange(len(p)),choice]
        outside = (w < -1e-8).any(1)
        supported = ~outside
        if outside.any():
            from ..reconstruction.contact_runtime import closest_on_triangles
            candidate_ids = candidates[outside]
            vertices = rest[tets[candidate_ids]]
            distance = np.full(candidate_ids.shape,np.inf)
            for a,b,c in ((0,1,2),(0,1,3),(0,2,3),(1,2,3)):
                _,_,d = closest_on_triangles(p[outside],vertices[:,:,a],vertices[:,:,b],vertices[:,:,c])
                distance = np.minimum(distance,d)
            candidate_weights = weights[outside]
            stable = (candidate_weights.min(2)>=-1)&(candidate_weights.max(2)<=2)&(abs(candidate_weights).sum(2)<=3)
            distance = np.where(stable,distance,np.inf)
            choice = distance.argmin(1)
            selected[outside] = candidate_ids[np.arange(len(choice)),choice]
            w[outside] = candidate_weights[np.arange(len(choice)),choice]
            supported[outside] = distance[np.arange(len(choice)),choice]<=.0001
        return selected,w,supported

    for start in range(0,len(points),128):
        p = points[start:start+128]
        selected,w = np.empty(len(p),np.int32),np.empty((len(p),4))
        pending = np.arange(len(p))
        # Centroid proximity is only a broad-phase heuristic. A thin or long
        # enclosing tetrahedron may sit beyond the nearest 64 centroids.
        for count in (64,256,1024):
            _,candidates = tree.query(p[pending],k=min(count,len(tets)))
            candidates = np.asarray(candidates).reshape(len(pending),-1)
            chosen,weights,supported = choose(p[pending],candidates)
            selected[pending[supported]] = chosen[supported]
            w[pending[supported]] = weights[supported]
            pending = pending[~supported]
            if not len(pending):
                break
            if count>=len(tets):
                break
        if len(pending):
            raise ValueError('surface lies outside its anatomy volume support (100 micrometres) or bounded search; repair the shell')
        ids_out.append(tets[selected])
        weights_out.append(w)
    return np.concatenate(ids_out), np.concatenate(weights_out)


def front_surface_anchor(skin, x, y, scale):
    """Choose a local frontal skin point without snapping to the back scalp."""
    skin=np.asarray(skin,np.float64)
    if (skin.ndim!=2 or skin.shape[1]!=3 or not len(skin) or not np.isfinite(skin).all()
            or not all(math.isfinite(v) for v in (x,y,scale)) or scale<=0):
        raise ValueError('invalid frontal surface anchor inputs')
    distance=np.square(skin[:,:2]-[x,y]).sum(1)
    near=np.flatnonzero(distance<=distance.min()+(.003*scale)**2)
    return skin[near[np.argmax(skin[near,2])]].copy()


def muscle_atlas(rest, tets, regions, parts, landmarks, scale):
    """Return sparse rest-space fiber fields and an authored control prior.

    Stress scales are animation defaults, not estimates of a person's muscle
    force. Overlapping occupancy is normalized per tetrahedron.
    """
    centers = rest[tets].mean(1)
    mouth, eye_y = np.asarray(landmarks['mouth']), landmarks['eye_y']
    names, controls, rows = [], [], []

    def field(name, region, center, sigma, direction, links, peak=3500.):
        index = len(names)
        names.append(name)
        controls.append(links)
        d = centers - np.asarray(center)
        weight = np.exp(-.5 * np.square(d / (np.asarray(sigma) * scale)).sum(1))
        weight *= regions == region
        ids = np.flatnonzero(weight > .12)
        if callable(direction):
            directions = direction(centers[ids])
        else:
            directions = np.tile(direction, (len(ids), 1)).astype(float)
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        rows.extend((int(t), index, *vector, peak * weight[t]) for t, vector in zip(ids, directions))

    # Anatomical families are independent on each side; medial rings use
    # separately excitable upper/lower, left/right sectors.
    facial = [
        ('frontalis', .020, eye_y + .026*scale, (.014,.025,.018), (0,1,0), 'browOuterUp', .6),
        ('corrugator', .011, eye_y + .010*scale, (.011,.010,.014), (1,.3,0), 'browDown', .7),
        ('orbicularis_oculi', .031, eye_y, (.018,.012,.016), (0,1,0), 'eyeBlink', 1.),
        ('levator_palpebrae', .031, eye_y + .006*scale, (.016,.008,.012), (0,1,0), 'eyeWide', .6),
        ('zygomaticus', .030, mouth[1] + .013*scale, (.024,.014,.014), (1,1,0), 'mouthSmile', 1.),
        ('levator_labii', .012, mouth[1] + .012*scale, (.010,.014,.014), (0,1,0), 'mouthUpperUp', .7),
        ('levator_anguli', .024, mouth[1] + .009*scale, (.012,.014,.014), (.3,1,0), 'mouthDimple', .6),
        ('risorius', .034, mouth[1], (.022,.009,.014), (1,0,0), 'mouthStretch', .8),
        ('buccinator', .040, mouth[1] + .001*scale, (.021,.021,.025), (1,0,0), 'cheekSquint', .5),
        ('depressor_anguli', .027, mouth[1] - .016*scale, (.013,.016,.014), (.3,-1,0), 'mouthFrown', .8),
        ('depressor_labii', .012, mouth[1] - .013*scale, (.013,.013,.014), (0,-1,0), 'mouthLowerDown', .8),
        ('mentalis', .008, mouth[1] - .025*scale, (.012,.012,.014), (0,1,0), 'mouthShrugLower', .6),
        ('nasalis', .012, mouth[1] + .027*scale, (.012,.010,.014), (1,0,0), 'noseSneer', .5),
    ]
    skin = parts.get('skin_exterior',parts['skin'])
    for side, sign in (('Left', 1), ('Right', -1)):
        for name, x, y, sigma, direction, control, strength in facial:
            point = front_surface_anchor(skin,mouth[0]+sign*x*scale,y,scale)
            direction = np.asarray(direction, float) * [sign, 1, 1]
            field(name + '_' + side, 0, point, sigma, direction, {control + side: strength})
        for upper, offset in ((True, .003), (False, -.003)):
            point = mouth + np.array([sign*.010, offset, .001])*scale
            # Marginal bands shorten across the lip seam. A circular tangent
            # makes much of this central band vertical and pulls the two lip
            # margins apart. Rounded-mouth control has separate lateral fibers.
            field('orbicularis_oris_' + ('upper_' if upper else 'lower_') + side, 0, point,
                  (.017,.007,.012), (1,0,0),
                  {'mouthPucker': .7, 'mouthFunnel': .5, 'mouthPress' + side: .6, 'mouthClose': .8})
        field('orbicularis_oris_lateral_'+side,0,mouth+np.array([sign*.025,0,0])*scale,
              (.008,.010,.012),(0,1,0),{'mouthPucker':.5,'mouthFunnel':.4},1500.)
    field('procerus', 0, front_surface_anchor(skin,mouth[0],eye_y+.009*scale,scale),
          (.009,.012,.020), (0,1,0), {'browDownLeft': .3,'browDownRight':.3})
    field('frontalis_medial',0,front_surface_anchor(skin,mouth[0],eye_y+.025*scale,scale),
          (.012,.024,.020),(0,1,0),{'browInnerUp':.8})
    tongue = parts['tongue']
    center = tongue.mean(0)
    for side, sign in (('Left', 1), ('Right', -1)):
        base = center + [sign*.005*scale, 0, 0]
        tongue_fields = [
            ('genioglossus_anterior', (0,-1,.3), {'tongueDown': .7}, (0,0,.010)),
            ('genioglossus_medial', (0,-1,1), {'tongueOut': .6}, (0,0,0)),
            ('genioglossus_posterior', (0,-.3,1), {'tongueOut': .8}, (0,0,-.012)),
            ('hyoglossus', (0,1,.2), {'tongueDown': .6}, (0,0,-.004)),
            ('styloglossus', (0,1,-1), {'tongueUp': .7}, (0,.003,-.004)),
            ('palatoglossus', (0,1,-.2), {'tongueUp': .5}, (0,.006,-.008)),
            ('superior_longitudinal', (0,0,1), {'tongueCurlUp': .8, 'tongueUp': .5}, (0,.004,.004)),
            ('inferior_longitudinal', (0,0,1), {'tongueDown': .6}, (0,-.004,.002)),
            ('transverse', (1,0,0), {'tongueOut': .8, 'tongue' + side: .5}, (0,0,.003)),
            ('vertical', (0,1,0), {'tongueOut': .5, 'tongueDown': .3}, (0,0,0)),
        ]
        for name, direction, links, offset in tongue_fields:
            field(name+'_'+side, 1, base+np.asarray(offset)*scale, (.014,.010,.020), direction, links, 2500.)
    jaw = []
    hinge = np.asarray(landmarks['jaw'])
    for side, sign in (('Left', 1), ('Right', -1)):
        # Origins and insertions are authored relative to the fitted hinge and
        # dental arch. Paths belong to this atlas, not an imported engine rig.
        for name, origin, insertion, link in (
            ('masseter', (sign*.035,.025,.010), (sign*.030,-.030,.025), {'jawOpen': -.6}),
            ('temporalis', (sign*.025,.050,-.015), (sign*.022,-.010,.020), {'jawOpen': -.4}),
            ('medial_pterygoid', (sign*.010,.018,.010), (sign*.025,-.025,.020), {'jawOpen': -.3}),
            ('lateral_pterygoid', (sign*.010,.003,.035), (sign*.025,0,0), {'jawForward': .8}),
            ('digastric', (sign*.010,-.060,.020), (sign*.012,-.025,.060), {'jawOpen': .8}),
        ):
            index = len(names)
            names.append(name+'_'+side)
            controls.append(link)
            o, p = hinge+np.asarray(origin)*scale, hinge+np.asarray(insertion)*scale
            jaw.append((index, *o, *p, 40.*scale**2, float(np.linalg.norm(o-p))))
    fibers = np.asarray(rows, np.float64).reshape(-1, 6)
    if len(fibers):
        occupancy = np.bincount(fibers[:, 0].astype(int), weights=fibers[:, 5], minlength=len(tets))
        fibers[:, 5] /= np.maximum(1, occupancy[fibers[:, 0].astype(int)] / 3500.)
    return names, controls, fibers, np.asarray(jaw, np.float64)


def fit_jaw(rest, target_frames, hinge, *, iterations=160, threads=4, fit_hinge=True):
    """Rigid dental fit; opening in GNM PCA is factored into mandible motion."""
    import torch
    from ..reconstruction.gnm_fit import cpu_threads
    with cpu_threads(threads):
        p = torch.as_tensor(rest, dtype=torch.float64, device='cpu')
        target = torch.as_tensor(target_frames, dtype=torch.float64, device='cpu')
        seed = torch.as_tensor(hinge, dtype=torch.float64, device='cpu')
        offset = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64, device='cpu'),requires_grad=fit_hinge)
        q = torch.nn.Parameter(torch.zeros((len(target), 3), dtype=torch.float64, device='cpu'))
        optimizer = torch.optim.Adam((offset, q) if fit_hinge else (q,), lr=.003)
        def posed():
            h = seed + offset
            r = p-h
            c, s = q[:, 0].cos()[:, None], q[:, 0].sin()[:, None]
            return torch.stack((r[None, :, 0].expand(len(q), -1)+h[0]+q[:, 1:2],
                                c*r[:, 1]-s*r[:, 2]+h[1]-.003333333333333333*q[:, :1],
                                s*r[:, 1]+c*r[:, 2]+h[2]+q[:, 2:3]), -1)
        for _ in range(iterations):
            optimizer.zero_grad()
            loss = ((posed()-target)/.001).square().mean() + (offset/.002).square().mean()*.05
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                offset.clamp_(-.002, .002)
                q[:, 0].clamp_(0, .48); q[:, 1].clamp_(-.004, .004); q[:, 2].clamp_(0, .006)
        with torch.no_grad():
            error = torch.linalg.vector_norm(posed()-target, dim=-1).numpy()*1000
        return (seed+offset).detach().numpy(), q.detach().numpy(), dict(rms_mm=float(np.sqrt(np.mean(error**2))),
                                                                 p95_mm=float(np.quantile(error, .95)))


def jaw_points(points,hinge,q):
    relative=points-hinge
    c,s=np.cos(q[:,0,None]),np.sin(q[:,0,None])
    return np.stack((np.broadcast_to(relative[None,:,0],(len(q),len(points)))+hinge[0]+q[:,1,None],
                     c*relative[:,1]-s*relative[:,2]+hinge[1]-.003333333333333333*q[:,:1],
                     s*relative[:,1]+c*relative[:,2]+hinge[2]+q[:,2,None]),-1)


def fit_expression_map(mapping,captures,model,*,threads=4):
    """Regularized CPU shape transfer; hold out a take or contiguous tail.

    Targets are fitted GNM motion, not independent 3D measurements. Region
    gates prevent a better mouth fit from degrading the rest of the face.
    """
    import torch
    from ..reconstruction.gnm_fit import cpu_threads
    initial=np.asarray(mapping['coefficients'],float)
    if not captures:
        return initial,dict(accepted=False,reason='no captured expression targets')
    if len(captures)>1:
        train=captures[:-1];held=captures[-1:]
    else:
        x,y=captures[0];cut=max(1,int(.8*len(x)))
        train=[(x[:cut],y[:cut])];held=[(x[cut:],y[cut:])]
    X,Y=np.concatenate([v[0] for v in train]),np.concatenate([v[1] for v in train])
    HX,HY=np.concatenate([v[0] for v in held]),np.concatenate([v[1] for v in held])
    if len(X)<2 or not len(HX):
        return initial,dict(accepted=False,reason='insufficient independent temporal holdout')
    with cpu_threads(threads):
        x=torch.as_tensor(X,dtype=torch.float64,device='cpu')
        y=torch.as_tensor(Y,dtype=torch.float64,device='cpu')
        seed=torch.as_tensor(initial,dtype=torch.float64,device='cpu')
        gram=x.T@x
        ridge=max(1e-4,float(gram.trace())/max(1,len(gram))*.05)
        update=torch.linalg.solve(gram+ridge*torch.eye(len(gram),dtype=torch.float64,device='cpu'),x.T@y+ridge*seed.T)
        proposed=update.T.clamp(-3,3).numpy()
    before=np.clip(HX@initial.T,-3,3)-HY
    after=np.clip(HX@proposed.T,-3,3)-HY
    region_masks={'mouth':model.group('upper_lip')|model.group('lower_lip')|model.group('tongue'),
                  'eyes':model.group('eye_sockets')}
    region_masks['other']=model.group('skin_exterior')&~region_masks['mouth']&~region_masks['eyes']
    def metric(delta,mask):
        ids=np.flatnonzero(mask)[::max(1,int(mask.sum())//256)]
        displacement=np.einsum('fe,evc->fvc',delta,model.data['expression_basis'][:,ids])
        return float(np.sqrt(np.mean(displacement**2))*1000)
    metrics={name:dict(before_mm=metric(before,mask),after_mm=metric(after,mask)) for name,mask in region_masks.items()}
    baseline=metric(before,model.group('skin_exterior'));result=metric(after,model.group('skin_exterior'))
    accepted=result<=baseline*.98 and all(v['after_mm']<=max(1e-6,v['before_mm']*1.05) for v in metrics.values())
    return (proposed if accepted else initial),dict(accepted=accepted,heldout_frames=len(HX),training_frames=len(X),
                before_mm=baseline,after_mm=result,regions=metrics,target='fitted_GNM_motion; not independent 3D truth')


def bond_semantic_seams(arrays):
    """Shared GNM tooth/gingiva vertices are one physical attachment, not
    colliding coincident surfaces. Opposing lips have different source IDs
    and must remain eligible for contact even when they touch at rest.
    """
    source=arrays['source_ids']
    order=np.argsort(source,kind='stable')
    cuts=np.r_[0,np.flatnonzero(np.diff(source[order]))+1,len(order)]
    mass=(arrays['inverse_mass'][arrays['embed_ids']]*arrays['embed_weights']**2).sum(1)
    for a,b in zip(cuts[:-1],cuts[1:]):
        ids=order[a:b]
        if len(ids)>1:
            chosen=ids[np.argmin(mass[ids])]
            arrays['embed_ids'][ids]=arrays['embed_ids'][chosen]
            arrays['embed_weights'][ids]=arrays['embed_weights'][chosen]


def surface_preflight(arrays):
    """Check the retained source surface without constructing tissue dynamics."""
    import ctypes as ct
    from .biomechanics_native import library,ptr,P,check
    lib=library();handle=P();count=ct.c_uint64()
    surface=np.ascontiguousarray(arrays['surface_rest'],np.float64)
    mass=np.zeros(len(surface))
    _,first,inverse=np.unique(arrays['source_ids'],return_index=True,return_inverse=True)
    faces=np.ascontiguousarray(first[inverse][arrays['faces']],np.int32)
    check(lib.lightphysics_surface_contact_create(ptr(surface),ptr(mass),len(surface),ptr(faces),len(faces),ct.byref(handle)))
    try:
        check(lib.lightphysics_surface_contact_count_intersections(handle,ptr(surface),ct.byref(count)))
    finally:
        lib.lightphysics_surface_contact_destroy(handle)
    return dict(proper_surface_crossings=int(count.value),passed=count.value==0,
                source='retained GNM surface; bonded semantic seams canonicalized')


def validate(arrays, meta):
    required = ('rest', 'tets', 'faces', 'inverse_mass', 'bone', 'fibers', 'jaw_muscles', 'jaw_hinge',
                'anchor_ids', 'anchor_bone', 'youngs_moduli', 'surface_rest', 'embed_ids', 'embed_weights', 'source_ids')
    if meta.get('schema') != SCHEMA or any(k not in arrays for k in required):
        raise ValueError('unsupported/incomplete anatomy package')
    for key in required:
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f'non-finite anatomy {key}')
    rest, tets, faces = arrays['rest'], arrays['tets'], arrays['faces']
    if rest.ndim != 2 or rest.shape[1] != 3 or len(rest) > 1000000 or len(rest) < 4:
        raise ValueError('invalid anatomy vertices')
    for key, size, limit in (('tets', 4, len(rest)), ('faces', 3, len(arrays['surface_rest'])), ('embed_ids', 4, len(rest))):
        value = arrays[key]
        if value.ndim != 2 or value.shape[1] != size or value.dtype.kind not in 'iu' or (value < 0).any() or (value >= limit).any():
            raise ValueError(f'invalid anatomy {key}')
    if not len(tets) or (volumes(rest, tets) < 1e-18).any():
        raise ValueError('invalid/degenerate anatomy volume')
    n, ns, nm = len(rest), len(arrays['surface_rest']), len(meta.get('muscles', []))
    for key in ('expression_coefficients','expression_seed_coefficients'):
        if key in arrays:
            value=arrays[key]
            if (value.ndim!=2 or not 1<=value.shape[0]<=1024 or
                    value.shape[1]!=len(meta.get('expression_controls',[])) or
                    not np.isfinite(value).all()):
                raise ValueError(f'invalid anatomy {key}')
    if 'expression_seed_coefficients' in arrays and (
            'expression_coefficients' not in arrays or
            arrays['expression_seed_coefficients'].shape!=arrays['expression_coefficients'].shape):
        raise ValueError('inconsistent expression calibration seed')
    if 'fit_importance' in arrays:
        importance=arrays['fit_importance']
        if (importance.shape!=(ns,) or importance.dtype.kind not in 'fiu' or not np.isfinite(importance).all()
                or (importance<0).any() or (importance>4).any()):
            raise ValueError('invalid inverse-fitting surface importance')
    if 'fit_eye_features' in arrays:
        mask=arrays['fit_eye_features']
        if mask.shape!=(ns,) or mask.dtype.kind not in 'biu' or not np.isin(mask,[0,1]).all():
            raise ValueError('invalid inverse-fitting eyelid mask')
    if 'bind_delta' in arrays and (arrays['bind_delta'].shape!=(17821,3) or not np.isfinite(arrays['bind_delta']).all()):
        raise ValueError('invalid fitted bind residual')
    if not 1 <= nm <= 256 or len(set(meta['muscles'])) != nm:
        raise ValueError('invalid anatomy muscle names')
    if arrays['inverse_mass'].shape != (n,) or (arrays['inverse_mass'] < 0).any() or arrays['bone'].shape != (n,):
        raise ValueError('invalid anatomy masses/bones')
    if (arrays['bone'] < -1).any() or (arrays['bone'] > 1).any() or arrays['jaw_hinge'].shape!=(3,):
        raise ValueError('invalid anatomy bone labels/jaw hinge')
    if 'jaw_track' in arrays and (arrays['jaw_track'].shape!=(3,) or not np.isfinite(arrays['jaw_track']).all()):
        raise ValueError('invalid fitted jaw condylar track')
    if arrays['youngs_moduli'].shape != (len(tets),) or (arrays['youngs_moduli'] <= 0).any():
        raise ValueError('invalid regional material')
    if arrays['embed_ids'].shape != (ns, 4) or arrays['embed_weights'].shape != (ns, 4) or not np.allclose(arrays['embed_weights'].sum(1), 1, atol=1e-7):
        raise ValueError('invalid surface embedding')
    if (arrays['embed_weights'] < -1).any() or (arrays['embed_weights'] > 2).any() or (abs(arrays['embed_weights']).sum(1)>3).any():
        raise ValueError('unstable surface embedding')
    extrapolated=np.flatnonzero((arrays['embed_weights'] < -1e-8).any(1))
    if len(extrapolated):
        from ..reconstruction.contact_runtime import closest_on_triangles
        for start in range(0,len(extrapolated),256):
            ids=extrapolated[start:start+256]
            vertices=rest[arrays['embed_ids'][ids]][:,None]
            distance=np.full((len(ids),1),np.inf)
            for a,b,c in ((0,1,2),(0,1,3),(0,2,3),(1,2,3)):
                _,_,d=closest_on_triangles(arrays['surface_rest'][ids],vertices[:,:,a],vertices[:,:,b],vertices[:,:,c])
                distance=np.minimum(distance,d)
            if (distance>.0001001).any():
                raise ValueError('surface embedding exceeds physical support envelope')
    reconstructed = (rest[arrays['embed_ids']] * arrays['embed_weights'][..., None]).sum(1)
    if not np.allclose(reconstructed, arrays['surface_rest'], atol=1e-7):
        raise ValueError('anatomy embedding changes neutral geometry')
    for key, width, index in (('fibers', 6, 1), ('jaw_muscles', 9, 0)):
        v = arrays[key]
        if v.ndim != 2 or v.shape[1] != width or (v[:, index] < 0).any() or (v[:, index] >= nm).any():
            raise ValueError(f'invalid anatomy {key}')
        if not np.equal(v[:,index],np.floor(v[:,index])).all():
            raise ValueError('muscle indices must be integers')
    if (arrays['fibers'][:, 0] < 0).any() or (arrays['fibers'][:, 0] >= len(tets)).any():
        raise ValueError('fiber refers to missing tetrahedron')
    fibers=arrays['fibers'];jaw=arrays['jaw_muscles']
    if not np.equal(fibers[:,0],np.floor(fibers[:,0])).all() or (fibers[:,5]<0).any() or (np.linalg.norm(fibers[:,2:5],axis=1)<1e-8).any():
        raise ValueError('invalid fiber direction/stress')
    if (jaw[:,7]<0).any() or (jaw[:,8]<=0).any():
        raise ValueError('invalid jaw muscle force/length')
    if arrays['source_ids'].shape!=(ns,) or arrays['source_ids'].dtype.kind not in 'iu' or (arrays['source_ids']<0).any() or (arrays['source_ids']>=17821).any():
        raise ValueError('invalid GNM source correspondence')
    for name in ('anchor_ids', 'anchor_bone'):
        if arrays[name].ndim != 1 or arrays[name].dtype.kind not in 'iu':
            raise ValueError('invalid anatomy anchors')
    if arrays['anchor_ids'].shape != arrays['anchor_bone'].shape or (arrays['anchor_ids'] < 0).any() or (arrays['anchor_ids'] >= n).any():
        raise ValueError('invalid anatomy attachment indices')
    if (arrays['anchor_bone']<0).any() or (arrays['anchor_bone']>1).any() or len(np.unique(arrays['anchor_ids']))!=len(arrays['anchor_ids']):
        raise ValueError('invalid anatomy attachment bones')
    if ('anchor_normals' in arrays)!=('anchor_tangent_compliance' in arrays):
        raise ValueError('incomplete sliding attachment configuration')
    if 'anchor_normals' in arrays:
        normals=arrays['anchor_normals'];tangent=arrays['anchor_tangent_compliance'];count=len(arrays['anchor_ids'])
        if (normals.shape!=(count,3) or tangent.shape!=(count,) or normals.dtype.kind!='f' or tangent.dtype.kind!='f'
                or not np.isfinite(normals).all() or not np.isfinite(tangent).all() or (tangent<0).any()):
            raise ValueError('invalid sliding attachment configuration')
        length=np.linalg.norm(normals,axis=1)
        if ((length>0)&(abs(length-1)>1e-5)).any() or ((length>0)&(arrays['anchor_bone']!=0)).any():
            raise ValueError('sliding attachment normals must be unit and fixed in the skull frame')
    if len(meta.get('control_prior',[]))!=nm:
        raise ValueError('invalid muscle control prior')
    if 'tongue_cables' in arrays:
        from .tongue_cables import validate_rows
        validate_rows(arrays['tongue_cables'],n,nm,inverse_mass=arrays['inverse_mass'],rest=arrays['rest'])


def load(rig_dir, package=None):
    folder = Path(package) if package else Path(rig_dir) / 'anatomy'
    meta = json.loads((folder/'anatomy.json').read_text())
    if meta.get('arrays_sha256') != digest(folder/'anatomy.npz'):
        raise ValueError('anatomy package hash mismatch')
    if meta.get('rig_sha256') != digest(Path(rig_dir)/'rig.json') or meta.get('rig_package_sha256') != digest(Path(rig_dir)/'rig_deformer.safetensors'):
        raise ValueError('anatomy belongs to a different rig; refit anatomy')
    with np.load(folder/'anatomy.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    validate(arrays, meta)
    return arrays, meta


def part_masks(model):
    masks = [('skin', model.group('skin'), 0), ('tongue', model.group('tongue'), 1)]
    for side in ('upper', 'lower'):
        masks.append(('gums_'+side, model.group(side+'_teeth_and_gums') & model.group('gums'), 2))
        masks.append(('teeth_'+side, model.group(side+'_teeth_and_gums') & model.group('teeth'), -1))
    for side in ('left', 'right'):
        masks.append(('eye_'+side, model.group(side+'_eye') & model.group('eye_exteriors'), -1))
    return masks


def validate_fitting_budget(fit_mouth,mouth_fairing,mesh_timeout):
    if (type(mesh_timeout) is not int or not 60<=mesh_timeout<=7200 or type(mouth_fairing) not in (int,float)
            or not math.isfinite(mouth_fairing) or not 0<=mouth_fairing<=.5):
        raise ValueError('mouth fairing must be 0..0.5 and mesh timeout an integer in 60..7200 seconds')
    if mouth_fairing and not fit_mouth:raise ValueError('mouth fairing requires mouth fitting')


def prepare(rig_dir, *, out=None, captures=(), threads=4, edge=.004, cancel=None, progress=None,
            fit_sockets=False, fit_mouth=False, fit_dentition=False, fit_tongue=False, fit_tongue_root=False,
            sliding_skin=False, tangent_fibers=False,harmonic_tongue_fibers=False,extrinsic_tongue_cables=False,mouth_fairing=0.,mesh_timeout=3600):
    from ..reconstruction.gnm_motion import rig_bind
    from .gnm_model import GNMModel
    from scipy.spatial import cKDTree
    rig_dir = Path(rig_dir)
    out = Path(out) if out else rig_dir/'anatomy'
    if out.exists():
        raise FileExistsError(f'anatomy destination exists: {out}; use a fresh --out')
    if any(type(flag) is not bool for flag in (fit_sockets,fit_mouth,fit_dentition,fit_tongue,fit_tongue_root,sliding_skin,tangent_fibers,harmonic_tongue_fibers,extrinsic_tongue_cables)):
        raise ValueError('geometry fitting and skin mechanics flags must be booleans')
    validate_fitting_budget(fit_mouth,mouth_fairing,mesh_timeout)
    if type(threads) is not int or not 1 <= threads <= 64 or not math.isfinite(edge) or not .001 <= edge <= .008:
        raise ValueError('invalid anatomy threads/edge length')
    stage = out.parent/('.anatomy-'+uuid.uuid4().hex+'.partial')
    stage.mkdir(parents=True)
    try:
        model = GNMModel()
        bind = rig_bind(rig_dir)
        rest, _ = model.evaluate(bind['identity'], bind_residual=bind['bind_residual'])
        scale = float(bind['scale'])
        rest = scale*rest@bind['alignment'].T+bind['offset']
        original_rest = rest.copy()
        bind_delta = np.zeros_like(rest)
        geometry_fit = dict(accepted=False, reason='socket conditioning not requested')
        mouth_fit = dict(accepted=False, reason='mouth conditioning not requested')
        dentition_fit = dict(accepted=False, reason='dentition conditioning not requested')
        tongue_fit = dict(accepted=False, reason='tongue conditioning not requested')
        tongue_root_fit = dict(accepted=False, reason='tongue-root conditioning not requested')
        if fit_sockets:
            from .geometry_fit import fit_sockets as condition_sockets
            if progress:
                progress(.02, 'fitting ocular cavities and eyelid clearance')
            fitted, geometry_fit = condition_sockets(model, rest)
            rest = fitted
            write_json(stage/'socket_fit.json', geometry_fit)
        if fit_mouth:
            from .oral_fit import fit_mouth as condition_mouth
            rest,mouth_fit = condition_mouth(model,rest,fairing=mouth_fairing,cancel=cancel,
                progress=(lambda f,t:progress(.03+.015*f,t)) if progress else None)
            write_json(stage/'mouth_fit.json',mouth_fit)
        if fit_dentition:
            from .oral_fit import fit_dentition as condition_dentition
            rest,dentition_fit = condition_dentition(model,rest,cancel=cancel,
                progress=(lambda f,t:progress(.045+.004*f,t)) if progress else None)
            write_json(stage/'dentition_fit.json',dentition_fit)
        if fit_tongue or fit_tongue_root:
            from .oral_fit import fit_tongue as condition_tongue
            rest,tongue_fit = condition_tongue(model,rest,cancel=cancel,
                progress=(lambda f,t:progress(.049+.0009*f,t)) if progress else None)
            write_json(stage/'tongue_fit.json',tongue_fit)
        if fit_tongue_root:
            from .oral_fit import fit_tongue_root as condition_root
            rest,tongue_root_fit=condition_root(model,rest,scale=scale,limit=.009*scale,cancel=cancel,
                progress=(lambda f,t:progress(.0491+.0008*f,t)) if progress else None)
            write_json(stage/'tongue_root_fit.json',tongue_root_fit)
        if fit_sockets or fit_mouth or fit_dentition or fit_tongue or fit_tongue_root:
            bind_delta = (rest-original_rest) @ np.linalg.inv(np.asarray(bind['alignment'],np.float64).T) / scale
        definition = json.loads((rig_dir/'rig.json').read_text())
        hinge = np.asarray(next(j['bind'] for j in definition['joints'] if j['name']=='jaw'))[:3, 3]
        dental = model.group('lower_teeth_and_gums') & model.group('teeth')
        capture_targets, receipts, expression_captures = [], [], []
        expression_mapping=definition.get('face_model_source',{}).get('expression_mapping')
        if expression_mapping is None:
            raise ValueError('anatomy fitting requires a geometry-matched GNM expression map')
        from .biomechanics_mapping import calibrate,capture_transforms
        expression_mapping,control_mapping_report=calibrate(rig_dir,model,bind,threads=threads)
        for capture in captures:
            folder = Path(capture)
            receipt = json.loads((folder/'motion.json').read_text())
            if receipt.get('motion_sha256') != digest(folder/'motion.npz'):
                raise ValueError('capture motion hash mismatch')
            if not receipt.get('quality_gate'):
                raise ValueError('anatomy capture failed its held-out fitting gate')
            with np.load(folder/'motion.npz', allow_pickle=False) as z:
                if not np.array_equal(z['identity'], bind['identity']):
                    raise ValueError('anatomy captures must have the same frozen identity')
                animation=json.loads((folder/'animation.json').read_text())
                if len(animation['frames'])!=len(z['expression']) or not np.allclose([f['t'] for f in animation['frames']],z['timestamps'],atol=1e-6):
                    raise ValueError('capture geometry/control timestamps differ')
                expression_captures.append((np.array([[f['v'].get(name,0) for name in expression_mapping['controls']] for f in animation['frames']]),z['expression'].copy()))
                for index in np.linspace(0, len(z['vertices'])-1, min(24, len(z['vertices'])), dtype=int):
                    rotations=z['rotations'][index]
                    translation=z['translation'][index] if 'translation' in z else np.zeros(3)
                    p, _ = model.evaluate(bind['identity'], z['expression'][index], rotations, translation, bind_residual=bind['bind_residual'])
                    p=scale*p@bind['alignment'].T+bind['offset']
                    head=capture_transforms(model,bind,rotations,translation)['head']
                    inverse_head=np.linalg.inv(head)
                    capture_targets.append((p@inverse_head[:3,:3].T+inverse_head[:3,3])[dental])
            receipts.append(dict(path=str(folder), sha256=receipt['motion_sha256']))
        jaw_fit = dict(source='authored_prior', accepted=False)
        if capture_targets:
            target=np.asarray(capture_targets)
            fitted,q,_ = fit_jaw(rest[dental][::2],target[:,::2],hinge,threads=threads)
            _,baseline,_ = fit_jaw(rest[dental][::2],target[:,::2],hinge,threads=threads,fit_hinge=False)
            withheld=rest[dental][1::2]
            before=float(np.sqrt(np.mean((jaw_points(withheld,hinge,baseline)-target[:,1::2])**2))*1000)
            after=float(np.sqrt(np.mean((jaw_points(withheld,fitted,q)-target[:,1::2])**2))*1000)
            jaw_fit=dict(source='native_GNM_dental_rigid_fit',accepted=after<=before*.98,
                         withheld_before_mm=before,withheld_after_mm=after)
            if jaw_fit['accepted']:
                hinge = fitted
        from .jaw_model import fit_track
        jaw_track,jaw_track_fit=fit_track(definition,hinge,rest[dental])
        write_json(stage/'jaw_track_fit.json',jaw_track_fit)
        expression_coefficients,expression_fit=fit_expression_map(expression_mapping,expression_captures,model,threads=threads)
        parts = []
        masks = part_masks(model)
        arrays = {k: [] for k in ('rest','tets','faces','inverse_mass','bone','anchor_ids','anchor_bone','youngs_moduli',
                                  'surface_rest','embed_ids','embed_weights','source_ids','regions')}
        offset, surface_offset, tet_offset = 0, 0, 0
        original_parts = {}
        mouth = rest[model.group('upper_lip') | model.group('lower_lip')].mean(0)
        for number, (name, mask, region) in enumerate(masks):
            if cancel is not None and cancel.is_set():
                raise RuntimeError('anatomy fitting cancelled')
            if progress:
                progress(.05+.8*number/len(masks), 'preparing '+name+' volume')
            ids = np.flatnonzero(mask)
            remap = np.full(len(rest), -1, np.int32); remap[ids] = np.arange(len(ids))
            triangles = model.data['triangles']
            faces = remap[triangles[mask[triangles].all(1)]]
            if name.startswith('gums_'):
                candidates=triangles[mask[triangles].all(1)]
                faces=faces[~model.group('teeth')[candidates].all(1)]
            # Remove unused semantic vertices before constructing a domain.
            used = np.unique(faces)
            ids = ids[used]
            remap2 = np.full(mask.sum(), -1, np.int32); remap2[used] = np.arange(len(used))
            faces = remap2[faces]
            p = rest[ids]
            original_parts[name] = p
            if not len(faces):
                raise ValueError(f'GNM anatomy has no {name} faces')
            if region >= 0:
                if region == 1:
                    closed_v, closed_f = tongue_domain(p,.00025*scale)
                    inner = p[np.argsort(p[:, 2])[:max(8, len(p)//12)]]
                else:
                    closed_v, closed_f, inner = shell(p, faces, .003*scale if region==0 else .0012*scale)
                # Positive orientation for winding-based volume extraction.
                volume = np.einsum('ij,ij->i', closed_v[closed_f[:, 0]],
                                   np.cross(closed_v[closed_f[:, 1]], closed_v[closed_f[:, 2]])).sum()/6
                if volume < 0:
                    closed_f = closed_f[:, ::-1]
                nodes, tets = tetrahedralize(closed_v, closed_f, stage, name, edge=edge if region==0 else min(edge,.002), threads=threads, cancel=cancel, preserve_surface=region>0,timeout=mesh_timeout)
                embedding, weights = embed(p, nodes, tets)
                mass = np.zeros(len(nodes))
                np.add.at(mass, tets.reshape(-1), np.repeat(volumes(nodes,tets)*1000/4, 4))
                if (mass <= 0).any():
                    raise ValueError('unconnected anatomy tetrahedral node')
                inv_mass = 1/mass
                distance, _ = cKDTree(inner).query(nodes)
                anchors = np.flatnonzero(distance < .00045*scale)
                if not len(anchors):
                    anchors = np.argsort(distance)[:max(4,len(nodes)//50)]
                bones = np.zeros(len(anchors), np.int32)
                if name.endswith('lower') or region==1:
                    bones[:] = 1
                elif region==0:
                    bones[:] = ((nodes[anchors,1] < mouth[1]-.007*scale) & (nodes[anchors,2] > hinge[2]+.015*scale)).astype(np.int32)
                arrays['tets'].append(tets+offset)
                arrays['youngs_moduli'].append(np.full(len(tets), 50000 if region==2 else 10000, np.float32))
                arrays['regions'].append(np.full(len(tets),region,np.int32))
                arrays['anchor_ids'].append(anchors+offset);arrays['anchor_bone'].append(bones)
                bone = np.full(len(nodes), -1, np.int32)
                tet_offset += len(tets)
            else:
                nodes, tets = p, np.empty((0,4),np.int32)
                embedding = np.repeat(np.arange(len(p))[:,None],4,axis=1)
                weights = np.tile([1.,0,0,0],(len(p),1))
                inv_mass = np.zeros(len(p))
                bone = np.full(len(p), 1 if name=='teeth_lower' else 0, np.int32)
            arrays['rest'].append(nodes);arrays['inverse_mass'].append(inv_mass);arrays['bone'].append(bone)
            arrays['faces'].append(faces+surface_offset);arrays['surface_rest'].append(p)
            arrays['embed_ids'].append(embedding+offset);arrays['embed_weights'].append(weights);arrays['source_ids'].append(ids)
            parts.append(dict(name=name, region=region, surface_start=surface_offset, surface_count=len(p),
                              face_start=sum(len(v) for v in arrays['faces'][:-1]), face_count=len(faces)))
            offset += len(nodes);surface_offset += len(p)
        arrays = {k: np.concatenate(v) for k,v in arrays.items()}
        bond_semantic_seams(arrays)
        source=arrays['source_ids']
        importance=np.ones(len(source),np.float32)
        importance[(model.group('skin')&~model.group('skin_exterior'))[source]]=0
        importance[(model.group('upper_lip')|model.group('lower_lip'))[source]]=4
        arrays['fit_importance']=importance
        arrays['fit_eye_features']=(model.group('eye_sockets')&model.group('skin_exterior'))[source]
        landmarks = dict(mouth=mouth.tolist(), eye_y=float(rest[model.group('eyes'),1].mean()), jaw=hinge.tolist())
        original_parts['skin_exterior']=rest[model.group('skin_exterior')]
        names, links, fibers, jaw = muscle_atlas(arrays['rest'], arrays['tets'], arrays['regions'], original_parts, landmarks, scale)
        arrays.update(fibers=fibers, jaw_muscles=jaw, jaw_hinge=hinge,jaw_track=jaw_track,expression_coefficients=expression_coefficients,
                      expression_seed_coefficients=np.asarray(expression_mapping['coefficients'],np.float64))
        if fit_sockets or fit_mouth or fit_dentition or fit_tongue or fit_tongue_root:
            arrays['bind_delta'] = bind_delta
        meta = dict(schema=SCHEMA, protocol='VHFULL1', backend='native_lightgeom_cpu',
                    muscles=names, control_prior=links, parts=parts, landmarks=landmarks,
                    rig_sha256=digest(rig_dir/'rig.json'), rig_package_sha256=digest(rig_dir/'rig_deformer.safetensors'),
                    model_sha256=bind['metadata']['model_sha256'], captures=receipts, jaw_fit=jaw_fit,jaw_track_fit=jaw_track_fit,
                    expression_controls=expression_mapping['controls'],expression_fit=expression_fit,
                    control_mapping=control_mapping_report,
                    anatomy_source='GNM+authored_priors', hidden_anatomy='inferred', geometry_fit=geometry_fit,mouth_fit=mouth_fit,dentition_fit=dentition_fit,tongue_fit=tongue_fit,tongue_root_fit=tongue_root_fit,
                    surface_support_envelope_mm=.1,tongue_volume_margin_mm=.25*scale,tongue_volume_prior='convex support envelope; folded original surface retained',
                    material_calibration='authored_animation_defaults', mesher=f'wildmeshing=={MESH_VERSION}',
                    muscle_atlas=MUSCLE_ATLAS,
                    facial_anchor_policy='frontmost exterior skin within a 3 mm XY neighborhood',
                    inverse_eye_features=dict(vertices=int(arrays['fit_eye_features'].sum()),source='GNM skin_exterior & eye_sockets'),
                    neutral_gravity='head_local_equilibrium; zero external acceleration', units='metres',
                    thread_budget=threads,meshing_timeout_seconds=mesh_timeout,geometry_only=True)
        if sliding_skin:
            from .skin_attachments import frames as attachment_frames,orbital_supports
            keep,orbital=orbital_supports(arrays,meta,scale)
            arrays['anchor_ids']=arrays['anchor_ids'][keep];arrays['anchor_bone']=arrays['anchor_bone'][keep]
            normals,tangent,receipt=attachment_frames(arrays,meta,model.group('skin_exterior')[source],scale)
            receipt.update(model='vhuman.skull_normal_tangential_tethers.v2_orbital',orbital_support_release=orbital)
            arrays.update(anchor_normals=normals,anchor_tangent_compliance=tangent)
            meta['skin_attachments']=receipt
        if tangent_fibers:
            from .skin_fibers import tangent_fields
            fibers,receipt=tangent_fields(arrays,meta,model.group('skin_exterior')[source],scale)
            arrays['fibers']=fibers;meta['muscle_geometry']=receipt
            meta['muscle_atlas']='vhuman.authored_surface_bands.v5_ocular'
        if harmonic_tongue_fibers:
            from .tongue_fibers import fit_fields
            fibers,receipt=fit_fields(arrays,meta,cancel=cancel)
            arrays['fibers']=fibers;meta['tongue_muscle_geometry']=receipt
            write_json(stage/'tongue_fibers.json',receipt)
        if extrinsic_tongue_cables:
            from .tongue_cables import fit as fit_external_muscles
            cables,receipt=fit_external_muscles(arrays,meta,scale=scale,cancel=cancel)
            arrays['tongue_cables']=cables;meta['tongue_extrinsic_cables']=receipt
            write_json(stage/'tongue_cables.json',receipt)
        validate(arrays, meta)
        meta['rest_surface_preflight']=surface_preflight(arrays)
        np.savez_compressed(stage/'anatomy.npz', **arrays)
        meta['arrays_sha256'] = digest(stage/'anatomy.npz')
        write_json(stage/'anatomy.json', meta)
        # Keep receipts/logs, not meshing work arrays, in the reviewable package.
        for path in stage.glob('*-input.npz'):
            path.unlink()
        for path in stage.glob('*-volume.npz'):
            path.unlink()
        (stage/'__tracked_surface.stl').unlink(missing_ok=True)
        stage.rename(out)
        if progress:
            progress(.99, 'anatomy ready; source geometry review required' if not meta['rest_surface_preflight']['passed'] else 'anatomy ready')
        return dict(package=str(out), vertices=len(arrays['rest']), tetrahedra=len(arrays['tets']),
                    muscles=len(names), arrays_sha256=meta['arrays_sha256'], geometry_only=True,
                    rest_surface_preflight=meta['rest_surface_preflight'])
    except BaseException:
        # Meshing failure logs remain inspectable; never promote a partial fit.
        raise


@contextmanager
def worker_cancellation():
    """Let the parent stop nested mesher/encoder process groups cleanly."""
    import signal
    cancel=threading.Event()
    saved={sig:signal.signal(sig,lambda *_:cancel.set()) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        yield cancel
    finally:
        for sig,handler in saved.items():
            signal.signal(sig,handler)


def run_worker(module, arguments, *, python=None, progress=None, cancel=None):
    """Explicit CPU worker boundary; never inherit auto GPU selection."""
    from .. import gpu
    from ..process import run
    py=Path(python) if python else ROOT/'tmp/vhuman-cpu-venv/bin/python'
    if not py.is_file():
        raise FileNotFoundError(f'CPU fitting interpreter missing: {py}; run setup_cpu_fit.sh --full')
    command=[str(py),'-m','server.vhuman.runtime','--backend','cpu','--device','0',
             '--models-root',str(gpu.model_path('')),'--module',module,*map(str,arguments)]
    tail=[]
    def line(text):
        tail.append(text)
        if len(tail)>30:
            del tail[0]
        if progress and text.startswith('@progress '):
            _,fraction,message=text.split(' ',2)
            progress(float(fraction),message)
    status=run(command,cancel=cancel or threading.Event(),on_line=line,cwd=ROOT,
               env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp')))
    if status:
        raise RuntimeError('CPU biomechanics worker failed: '+'\n'.join(tail))


def anatomy_job(service,request,progress,cancel,*,python=None):
    if any(type(request.get(flag,False)) is not bool for flag in ('fit_sockets','fit_mouth','fit_dentition','fit_tongue','fit_tongue_root','sliding_skin','tangent_fibers','harmonic_tongue_fibers','extrinsic_tongue_cables')):
        raise ValueError('geometry fitting and skin mechanics flags must be booleans')
    validate_fitting_budget(request.get('fit_mouth',False),request.get('mouth_fairing',0.),request.get('mesh_timeout',3600))
    rig=service.rig_file(request.get('head_id'),'rig.json').parent
    captures=[service.take_file(request['head_id'],take,'motion.json').parent
              for take in request.get('take_ids',[])]
    threads=request.get('threads',4)
    arguments=['fit',rig,'--threads',threads,'--mouth-fairing',request.get('mouth_fairing',0.),
               '--mesh-timeout',request.get('mesh_timeout',3600)]
    if request.get('fit_sockets',False):
        arguments.append('--fit-sockets')
    if request.get('fit_mouth',False):
        arguments.append('--fit-mouth')
    if request.get('fit_dentition',False):
        arguments.append('--fit-dentition')
    if request.get('fit_tongue',False):
        arguments.append('--fit-tongue')
    if request.get('fit_tongue_root',False):
        arguments.append('--fit-tongue-root')
    if request.get('sliding_skin',False):
        arguments.append('--sliding-skin')
    if request.get('tangent_fibers',False):
        arguments.append('--tangent-fibers')
    if request.get('harmonic_tongue_fibers',False):
        arguments.append('--harmonic-tongue-fibers')
    if request.get('extrinsic_tongue_cables',False):
        arguments.append('--extrinsic-tongue-cables')
    if captures:
        arguments+=['--captures',*captures]
    run_worker('server.vhuman.rig.anatomy',arguments,python=python,progress=progress,cancel=cancel)
    meta=json.loads((rig/'anatomy/anatomy.json').read_text())
    return dict(head_id=request['head_id'],package=str(rig/'anatomy'),report=meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('rig', type=Path);fit.add_argument('--out',type=Path)
    fit.add_argument('--captures', nargs='*',type=Path,default=[])
    fit.add_argument('--fit-sockets',action='store_true',help='condition inferred ocular cavities and bounded eyelid clearance')
    fit.add_argument('--fit-mouth',action='store_true',help='separate overlapping lips and condition inferred oral lining')
    fit.add_argument('--mouth-fairing',type=float,default=0.,help='optional hidden-lining displacement fairing (0..0.5); requires --fit-mouth')
    fit.add_argument('--mesh-timeout',type=int,default=3600,help='per-region CPU meshing timeout in seconds (60..7200)')
    fit.add_argument('--fit-dentition',action='store_true',help='separate overlapping tooth crowns by bounded rigid translations')
    fit.add_argument('--fit-tongue',action='store_true',help='untangle local tongue folds with bounded area-preserving fairing')
    fit.add_argument('--fit-tongue-root',action='store_true',help='fit inferred ventral tongue base inside the lower arch; includes fold fitting')
    fit.add_argument('--sliding-skin',action='store_true',help='use authored skull-normal support and tangential facial skin tethers')
    fit.add_argument('--tangent-fibers',action='store_true',help='fit facial fibers to the exterior skin shell tangent planes')
    fit.add_argument('--harmonic-tongue-fibers',action='store_true',help='fit authored tongue compartments and harmonic volume directions on CPU')
    fit.add_argument('--extrinsic-tongue-cables',action='store_true',help='add authored external tongue muscle paths with distributed tissue insertions')
    fit.add_argument('--threads',type=int,default=4);fit.add_argument('--edge',type=float,default=.004)
    worker = sub.add_parser('mesh')
    worker.add_argument('source',type=Path);worker.add_argument('destination',type=Path)
    worker.add_argument('--threads',type=int,default=4);worker.add_argument('--edge',type=float,default=.004)
    worker.add_argument('--preserve-surface',action='store_true')
    args = ap.parse_args(argv)
    if args.command=='mesh':
        mesh_worker(args.source,args.destination,edge=args.edge,threads=args.threads,preserve_surface=args.preserve_surface)
    else:
        with worker_cancellation() as cancel:
            result=prepare(args.rig,out=args.out,captures=args.captures,threads=args.threads,edge=args.edge,cancel=cancel,fit_sockets=args.fit_sockets,fit_mouth=args.fit_mouth,fit_dentition=args.fit_dentition,fit_tongue=args.fit_tongue,fit_tongue_root=args.fit_tongue_root,sliding_skin=args.sliding_skin,tangent_fibers=args.tangent_fibers,harmonic_tongue_fibers=args.harmonic_tongue_fibers,extrinsic_tongue_cables=args.extrinsic_tongue_cables,mouth_fairing=args.mouth_fairing,mesh_timeout=args.mesh_timeout,
                           progress=lambda fraction,text: print(f'@progress {fraction:.3f} {text}',flush=True))
        print(json.dumps(result))


if __name__=='__main__':
    main()
