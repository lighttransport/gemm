"""Experimental generated-view fitting; requested angles are never calibration."""
from __future__ import annotations

import copy
import json
import math
import uuid
from pathlib import Path

import numpy as np

from . import fitting, observations
from .reference import Camera, rasterize
from .correspondence import occlusion_weight


class MultiviewRejected(ValueError):
    pass


def withheld_error(vertices, camera, view):
    """The exact dense landmark split withheld by fit_staged, in 1024px units."""
    names = sorted(n for n in view['anchors'] if n.startswith('mp_'))[::10]
    if len(names) < 20:
        raise MultiviewRejected('dense frontal landmarks required for independent fit validation')
    selected = {n: view['anchors'][n] for n in names if view['anchors'][n].get('weight', 1) > 0}
    rows = fitting.anchor_indices(vertices, camera, selected)
    errors = []
    for row in rows:
        if row[3] <= 0 or occlusion_weight(view, row[2]) <= 0:
            continue
        projected, depth = camera.project(fitting.attached_point(vertices, row, selected)[None])
        if depth[0] <= 0:
            return float('inf')
        errors.append(float(np.linalg.norm(projected[0]-row[2])))
    if not errors:
        raise MultiviewRejected('no usable withheld frontal landmarks')
    return float(np.mean(errors))*1024/max(view['size'])


def check_front(vertices, camera, view, gate):
    error = withheld_error(vertices, camera, view)
    baseline = float(gate['baseline_error_1024px'])
    limit = baseline+max(2., baseline*.1)
    if not np.isfinite(error) or error > limit:
        raise MultiviewRejected(f'frontal consistency rejected: {error:.2f}px exceeds {limit:.2f}px '
                                f'(baseline {baseline:.2f}px, normalised to 1024)')
    return dict(baseline_error_1024px=baseline, candidate_error_1024px=error,
                limit_1024px=limit, accepted=True)


def visible_points(vertices, triangles, points, camera, size):
    """Visibility in the fitted head, including self occlusion and back faces."""
    from ..rig.common import vertex_normals
    from scipy.spatial import cKDTree
    scale = min(1., 192/max(size))
    small_size = tuple(max(1, round(s*scale)) for s in size)
    _, _, depth = rasterize(vertices, triangles, camera.scaled(scale), small_size)
    xy, z = camera.project(points)
    normal = vertex_normals(vertices, triangles)[cKDTree(vertices).query(points)[1]]
    ray = camera.origin-points
    facing = np.sum(normal*ray, axis=1) > 0
    ix, iy = np.floor(xy*scale).astype(int).T
    inside = (ix >= 0)&(iy >= 0)&(ix < small_size[0])&(iy < small_size[1])&(z > .02)
    sampled = depth[np.clip(iy, 0, small_size[1]-1), np.clip(ix, 0, small_size[0]-1)]
    return inside & facing & (abs(sampled-z) < .004)


def estimate_camera(vertices, triangles, view, front_camera, yaw_deg):
    """Robust fixed-intrinsic PnP, with requested pose as an initialization only."""
    import cv2
    from .dense_landmarks import MP68
    from scipy.spatial.transform import Rotation

    w, h = view['size']
    focal = .5*w/math.tan(math.radians(10))
    k = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1.]], np.float64)
    excluded = {f'mp_{int(n):03d}' for n in MP68[:17]}  # silhouette points slide with pose
    anchors = {n: a for n, a in view['anchors'].items() if n.startswith('mp_') and n not in excluded
               and int(n[3:]) < 468 and a.get('weight', 1) > .05}
    rows = fitting.anchor_indices(vertices, front_camera, anchors)
    if len(rows) < 12:
        raise MultiviewRejected('fewer than 12 usable anatomical side-view landmarks')
    points = np.array([fitting.attached_point(vertices, row, anchors) for row in rows], np.float64)
    pixels = np.array([row[2] for row in rows], np.float64)
    centre = np.mean(points, axis=0)
    orbit = Rotation.from_rotvec(front_camera.rotation[1]*math.radians(yaw_deg)).as_matrix()
    initial = Camera(focal, w/2, h/2,
                     centre+orbit@(front_camera.origin-centre), front_camera.rotation@orbit.T)
    convert = np.diag([1., -1., -1.])
    rvec = cv2.Rodrigues(convert@initial.rotation)[0]
    tvec = -(convert@initial.rotation@initial.origin)[:, None]
    eligible = visible_points(vertices, triangles, points, initial, (w, h))
    if eligible.sum() < 12:
        raise MultiviewRejected('requested side has fewer than 12 visible anatomical landmarks')
    inliers = None
    for _ in range(2):
        chosen = np.flatnonzero(eligible)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(points[chosen], pixels[chosen], k, None,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True, iterationsCount=150,
            reprojectionError=12*max(w, h)/1024, confidence=.999, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < 12:
            raise MultiviewRejected('camera fit has fewer than 12 reprojection inliers')
        rcv = cv2.Rodrigues(rvec)[0]
        camera = Camera(focal, w/2, h/2, (-rcv.T@tvec).ravel(), convert@rcv)
        eligible = visible_points(vertices, triangles, points, camera, (w, h))
        if eligible.sum() < 12:
            raise MultiviewRejected('estimated camera has insufficient visible landmarks')
    back = camera.rotation[2]
    actual_yaw = math.degrees(math.atan2(back@front_camera.rotation[0], back@front_camera.rotation[2]))
    delta = abs((actual_yaw-yaw_deg+180) % 360-180)
    projected, _ = camera.project(points)
    error = np.linalg.norm(projected-pixels, axis=1)*1024/max(w, h)
    accepted = eligible & (error <= 24)
    if delta > 35 or accepted.sum() < 12 or np.median(error[accepted]) > 15:
        raise MultiviewRejected(f'pose/landmark check rejected: requested {yaw_deg}°, fitted {actual_yaw:.1f}°')
    # No far-side hallucinated tracker points, sliding contour correspondences,
    # or provisional nearest-vertex attachments enter the synthetic fit.
    view['anchors'] = {row[0]: dict(anchors[row[0]], weight=float(anchors[row[0]].get('weight', 1))*.35)
                       for row, take in zip(rows, accepted) if take}
    view.update(camera=camera.as_dict(), camera_kind='estimated', synthetic=True)
    return dict(requested_yaw_deg=yaw_deg, estimated_yaw_deg=actual_yaw, inliers=int(accepted.sum()),
                median_error_1024px=float(np.median(error[accepted])),
                p95_error_1024px=float(np.percentile(error[accepted], 95)),
                intrinsics='assumed 20 degree horizontal FOV; principal point at image centre',
                metric_scale='front anchor assumed IPD; not measured scale')


def run_generated(folder, input_file, **options):
    """Keep a usable baseline even when an experimental multiview candidate fails."""
    from .pipeline import run
    folder, input_file = Path(folder), Path(input_file)
    inputs = json.loads(input_file.read_text())
    if inputs.get('format') != 'vhuman.generated_face_inputs.v1' or not 1 <= len(inputs.get('views', [])) <= 5:
        raise ValueError('invalid generated face input bundle')
    for row in inputs['views']:
        image = (input_file.parent/row['file']).resolve()
        if not image.is_relative_to(input_file.parent.resolve()) or observations.sha256(image) != row['sha256']:
            raise ValueError('generated reconstruction input checksum mismatch')
    if options.get('face_model') != 'gnm_v3':
        raise ValueError('generated multiview fitting requires GNM v3')
    provenance = copy.deepcopy(inputs)
    baseline_id = options.pop('run_id', None) or uuid.uuid4().hex[:16]
    baseline = run(folder, run_id=baseline_id, input_provenance=provenance, **options)
    review = dict(format='vhuman.generated_multiview_review.v1', synthetic=True,
                  baseline_run_id=baseline_id, selected_run_id=baseline_id, status='front_only', views=[])
    if len(inputs['views']) > 1:
        try:
            base = folder/'reconstruction'/baseline_id
            front_doc = observations.load(base/'observations.json')
            front = front_doc['views'][0]
            front.update(camera=baseline['geometry']['fitted_cameras'][0], camera_kind='estimated', synthetic=True,
                         generated_view_id=inputs['views'][0]['id'], requested_yaw_deg=0)
            front_camera = Camera.from_dict(front['camera'])
            with np.load(base/'geometry.npz', allow_pickle=False) as geometry:
                points, triangles = geometry['captured'][0].copy(), geometry['triangles'].copy()
            front_error = withheld_error(points, front_camera, front)
            views = [front]
            for row in inputs['views'][1:]:
                record = dict(view_id=row['id'], requested_yaw_deg=row['yaw_deg'])
                try:
                    image = input_file.parent/row['file']
                    view = observations.observe(image, dense=True)['views'][0]
                    view['image_path'] = str(image)
                    view.update(generated_view_id=row['id'], requested_yaw_deg=row['yaw_deg'])
                    record.update(estimate_camera(points, triangles, view, front_camera, row['yaw_deg']), accepted=True)
                    views.append(view)
                except (ValueError, RuntimeError) as exc:
                    record.update(accepted=False, reason=str(exc))
                review['views'].append(record)
            if len(views) == 1:
                raise MultiviewRejected('no side view passed camera and visibility checks')
            work = folder/'generated_inputs'
            work.mkdir(exist_ok=True)
            doc = dict(front_doc, views=views, provenance=provenance)
            # Absolute paths are internal to the offline worker, never accepted from HTTP.
            for view in views:
                view['image'] = view['image_path']
                for key in ('exclusion_mask', 'silhouette_mask'):
                    if view.get(key+'_path'):
                        view[key] = view[key+'_path']
            obs_path = work/'observations.json'
            obs_path.write_text(json.dumps(doc, indent=2))
            seed = dict(baseline['geometry'])
            n = len(views)
            seed['pose_translations'] = [[0., 0., 0.] for _ in range(n)]
            seed['pose_rotations'] = [[0., 0., 0.] for _ in range(n)]
            seed['expression_coefficients'] = [baseline['geometry']['expression_coefficients'][0] for _ in range(n)]
            joint_id = uuid.uuid4().hex[:16]
            joint_options = dict(options, observation_file=obs_path)
            joint = run(folder, run_id=joint_id, input_provenance=provenance, fit_seed=seed,
                fixed_pose_views=(0,), protect_front=True,
                front_gate=dict(baseline_error_1024px=front_error), **joint_options)
            cameras = [Camera.from_dict(c) for c in joint['geometry']['fitted_cameras']]
            accepted = [r for r in review['views'] if r['accepted']]
            for record, camera in zip(accepted, cameras[1:]):
                back = camera.rotation[2]
                record['fitted_yaw_deg'] = math.degrees(math.atan2(back@cameras[0].rotation[0], back@cameras[0].rotation[2]))
            review.update(status='accepted', selected_run_id=joint_id, multiview_run_id=joint_id)
        except Exception as exc:
            review.update(status='rejected' if isinstance(exc, ValueError) else 'failed', reason=str(exc))
    for rid in {baseline_id, review['selected_run_id']}:
        (folder/'reconstruction'/rid/'multiview_review.json').write_text(json.dumps(review, indent=2))
    baseline['generated_multiview'] = review
    return baseline
