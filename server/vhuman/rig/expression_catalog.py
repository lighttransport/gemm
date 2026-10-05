"""Generate seven basic emotions and synchronized observation/debug movies.

The mesh is the subject's fitted GNM rig, animated by temporally fitted
MediaPipe controls. GNM PCA coefficients are exported separately; this is
monocular rig fitting, not an independently recovered 3D ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..face_assets import asset_path, sha256
from ..face_parsing import FaceParser, LABELS
from ..native_landmarks import FaceLandmarker
from ..video_backend import select
from .exprdata import KEEP

EMOTIONS = {
    'neutral': 'The person maintains a calm neutral expression, relaxed brows and closed lips.',
    'happy': 'The person becomes happy, smiling broadly with raised cheeks and joyful eyes.',
    'sad': 'The person becomes sad, inner brows raised and mouth corners turned downward.',
    'angry': 'The person becomes angry, eyebrows lowered and drawn together, eyes tense and lips pressed.',
    'fear': 'The person becomes afraid, eyebrows raised and drawn together, eyes wide and lips stretched apart.',
    'surprise': 'The person becomes surprised, eyebrows raised high, eyes wide and mouth open in astonishment.',
    'disgust': 'The person shows disgust, wrinkling the nose and raising the upper lip, eyebrows lowered.',
}
MOVIES = ('generated', 'landmarks', 'gnm_mesh', 'face_parsing', 'four_panel')


def gnm_anchors(source):
    """Semantic anchors from GNM's own lip/chin/socket groups and vertex order."""
    with np.load(asset_path('gnm'), allow_pickle=False) as z:
        names = z['vertex_group_names'].tolist()
        exterior = z['vertex_groups'][names.index('skin_exterior')] > .5
        masks = {name: z['vertex_groups'][names.index(name), exterior] > .5
                 for name in ('upper_lip', 'lower_lip', 'chin_region', 'eye_sockets')}
    v = source.vertices
    upper = np.flatnonzero(masks['upper_lip'] & (np.abs(v[:, 0]) < .002))
    lower = np.flatnonzero(masks['lower_lip'] & (np.abs(v[:, 0]) < .002))
    chin = np.flatnonzero(masks['chin_region'] & (np.abs(v[:, 0]) < .006))
    lips = np.flatnonzero(masks['upper_lip'] | masks['lower_lip'])
    return [np.array([v[:, 2].argmax()]), np.array([upper[v[upper, 1].argmin()]]),
            np.array([lower[v[lower, 1].argmax()]]), np.array([chin[v[chin, 1].argmin()]]),
            np.array([lips[v[lips, 0].argmin()]]), np.array([lips[v[lips, 0].argmax()]]),
            np.flatnonzero(masks['eye_sockets'] & (v[:, 0] < 0)),
            np.flatnonzero(masks['eye_sockets'] & (v[:, 0] > 0))]


def write_json(path, value):
    stage = path.with_suffix('.partial')
    stage.write_text(json.dumps(value, indent=2))
    stage.replace(path)


def similarity_camera(base, target):
    """Least-squares positive-scale, proper-rotation 2D cameras, row vectors."""
    base, target = np.asarray(base), np.asarray(target)
    if base.shape != target.shape or base.ndim != 3 or base.shape[-1] != 2:
        raise ValueError('camera anchors must have matching (frames, anchors, 2) shapes')
    if not np.isfinite(base).all() or not np.isfinite(target).all():
        raise ValueError('camera anchors must be finite')
    bc, tc = base.mean(1, keepdims=True), target.mean(1, keepdims=True)
    a, b = base - bc, target - tc
    variance = np.square(a).sum((1, 2))
    if (variance <= 1e-12).any():
        raise ValueError('degenerate camera anchors')
    u, singular, vt = np.linalg.svd(a.transpose(0, 2, 1) @ b)
    fix = np.broadcast_to(np.eye(2), u.shape).copy()
    fix[:, 1, 1] = np.linalg.det(u @ vt)
    rotation = u @ fix @ vt
    scale = (singular * np.diagonal(fix, axis1=1, axis2=2)).sum(1) / variance
    if (scale <= 0).any():
        raise ValueError('camera scale must be positive')
    angle = np.arctan2(rotation[:, 0, 1], rotation[:, 0, 0])
    center = tc[:, 0] - scale[:, None] * (bc @ rotation)[:, 0]
    return scale, angle, center


def fit_pca_residual(matrix, residual, ridge=1e-7, prior=None):
    """Minimum-norm regional PCA correction, solved in landmark space.

    matrix: (frames, observed coordinates, PCA coefficients). The coefficient
    bound retains the model's established [-3, 3] expression domain.
    """
    matrix, residual = np.asarray(matrix, np.float64), np.asarray(residual, np.float64)
    if matrix.ndim != 3 or residual.shape != matrix.shape[:2] or ridge <= 0:
        raise ValueError('invalid PCA observation system')
    if not np.isfinite(matrix).all() or not np.isfinite(residual).all():
        raise ValueError('PCA observations must be finite')
    gram = matrix @ matrix.transpose(0, 2, 1)
    dual = np.linalg.solve(gram + ridge * np.eye(matrix.shape[1]), residual[..., None])
    coefficients = (matrix.transpose(0, 2, 1) @ dual)[..., 0]
    lower, upper = -3., 3.
    if prior is not None:
        prior = np.asarray(prior, np.float64)
        if prior.shape != coefficients.shape or not np.isfinite(prior).all() or (np.abs(prior) > 3.000001).any():
            raise ValueError('invalid PCA prior')
        lower, upper = np.maximum(-3., -3. - prior), np.minimum(3., 3. - prior)
        # Projected accelerated gradient redistributes motion to unsaturated
        # regional modes; clipping an unconstrained solution loses that motion.
        coefficients = np.clip(coefficients, lower, upper)
        accelerated, momentum = coefficients.copy(), 1.
        step = 1. / (np.linalg.eigvalsh(gram)[:, -1] + ridge)
        for _ in range(400):
            error = np.einsum('fde,fe->fd', matrix, accelerated) - residual
            gradient = np.einsum('fde,fd->fe', matrix, error) + ridge * accelerated
            updated = np.clip(accelerated - step[:, None] * gradient, lower, upper)
            next_momentum = (1. + np.sqrt(1. + 4. * momentum * momentum)) / 2.
            accelerated = updated + ((momentum - 1.) / next_momentum) * (updated - coefficients)
            coefficients, momentum = updated, next_momentum
    return np.clip(coefficients, lower, upper).astype(np.float32)


class Movie:
    def __init__(self, path, width, height, fps):
        self.proc = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
            '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
            '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)

    def frame(self, image):
        self.proc.stdin.write(np.ascontiguousarray(image, np.uint8).tobytes())

    def close(self):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        if self.proc.wait():
            raise RuntimeError('video encoder failed')


def debug(clip, rig_dir, out, emotion, *, identity_coefficients=None):
    import cv2
    import torch
    from . import safetensors as st
    from .face_models import load
    from .gnm_expression import evaluate
    from .torchrig import TorchRig
    from .video_fit import ANCHORS

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    images, faces = [], []
    try:
        with FaceLandmarker(asset_path('mediapipe')) as tracker:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                images.append(bgr)
                detected = tracker.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                faces.append(detected[0] if len(detected) == 1 else None)
    finally:
        cap.release()
    if not images or not np.isfinite(fps) or fps <= 0:
        raise ValueError('invalid generated video')
    valid = np.array([f is not None for f in faces])
    if not valid.any():
        raise ValueError('no single face observed in generated clip')
    definition = json.loads((rig_dir / 'rig.json').read_text())
    if definition.get('face_model') != 'gnm_v3':
        raise ValueError('catalog mesh requires a fitted GNM v3 rig')
    names = [c['name'] for c in definition['controls']]
    direct = np.zeros((len(images), len(names)), np.float32)
    landmarks = np.zeros((len(images), len(ANCHORS), 2), np.float32)
    for i, face in enumerate(faces):
        if face is None:
            continue
        direct[i] = [face['blendshapes'].get(n, 0) for n in names]
        for j, (_, index) in enumerate(ANCHORS):
            landmarks[i, j] = face['landmarks'][list(index) if isinstance(index, tuple) else [index], :2].mean(0)
    ids = np.flatnonzero(valid)
    for array in (direct, landmarks):
        flat = array.reshape(len(images), -1)
        for j in range(flat.shape[1]):
            flat[:, j] = np.interp(np.arange(len(images)), ids, flat[ids, j])
    package, metadata = st.load(rig_dir / 'rig_deformer.safetensors')
    morphs = json.loads(metadata['morphs'])
    shapes = {n: package['morph'][i] for i, n in enumerate(morphs)}
    source = load('gnm_v3')
    if len(source.vertices) != definition['face_model_source']['vertices'] or len(package['rest']) < len(source.vertices):
        raise ValueError('GNM rig topology does not match pinned source')
    groups = gnm_anchors(source)
    indices = np.concatenate(groups)
    cuts = np.cumsum([0] + [len(group) for group in groups])
    anchor_rig = TorchRig(definition, package['rest'][indices],
        {n: delta[indices] for n, delta in shapes.items()}, package['skin.joints'][indices],
        package['skin.weights'][indices], device='cpu')

    alignment = definition['face_model_source']['expression_basis_alignment']
    identity_basis = alignment['scale'] * np.einsum('ivc,dc->ivd', source.identity_basis,
                                                   np.asarray(alignment['rotation']))
    initial_identity = np.zeros(len(identity_basis), np.float32) if identity_coefficients is None else np.asarray(identity_coefficients, np.float32)
    if initial_identity.shape != (len(identity_basis),) or not np.isfinite(initial_identity).all() or (np.abs(initial_identity) > 3).any():
        raise ValueError('invalid shared GNM identity coefficients')
    identity = torch.nn.Parameter(torch.tensor(initial_identity), requires_grad=identity_coefficients is None)
    anchor_basis = torch.tensor(identity_basis[:, indices], dtype=torch.float32)

    def anchor_positions(values):
        offset = torch.einsum('i,ivc->vc', identity, anchor_basis)[None]
        pos = anchor_rig(values, pre=offset)['pos']
        return torch.stack([pos[:, cuts[j]:cuts[j + 1]].mean(1) for j in range(8)], 1)
    torch.set_num_threads(4)
    prior = torch.tensor(direct)
    target = torch.tensor(landmarks)
    with torch.no_grad():
        base = anchor_positions(prior)[..., :2] * torch.tensor([1., -1.])
        initial_scale, initial_angle, initial_center = [torch.tensor(v, dtype=torch.float32)
            for v in similarity_camera(base.numpy(), target.numpy())]
    controls = torch.nn.Parameter(prior.clone())
    log_scale = torch.nn.Parameter(initial_scale.log())
    angle = torch.nn.Parameter(initial_angle)
    center = torch.nn.Parameter(initial_center)

    def project(pos):
        c, s = angle.cos(), angle.sin()
        matrix = torch.stack((c, s, -s, c), -1).reshape(-1, 2, 2)
        xy = pos[..., :2] * torch.tensor([1., -1.])
        return (xy @ matrix) * log_scale.exp()[:, None, None] + center[:, None]

    confidence = torch.tensor(valid.astype(np.float32))[:, None, None]
    parameters = [{'params': [controls, log_scale, angle, center], 'lr': .008}]
    if identity.requires_grad:
        parameters.append({'params': [identity], 'lr': .06})
    optimizer = torch.optim.Adam(parameters)
    with torch.no_grad():
        before = torch.linalg.vector_norm(project(anchor_positions(prior)) - target, dim=-1)
    for _ in range(180):
        optimizer.zero_grad()
        prediction = project(anchor_positions(controls))
        loss = ((prediction - target).square() * confidence).mean()
        loss += .0002 * (controls - prior).square().mean()
        loss += .0001 * (controls[1:] - controls[:-1]).square().mean()
        loss += .000001 * identity.square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            controls.clamp_(anchor_rig.lo, anchor_rig.hi)
            identity.clamp_(-3, 3)
    with torch.no_grad():
        anchor_result = anchor_rig(controls, pre=torch.einsum('i,ivc->vc', identity, anchor_basis)[None])
        baseline_anchors = torch.stack([anchor_result['pos'][:, cuts[j]:cuts[j + 1]].mean(1) for j in range(8)], 1)
        baseline_projected = project(baseline_anchors)
        expression_basis = alignment['scale'] * np.einsum('evc,dc->evd', source.expression_basis,
                                                         np.asarray(alignment['rotation']))
        local_basis = torch.tensor(expression_basis[:, indices], dtype=torch.float32)
        posed_basis = torch.einsum('svcd,evd->sevc', anchor_result['blend'], local_basis)
        averaged = torch.stack([posed_basis[:, :, cuts[j]:cuts[j + 1]].mean(2) for j in range(8)], 2)
        c, s = angle.cos(), angle.sin()
        camera_rotation = torch.stack((c, s, -s, c), -1).reshape(-1, 2, 2)
        projected_basis = (averaged[..., :2] * torch.tensor([1., -1.])) @ camera_rotation[:, None]
        projected_basis *= log_scale.exp()[:, None, None, None]
        matrix = projected_basis.permute(0, 2, 3, 1).reshape(len(images), 16, -1).numpy()
        prior_gnm = np.stack([evaluate(definition['face_model_source']['expression_mapping'],
                              dict(zip(names, map(float, row)))) for row in controls.detach().numpy()])
        correction = fit_pca_residual(matrix, (target - baseline_projected).reshape(len(images), 16).numpy(), prior=prior_gnm)
        # A short symmetric temporal filter avoids fitting tracker jitter.
        if len(correction) > 2:
            correction[1:-1] = (.2 * correction[:-2] + .6 * correction[1:-1] + .2 * correction[2:])
        correction = np.clip(correction, np.maximum(-3., -3. - prior_gnm),
                             np.minimum(3., 3. - prior_gnm)).astype(np.float32)
        corrected = baseline_projected + torch.tensor(np.einsum('sde,se->sd', matrix, correction).reshape(len(images), 8, 2))
        after = torch.linalg.vector_norm(corrected - target, dim=-1)
    fitted = controls.detach().numpy()
    report = dict(backend='cpu', device='cpu', frames=len(images), valid_frames=int(valid.sum()),
        landmark_error_before_normalized=float(before[valid].mean()),
        landmark_error_after_normalized=float(after[valid].mean()),
        identity_coefficients=identity.detach().tolist(),
        expression_fit='Transferred controls plus bounded GNM regional PCA residual fit to semantic landmarks.',
        identity_mode='fit once from neutral' if identity_coefficients is None else 'fixed shared neutral identity',
        projection='Shared GNM identity PCA refinement and per-frame orthographic camera jointly fitted with rig controls; no per-landmark offsets.')
    rig = TorchRig(definition, package['rest'], shapes, package['skin.joints'],
                   package['skin.weights'], device='cpu')
    identity_offset = np.zeros_like(package['rest'])
    identity_offset[:len(source.vertices)] = np.einsum('i,ivc->vc', identity.detach().numpy(), identity_basis)
    h, w = images[0].shape[:2]
    parser = FaceParser()
    palette = np.array([(0, 0, 0), (190, 170, 130), (20, 190, 255), (40, 140, 255),
        (70, 240, 70), (130, 230, 40), (220, 100, 40), (180, 80, 220),
        (220, 60, 180), (0, 220, 220), (0, 180, 255), (80, 70, 220),
        (80, 110, 255), (150, 80, 230), (170, 160, 80), (60, 170, 180),
        (170, 90, 30), (130, 90, 160), (100, 160, 160)], np.uint8)
    writers = {n: Movie(out / f'{n}.mp4', w * (2 if n == 'four_panel' else 1),
                       h * (2 if n == 'four_panel' else 1), fps) for n in MOVIES}
    observations, parsing, mesh_positions = [], [], []
    try:
        for i, image in enumerate(images):
            if i % 5 == 0:
                print(f'{emotion}: debug frame {i + 1}/{len(images)}', flush=True)
            landmark = (image.astype(np.float32) * .35).astype(np.uint8)
            if faces[i] is not None:
                points = faces[i]['landmarks'][:, :2] * [w, h]
                for x, y in np.rint(points).astype(int):
                    cv2.circle(landmark, (x, y), 1, (80, 255, 100), -1)
            labels = parser(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            parsing.append(labels)
            parsed = palette[labels]
            with torch.no_grad():
                pre = identity_offset.copy()
                pre[:len(source.vertices)] += np.einsum('e,evc->vc', correction[i], expression_basis)
                pos = rig(torch.tensor(fitted[i:i + 1]), pre=torch.tensor(pre)[None])['pos'][0].numpy()
            mesh_positions.append(pos[:len(source.vertices)].copy())
            c, s = float(angle[i].detach().cos()), float(angle[i].detach().sin())
            camera_rotation = np.array([[c, s], [-s, c]])
            xy = np.stack((pos[:, 0], -pos[:, 1]), -1) @ camera_rotation
            xy = xy * float(log_scale[i].detach().exp()) + center[i].detach().numpy()
            xy = xy * [w, h]
            tri = source.triangles
            from .preview import render
            screen = np.column_stack((xy[:, 0] - w / 2, h / 2 - xy[:, 1],
                                      pos[:, 2] * float(log_scale[i].detach().exp()) * w))
            side = max(w, h)
            raster = render(screen, tri, size=side, center=[0, 0, 0], extent=side / 2,
                            wire=True, bg=(22, 22, 22))
            mesh = raster[(side - h) // 2:(side + h) // 2,
                          (side - w) // 2:(side + w) // 2, ::-1].copy()
            panels = dict(generated=image.copy(), landmarks=landmark,
                          gnm_mesh=mesh, face_parsing=parsed)
            for label, title in enumerate(LABELS):
                x, y = (label // 10) * (w // 2) + 8, h - 162 + (label % 10) * 16
                cv2.rectangle(parsed, (x - 2, y - 11), (x + 215, y + 3), (12, 12, 12), -1)
                cv2.rectangle(parsed, (x, y - 8), (x + 8, y), tuple(map(int, palette[label])), -1)
                cv2.putText(parsed, title, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, .35,
                            (255, 255, 255), 1, cv2.LINE_AA)
            for name, panel in panels.items():
                cv2.rectangle(panel, (0, 0), (w, 44), (12, 12, 12), -1)
                cv2.putText(panel, f'{emotion.upper()} | {name.replace("_", " ")}',
                            (9, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
                if not valid[i] and name != 'generated':
                    cv2.putText(panel, 'FACE MISSING: interpolated fit', (8, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 30, 255), 1)
                writers[name].frame(panel)
            writers['four_panel'].frame(np.vstack((np.hstack((panels['generated'], panels['landmarks'])),
                                                    np.hstack((panels['gnm_mesh'], panels['face_parsing'])))))
            controls = dict(zip(names, map(float, fitted[i])))
            observations.append(dict(frame=i, seconds=i / fps, visible=bool(valid[i]),
                landmarks=None if faces[i] is None else faces[i]['landmarks'].tolist(),
                measured_controls=None if faces[i] is None else faces[i]['blendshapes'],
                fitted_controls=controls,
                gnm_coefficients=(prior_gnm[i] + correction[i]).tolist(),
                gnm_residual_coefficients=correction[i].tolist(),
                landmark_error_normalized=float(after[i].mean()),
                camera=dict(scale=float(log_scale[i].detach().exp()),
                            roll_radians=float(angle[i].detach()), center=center[i].detach().tolist())))
    finally:
        encoder_errors = []
        for writer in writers.values():
            try:
                writer.close()
            except RuntimeError as exc:
                encoder_errors.append(exc)
        if encoder_errors:
            raise RuntimeError('debug video encoding failed') from encoder_errors[0]
    np.savez_compressed(out / 'face_parsing.npz', labels=np.stack(parsing))
    write_json(out / 'parsing_palette.json', {name: dict(index=i, rgb=palette[i, ::-1].tolist())
                                             for i, name in enumerate(LABELS)})
    np.savez_compressed(out / 'mesh_fit.npz', vertices=np.stack(mesh_positions),
                        triangles=source.triangles, triangle_uvs=source.triangle_uvs,
                        identity_coefficients=identity.detach().numpy(), fitted_controls=fitted,
                        expression_residual_coefficients=correction)
    report.update(fps=fps, width=w, height=h, source_sha256=sha256(clip),
        rig_sha256=sha256(rig_dir / 'rig_deformer.safetensors'),
        gnm_weight_sha256=source.provenance['sha256'],
        mesh_units='metres',
        mesh_interpretation='Subject-fitted GNM topology with transferred controls and joint skinning; monocular weak-perspective fit.',
        gnm_coefficient_interpretation='Control-to-PCA projection plus independently fitted regional PCA residual; transferred control skinning remains nonlinear.',
        observations=observations)
    write_json(out / 'observations.json', report)
    return {k: v for k, v in report.items() if k != 'observations'}


def audit_catalog(out):
    """Check movie synchronization and independently reproject saved GNM vertices."""
    from .face_models import load
    from .video_fit import ANCHORS
    source = load('gnm_v3')
    groups = gnm_anchors(source)
    with np.load(asset_path('gnm'), allow_pickle=False) as weights:
        identity_names = weights['identity_names'][:len(source.identity_basis)].tolist()
    write_json(out / 'gnm_schema.json', dict(model=source.provenance, mesh_units='metres',
               identity_names=identity_names, expression_names=source.expression_names))
    manifest = json.loads((out / 'catalog.json').read_text())
    expected_frames = manifest['frames']
    identity = None
    checks = {}
    for name in EMOTIONS:
        folder = out / name
        observations = json.loads((folder / 'observations.json').read_text())
        if len(observations['observations']) != expected_frames:
            raise ValueError(f'{name}: observation count does not match generation')
        coefficients = np.array(observations['identity_coefficients'])
        if identity is None:
            identity = coefficients
        elif not np.array_equal(identity, coefficients):
            raise ValueError(f'{name}: GNM identity changed across expressions')
        with np.load(folder / 'mesh_fit.npz', allow_pickle=False) as mesh:
            vertices = mesh['vertices']
            if vertices.shape != (expected_frames, len(source.vertices), 3) or not np.isfinite(vertices).all():
                raise ValueError(f'{name}: invalid fitted mesh sequence')
            if not np.array_equal(mesh['triangles'], source.triangles):
                raise ValueError(f'{name}: GNM topology changed')
            pca = mesh['expression_residual_coefficients']
            if pca.shape != (expected_frames, len(source.expression_basis)) or not np.isfinite(pca).all():
                raise ValueError(f'{name}: invalid regional PCA residuals')
            errors = []
            for i, record in enumerate(observations['observations']):
                coefficients = np.array(record['gnm_coefficients'])
                if not np.isfinite(coefficients).all() or (np.abs(coefficients) > 3.00001).any():
                    raise ValueError(f'{name}: GNM expression coefficient outside model domain')
                if not np.allclose(pca[i], record['gnm_residual_coefficients'], rtol=0, atol=1e-7):
                    raise ValueError(f'{name}: saved regional PCA residuals do not match observations')
                if not record['visible']:
                    continue
                anchors = np.array([vertices[i, group].mean(0) for group in groups])
                cam = record['camera']
                c, s = np.cos(cam['roll_radians']), np.sin(cam['roll_radians'])
                projected = (anchors[:, :2] * [1, -1]) @ np.array([[c, s], [-s, c]])
                projected = projected * cam['scale'] + cam['center']
                dense = np.array(record['landmarks'])
                observed = np.array([dense[list(index) if isinstance(index, tuple) else [index], :2].mean(0)
                                     for _, index in ANCHORS])
                error = np.linalg.norm(projected - observed, axis=1).mean()
                if abs(error - record['landmark_error_normalized']) > 2e-6:
                    raise ValueError(f'{name}: saved mesh does not reproduce reported fitting error')
                errors.append(float(error))
        with np.load(folder / 'face_parsing.npz', allow_pickle=False) as parsing:
            labels = parsing['labels']
            if labels.shape != (expected_frames, 832, 480) or labels.max() >= len(LABELS):
                raise ValueError(f'{name}: invalid parsing sequence')
        checks[name] = dict(frames=expected_frames, visible_frames=len(errors),
                            mean_mesh_error_normalized=float(np.mean(errors)))
    movies = {}
    for name in [*EMOTIONS, None]:
        for category in MOVIES:
            path = out / name / f'{category}.mp4' if name else out / f'{category}_all.mp4'
            probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-count_frames', '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames',
                '-of', 'json', str(path)], capture_output=True, text=True, check=True)
            stream = json.loads(probe.stdout)['streams'][0]
            multiplier = 2 if category == 'four_panel' else 1
            frames = expected_frames * (1 if name else len(EMOTIONS))
            if (stream['width'], stream['height'], int(stream['nb_read_frames']), Fraction(stream['avg_frame_rate'])) != (
                    480 * multiplier, 832 * multiplier, frames, Fraction(24)):
                raise ValueError(f'{path}: wrong geometry, frame count, or frame rate')
            movies[str(path.relative_to(out))] = dict(frames=frames, sha256=sha256(path))
    report = dict(format='vhuman.expression_catalog_audit.v1', status='passed',
                  expressions=checks, movies=movies, shared_gnm_identity=True)
    write_json(out / 'audit.json', report)
    return report


def run(args):
    from PIL import Image, ImageOps
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    ref = out / 'reference.png'
    image = Image.open(args.portrait).convert('RGBA')
    background = Image.new('RGBA', image.size, (96, 96, 96, 255))
    background.alpha_composite(image)
    image = background.convert('RGB')
    encoded = io.BytesIO()
    ImageOps.fit(image, (480, 832)).save(encoded, format='PNG')
    reference_bytes = encoded.getvalue()
    if ref.exists():
        if sha256(ref) != hashlib.sha256(reference_bytes).hexdigest():
            raise ValueError('existing catalog reference does not match the requested portrait')
    else:
        ref.write_bytes(reference_bytes)
    manifest = dict(format='vhuman.expression_catalog.v1', state='running',
                    emotions=list(EMOTIONS), backend=args.backend, preset=args.preset,
                    frames=args.frames, seed=args.seed, portrait_sha256=sha256(args.portrait),
                    rig_sha256=sha256(args.rig / 'rig_deformer.safetensors'),
                    prompts=EMOTIONS, synthetic=True, reviewed=False, expressions={})
    old = out / 'catalog.json'
    if old.exists():
        previous = json.loads(old.read_text())
        for key in ('portrait_sha256', 'backend', 'preset', 'frames', 'seed', 'rig_sha256', 'prompts'):
            if key in previous and previous[key] != manifest[key]:
                raise ValueError(f'cannot resume catalog with changed {key}')
        manifest['expressions'] = previous['expressions']
    backend = select(args.backend)
    for index, (name, prompt) in enumerate(EMOTIONS.items()):
        folder = out / name
        folder.mkdir(exist_ok=True)
        clip = folder / 'video/clip.mp4'
        print(f'{name}: {"checking existing generation" if clip.exists() else "generating"}', flush=True)
        full_prompt = KEEP + ' ' + prompt + ' Smoothly develop the expression and hold it. Silent, static camera.'
        if not clip.exists():
            try:
                result = backend.generate(image=ref, out=folder / 'video', prompt=full_prompt,
                    preset=args.preset, frames=args.frames, seed=args.seed + index,
                    device=0, allow_experimental=True)
            except Exception as exc:
                manifest.update(state='failed', failed_expression=name, error=str(exc))
                write_json(old, manifest)
                raise
            manifest['expressions'][name] = {'generation': result}
            write_json(old, manifest)
        generation = json.loads((folder / 'video/manifest.json').read_text())
        recorded_prompt = ('The person in <Picture 1>. ' if backend.variant == 'ref2va' else '') + full_prompt
        expected = dict(seed=args.seed + index, frames=args.frames, prompt=recorded_prompt,
                        variant=backend.variant, euler_updates={'fast5': 5, 'fast12': 12, 'quality': 39}[args.preset])
        if any(generation.get(k) != v for k, v in expected.items()):
            raise ValueError(f'{name}: existing generation settings do not match this catalog')
        if not generation.get('references') or generation['references'][0]['sha256'] != sha256(ref):
            raise ValueError(f'{name}: existing generation uses a different portrait')
        print(f'{name}: fitting and rendering debug movies', flush=True)
        if args.rerender_debug or not (folder / 'observations.json').exists():
            shared = None if name == 'neutral' else json.loads((out / 'neutral/observations.json').read_text()).get('identity_coefficients')
            stage = folder / '.debug-partial'
            stage.mkdir(exist_ok=False)
            try:
                report = debug(clip, args.rig, stage, name, identity_coefficients=shared)
                for artifact in sorted(stage.iterdir(), key=lambda p: p.name == 'observations.json'):
                    artifact.replace(folder / artifact.name)
            finally:
                shutil.rmtree(stage)
            manifest['expressions'].setdefault(name, {})['debug'] = report
            write_json(old, manifest)
        else:
            report = json.loads((folder / 'observations.json').read_text())
            if report['source_sha256'] != sha256(clip) or report['rig_sha256'] != manifest['rig_sha256']:
                raise ValueError('stale debug exports; use --rerender-debug')
            manifest['expressions'].setdefault(name, {})['debug'] = {k: v for k, v in report.items() if k != 'observations'}
    for movie in MOVIES:
        listing = out / f'{movie}_concat.txt'
        listing.write_text(''.join(f"file '{name}/{movie}.mp4'\n" for name in EMOTIONS))
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'concat',
            '-safe', '0', '-i', str(listing), '-c', 'copy', '-movflags', '+faststart',
            str(out / f'{movie}_all.mp4')], check=True)
    manifest['state'] = 'auditing'
    write_json(old, manifest)
    try:
        audit_catalog(out)
    except Exception as exc:
        manifest.update(state='failed', error=str(exc))
        write_json(old, manifest)
        raise
    manifest['state'] = 'complete'
    write_json(old, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portrait', type=Path, required=True)
    parser.add_argument('--rig', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', choices=('h3', 'h3-fl2va'), default='h3-fl2va')
    parser.add_argument('--preset', choices=('fast5', 'fast12', 'quality'), default='fast5')
    parser.add_argument('--frames', type=int, default=22)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rerender-debug', action='store_true', help='re-fit and render debug videos without regenerating completed clips')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
