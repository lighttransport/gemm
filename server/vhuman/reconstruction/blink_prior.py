"""Authored, localized blink proposals for GNM-style head coordinates.

This is a geometric prior, not recovered expression evidence. Callers must
validate topology, optical contact and appearance before accepting a proposal.
"""
from collections.abc import Mapping
import numpy as np


CONTOURS = {
    'right': dict(lower=[33, 7, 163, 144, 145, 153, 154, 155, 133],
                  upper=[33, 246, 161, 160, 159, 158, 157, 173, 133],
                  brow=[70, 63, 105, 66, 107]),
    'left': dict(lower=[263, 249, 390, 373, 374, 380, 381, 382, 362],
                 upper=[263, 466, 388, 387, 386, 385, 384, 398, 362],
                 brow=[336, 296, 334, 293, 300]),
}


def _hermite(x, x0, x1, y0, y1, d0, d1):
    width = np.maximum(x1 - x0, 1e-8)
    t = np.clip((x - x0) / width, 0, 1)
    return ((2*t**3 - 3*t**2 + 1)*y0 + (t**3 - 2*t**2 + t)*width*d0
            + (-2*t**3 + 3*t**2)*y1 + (t**3 - t**2)*width*d1)


def propose_blink(vertices, movable, landmarks, eye_centers, strength, *,
                  head_rotation=None, remaining_aperture=.025,
                  angular_span_degrees=35., brow_margin_degrees=3.):
    """Return proposed vertices and a diagnostic report without mutating input.

    Coordinates are metres. In the head frame, X is horizontal, Y up and Z
    forward. ``head_rotation`` maps that frame into the input/world frame.
    ``landmarks`` contains MediaPipe-indexed 3D attachments; ``eye_centers``
    maps anatomical left/right names to their fitted optical centers. The
    movable mask normally includes exterior skin and the native eye sockets,
    but excludes optical eyes and teeth. Optical surfaces are not modified.

    The angular map is monotone for each fixed X/radius. That property does
    not guarantee that its piecewise-linear mesh remains intersection-free.
    A positive residual aperture avoids collapsing the entire opening to a
    line. Brow anchors and eye-corner falloff limit surrounding deformation.
    """
    vertices = np.asarray(vertices, dtype=float)
    movable = np.asarray(movable)
    landmarks = np.asarray(landmarks, dtype=float)
    rotation = np.eye(3) if head_rotation is None else np.asarray(head_rotation, dtype=float)
    settings = [strength, remaining_aperture, angular_span_degrees, brow_margin_degrees]
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all() or movable.shape != (len(vertices),)
            or movable.dtype.kind != 'b' or landmarks.ndim != 2
            or landmarks.shape[1] != 3 or len(landmarks) < 467
            or not np.isfinite(landmarks).all() or rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)
            or not np.isfinite(settings).all() or not 0 <= strength <= 1
            or not 0 < remaining_aperture <= 1
            or not 3 < angular_span_degrees < 90 or not 0 <= brow_margin_degrees < 30):
        raise ValueError('invalid blink geometry, mask, coordinate frame or settings')
    if not isinstance(eye_centers, Mapping) or not set(CONTOURS).issubset(eye_centers):
        raise ValueError('left and right optical eye centers are required')
    centers = {}
    for side in CONTOURS:
        center = np.asarray(eye_centers[side], dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('invalid optical eye center: ' + side)
        centers[side] = center @ rotation
    local, points = vertices @ rotation, landmarks @ rotation
    proposed = local.copy()
    span = np.deg2rad(angular_span_degrees)
    affected = {}
    for side, contour in CONTOURS.items():
        center = centers[side]
        relative = local - center
        theta = np.arctan2(relative[:, 1], relative[:, 2])
        radius = np.linalg.norm(relative[:, 1:], axis=1)

        def curve(indices):
            p = points[indices] - center
            order = np.argsort(p[:, 0])
            if np.any(np.diff(p[order, 0]) <= 1e-9):
                raise ValueError('eye contour must have distinct horizontal coordinates')
            return np.interp(relative[:, 0], p[order, 0],
                             np.arctan2(p[order, 1], p[order, 2]))

        low, upper = curve(contour['lower']), curve(contour['upper'])
        brow = curve(contour['brow'])
        middle, gap = (low + upper)*.5, np.maximum(upper - low, 0)
        low, upper = middle - gap*.5, middle + gap*.5
        seam = .75*low + .25*upper
        fade = np.clip(gap / max(float(gap.max())*.25, 1e-8), 0, 1)
        fade = fade*fade*(3 - 2*fade)
        compression = 1 - (1 - remaining_aperture)*fade
        lower_target = seam + compression*(low - seam)
        upper_target = seam + compression*(upper - seam)
        anchor = np.maximum(upper + np.deg2rad(3),
                            np.minimum(upper + span, brow - np.deg2rad(brow_margin_degrees)))
        angle = np.where(theta < low,
            _hermite(theta, low-span, low, low-span, lower_target, 1., compression),
            np.where(theta > upper,
                _hermite(theta, upper, anchor, upper_target, anchor, compression, 1.),
                seam + compression*(theta-seam)))
        angle = np.where((theta < low-span) | (theta > anchor), theta, angle)
        corners = points[[contour['lower'][0], contour['lower'][-1]], 0]
        support = (movable & (local[:, 0] >= corners.min())
                   & (local[:, 0] <= corners.max()) & (radius < .035))
        blend = np.clip((.035-radius)/.01, 0, 1)*strength
        angle = theta + blend*(angle-theta)
        proposed[support, 1] = center[1] + radius[support]*np.sin(angle[support])
        proposed[support, 2] = center[2] + radius[support]*np.cos(angle[support])
        affected[side] = int(np.sum(support & (abs(angle-theta) > 1e-8)))
    delta = (proposed-local) @ rotation.T
    # Preserve exactly stationary vertices, including the zero-strength pose.
    delta[~movable] = 0
    if strength == 0 or remaining_aperture == 1:
        delta[:] = 0
    result = vertices + delta
    return result, dict(strength=float(strength), remaining_aperture=remaining_aperture,
        affected_vertices=affected, maximum_displacement_mm=float(np.linalg.norm(delta, axis=1).max()*1000),
        observed_motion=False, requires_geometry_and_contact_validation=True)
