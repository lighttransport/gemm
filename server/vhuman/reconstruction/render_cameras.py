"""Named Cycles camera presets for the quality A/B study (numpy only; importable inside Blender).

A preset orbits the portrait camera about a pivot (head centre or a part centre) by yaw/pitch, scales its
distance and focal length, and re-aims it at the target. Positive yaw matches ANGLES/multiview: the subject's
left side turns toward the camera. 'portrait' reproduces the fitted source camera exactly.
"""
import numpy as np

# name: (yaw_deg, pitch_deg, distance_scale, focal_scale, target)
CAMERAS = {
    'portrait':      (0, 0, 1., 1., 'head'),
    'front':         (0, 0, 1., 1., 'head'),
    'yaw_p40':       (40, 0, 1., 1., 'head'),
    'yaw_m40':       (-40, 0, 1., 1., 'head'),
    'yaw_p90':       (90, 0, 1., 1., 'head'),
    'yaw_m90':       (-90, 0, 1., 1., 'head'),
    'yaw_p135':      (135, 0, 1., 1., 'head'),
    'yaw_m135':      (-135, 0, 1., 1., 'head'),
    'back':          (180, 0, 1., 1., 'head'),
    'ear_left':      (95, 10, .55, 1.8, 'ear_left'),
    'ear_right':     (-95, 10, .55, 1.8, 'ear_right'),
    'ear_behind_left':  (140, 5, .55, 1.8, 'ear_left'),
    'ear_behind_right': (-140, 5, .55, 1.8, 'ear_right'),
    'eye_left':      (8, -3, .45, 2.6, 'eye_left'),
    'eye_right':     (-8, -3, .45, 2.6, 'eye_right'),
    'mouth':         (0, -8, .5, 2.2, 'mouth'),
}
HEAD_SET = ('front', 'yaw_p40', 'yaw_m40', 'yaw_p90', 'yaw_m90', 'yaw_p135', 'yaw_m135', 'back')
CLOSEUP_SET = ('ear_left', 'ear_right', 'ear_behind_left', 'ear_behind_right', 'eye_left', 'eye_right', 'mouth')


def rotation_y(deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_axis(axis, deg):
    axis = np.asarray(axis, float) / max(np.linalg.norm(axis), 1e-12)
    a = np.deg2rad(deg); k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def targets(skin, eyes=None, ears=None):
    """Target points in head space from skin vertices (N,3) and optional eye/ear vertex dicts {'left':(M,3),...}.

    Fallbacks derive ears from the lateral extremes of the skin at eye height and the mouth from the
    front-most skin below the eye line; they are approximate framing aids, not anatomical fits."""
    skin = np.asarray(skin, float)
    lo, hi = skin.min(0), skin.max(0)
    head = (lo + hi) / 2
    out = {'head': head}
    eye_y = None
    for side in ('left', 'right'):
        pts = None if not eyes else eyes.get(side)
        if pts is not None and len(pts):
            out['eye_' + side] = np.asarray(pts, float).mean(0)
    if 'eye_left' in out and 'eye_right' in out:
        eye_y = (out['eye_left'][1] + out['eye_right'][1]) / 2
    else:
        eye_y = lo[1] + .62 * (hi[1] - lo[1])
    band = skin[np.abs(skin[:, 1] - eye_y) < .02 * max(hi[1] - lo[1], 1e-3) * 10]
    if len(band) < 10:
        band = skin
    for side, sign in (('left', 1), ('right', -1)):
        pts = None if not ears else ears.get(side)
        if pts is not None and len(pts):
            out['ear_' + side] = np.asarray(pts, float).mean(0)
        else:
            # +x is the subject's left in the GNM/H frame used by the fit (mirrored by the camera).
            edge = band[np.argsort(sign * band[:, 0])[-max(5, len(band) // 50):]]
            out['ear_' + side] = edge.mean(0)
    height = hi[1] - lo[1]
    below = skin[(skin[:, 1] < eye_y - .22 * height) & (skin[:, 1] > eye_y - .42 * height)]
    if len(below) < 10:
        below = skin
    front = below[np.argsort(below[:, 2])[-max(5, len(below) // 50):]]
    mouth = front.mean(0); mouth[0] = head[0] if 'eye_left' not in out else (out['eye_left'][0] + out['eye_right'][0]) / 2
    out['mouth'] = mouth
    return out


def place(name, portrait_origin, portrait_rotation, portrait_focal, points, *, yaw=None, pitch=None,
          distance_scale=None, focal_scale=None, target=None):
    """Return (origin, rotation(3x3 rows = camera right/up/back in world), focal) for a preset.

    portrait_rotation rows follow reference.Camera (x right, y up, z back/toward camera)."""
    yaw0, pitch0, dist0, focal0, target0 = CAMERAS[name]
    yaw = yaw0 if yaw is None else yaw; pitch = pitch0 if pitch is None else pitch
    distance_scale = dist0 if distance_scale is None else distance_scale
    focal_scale = focal0 if focal_scale is None else focal_scale
    target = target0 if target is None else target
    origin = np.asarray(portrait_origin, float); rotation = np.asarray(portrait_rotation, float)
    aim = np.asarray(points[target], float)
    pivot = np.asarray(points['head'], float)
    if name == 'portrait' or (yaw == 0 and pitch == 0 and distance_scale == 1 and focal_scale == 1 and target == 'head'):
        return origin, rotation, float(portrait_focal)
    # Orbit the camera about the pivot, then re-aim at the target, keeping the portrait camera's up vector.
    offset = origin - pivot
    r = rotation_y(yaw)
    right_axis = r @ rotation[0]
    r = rotation_axis(right_axis, -pitch) @ r
    new_origin = aim + r @ offset * distance_scale if target != 'head' else pivot + r @ offset * distance_scale
    back = new_origin - aim; back /= max(np.linalg.norm(back), 1e-12)
    up_hint = rotation[1]
    right = np.cross(up_hint, back); right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(back, right)
    return new_origin, np.stack((right, up, back)), float(portrait_focal) * focal_scale
