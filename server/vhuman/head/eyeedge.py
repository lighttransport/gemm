"""Original wet-margin ribbon and medial-corner geometry.

Coordinates follow the draped head cut. Smooth normals, low roughness and
glTF alpha blending approximate a thin lacrimal film."""
from __future__ import annotations

import math
import numpy as np

from ..eye.geometry import Mesh
from ..eye.optics import ANATOMICAL
from .camera import ray_mesh

TEAR_RADIUS_M = 0.00012
MAX_MARGIN_STEP_M = 0.0015  # do not bridge separate socket layers


def _mesh(name, positions, normals, triangles):
    return Mesh(name, np.asarray(positions, np.float32), np.asarray(normals, np.float32),
                np.zeros((len(positions), 2), np.float32), np.asarray(triangles, np.uint32).reshape(-1, 3))


def margin(cut_points, cam, eye, contour, bins=180):
    """Sample the front cut's depth on smooth contour rays; reject layer spikes.

    A periodic median removes isolated socket-layer hits without low-passing
    the whole lid. Missing angular bins interpolate depth, never XYZ (which
    would shrink the almond). An empty/partial cut produces no margin.
    """
    if len(cut_points) < 3 or contour is None:
        return None
    pix = cam.project(cut_points)
    th = np.arctan2(pix[:, 1] - eye.cy, pix[:, 0] - eye.cx)
    ids = ((th + math.pi) * bins / (2 * math.pi)).astype(int) % bins
    depth = np.full(bins, np.inf)
    np.minimum.at(depth, ids, np.linalg.norm(cut_points - cam.origin, axis=1))
    valid = np.isfinite(depth)
    if valid.sum() < bins // 4:
        return None
    grid = np.arange(bins)
    depth = np.interp(grid, grid[valid], depth[valid], period=bins)
    depth = np.median(np.stack([np.roll(depth, i) for i in range(-2, 3)]), axis=0)
    theta = (grid + 0.5) * (2 * math.pi / bins) - math.pi
    r = contour(theta)
    _, rays = cam.rays(eye.cx + r * np.cos(theta), eye.cy + r * np.sin(theta))
    return cam.origin + rays * depth[:, None]


def tearline(points, pose, cam):
    """A rounded strip, with a narrow crescent cross-section facing the camera.

    Open backs sit in the lining. This avoids drawing both sides of a
    translucent tube and gives the thin film a smooth specular highlight.
    """
    if points is None or len(points) < 3:
        return None
    count = len(points)
    forward = np.roll(points, -1, axis=0) - points
    length = np.linalg.norm(forward, axis=1)
    connected = (length > 1e-12) & (length <= MAX_MARGIN_STEP_M * pose.units_per_m)
    # Interior rings use a centred secant. At a gap, use only the connected
    # side: a skipped layer must not tilt the strip's end or its highlight.
    steps = np.where(connected[:, None], forward, 0.0)
    tangent = steps + np.roll(steps, 1, axis=0)
    tangent_length = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent /= np.maximum(tangent_length, 1e-12)
    front = cam.origin - points
    front /= np.linalg.norm(front, axis=1, keepdims=True)
    across = np.cross(tangent, front)
    across_length = np.linalg.norm(across, axis=1, keepdims=True)
    across /= np.maximum(across_length, 1e-12)
    usable = (tangent_length[:, 0] > 1e-12) & (across_length[:, 0] > 1e-12)
    connected &= usable & np.roll(usable, -1)
    front = np.cross(across, tangent)
    angles = np.linspace(-math.pi / 2, math.pi / 2, 7)
    normals = front[:, None] * np.cos(angles)[None, :, None] + across[:, None] * np.sin(angles)[None, :, None]
    p = points[:, None] + normals * (TEAR_RADIUS_M * pose.units_per_m)
    tri = []
    for i in range(count):
        j = (i + 1) % count
        # Do not bridge a socket layer jump with a visible spike.
        if not connected[i]:
            continue
        for a in range(6):
            v, w = i * 7 + a, j * 7 + a
            tri.extend([(v, v + 1, w), (v + 1, w + 1, w)])
    if not tri:
        return None
    return _mesh(f"eye_{pose.side}_tearline", p.reshape(-1, 3), normals.reshape(-1, 3), tri)


def caruncle(eye, pose, cam, contour, positions, triangles):
    """Small flattened pink mound embedded in the retained medial canthus.

    Anchor to a real surface hit, not the eye sphere: the medial tip can
    lie beyond the globe's silhouette. Skip if there is no local support.
    """
    if contour is None or len(triangles) == 0:
        return None
    theta = 0.0 if eye.side == "right" else math.pi
    p = positions[triangles]
    anchor = None
    # The cut can include the whole medial tip on a round mock head. Search
    # outward to the first retained surface, then inset the mound into it.
    for fraction in (0.90, 0.94, 0.98, 1.02, 1.06):
        x = eye.cx + math.cos(theta) * float(contour(theta)) * fraction
        o, d = cam.rays(x, eye.cy)
        t, _ = ray_mesh(o, d, p[:, 0], p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        if math.isfinite(t) and np.linalg.norm(o + d * t - pose.center) < 2 * ANATOMICAL.sclera_radius * pose.units_per_m:
            anchor = o + d * t
            break
    if anchor is None:
        return None
    # UV sphere with unique poles: no zero-area pole triangles.
    segments, rings = 32, 12
    a = np.linspace(0, math.pi, rings + 1)[1:-1]
    f = np.arange(segments) * (2 * math.pi / segments)
    aa, ff = np.meshgrid(a, f, indexing="ij")
    unit = np.concatenate([[[0, 0, 1]], np.stack([np.sin(aa) * np.cos(ff), np.sin(aa) * np.sin(ff),
                                                 np.cos(aa)], -1).reshape(-1, 3), [[0, 0, -1]]])
    radii = np.array([0.00085, 0.00048, 0.00045]) * pose.units_per_m
    center = anchor - pose.rotation[:, 2] * (0.00015 * pose.units_per_m)
    vertices = center + (unit * radii) @ pose.rotation.T
    normal = (unit / radii) @ pose.rotation.T
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    tri = []
    for j in range(segments):
        n = (j + 1) % segments
        tri.append((0, 1 + j, 1 + n))
        for i in range(rings - 2):
            v, w = 1 + i * segments + j, 1 + i * segments + n
            tri.extend([(v, v + segments, w), (w, v + segments, w + segments)])
        v, w = 1 + (rings - 2) * segments + j, 1 + (rings - 2) * segments + n
        tri.append((v, len(unit) - 1, w))
    return _mesh(f"eye_{pose.side}_caruncle", vertices, normal, tri)


def material(name, lining_rgb):
    """Portable wet materials; colours derive from this portrait's lid tint."""
    from ..eye.optics import srgb_to_linear
    rgb = srgb_to_linear(np.asarray(lining_rgb) / 255)
    film = name.endswith("tearline")
    rgb = rgb * np.array([1.08, 0.80, 0.85])
    return {"name": name, "alphaMode": "BLEND" if film else "OPAQUE",
            "pbrMetallicRoughness": {"baseColorFactor": [*np.clip(rgb, 0, 1).tolist(), 0.28 if film else 1.0],
                                     "metallicFactor": 0.0, "roughnessFactor": 0.10 if film else 0.24},
            "extensions": {"KHR_materials_ior": {"ior": 1.336}}}
