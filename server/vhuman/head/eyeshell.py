"""Procedural lid occlusion over an analytic eye cap.

A Gaussian falloff from the detected fissure gives a soft contact shadow.
Width varies with portrait direction; the shell is an unlit black alpha layer."""
from __future__ import annotations

import math

import numpy as np

from ..eye import optics
from ..eye.geometry import Mesh
from .lids import eye_contour

SHELL_OFFSET_M = 0.0002
CAP_DEG = 80.0
# Synthetic Gaussian width as a fraction of the detected fissure height.
BLUR = {"top": 0.35, "bottom": 0.15, "inner": 0.2, "outer": 0.2}
STRENGTH = 0.45                 # alpha at the lid edge; neutral alpha blend multiplies by 1 - alpha
# Nonzero unlit RGB adds light to dark irises instead of only occluding them.
# Neutral black makes the portable alpha blend a pure shadow multiply.
TINT = (0.0, 0.0, 0.0)
TEX = 256


def shell_mesh(rings: int = 40, segments: int = 96) -> Mesh:
    """The cap (eye-local metres, +Z gaze): the eye's outer surface, offset out."""
    prof = optics.ANATOMICAL
    alphas = np.linspace(0.0, math.radians(CAP_DEG), rings + 1)[1:]
    phis = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    a, f = np.meshgrid(alphas, phis, indexing="ij")
    dirs = np.stack([np.sin(a) * np.cos(f), np.sin(a) * np.sin(f), np.cos(a)], -1).reshape(-1, 3)
    rho = optics.surface_radius(a.reshape(-1), prof) + SHELL_OFFSET_M
    pos = np.concatenate([[[0.0, 0.0, prof.apex_z + SHELL_OFFSET_M]], dirs * rho[:, None]])
    nrm = np.concatenate([[[0.0, 0.0, 1.0]], dirs])
    from ..eye.geometry import _grid_indices
    k = np.arange(segments)
    fan = np.stack([np.zeros(segments, int), 1 + k, 1 + (k + 1) % segments], -1)
    idx = np.concatenate([fan, _grid_indices(rings, segments, 1)]).astype(np.uint32)
    return Mesh("eyeshell", pos.astype(np.float32), nrm.astype(np.float32), np.zeros((len(pos), 2), np.float32), idx)


def _blur_width(theta, medial_sign: float, height: float):
    """Blur (pixels) by direction in the image: up = -y, medial = towards the nose."""
    up = np.clip(-np.sin(theta), 0, None)
    down = np.clip(np.sin(theta), 0, None)
    med = np.clip(np.cos(theta) * medial_sign, 0, None)
    lat = np.clip(-np.cos(theta) * medial_sign, 0, None)
    w = (up * BLUR["top"] + down * BLUR["bottom"] + med * BLUR["inner"] + lat * BLUR["outer"])
    return np.maximum(w / np.maximum(up + down + med + lat, 1e-6), 0.05) * height


def build(eye, pose, cam) -> tuple[Mesh, np.ndarray, dict]:
    """The eye's shell mesh (UVs set) and its RGBA occlusion texture."""
    mesh = shell_mesh()
    k = pose.units_per_m
    world = pose.center + (mesh.positions.astype(np.float64) * k) @ pose.rotation.T
    pix = cam.project(world)
    half = 4.4 * eye.r
    x0, y0 = eye.cx - half, eye.cy - half
    mesh.uvs = np.stack([(pix[:, 0] - x0) / (2 * half), (pix[:, 1] - y0) / (2 * half)], -1).astype(np.float32)
    contour = eye_contour(eye)
    # texel centres in portrait pixels
    c = (np.arange(TEX) + 0.5) / TEX * 2 * half
    xx, yy = np.meshgrid(x0 + c, y0 + c)
    theta = np.arctan2(yy - eye.cy, xx - eye.cx)
    r = np.hypot(xx - eye.cx, yy - eye.cy)
    if contour is None:
        edge = np.full_like(r, eye.r)
        height = 2 * eye.r
    else:
        edge = contour(theta)
        height = float(contour(-math.pi / 2) + contour(math.pi / 2))
    inside = edge - r                                  # pixels inside the opening (negative: under a lid)
    medial = 1.0 if eye.side == "right" else -1.0      # subject's right eye: the nose is at +x (image right)
    width = _blur_width(theta, medial, height)
    distance = np.maximum(inside, 0.) / width
    alpha = np.exp(-2. * distance ** 2) * STRENGTH
    rgba = np.zeros((TEX, TEX, 4), np.float32)
    rgba[..., :3] = optics.linear_to_srgb(np.array(TINT))
    rgba[..., 3] = alpha
    info = {"fissure_height_px": round(height, 2), "box_px": round(2 * half, 2),
            "strength": STRENGTH, "tint_linear": list(TINT), "blur": dict(BLUR)}
    return mesh, (np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8), info
