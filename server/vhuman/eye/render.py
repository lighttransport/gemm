"""CPU ray-traced eye preview: sphere intersections, Snell refraction,
Lambertian iris lighting and Schlick dielectric reflection. Portable meshes
and the WebGL preview share the same physical parameters and angular UV map."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import iris, optics, sclera
from . import params as P
from .noise import smoothstep



@dataclass
class Camera:
    fov_deg: float = 0.0          # 0: orthographic
    distance: float = 0.12        # metres from the eye centre (perspective)
    half_width: float = 0.0145    # orthographic half extent (metres)
    yaw_deg: float = 0.0          # gaze rotation (eye turns)
    pitch_deg: float = 0.0


def gaze_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Eye-to-world rotation: yaw about +Y (positive looks to +X), then pitch
    about +X (positive looks up)."""
    y, x = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(x), math.sin(x)], [0, -math.sin(x), math.cos(x)]])
    return ry @ rx


def _sphere_entry(o, d, center, radius):
    oc = o - center
    b = np.sum(oc * d, -1)
    c = np.sum(oc * oc, -1) - radius * radius
    disc = b * b - c
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def primary_rays(size: int, cam: Camera, spp: int, seed: int = 0, *, jitter: bool = True):
    """World-space rays (origin, direction) for size x size pixels, spp each."""
    g = np.random.default_rng(seed)
    n = int(round(math.sqrt(spp)))
    offs = [((i + 0.5) / n, (j + 0.5) / n) for i in range(n) for j in range(n)]
    ys, xs = np.mgrid[0:size, 0:size]
    rays_o, rays_d = [], []
    for ox, oy in offs:
        jx = xs + ox + (g.uniform(-0.5, 0.5, xs.shape) / n if jitter else 0.0)
        jy = ys + oy + (g.uniform(-0.5, 0.5, ys.shape) / n if jitter else 0.0)
        sx = (jx / size) * 2 - 1
        sy = 1 - (jy / size) * 2
        if cam.fov_deg > 0:
            h = math.tan(math.radians(cam.fov_deg) / 2)
            d = np.stack([sx * h, sy * h, -np.ones_like(sx)], -1)
            o = np.broadcast_to(np.array([0.0, 0.0, cam.distance]), d.shape)
        else:
            o = np.stack([sx * cam.half_width, sy * cam.half_width, np.full_like(sx, 0.05)], -1)
            d = np.broadcast_to(np.array([0.0, 0.0, -1.0]), o.shape)
        rays_o.append(o.reshape(-1, 3))
        rays_d.append(optics.normalize(d.reshape(-1, 3)))
    return np.stack(rays_o), np.stack(rays_d)


def iris_hit(o, d, p: dict, profile=None):
    """Refracted ray (in the chamber) to the iris surface: hit points, the
    radius, and whether the chamber wall (beyond the limbus) was hit."""
    profile = profile or optics.profile_from_params(p)
    o_ = p["optics"]
    z_iris = profile.iris_z(o_["chamber_depth"])
    h, r_l = o_["iris_convexity"], profile.limbus_radius
    t = (z_iris + h - o[..., 2]) / np.minimum(d[..., 2], -1e-6)
    for _ in range(2):          # the convex iris: refine on z(r) = z_iris + h (1 - r / r_l)
        q = o + t[..., None] * d
        r = np.hypot(q[..., 0], q[..., 1])
        z_t = z_iris + h * (1.0 - np.minimum(r, r_l) / r_l)
        t = (z_t - o[..., 2]) / np.minimum(d[..., 2], -1e-6)
    q = o + t[..., None] * d
    r = np.hypot(q[..., 0], q[..., 1])
    return q, r, r > r_l


def shade(o, d, p: dict, tex_iris, tex_sclera, profile=None, exposure: float = 1.0,
          background: bool = True, rot: np.ndarray | None = None):
    """Linear radiance and coverage for eye-space rays (N, 3). `rot` is the
    eye-to-world rotation: the studio (lights, environment) stays fixed in
    the world when the eye turns."""
    pr = profile or optics.profile_from_params(p)
    rot = np.eye(3) if rot is None else rot
    to_world = lambda v: v @ rot.T          # noqa: E731  (row vectors)
    t_s = _sphere_entry(o, d, np.zeros(3), pr.sclera_radius)
    t_c = _sphere_entry(o, d, np.array([0.0, 0.0, pr.cornea_center_z]), pr.cornea_radius)
    t = np.minimum(t_s, t_c)
    hit = np.isfinite(t)
    out = np.zeros(o.shape, np.float64)
    if background:
        out[~hit] = optics.environment(to_world(d[~hit]))
    if not hit.any():
        return out, hit
    o, d, t, on_cornea = o[hit], d[hit], t[hit], (t_c <= t_s)[hit]
    x = o + t[:, None] * d
    out[hit] = shade_points(x, d, p, tex_iris, tex_sclera, pr, rot=rot, on_cornea=on_cornea)
    return out * exposure, hit


def project_to_surface(x, d, profile, back: float = 0.004):
    """Mesh facets lie inside the analytic surface: re-intersect the two
    spheres along the view ray through each facet point so the shading uses
    the exact surface point (the mesh shader does the same). Points whose
    ray misses both spheres (silhouette facets) are kept."""
    x, d = np.asarray(x, np.float64), np.asarray(d, np.float64)
    o = x - d * back
    t_s = _sphere_entry(o, d, np.zeros(3), profile.sclera_radius)
    t_c = _sphere_entry(o, d, np.array([0.0, 0.0, profile.cornea_center_z]), profile.cornea_radius)
    t = np.minimum(t_s, t_c)
    hit = np.isfinite(t)
    return np.where(hit[:, None], o + np.where(hit, t, 0.0)[:, None] * d, x), hit


def surface_normal(x, profile):
    """Analytic normal of the two-sphere surface at eye-space points: cornea
    sphere inside the limbus angle, sclera sphere outside (mesh shading)."""
    x = np.asarray(x, np.float64)
    alpha = np.arccos(np.clip(optics.normalize(x)[:, 2], -1, 1))
    on_cornea = alpha <= profile.alpha_limbus
    n = np.where(on_cornea[:, None], optics.normalize(x - np.array([0.0, 0.0, profile.cornea_center_z])),
                 optics.normalize(x))
    return n, on_cornea


def shade_points(x, d, p: dict, tex_iris, tex_sclera, pr, *, rot: np.ndarray | None = None,
                 on_cornea: np.ndarray | None = None):
    """Linear radiance at eye-space surface points x (N, 3) seen along d.

    Shared by the ray tracer (x from sphere intersections) and the mesh
    renderer (x from rasterised runtime-shell triangles). The normal is
    always analytic; on_cornea defaults to the limbus-angle classification."""
    rot = np.eye(3) if rot is None else rot
    to_world = lambda v: v @ rot.T          # noqa: E731  (row vectors)
    x, d = np.asarray(x, np.float64), np.asarray(d, np.float64)
    n, by_alpha = surface_normal(x, pr)
    if on_cornea is not None and not np.array_equal(on_cornea, by_alpha):
        n = np.where(on_cornea[:, None], optics.normalize(x - np.array([0.0, 0.0, pr.cornea_center_z])),
                     optics.normalize(x))
    v = -d
    ndv = np.clip(np.sum(n * v, -1), 1e-4, 1.0)
    dirs = optics.normalize(x)
    alpha = np.arccos(np.clip(dirs[:, 2], -1, 1))
    r_uv = optics.uv_radius(alpha, pr)
    phi_s = np.arctan2(dirs[:, 1], dirs[:, 0])
    uv = np.stack([0.5 + r_uv * np.cos(phi_s), 0.5 - r_uv * np.sin(phi_s)], -1)
    cornea_w = 1.0 - smoothstep(optics.CORNEA_SIZE_REF - 0.004, optics.CORNEA_SIZE_REF + 0.004, r_uv)
    o_ = p["optics"]
    fres = optics.fresnel_schlick(ndv, ((o_["ior"] - 1) / (o_["ior"] + 1)) ** 2)
    lights = [(l_dir @ rot, e) for l_dir, e in optics.studio_lights()]      # world -> eye frame

    # Sclera: tinted albedo, normal-mapped, wrapped diffuse near the limbus.
    ruv = sclera.rotate_uv(uv, p["sclera"]["rotation"])
    m = sclera.sample(tex_sclera.masks, ruv)
    alb_s = sclera.colorize(m, r_uv, p)
    nm = sclera.sample(np.concatenate([tex_sclera.normal, np.zeros(tex_sclera.normal.shape[:2] + (1,), np.float32)],
                                      -1), ruv)[:, :3]
    tan = _uv_tangent(dirs, alpha, phi_s, pr)
    bit = np.cross(n, tan)
    n_s = optics.normalize(tan * nm[:, :1] + bit * nm[:, 1:2] + n * nm[:, 2:3])
    wrap = 0.2 + 0.6 * sclera.transmission_amount(r_uv, p)
    diff_s = np.zeros_like(alb_s)
    for l_dir, e in lights:
        ndl = np.sum(n_s * l_dir, -1)
        diff_s += e * optics.soft_wrap(ndl, wrap)[:, None] / math.pi
    diff_s += optics.ambient(to_world(n_s))
    col_s = alb_s * diff_s

    # Cornea: refract into the chamber and shade the iris with Lambertian diffuse.
    tdir = optics.refract(d, n, 1.0 / o_["ior"])
    q, r_h, wall = iris_hit(x, tdir, p, pr)
    r_iris = optics.iris_radius(p, pr)
    t_iris = r_h / r_iris
    phi = np.arctan2(q[:, 1], q[:, 0])
    alb_i = iris.iris_plane_color(tex_iris, t_iris, phi, p, sclera.sampler(tex_sclera, p))
    alb_i = np.where(wall[:, None], iris.ANGLE_COLOR, alb_i)
    phi_r = phi - 2 * math.pi * p["iris"]["rotation"]
    t_tex = np.minimum(optics.pupil_scale(np.minimum(t_iris, 1.0), P.pupil_scale(p)), 1.0)
    nmap = iris.sample_masks(np.concatenate([tex_iris.normal, np.zeros(tex_iris.normal.shape[:2] + (1,),
                                                                         np.float32)], -1), t_tex, phi_r)[:, :3]
    c, s_ = np.cos(2 * math.pi * p["iris"]["rotation"]), np.sin(2 * math.pi * p["iris"]["rotation"])
    nx = nmap[:, 0] * c - nmap[:, 1] * s_
    ny = nmap[:, 0] * s_ + nmap[:, 1] * c
    iris_n = optics.normalize(np.stack([nx * 0.6, ny * 0.6, nmap[:, 2]], -1))
    iris_mask = iris.cornea_mask(np.minimum(t_iris, 1.0) * p["cornea"]["size"], p)
    caustic_n = optics.caustic_normal(n, iris_mask, t_iris)
    diff_i = np.zeros_like(alb_i)
    for l_dir, e in lights:
        lobe = optics.caustic(caustic_n, iris_n, l_dir, o_["caustic_strength"], o_["caustic_sharpness"])
        diff_i += e * (optics.iris_irradiance(iris_n, l_dir) + lobe)[:, None] / math.pi
    shadow = iris.sample_masks(tex_iris.masks, t_tex, phi_r)[:, 1:2]
    ao = 1.0 - 0.5 * (1.0 - shadow) * p["iris"]["shadow_details"]
    diff_i += optics.ambient(to_world(iris_n)) * ao
    col_i = alb_i * diff_i
    t_trans = 1.0 - optics.fresnel_schlick(np.clip(np.sum(-tdir * n, -1), 0, 1), ((o_["ior"] - 1) / (o_["ior"] + 1)) ** 2)
    col_i = col_i * t_trans[:, None]

    refl = d - 2 * np.sum(d * n, -1, keepdims=True) * n
    env = (optics.environment(to_world(refl), o_["cornea_roughness"]) * cornea_w[:, None]
           + optics.environment(to_world(refl), o_["sclera_roughness"]) * (1 - cornea_w[:, None]))
    spec = fres[:, None] * env
    return (col_i * cornea_w[:, None] + col_s * (1 - cornea_w[:, None])) * (1 - fres[:, None]) + spec


def rasterize(mesh, size: int, cam: Camera, rot: np.ndarray):
    """Orthographic z-buffer rasterisation of an eye-space mesh (pixel
    centres): hit mask (size*size,), eye-space surface points and view
    directions of the hits. Only the CPU camera (fov 0) is supported."""
    if cam.fov_deg > 0:
        raise ValueError("mesh rasterisation supports the orthographic CPU camera only")
    pos = mesh.positions.astype(np.float64) @ rot.T                 # eye -> world
    tri = mesh.indices.astype(np.int64)
    px = (pos[:, 0] / cam.half_width + 1) * 0.5 * size - 0.5          # pixel coordinates of the vertices
    py = (1 - pos[:, 1] / cam.half_width) * 0.5 * size - 0.5
    depth = np.full((size, size), -np.inf)
    xe = np.zeros((size, size, 3))
    a, b, c = (tri[:, k] for k in range(3))
    for t in range(len(tri)):
        xs, ys = px[[a[t], b[t], c[t]]], py[[a[t], b[t], c[t]]]
        x0, x1 = max(int(math.ceil(xs.min())), 0), min(int(math.floor(xs.max())), size - 1)
        y0, y1 = max(int(math.ceil(ys.min())), 0), min(int(math.floor(ys.max())), size - 1)
        if x0 > x1 or y0 > y1:
            continue
        gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        det = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if abs(det) < 1e-12:
            continue
        w1 = ((gx - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (gy - ys[0])) / det
        w2 = ((xs[1] - xs[0]) * (gy - ys[0]) - (gx - xs[0]) * (ys[1] - ys[0])) / det
        w0 = 1 - w1 - w2
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        z = w0 * pos[a[t], 2] + w1 * pos[b[t], 2] + w2 * pos[c[t], 2]
        sel = inside & (z > depth[y0:y1 + 1, x0:x1 + 1])
        depth[y0:y1 + 1, x0:x1 + 1][sel] = z[sel]
        ids = (a[t], b[t], c[t])
        world = sum(w[..., None] * pos[i] for w, i in zip((w0, w1, w2), ids))
        xe[y0:y1 + 1, x0:x1 + 1][sel] = world[sel]
    hit = np.isfinite(depth).reshape(-1)
    x_eye = xe.reshape(-1, 3)[hit] @ rot                                 # world -> eye
    d_eye = np.broadcast_to(np.array([0.0, 0.0, -1.0]) @ rot, x_eye.shape)
    return hit, x_eye, d_eye


def render_mesh(p: dict, mesh, size: int = 384, cam: Camera | None = None, *, res: int = 1024,
                exposure: float = 1.0, iris_tex=None) -> np.ndarray:
    """The runtime mesh eye shaded with shade_points (the mesh shader's
    reference): tone-mapped sRGB float image (size, size, 4) with the studio
    background, comparable with render(spp=1, jitter=False)."""
    p = P.validate(p)
    cam = cam or Camera()
    tex_i, tex_s = iris_tex or iris.build(p, res), sclera.build(p, res)
    rot = gaze_matrix(cam.yaw_deg, cam.pitch_deg)
    pr = optics.profile_from_params(p)
    hit, x, d = rasterize(mesh, size, cam, rot)
    ro, rd = primary_rays(size, cam, 1, jitter=False)
    acc = np.zeros((size * size, 3))
    acc[~hit] = optics.environment(rd[0][~hit])
    if hit.any():
        x, _ = project_to_surface(x, d, pr)
        acc[hit] = shade_points(x, d, p, tex_i, tex_s, pr, rot=rot)
    rgb = optics.tonemap_neutral(acc * exposure)
    # alpha: the mesh coverage, comparable with render(transparent=True) where both cover
    return np.concatenate([rgb, hit[:, None].astype(np.float32)], -1).reshape(size, size, 4).astype(np.float32)


def _uv_tangent(dirs, alpha, phi, profile):
    """Unit direction of increasing u on the sphere (the analytic tangent)."""
    h = 1e-4
    dr = (optics.uv_radius(alpha + h, profile) - optics.uv_radius(alpha - h, profile)) / (2 * h)
    r = optics.uv_radius(alpha, profile)
    a_hat = np.stack([np.cos(alpha) * np.cos(phi), np.cos(alpha) * np.sin(phi), -np.sin(alpha)], -1)
    p_hat = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], -1)
    du_da = dr * np.cos(phi)
    du_dp = -r * np.sin(phi) / np.maximum(np.sin(alpha), 1e-3)
    t = a_hat * du_da[:, None] + p_hat * du_dp[:, None]
    bad = np.linalg.norm(t, axis=-1) < 1e-9
    t[bad] = [1.0, 0.0, 0.0]
    return optics.normalize(t)


def render(p: dict, size: int = 512, cam: Camera | None = None, *, spp: int = 4, res: int = 1024,
           exposure: float = 1.0, transparent: bool = False, seed: int = 0, chunk: int = 1 << 18,
           iris_tex=None, jitter: bool = True) -> np.ndarray:
    """Tone-mapped sRGB float image (size, size, 4). `iris_tex` overrides
    the procedural iris (an iris plate)."""
    p = P.validate(p)
    cam = cam or Camera()
    tex_i, tex_s = iris_tex or iris.build(p, res), sclera.build(p, res)
    rot = gaze_matrix(cam.yaw_deg, cam.pitch_deg)
    ro, rd = primary_rays(size, cam, spp, seed, jitter=jitter)
    acc = np.zeros((size * size, 3))
    cov = np.zeros(size * size)
    for k in range(ro.shape[0]):
        o = ro[k] @ rot            # world -> eye frame (rows: x_eye = R^T x_world)
        d = rd[k] @ rot
        for a in range(0, len(o), chunk):
            col, hit = shade(o[a:a + chunk], d[a:a + chunk], p, tex_i, tex_s, exposure=exposure,
                             background=not transparent, rot=rot)
            acc[a:a + chunk] += col
            cov[a:a + chunk] += hit
    acc /= ro.shape[0]
    cov /= ro.shape[0]
    if transparent:     # colour of the covered part; coverage goes to alpha
        acc = acc / np.maximum(cov[:, None], 1e-6)
    rgb = optics.tonemap_neutral(acc)
    img = np.concatenate([rgb, cov[:, None] if transparent else np.ones((len(cov), 1))], -1)
    return img.reshape(size, size, 4).astype(np.float32)


def save(img: np.ndarray, path) -> None:
    from PIL import Image
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
