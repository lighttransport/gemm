"""Image-free synthetic identities from the GNM v3 identity space.

A seed (or explicit coefficients) samples bounded GNM identity modes; the
neutral full anatomy is evaluated, a procedural skin and painted eyes are
authored in the model's own UV atlas (tone chart, cellular pores, lip tint),
and the result is written as the inputs of the existing head pipeline: a
Pixal3D-style `pixal3d.glb` and a software-rendered neutral `portrait.png`
(no diffusion model, no photograph). `head-fit` and the rig builder then run
unchanged, so synthetic, photographed and generated heads share one path.
Outputs are tagged `source: synthetic` with the seed and coefficients; no
image licence applies. GNM v3 is Apache-2.0 (hash pinned).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "vhuman.synthetic_identity.v1"
IDENTITY_MODES = 170          # the fitted modes of the GNM loader (rig/face_models.py)
IRIS_COLORS = {"brown": (.22, .12, .07), "dark_brown": (.1, .05, .03), "hazel": (.32, .22, .1),
               "green": (.2, .32, .18), "blue": (.25, .4, .58), "gray": (.4, .42, .45)}
GLB_HEAD_EXTENT = 1.0 / 1.1   # Pixal3D-style framing: the portrait crop is 1.1 x the head's extent


def sample_identity(seed: int, strength: float = 1.0, modes: int = IDENTITY_MODES, dim: int = 253) -> np.ndarray:
    """Bounded Gaussian identity coefficients (PCA-like unit scale, clipped to
    +-2 sigma, scaled by `strength`); modes beyond `modes` stay neutral."""
    if not 0.0 <= strength <= 2.0:
        raise ValueError("strength must be within [0, 2]")
    g = np.random.default_rng(int(seed))
    beta = np.zeros(dim)
    beta[:modes] = np.clip(g.standard_normal(modes), -2.0, 2.0) * strength
    return beta


def evaluate(model, beta: np.ndarray):
    """Neutral full anatomy (vertices, joints) in GNM metres (+Y up, face +Z)."""
    beta = np.asarray(beta, np.float64)
    if beta.shape != (model.identity_dim,) or not np.isfinite(beta).all() or np.abs(beta).max() > 4.0:
        raise ValueError("invalid identity coefficients")
    vertices, joints = model.evaluate(beta)
    return np.asarray(vertices, np.float64), np.asarray(joints, np.float64)


def _vertex_normals(pos, tris):
    n = np.zeros_like(pos)
    fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def eyeball_colours(model, vertices, joints, iris="brown"):
    """Per-vertex linear colour of the GNM eyeballs' outer surface: iris and
    pupil by radial distance from each eye's gaze axis (the model's iris
    group lies behind the cornea), sclera elsewhere. The eyeballs share the
    skin's UV square, so the portrait render colours them by vertex."""
    sclera = np.array([.74, .70, .66])
    iris_rgb = np.array(IRIS_COLORS.get(iris, IRIS_COLORS["brown"]))
    c = np.tile(sclera, (len(vertices), 1))
    for name, joint in (("left_eye", 2), ("right_eye", 3)):
        ids = np.flatnonzero(model.group(name))
        rel = vertices[ids] - joints[joint]
        radial = np.hypot(rel[:, 0], rel[:, 1])
        front = rel[:, 2] > 0
        ring = np.clip((0.0060 - radial) / 0.0006, 0, 1) * front
        pupil = np.clip((0.0022 - radial) / 0.0005, 0, 1) * front
        col = sclera * (1 - ring)[:, None] + iris_rgb * ring[:, None]
        c[ids] = col * (1 - pupil)[:, None] + np.array([.003, .003, .003]) * pupil[:, None]
    return c


def paint_atlas(model, pos, tris, uvs, res: int, *, seed: int, tone=(0.5, 0.35)):
    """sRGB base colour in the GNM skin atlas: tone chart + cellular pores +
    lip tint (the skin's UV layout covers the square once; eyeballs, teeth
    and tongue have their own overlapping layouts and are not painted here)."""
    from ..head.skin import cellular, tone_color
    from ..rig.bake import rasterize_uv
    from ..eye.optics import linear_to_srgb
    tid, bary = rasterize_uv(uvs.reshape(-1, 2), np.arange(uvs.size // 2).reshape(-1, 3), res)
    valid = tid >= 0
    y, x = np.nonzero(valid)
    t = tid[y, x]
    w = bary[y, x].astype(np.float64)
    corners = tris[t]                                            # (n, 3) vertex ids
    p3 = (pos[corners] * w[:, :, None]).sum(1)
    base = np.array(tone_color(*tone), np.float64)
    colour = np.tile(base, (len(y), 1))
    pores, _ = cellular(p3, 0.0011, seed + 17, density=0.9)
    colour *= (1.0 - 0.10 * np.clip(pores, 0, 1))[:, None]
    mottle, _ = cellular(p3, 0.012, seed + 29, density=0.6)
    colour *= (1.0 + 0.05 * (np.clip(mottle, 0, 1) - 0.5))[:, None]
    groups = {name: model.group(name) for name in ("upper_lip", "lower_lip", "mouth_sock")}
    lip = ((groups["upper_lip"] | groups["lower_lip"])[corners] * w).sum(1)
    colour = colour * (1 - 0.6 * lip)[:, None] + np.array([.5, .2, .18]) * (0.6 * lip)[:, None]
    sock = (groups["mouth_sock"][corners] * w).sum(1)
    colour = colour * (1 - sock)[:, None] + np.array([.12, .03, .03]) * sock[:, None]
    atlas = np.full((res, res, 3), 0, np.uint8)
    atlas[y, x] = np.uint8(np.clip(linear_to_srgb(colour) * 255 + .5, 0, 255))
    # gutters: nearest painted texel so bilinear sampling does not pick black
    from scipy.ndimage import distance_transform_edt
    _, nearest = distance_transform_edt(~valid, return_indices=True)
    atlas = atlas[nearest[0], nearest[1]]
    return atlas


def render_portrait(pos_glb, nrm_glb, tris, uvs, atlas, cam, size: int = 1024, extra=None):
    """Software raster of the head GLB through the Pixal3D camera: textured,
    Lambertian + ambient, alpha where the head covers the pixel. `extra` =
    (triangles, per-vertex linear colours) drawn with vertex colours (eyeballs)."""
    from ..eye.optics import linear_to_srgb, srgb_to_linear
    pix = cam.project(pos_glb)
    grid = pos_glb * np.array([-1.0, 1.0, -1.0])
    depth_v = cam.distance - grid[:, 2]
    light = np.array([0.35, 0.45, -1.0]); light /= np.linalg.norm(light)
    shade_v = 0.35 + 0.65 * np.clip(nrm_glb @ light, 0, 1)
    tex = srgb_to_linear(atlas.astype(np.float64) / 255.0)
    res = atlas.shape[0]
    img = np.zeros((size, size, 3))
    depth = np.full((size, size), np.inf)
    uv_t = uvs.reshape(-1, 3, 2)
    n_tex = len(tris)
    if extra is not None:
        tris = np.concatenate([tris, extra[0]])
        vcol = extra[1]
    for i, (a, b, c) in enumerate(tris):
        xs, ys = pix[[a, b, c], 0], pix[[a, b, c], 1]
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
        z = w0 * depth_v[a] + w1 * depth_v[b] + w2 * depth_v[c]
        sel = inside & (z < depth[y0:y1 + 1, x0:x1 + 1])
        if not sel.any():
            continue
        depth[y0:y1 + 1, x0:x1 + 1][sel] = z[sel]
        s = (w0 * shade_v[a] + w1 * shade_v[b] + w2 * shade_v[c])[sel]
        if i < n_tex:
            uv = (w0[..., None] * uv_t[i, 0] + w1[..., None] * uv_t[i, 1] + w2[..., None] * uv_t[i, 2])[sel]
            tx = np.clip((uv[:, 0] * res).astype(int), 0, res - 1)
            ty = np.clip((uv[:, 1] * res).astype(int), 0, res - 1)
            img[y0:y1 + 1, x0:x1 + 1][sel] = tex[ty, tx] * s[:, None]
        else:
            col = (w0[..., None] * vcol[a] + w1[..., None] * vcol[b] + w2[..., None] * vcol[c])[sel]
            img[y0:y1 + 1, x0:x1 + 1][sel] = col * s[:, None]
    alpha = np.isfinite(depth)
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3] = np.uint8(np.clip(linear_to_srgb(img) * 255 + .5, 0, 255))
    rgba[..., 3] = np.where(alpha, 255, 0)
    return rgba


def head_inputs(out, seed: int = 1, *, strength: float = 1.0, identity=None, tone=(0.5, 0.35), iris: str = "brown",
                fov_deg: float = 20.0, size: int = 1024, texture_res: int = 1024, model=None) -> dict:
    """Write pixal3d.glb, portrait.png and synthetic.json for a sampled identity."""
    from PIL import Image
    from ..eye.geometry import Mesh
    from ..eye.glb import GLBBuilder
    from ..head.camera import PixalCamera
    from ..rig.gnm_model import GNMModel
    model = model or GNMModel()
    beta = np.asarray(identity, np.float64) if identity is not None else sample_identity(seed, strength, dim=model.identity_dim)
    vertices, joints = evaluate(model, beta)
    all_tris = model.data["triangles"]
    # The exterior skin with its eye-socket lining (the head fit casts rays through the
    # portrait's eyes and needs a surface there) but without the mouth cavity, which would
    # attract the template registration's correspondences.
    outer = model.group("skin_exterior") | model.group("eye_sockets") | model.group("eye_interiors")
    skin = outer[all_tris].all(1)
    eyeballs = (model.group("left_eye") | model.group("right_eye"))[all_tris].all(1)
    tris = all_tris[skin]
    uvs = model.data["triangle_uvs"][skin].astype(np.float64)
    atlas = paint_atlas(model, vertices, tris, uvs, texture_res, seed=seed, tone=tone)
    # GNM metres (+Y up, face +Z) -> the Pixal3D GLB frame (face -Z, head centred, framed like Pixal3D output)
    used = np.unique(np.concatenate([tris, all_tris[eyeballs]]))
    centre = (vertices[used].min(0) + vertices[used].max(0)) / 2
    extent = float((vertices[used].max(0) - vertices[used].min(0)).max())
    k = GLB_HEAD_EXTENT / extent                                  # GLB units per metre
    S = np.array([-1.0, 1.0, -1.0])
    pos_glb = (vertices - centre) * k * S
    nrm_glb = _vertex_normals(vertices, np.concatenate([tris, all_tris[eyeballs]])) * S
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # unweld per triangle corner for the UV atlas
    corner_ids = tris.reshape(-1)
    mesh_pos = pos_glb[corner_ids].astype(np.float32)
    mesh_nrm = nrm_glb[corner_ids].astype(np.float32)
    mesh_uv = uvs.reshape(-1, 2).astype(np.float32)
    mesh_tri = np.arange(len(corner_ids), dtype=np.uint32).reshape(-1, 3)
    b = GLBBuilder("vhuman synthetic head")
    tex = b.texture(atlas, "synthetic_basecolor")
    mr = b.texture(np.full((16, 16, 3), [0, 140, 0], np.uint8), "synthetic_mr")
    mat = b.material({"name": "pixal3d", "pbrMetallicRoughness": {"baseColorTexture": {"index": tex},
                                                                   "metallicRoughnessTexture": {"index": mr}}})
    b.node("mesh", mesh=b.mesh(Mesh("head", mesh_pos, mesh_nrm, mesh_uv, mesh_tri), mat))
    b.write(out / "pixal3d.glb")
    # the portrait: a camera whose crop is the whole image, then the Pixal3D framing from its own alpha
    cam = PixalCamera(math.radians(fov_deg), 1.0, 0.0, 0.0, float(size))
    rgba = render_portrait(pos_glb, nrm_glb, tris, uvs, atlas, cam, size, extra=(all_tris[eyeballs], eyeball_colours(model, vertices, joints, iris)))
    Image.fromarray(rgba, "RGBA").save(out / "portrait.png")
    cam2 = PixalCamera.from_portrait(out / "portrait.png", math.radians(fov_deg))
    framing = dict(side=cam2.side, left=cam2.left, top=cam2.top, intended_side=float(size))
    record = {"schema": SCHEMA, "source": "synthetic", "seed": int(seed), "strength": float(strength),
              "identity_modes": int(IDENTITY_MODES), "identity": beta.round(6).tolist(), "tone": list(tone), "iris": iris,
              "fov_deg": fov_deg, "glb_units_per_metre": k, "head_extent_m": extent, "framing": framing,
              "texture_res": texture_res, "triangles": int(len(tris)),
              "model": {"name": "gnm_v3", "license": "Apache-2.0", "source": "https://huggingface.co/google/gnm-v3"},
              "license": "no image licence: procedural identity and appearance",
              "limitations": ["procedural skin tone, pores and lips only: no hair, brows or lashes in the texture",
                              "the portrait is a software render used by the head fit, not a photograph",
                              "the GLB carries the outer skin only; the eyes are rendered into the portrait and refitted analytically"]}
    (out / "synthetic.json").write_text(json.dumps(record, indent=1))
    return record


def create_head(work, seed: int = 1, *, subject: str | None = None, **kw) -> dict:
    """A head record in `work/heads/<id>` (like `head-fit`) from a synthetic identity:
    the fitted analytic eyes, lids and skin maps, ready for the rig builder."""
    import time
    import uuid
    from ..head import fit
    folder = Path(work) / "heads" / uuid.uuid4().hex[:12]
    started = time.perf_counter()
    record = head_inputs(folder, seed, **kw)
    rec = fit.fit_head(folder / "portrait.png", folder / "pixal3d.glb", folder, fov_deg=record["fov_deg"], res=1024)
    head = {"id": folder.name, "subject": subject or f"synthetic identity {seed}", "quality": "synthetic",
            "fov": record["fov_deg"], "created": time.time(), "source": "synthetic", "synthetic": record,
            "fit": {k: rec[k] for k in ("fit", "plate", "eyes", "seconds")}, "export": rec["export"],
            "iris": rec["iris"], "skin": rec["skin"], "seconds": round(time.perf_counter() - started, 2)}
    (folder / "head.json").write_text(json.dumps(head, indent=1, default=float))
    return {"id": folder.name, "folder": str(folder), "seed": seed, "fit": rec["fit"], "seconds": head["seconds"]}
