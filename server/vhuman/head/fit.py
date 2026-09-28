"""Fit analytic eyeballs into a Pixal3D head, and open the eyelids.

The portrait's eyes (landmarks.py) are cast as rays through Pixal3D's camera
(camera.py) onto the reconstructed head:
- scale: the iris centres' distance against an synthetic IPD default (63 mm),
  combined with the iris width against the default model limbus diameter
  (12 mm);
- pose: each cornea apex just behind the painted eye's surface, gazing at
  the camera (the portrait was asked to look into the lens), so the two
  eyes converge;
- opening: the head's front triangles that project inside the eye opening
  (the palpebral fissure found in the portrait) and lie on the eye's
  surface patch are removed, so the eyeball shows through the lids.
The result is one GLB: the carved head with Pixal3D's textures, plus the
two eyes (our shell and iris meshes) under node transforms.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ..eye import assets, optics
from ..eye import params as P
from ..eye.glb import GLBBuilder
from . import cleanup, eyeedge, eyeshell, iris_match, lids
from . import landmarks as L
from . import skin, texture
from .camera import PixalCamera, mesh_from_glb, ray_mesh

IPD_M = 0.063               # synthetic interpupillary distance default
IRIS_M = 2 * optics.ANATOMICAL.limbus_radius  # match the synthetic eye geometry
INSET_M = 0.0006            # the cornea apex at least this far behind the painted eye surface
LID_MARGIN_M = 0.0004       # the eyelid margins stay this far outside the eyeball
MAX_EXTRA_DEPTH_M = 0.004   # at most this much deeper than the painted surface
CARVE_GROW_M = 0.0003       # head triangles this close to the eyeball's surface go too (the painted eye)


@dataclass
class EyePose:
    side: str
    surface: np.ndarray          # the painted iris centre on the head (GLB)
    center: np.ndarray           # eyeball centre (GLB)
    rotation: np.ndarray         # 3x3, columns = eye x, y, z (gaze) in GLB
    units_per_m: float

    def as_dict(self) -> dict:
        return {"side": self.side, "surface": self.surface.round(6).tolist(), "center": self.center.round(6).tolist(),
                "rotation": self.rotation.round(6).tolist(), "units_per_m": round(self.units_per_m, 5)}


def _tri_arrays(mesh):
    P_, T = mesh["positions"], mesh["triangles"]
    v0 = P_[T[:, 0]]
    return v0, P_[T[:, 1]] - v0, P_[T[:, 2]] - v0


def _hit(cam, x, y, tri) -> np.ndarray | None:
    o, d = cam.rays(x, y)
    t, _ = ray_mesh(o, d, *tri)
    return None if not math.isfinite(t) else o + t * d


def _quat(rot: np.ndarray) -> list:
    """Rotation matrix -> glTF quaternion (x, y, z, w)."""
    m = rot
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return (q / np.linalg.norm(q)).tolist()


def fit_eyes(cam: PixalCamera, mesh: dict, eyes: list[L.Eye], ipd_m: float = IPD_M, iris_m: float = IRIS_M,
             inset_m: float = INSET_M) -> tuple[list[EyePose], dict]:
    tri = _tri_arrays(mesh)
    surf, widths = {}, []
    for e in eyes:
        s = _hit(cam, e.cx, e.cy, tri)
        if s is None:
            raise ValueError(f"the {e.side} eye's ray misses the head mesh")
        surf[e.side] = s
        lo, hi = _hit(cam, e.cx - e.r, e.cy, tri), _hit(cam, e.cx + e.r, e.cy, tri)
        if lo is not None and hi is not None:
            widths.append(float(np.linalg.norm(hi - lo)))
    ipd_units = float(np.linalg.norm(surf["right"] - surf["left"]))
    k_ipd = ipd_units / ipd_m
    k_iris = (float(np.mean(widths)) / iris_m) if widths else k_ipd
    k = math.sqrt(k_ipd * k_iris)                      # mesh units per metre
    apex = optics.ANATOMICAL.apex_z
    radius = optics.ANATOMICAL.sclera_radius
    poses, depths = [], {}
    for e in eyes:
        s = surf[e.side]
        gaze = cam.origin - s
        gaze /= np.linalg.norm(gaze)
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, gaze)
        x /= np.linalg.norm(x)
        y = np.cross(gaze, x)
        rot = np.stack([x, y, gaze], 1)
        center = s - gaze * (inset_m + apex) * k
        # Pixal3D paints the eye on a surface that stands in for the visible
        # eyeball: fit the sphere (radius fixed) to the painted points inside
        # the opening, sliding along the gaze line.
        painted = _painted_points(cam, mesh, e, s, k)
        fitted = None
        if len(painted) >= 30:
            ss = np.linspace(0.7 * radius * k, 1.6 * radius * k, 181)
            cost = [np.median(np.abs(np.linalg.norm(painted - (s - gaze * t_), axis=1) - radius * k)) for t_ in ss]
            fitted = float(ss[int(np.argmin(cost))])
            center = s - gaze * fitted
        # Push the eyeball back until the eyelid margins (hits along the eye
        # opening's contour) lie outside it: the lids then cover its edge.
        rim = _rim_points(cam, e, tri)
        extra = 0.0
        rho = (radius + LID_MARGIN_M) * k
        for q in rim:
            d = q - center
            if np.linalg.norm(d) >= rho:
                continue
            gd = float(gaze @ d)
            extra = max(extra, -gd + math.sqrt(max(gd * gd + rho * rho - float(d @ d), 0.0)))
        extra = min(extra, MAX_EXTRA_DEPTH_M * k)
        depths[e.side] = {"rim_points": len(rim), "extra_depth_mm": round(extra / k * 1000, 3),
                          "painted_points": len(painted),
                          "center_depth_mm": None if fitted is None else round(fitted / k * 1000, 3)}
        poses.append(EyePose(e.side, s, center - gaze * extra, rot, k))
    info = {"units_per_m": k, "k_ipd": k_ipd, "k_iris": k_iris, "ipd_units": ipd_units,
            "iris_width_units": widths, "head_height_m": float(np.ptp(mesh["positions"][:, 1]) / k),
            "depth": depths}
    return poses, info


def _painted_points(cam: PixalCamera, mesh: dict, eye: L.Eye, surface: np.ndarray, k: float) -> np.ndarray:
    """Head vertices that make up the painted eye: projecting inside the
    opening (eroded, away from the lids), facing the camera, near the iris."""
    P_ = mesh["positions"]
    near = np.linalg.norm(P_ - surface, axis=1) < 0.016 * k
    idx = np.flatnonzero(near)
    if len(idx) == 0:
        return np.zeros((0, 3))
    pix = cam.project(P_[idx])
    opening = np.asarray(Image.fromarray((eye.opening * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5))) > 127
    h, w = opening.shape
    xi = np.clip(np.round(pix[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pix[:, 1]).astype(int), 0, h - 1)
    facing = np.einsum("ij,ij->i", mesh["normals"][idx], cam.origin - P_[idx]) > 0
    pts = P_[idx][opening[yi, xi] & facing]
    # the front-most layer only: the nearest point per pixel
    order = np.argsort(np.linalg.norm(pts - cam.origin, axis=1))
    keys = set()
    front = []
    pix = cam.project(pts[order])
    for pnt, (px, py) in zip(pts[order], pix):
        key = (int(px), int(py))
        if key not in keys:
            keys.add(key)
            front.append(pnt)
    return np.array(front)


def _rim_points(cam: PixalCamera, eye: L.Eye, tri, count: int = 48) -> list:
    """3D hits along the eye opening's contour (the eyelid margins)."""
    m = eye.opening
    inner = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))) > 127
    ys, xs = np.nonzero(m & ~inner)
    if len(xs) == 0:
        return []
    ang = np.arctan2(ys - eye.cy, xs - eye.cx)
    order = np.argsort(ang)
    pick = order[np.linspace(0, len(order) - 1, min(count, len(order))).astype(int)]
    hits = []
    for x, y in zip(xs[pick], ys[pick]):
        h = _hit(cam, float(x), float(y), tri)
        if h is not None:
            hits.append(h)
    return hits


def carve(mesh: dict, cam: PixalCamera, eyes: list[L.Eye], poses: list[EyePose], depth_m: float = 0.015) -> np.ndarray:
    """Keep-mask over the head's triangles: drop the front-facing triangles
    whose three vertices project inside an eye's fissure (and its eyeball's
    silhouette), near that eye."""
    P_, T = mesh["positions"], mesh["triangles"]
    cen = P_[T].mean(1)
    pix = cam.project(P_)
    n = np.cross(P_[T[:, 1]] - P_[T[:, 0]], P_[T[:, 2]] - P_[T[:, 0]])
    facing = np.einsum("ij,ij->i", n, cam.origin - cen) > 0
    keep = np.ones(len(T), bool)
    for e, pose in zip(eyes, poses):
        h, w = e.opening.shape
        # one pixel inside the fissure, and inside the eyeball's silhouette
        # (beyond it, at the canthi, lids.wrap sinks the surface instead of
        # leaving a hole): the lid triangles left overlap the eyeball's edge
        mask = e.fissure if e.fissure is not None else e.opening
        opening = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))) > 127
        yy, xx = np.mgrid[0:h, 0:w]
        sx, sy = cam.project(pose.center)
        s_r = cam.focal * optics.ANATOMICAL.sclera_radius * pose.units_per_m / float(np.linalg.norm(pose.center - cam.origin))
        opening &= np.hypot(xx - sx, yy - sy) < 0.97 * s_r
        xi = np.clip(np.round(pix[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pix[:, 1]).astype(int), 0, h - 1)
        inside_v = opening[yi, xi]
        inside = inside_v[T].all(1)
        near = np.linalg.norm(cen - pose.surface, axis=1) < depth_m * pose.units_per_m
        keep &= ~(inside & facing & near)
        # The painted eye: anything inside the eyeball's volume (sclera ball
        # or cornea ball) would show in front of the real eye from any other
        # viewpoint. The lid margins were kept outside it (fit_eyes).
        q = (cen - pose.center) @ pose.rotation / pose.units_per_m          # eye-local metres
        prof = optics.ANATOMICAL
        grow = CARVE_GROW_M
        in_sclera = np.linalg.norm(q, axis=1) < prof.sclera_radius + grow
        in_cornea = np.linalg.norm(q - np.array([0.0, 0.0, prof.cornea_center_z]), axis=1) < prof.cornea_radius + grow
        keep &= ~(in_sclera | in_cornea)
    return keep


def choose_iris(eye_color: dict, plates: list[dict] | None) -> tuple[dict, str | None]:
    """Eye parameters for a portrait's iris colour: the nearest Qwen plate
    by colour when there is a library, else the chart suggestion."""
    p = P.defaults()
    p["sclera"]["skin_u"] = 0.8           # mild synthetic tint linked to portrait skin tone
    sug = (eye_color or {}).get("suggested") or {}
    if sug:
        p["iris"].update(primary_color_u=sug["primary_color_u"], primary_color_v=sug["primary_color_v"],
                         secondary_color_u=max(0.0, sug["primary_color_u"] - 0.12),
                         secondary_color_v=max(0.0, sug["primary_color_v"] - 0.1),
                         blend_method="Radial", color_blend=0.5, color_blend_softness=0.25)
    plate_id = None
    target = np.array((eye_color or {}).get("median_linear") or [])
    if plates and target.size == 3:
        best, best_d = None, math.inf
        for rec in plates:
            c = (rec.get("colors") or {}).get("primary_linear")
            if not c:
                continue
            d = iris_match.color_distance(target, c)
            if d < best_d:
                best, best_d = rec, d
        if best is not None:
            plate_id = best["id"]
            p["iris"].update(iris_match.color_controls(target, best["colors"]["primary_linear"]))
    return P.validate(p), plate_id


LINING_SHADE = np.array([0.78, 0.62, 0.62])   # skin -> lid margin / canthus: darker, pinker (linear multipliers)
LINING_SKIRT_SHADE = 0.55                      # deep lining shadow; the margin remains pinker


def lining_color(portrait, eyes) -> np.ndarray:
    """sRGB (0-255) for the lid margins and canthi: the skin around the
    eyes (a ring 2.2-3.5 iris radii out), shaded darker and pinker."""
    img = Image.open(portrait) if not isinstance(portrait, Image.Image) else portrait
    rgba = np.asarray(img.convert("RGBA"), np.float32) / 255
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    ring = np.zeros((h, w), bool)
    for e in eyes:
        d = np.hypot(xx - e.cx, yy - e.cy)
        ring |= (d > 2.2 * e.r) & (d < 3.5 * e.r)
    ring &= rgba[..., 3] > 0.9
    skin = np.median(rgba[..., :3][ring], axis=0) if ring.any() else np.array([0.8, 0.6, 0.5])
    lin = optics.srgb_to_linear(skin) * LINING_SHADE
    return np.clip(optics.linear_to_srgb(lin) * 255, 0, 255)


def export_head(out: Path, mesh: dict, keep: np.ndarray, poses: list[EyePose], eye_params: list[dict],
                res: int = 1024, iris_tex=None, eyes: list | None = None, cam: PixalCamera | None = None,
                tints: list | None = None, lining_rgb=None, skin_params=None, portrait=None) -> dict:
    """One GLB: the carved head (Pixal3D's textures copied), two eyes and
    their eyeshells (when `eyes` and `cam` are given)."""
    g = mesh["glb"]
    b = GLBBuilder("vhuman head")
    # Pad the original atlas before baking; embed only the final maps.
    padding, source_png = {}, {}
    skin_info, tangents = None, None
    mat = json.loads(json.dumps(mesh["material"]))
    pbr = mat.setdefault("pbrMetallicRoughness", {})
    for key in ("baseColorTexture", "metallicRoughnessTexture"):
        if key not in pbr:
            continue
        src = g.doc["textures"][pbr[key]["index"]]["source"]
        view = g.doc["bufferViews"][g.doc["images"][src]["bufferView"]]
        start = view.get("byteOffset", 0)
        # Skip mipmaps, which bleed between the densely packed charts.
        source_png[key], padding[key] = texture.pad_png(
            bytes(g.bin[start:start + view["byteLength"]]), mesh["uvs"], mesh["triangles"],
            tints=tints if key == "baseColorTexture" else None)
    if portrait is not None and skin.validate(skin_params)["enabled"] and "baseColorTexture" in source_png:
        skin_info, tangents = skin.bake(mesh, mesh["triangles"][keep], source_png["baseColorTexture"],
            source_png.get("metallicRoughnessTexture"), portrait, eyes, poses, cam, skin_params, out.parent,
            roughness_factor=pbr.get("roughnessFactor", 1.), metallic_factor=pbr.get("metallicFactor", 1.))
        for key, name in (("baseColorTexture", "skin_basecolor.png"), ("metallicRoughnessTexture", "skin_orm.png")):
            pbr[key] = {"index": b.texture_png((out.parent / name).read_bytes(), name, mipmaps=False)}
        pbr["metallicFactor"] = 1.
        pbr["roughnessFactor"] = 1.
        mat["normalTexture"] = {"index": b.texture_png((out.parent / "skin_normal.png").read_bytes(),
                                                       "skin_normal", mipmaps=False)}
    else:
        for key, png in source_png.items():
            pbr[key]["index"] = b.texture_png(png, f"head_{key}", mipmaps=False)
    mat["name"] = "head"
    head_mat = b.material(mat)
    from ..eye.geometry import Mesh
    tris = mesh["triangles"][keep].astype(np.uint32)
    head = Mesh("head", mesh["positions"].astype(np.float32), mesh["normals"].astype(np.float32),
                mesh["uvs"].astype(np.float32), tris, tangents)
    b.node("head", mesh=b.mesh(head, head_mat))
    lin = mesh.get("lid_lining")
    if lin is not None and len(lin["triangles"]):
        # the lid lining (lids.wrap): flat, shadowed skin, seen from both sides
        rgb = optics.srgb_to_linear(np.asarray(lining_rgb if lining_rgb is not None else (150, 100, 85)) / 255)
        # A linear-light gradient: pinker at the margin, darker at the globe.
        v = np.linspace(0, 1, 32)
        shade = 0.90 + (LINING_SKIRT_SHADE - 0.90) * v * v * (3 - 2 * v)
        gradient = optics.linear_to_srgb(rgb[None, :] * shade[:, None])
        rgba = np.full((32, 2, 4), 255, np.uint8)
        rgba[:, :, :3] = np.round(np.clip(gradient, 0, 1)[:, None, :] * 255).astype(np.uint8)
        lining_tex = b.texture(rgba, "lid_lining_gradient")
        lin_mat = b.material({"name": "lid_lining", "doubleSided": True,
                              "pbrMetallicRoughness": {"baseColorTexture": {"index": lining_tex},
                                                       "metallicFactor": 0.0, "roughnessFactor": 0.45}})
        lm = Mesh("lid_lining", lin["positions"].astype(np.float32), lin["normals"].astype(np.float32),
                  lin["uvs"].astype(np.float32), lin["triangles"].astype(np.uint32))
        b.node("lid_lining", mesh=b.mesh(lm, lin_mat))
    for wet in mesh.get("eye_edges", []):
        wet_mat = b.material(eyeedge.material(wet.name, lining_rgb if lining_rgb is not None else (150, 100, 85)))
        b.node(wet.name, mesh=b.mesh(wet, wet_mat))
    eyes_info = []
    for n, (pose, p) in enumerate(zip(poses, eye_params)):
        shell_mesh, iris_mesh = assets.add_eye_meshes(b, p, res, iris_tex=iris_tex, name=f"eye_{pose.side}",
                                                      volume=False)
        kids = [b.node(f"eye_{pose.side}_shell", mesh=shell_mesh, root=False),
                b.node(f"eye_{pose.side}_iris", mesh=iris_mesh, root=False)]
        occ_info = None
        if eyes is not None and cam is not None:
            occ_mesh, occ_tex, occ_info = eyeshell.build(eyes[n], pose, cam)
            occ_mat = b.material({"name": f"eye_{pose.side}_occlusion", "alphaMode": "BLEND",
                                  "pbrMetallicRoughness": {"baseColorTexture": {"index": b.texture(
                                      occ_tex, f"eye_{pose.side}_occlusion")}, "metallicFactor": 0.0},
                                  "extensions": {"KHR_materials_unlit": {}}})
            kids.append(b.node(f"eye_{pose.side}_eyeshell", mesh=b.mesh(occ_mesh, occ_mat), root=False))
        b.node(f"eye_{pose.side}", translation=pose.center, rotation=_quat(pose.rotation),
               scale=[pose.units_per_m] * 3, children=kids)
        eye_info = pose.as_dict()
        if occ_info is not None:
            eye_info["occlusion"] = occ_info
        eyes_info.append(eye_info)
    size = b.write(out)
    return {"bytes": size, "triangles_kept": int(keep.sum()), "triangles_removed": int((~keep).sum()),
            "eyes": eyes_info, "texture_padding": padding, "skin": skin_info}


def fit_head(portrait, head_glb, out_dir, fov_deg: float = 20.0, plates: list[dict] | None = None,
             plate_loader=None, res: int = 1024, iris_info: dict | None = None, skin_params: dict | None = None) -> dict:
    """The whole fit: eyes in the portrait, rays onto the head, carve, export."""
    skin_params = skin.validate(skin_params)
    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eyes = L.find_eyes(portrait)
    cam = PixalCamera.from_portrait(portrait, math.radians(fov_deg))
    mesh = mesh_from_glb(Path(head_glb))
    poses, info = fit_eyes(cam, mesh, eyes)
    keep = carve(mesh, cam, eyes, poses)
    # Cut the fissures along the smoothed contour and drape the lids onto the eyeballs.
    mesh, keep, wrap_info, lining = lids.wrap(mesh, keep, cam, eyes, poses)
    info["lids"] = wrap_info
    mesh, info["upper_lid_cleanup"] = cleanup.relax_upper_lids(mesh, keep, cam, eyes, poses)
    keep, info["cleanup"] = cleanup.remove_slivers(mesh, keep, cam, eyes, poses)
    # The canthi's tips (inside the fissure, beyond the eyeball's silhouette)
    # are kept, but Pixal3D painted them as sclera: recolour them (shadowed,
    # pinkish skin) in the head texture.
    lining_tris = mesh["triangles"][keep & lining[mesh["triangles"]].any(1)]
    lining_rgb = lining_color(portrait, eyes)
    color = iris_match.target_color(eyes)
    params, plate_id = choose_iris(color, plates)
    iris_tex = plate_loader(plate_id, res) if (plate_id and plate_loader) else None
    per_eye = []
    for pose in poses:
        q = json.loads(json.dumps(params))
        q["optics"]["side"] = pose.side
        per_eye.append(q)
    result = export_head(out / "head_eyes.glb", mesh, keep, poses, per_eye, res, iris_tex, eyes=eyes, cam=cam,
                         tints=[(lining_tris, lining_rgb)], lining_rgb=lining_rgb, skin_params=skin_params, portrait=portrait)
    info["lining"] = {"triangles": int(len(lining_tris)), "srgb": [round(float(c)) for c in lining_rgb]}
    selection = dict(iris_info or {"requested": "library", "source": "library"}, selected=plate_id)
    if plate_id:
        chosen = next(rec for rec in plates if rec["id"] == plate_id)
        selection["color_distance"] = iris_match.color_distance(color.get("median_linear", []),
                                                                chosen.get("colors", {}).get("primary_linear", []))
        matched = iris_match.controlled_color(chosen["colors"]["primary_linear"], params["iris"])
        selection["matched_primary_linear"] = matched.tolist()
        selection["matched_color_distance"] = iris_match.color_distance(color.get("median_linear", []), matched)
    else:
        selection["source"] = "procedural"
    record = {"skin": result["skin"], "iris": selection, "camera": cam.as_dict(), "eyes": [e.as_dict() for e in eyes], "fit": info,
              "eye_params": params, "plate": plate_id, "export": result,
              "seconds": round(time.perf_counter() - started, 2)}
    (out / "fit.json").write_text(json.dumps(record, indent=1, default=float))
    overlay(portrait, eyes, out / "landmarks.png")
    return record


def overlay(portrait, eyes, out: Path) -> None:
    """The portrait with the detected irises, pupils, openings and corners."""
    from PIL import ImageDraw
    im = Image.open(portrait).convert("RGBA")
    bg = Image.new("RGBA", im.size, (60, 62, 70, 255))
    bg.alpha_composite(im)
    for e in eyes:
        ov = np.zeros((im.size[1], im.size[0], 4), np.uint8)
        if e.fissure is not None:
            ov[e.fissure] = (0, 140, 255, 50)
        ov[e.opening] = (0, 255, 0, 60)
        bg.alpha_composite(Image.fromarray(ov))
        d = ImageDraw.Draw(bg)
        d.ellipse([e.cx - e.r, e.cy - e.r, e.cx + e.r, e.cy + e.r], outline=(255, 60, 60, 255), width=2)
        d.ellipse([e.cx - e.pupil_r, e.cy - e.pupil_r, e.cx + e.pupil_r, e.cy + e.pupil_r],
                  outline=(255, 230, 0, 255), width=1)
        for c in e.corners:
            d.ellipse([c[0] - 3, c[1] - 3, c[0] + 3, c[1] + 3], fill=(0, 140, 255, 255))
    bg.convert("RGB").save(out)
