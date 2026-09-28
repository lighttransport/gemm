"""Data-driven expressions: the subject's own expression portraits.

1. Generation (GPU job `expressions`): Qwen-Image 2.1 edits the neutral
   portrait (composited on grey) into each expression of EXPRESSIONS while
   keeping identity, framing and lighting. Files: <head>/rig/expressions/
   ref.png, <name>.png, manifest.json.
2. Shapes (rig build, PyTorch): dense optical flow (OpenCV DIS) from the
   neutral to each expression portrait, with a forward/backward consistency
   confidence and the head's rigid drift (skull band + ears) removed. A
   pre-skinning correction is solved so that the rig's surface for that
   expression's controls, projected through Pixal3D's camera, lands where
   the flow says: pixel error (confidence-weighted, front-facing vertices),
   a Laplacian smoothness term, a weak prior against motion along the view
   ray (not observable) and zero outside the controls' region. The
   correction is split back onto the controls in proportion to their
   procedural motion (and the side masks for left/right pairs).
3. Wrinkle maps (wrinkles.py) use the same flow to warp each expression back
   onto the neutral image.
The prompts describe expressions in plain words; the controls they map to
are our lr.face.v1 names.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

KEEP = ("Keep the same person, identity, age, skin, hairstyle, head position, head size, camera framing, "
        "lighting and plain grey background exactly as in the reference photo; frontal view, looking into the "
        "camera, only the facial expression changes.")

# name -> (prompt, controls it shows at weight 1..0, wrinkle group)
EXPRESSIONS = {
    "smile": ("The person smiles broadly with lips closed, cheeks raised.",
              {"mouthSmileLeft": 1, "mouthSmileRight": 1, "cheekSquintLeft": .5, "cheekSquintRight": .5}, "smile"),
    "brows_up": ("The person raises both eyebrows high in surprise, forehead wrinkled, mouth closed and relaxed.",
                 {"browInnerUp": 1, "browOuterUpLeft": 1, "browOuterUpRight": 1}, "brow_up"),
    "brows_down": ("The person frowns angrily, eyebrows pulled down and together, vertical creases between the "
                   "brows, mouth closed and relaxed.", {"browDownLeft": 1, "browDownRight": 1}, "brow_down"),
    "sneer": ("The person wrinkles the nose in disgust, nostrils flared and upper lip slightly raised, mouth "
              "closed.", {"noseSneerLeft": 1, "noseSneerRight": 1}, "brow_down"),
    "squint": ("The person squints, narrowing both eyes, lower eyelids raised, mouth relaxed.",
               {"eyeSquintLeft": 1, "eyeSquintRight": 1, "cheekSquintLeft": .5, "cheekSquintRight": .5}, "smile"),
    "frown": ("The person looks sad, mouth corners pulled down, lips closed.",
              {"mouthFrownLeft": 1, "mouthFrownRight": 1}, "mouth"),
    "pucker": ("The person purses the lips forward as for a kiss, lips closed.", {"mouthPucker": 1}, "mouth"),
    "funnel": ("The person rounds the lips into an 'oo' shape, slightly open and pushed forward.",
               {"mouthFunnel": 1}, "mouth"),
    "stretch": ("The person stretches the lips wide horizontally in a tense grimace, lips closed.",
                {"mouthStretchLeft": 1, "mouthStretchRight": 1}, "mouth"),
    "press": ("The person presses the lips tightly together.", {"mouthPressLeft": 1, "mouthPressRight": 1}, "mouth"),
    "cheek_puff": ("The person puffs both cheeks out with air, lips closed.", {"cheekPuff": 1}, "mouth"),
    "jaw_open": ("The person opens the mouth wide as if saying 'ah', teeth visible, eyebrows relaxed.",
                 {"jawOpen": 1}, "mouth"),
}


# ---- 1. generation (server process; Qwen through the native backend) -------------------------

def expressions_job(service, request: dict, progress, cancel, python=None, mock=False) -> dict:
    """{head_id, names (default all), preset (low8|fast12), steps, seed}"""
    import sys
    from .. import gpu, qwen
    head_id = request.get("head_id")
    folder = service.head_file(head_id, "head.json").parent
    names = request.get("names") or list(EXPRESSIONS)
    bad = [n for n in names if n not in EXPRESSIONS]
    if bad:
        raise ValueError(f"unknown expressions: {bad}")
    preset = request.get("preset", "low8")
    if preset not in ("low8", "fast12"):
        raise ValueError("preset must be low8 or fast12")
    steps = int(request.get("steps", 20))
    seed = int(request.get("seed", 7))
    out = folder / "rig" / "expressions"
    out.mkdir(parents=True, exist_ok=True)
    ref = out / "ref.png"
    im = Image.open(folder / "portrait.png").convert("RGBA")
    bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
    bg.alpha_composite(im)
    bg.convert("RGB").save(ref)
    manifest = {"ref": "ref.png", "size": list(im.size), "expressions": {}, "model": "Qwen-Image 2.1 (edit)",
                "license": qwen.LICENSE, "created": time.time()}
    lock = (service.work / "mock-gpu.lock") if mock else gpu.LOCK_PATH
    progress(0.01, "waiting for the GPU")
    need = 8192 if preset == "low8" else gpu.QWEN_MIN_FREE_MIB
    with gpu.device_session(need, cancel, lock_path=lock, check_memory=not mock):
        qwen._import_qimg21()
        from qimg21_i23d.backends import GenRequest
        backend = qwen.make_backend(python, mock, preset=preset)
        try:
            for i, name in enumerate(names):
                if cancel.is_set():
                    raise gpu.Cancelled("cancelled")
                progress(0.02 + 0.96 * i / len(names), f"expression {name}")
                prompt, controls, group = EXPRESSIONS[name]
                t = time.perf_counter()
                if mock:
                    _mock_expression(ref, out / f"{name}.png", name)
                else:
                    backend.generate(GenRequest(prompt=prompt + " " + KEEP, out=out / f"{name}.png", width=im.size[0],
                                                height=im.size[1], steps=steps, seed=seed, references=(ref,),
                                                true_cfg_scale=4.0))
                manifest["expressions"][name] = {"file": f"{name}.png", "prompt": prompt, "controls": controls,
                                                 "wrinkle_group": group, "seconds": round(time.perf_counter() - t, 1)}
                (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        finally:
            backend.close()
    del sys
    return {"id": head_id, "expressions": list(manifest["expressions"]), "folder": str(out)}


def _mock_expression(ref: Path, out: Path, name: str):
    """No GPU: a smooth synthetic warp of the reference (for tests)."""
    a = np.asarray(Image.open(ref).convert("RGB"), np.float32)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    amp = 3.0 + 2.0 * (hash(name) % 5)
    dy = amp * np.exp(-(((xx - w * .5) / (w * .15)) ** 2 + ((yy - h * .62) / (h * .08)) ** 2))
    src_y = np.clip(yy + dy, 0, h - 1).astype(int)
    Image.fromarray(a[src_y, xx.astype(int)].astype(np.uint8)).save(out)


# ---- 2. flow ----------------------------------------------------------------------------------

def flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense flow a -> b (pixels, (H, W, 2)): b(p + F(p)) ~ a(p). OpenCV DIS."""
    import cv2
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0)
    return dis.calc(ga, gb, None)


def sample(field: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Bilinear sample of (H, W, C) at pixel coordinates (N, 2) (x, y)."""
    h, w = field.shape[:2]
    x = np.clip(px[:, 0], 0, w - 1.001)
    y = np.clip(px[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    tx, ty = (x - x0)[:, None], (y - y0)[:, None]
    f = field.reshape(h, w, -1)
    return ((f[y0, x0] * (1 - tx) + f[y0, x0 + 1] * tx) * (1 - ty) + (f[y0 + 1, x0] * (1 - tx) + f[y0 + 1, x0 + 1] * tx) * ty)


def consistent_flow(n_img: np.ndarray, e_img: np.ndarray):
    """Forward flow and its confidence from the forward/backward round trip."""
    fwd = flow(n_img, e_img)
    bwd = flow(e_img, n_img)
    h, w = fwd.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    p = np.stack([xx + fwd[..., 0], yy + fwd[..., 1]], -1).reshape(-1, 2)
    back = sample(bwd, p).reshape(h, w, 2)
    err = np.linalg.norm(fwd + back, axis=-1)
    return fwd, np.exp(-(err / 1.5) ** 2)


# ---- 3. shape fitting -----------------------------------------------------------------------

class Projector:
    """Head frame (metres) -> portrait pixels through Pixal3D's camera (torch)."""

    def __init__(self, cam, frame, device):
        import torch
        self.t = torch
        self.origin = torch.tensor(frame.origin, dtype=torch.float32, device=device)
        self.k = float(frame.k)
        self.cam = cam

    def __call__(self, p):
        torch = self.t
        flip = torch.tensor([-1.0, 1.0, -1.0], device=p.device)
        g = p * flip * self.k + self.origin                     # GLB
        q = g * flip                                            # grid frame
        c = self.cam
        depth = c.distance - q[..., 2]
        cx = c.focal * q[..., 0] / depth + c.side / 2
        cy = -c.focal * q[..., 1] / depth + c.side / 2
        return torch.stack([(cx + c.left) / c.scale, (cy + c.top) / c.scale], -1)


def rigid_drift(px0: np.ndarray, target: np.ndarray, stable: np.ndarray):
    """2D similarity (scale, rotation, translation) best mapping px0 -> target
    on the stable vertices; returns a function applying it."""
    a, b = px0[stable], target[stable]
    ca, cb = a.mean(0), b.mean(0)
    A, B = a - ca, b - cb
    U, S, Vt = np.linalg.svd(B.T @ A)
    R = U @ Vt
    s = S.sum() / max((A ** 2).sum(), 1e-9)
    t = cb - s * ca @ R.T
    return lambda p: s * p @ R.T + t


FOLD_AREA = 0.1          # fold: projected area below this share of the procedural pose's
FOLD_ROUNDS = 8
FOLD_SPREAD = 6          # rings of falloff around a fold
DEBUG = False
MIN_INTENSITY = 0.75
INTENSITIES = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0)   # joint-driven expressions


def fit(tmpl, pos, subj, feat, shapes: dict, rig_def: dict, skel: dict, joints, weights, folder: Path,
        iters: int = 500, log=print, smooth: float = 40.0) -> tuple[dict, dict]:
    """Refine `shapes` from the expression portraits in folder. Returns the new
    shapes and a report (per expression: pixel error of the procedural and the
    fitted rig on the confident, front-facing vertices)."""
    import torch
    from ..head.camera import PixalCamera
    from .common import edges, vertex_normals
    from . import template as T
    from .torchrig import TorchRig
    manifest = json.loads((folder / "manifest.json").read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cam = PixalCamera.from_portrait(subj.portrait, math.radians(subj.fit["camera"]["fov_deg"]))
    proj = Projector(cam, subj.frame, dev)
    n_img = np.asarray(Image.open(folder / manifest["ref"]).convert("RGB"))
    V = len(pos)
    skin_t = tmpl.tri_mat == 0
    nrm = vertex_normals(pos, tmpl.tris[skin_t])
    cam_h = subj.frame.to_h(cam.origin)
    view = cam_h - pos
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    facing = ((nrm * view).sum(1) > 0.25) & np.isin(tmpl.kind, [0, 1, 2, 3])
    px0 = proj(torch.tensor(pos, dtype=torch.float32, device=dev)).cpu().numpy()
    stable = facing & ((pos[:, 1] > 0.075) | (np.abs(pos[:, 0]) > 0.075))      # skull top, ears
    E = edges(tmpl.tris)
    ei, ej = torch.tensor(E[:, 0], device=dev), torch.tensor(E[:, 1], device=dev)
    nbrs = [[] for _ in range(V)]
    for a, b in E:
        nbrs[a].append(b)
        nbrs[b].append(a)
    new = {k: v.copy() for k, v in shapes.items()}
    adj = torch.sparse_coo_tensor(torch.tensor(np.concatenate([E, E[:, ::-1]]).T.copy(), device=dev),
                                  torch.ones(2 * len(E), device=dev), (V, V)).coalesce()
    deg = torch.sparse.sum(adj, 1).to_dense().clamp(min=1)
    report = {}
    side_left = np.clip(0.5 + pos[:, 0] / 0.012, 0, 1)
    for name, spec in manifest["expressions"].items():
        f = folder / spec["file"]
        if not f.exists():
            continue
        t0 = time.perf_counter()
        e_img = np.asarray(Image.open(f).convert("RGB").resize(n_img.shape[1::-1]))
        F, conf = consistent_flow(n_img, e_img)
        target = px0 + sample(F, px0)
        c_v = sample(conf[..., None], px0)[:, 0] * facing
        if stable.sum() > 20:
            drift = rigid_drift(px0, target, stable & (c_v > 0.5))
            target = target - (drift(px0) - px0)
        controls = {k: float(v) for k, v in spec["controls"].items()}
        # region: where the controls' procedural shapes move, grown over the mesh
        mag = sum(w * np.linalg.norm(shapes[c], axis=1) for c, w in controls.items() if c in shapes)
        joint_driven = any(e["input"] == c for e in rig_def["joint_matrix"] for c in controls)
        region = (mag > 1e-4) if np.ndim(mag) else np.zeros(V, bool)
        if joint_driven:
            region |= (np.asarray(weights)[:, 3] > 0.05) | (tmpl.group == 2)
        for _ in range(4):
            grow = region.copy()
            for i in np.flatnonzero(region):
                grow[nbrs[i]] = True
            region = grow
        # the lid linings and the mouth bag have no data but must follow the skin
        # (smoothness carries them); else a moved lip corner uncovers the bag
        region &= tmpl.kind != T.KIND["cap"]
        tr = TorchRig(rig_def, pos, shapes, joints, weights, device=dev)
        x = torch.zeros(1, len(tr.controls), device=dev)
        for c, w in controls.items():
            if c in tr.controls:
                x[0, tr.controls.index(c)] = w
        # the lips' inner rings (and, when the jaw opens, the lips as a whole) are
        # where the mouth's interior appears: flow is unreliable there
        mouth = tmpl.group == 2
        unreliable = mouth & (tmpl.ring < (4 if joint_driven else 1))
        tgt = torch.tensor(target, dtype=torch.float32, device=dev)
        cw = torch.tensor(c_v * region * ~unreliable, dtype=torch.float32, device=dev)
        vw = torch.tensor(view, dtype=torch.float32, device=dev)
        reg = torch.tensor(region, device=dev)[:, None]
        delta = torch.zeros(V, 3, device=dev, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=2e-4)
        # folds: skin triangles whose area projected on the procedural pose's
        # normal falls below FOLD_AREA of the procedural area (flipped or crushed)
        ftri = torch.tensor(tmpl.tris[skin_t][region[tmpl.tris[skin_t]].any(1)], device=dev)

        def tri_a(p):                    # area vectors (2 x area along the normal)
            return torch.cross(p[ftri[:, 1]] - p[ftri[:, 0]], p[ftri[:, 2]] - p[ftri[:, 0]], dim=-1)
        # joint-driven expressions: the image may show a smaller or larger joint
        # motion than the control at 1; find that intensity first (grid), fit the
        # remaining correction there and scale it back to the control's range
        intensity = 1.0
        if joint_driven:
            with torch.no_grad():
                errs = {k: float(((proj(tr(x * k)["pos"][0]) - tgt).norm(dim=-1) * cw).sum() / cw.sum().clamp(min=1))
                        for k in INTENSITIES}
            intensity = min(errs, key=errs.get)
            x = x * intensity
        with torch.no_grad():
            p_proc = tr(x)["pos"][0]
            px_proc = proj(p_proc)
            err0 = float(((px_proc - tgt).norm(dim=-1) * cw).sum() / cw.sum().clamp(min=1))
            a_proc = tri_a(p_proc)
            s_proc = a_proc.norm(dim=-1)
            n_proc = a_proc / s_proc[:, None].clamp(min=1e-12)
            ok_t = s_proc > 0.05 * s_proc.median()          # degenerate triangles: not judged
        for it in range(iters):
            d = delta * reg
            p = tr(x, pre=d[None])["pos"][0]
            px = proj(p)
            e_px = (((px - tgt) ** 2).sum(-1) * cw).sum() / cw.sum().clamp(min=1)
            lap = d[ei] - d[ej]
            e_s = (lap ** 2).sum(-1).mean() * 1e6 * smooth
            e_z = ((d * vw).sum(-1) ** 2).mean() * 1e6 * 0.5
            e_0 = (d ** 2).sum(-1).mean() * 1e6 * 0.02
            loss = e_px + e_s + e_z + e_0
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            # fold repair: shrink the correction around folded triangles (grown
            # 2 rings, halved per round) until no new folds remain
            dd = delta * reg
            atten = torch.ones(V, device=dev)
            folds0 = None
            for r in range(FOLD_ROUNDS + 1):
                p = tr(x, pre=(dd * atten[:, None])[None])["pos"][0]
                bad = ok_t & ((tri_a(p) * n_proc).sum(-1) < FOLD_AREA * s_proc)
                folds0 = int(bad.sum()) if folds0 is None else folds0
                if DEBUG:
                    print(name, "fold round", r, int(bad.sum()))
                if not bad.any() or r == FOLD_ROUNDS:
                    break
                m = torch.zeros(V, device=dev)
                m[ftri[bad].flatten()] = 1.0
                for _ in range(FOLD_SPREAD):             # smooth falloff: no step for new folds
                    m = torch.maximum(m, (adj @ m[:, None])[:, 0] / deg)
                m = torch.clamp(m, 0, 1)
                atten = atten * (1 - (0.5 if r < FOLD_ROUNDS - 1 else 1.0) * m)   # last round: remove
            folds1 = int(bad.sum())
            d = (dd * atten[:, None]).cpu().numpy().astype(np.float64)
            px_fit = proj(tr(x, pre=(dd * atten[:, None])[None])["pos"][0])
            err1 = float(((px_fit - tgt).norm(dim=-1) * cw).sum() / cw.sum().clamp(min=1))
        # split the correction over the controls (their procedural share; sides for L/R)
        shares = {}
        for c, w in controls.items():
            if c not in new or w < 1.0:          # primary controls only (secondary ones are shared)
                continue
            side = side_left if c.endswith("Left") else (1 - side_left) if c.endswith("Right") else np.ones(V)
            shares[c] = w * (np.linalg.norm(shapes[c], axis=1) + 1e-5 * side)
        tot = sum(shares.values()) if shares else None
        # a partial joint motion says little about the control's full range (the
        # correction would be extrapolated); open-mouth flow is unreliable anyway
        skipped = joint_driven and intensity < MIN_INTENSITY
        if skipped:
            tot = None
        if tot is not None:
            for c, s in shares.items():
                new[c] = new[c] + d * (s / np.maximum(tot, 1e-12))[:, None] / (controls[c] * intensity)
        report[name] = {"controls": controls, "intensity": intensity, "applied": not skipped, "px_error_procedural": round(err0, 3), "px_error_fitted": round(err1, 3),
                        "folds_repaired": [folds0, folds1], "vertices": int(region.sum()), "confident": int((c_v * region > 0.5).sum()),
                        "max_correction_mm": round(float(np.linalg.norm(d, axis=1).max() * 1000), 2),
                        "seconds": round(time.perf_counter() - t0, 1)}
        if log:
            log(f"expression {name}: {report[name]}")
    return new, report
