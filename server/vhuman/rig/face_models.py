"""Permissively licensed face topologies fitted onto a vhuman subject.

The procedural rig remains the animation correspondence scaffold. Its fitted
surface, controls and skinning are transferred onto the selected source's own
vertices/UVs; GNM PCA or ICT authored shapes inform the exported controls.
Weights are cached outside tracked source, under tmp/vhuman-rig/models.
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import bake, template

ROOT = Path(__file__).resolve().parents[3]
MODEL_CACHE = ROOT / "tmp/vhuman-rig/models"
GNM_SHA256 = "61d78bbfb4ad8e0b38495804a4caef3214d3df00f8c3f68761e63b41ce3747eb"
GNM_REVISION = "c01e90d298d82301f9fd18f54806be751775cb7c"
ICT_REVISION = "da5f95a607f5e6b37755b38d3385d7f2853732e5"
SOURCES = ("gnm_v3", "ict_facekit_light", "procedural")


@dataclass
class Source:
    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_uvs: np.ndarray
    identity_basis: np.ndarray | None
    expression_basis: np.ndarray | None
    authored_shapes: dict[str, np.ndarray]
    fit_mask: np.ndarray
    provenance: dict


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _gnm_path(cache: Path) -> Path:
    path = cache / "gnm-v3/gnm_head.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/google/gnm-v3/resolve/{GNM_REVISION}/v3_0/gnm_head.npz"
        partial = path.with_suffix(".partial")
        try:
            urllib.request.urlretrieve(url, partial)
            if _sha256(partial) != GNM_SHA256:
                raise ValueError("GNM v3 weight hash mismatch")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    if _sha256(path) != GNM_SHA256:
        raise ValueError(f"GNM v3 weight hash mismatch: {path}")
    return path


def _ict_path(cache: Path) -> Path:
    source = cache / "ict-facekit/source"
    path = source / "FaceXModel/generic_neutral_mesh.obj"
    if path.exists():
        return path.parent
    source.parent.mkdir(parents=True, exist_ok=True)
    # An immutable revision keeps the morph vertex order and topology stable.
    url = f"https://github.com/USC-ICT/ICT-FaceKit/archive/{ICT_REVISION}.zip"
    archive = source.parent / "ict-facekit.zip"
    try:
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as z:
            prefix = f"ICT-FaceKit-{ICT_REVISION}/"
            for member in z.infolist():
                if not member.filename.startswith(prefix) or member.is_dir():
                    continue
                rel = Path(member.filename[len(prefix):])
                if not rel.parts or rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("unsafe ICT-FaceKit archive member")
                dst = source / rel
                if not dst.resolve().is_relative_to(source.resolve()):
                    raise ValueError("unsafe ICT-FaceKit archive member")
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, dst.open("wb") as out:
                    import shutil
                    shutil.copyfileobj(src, out)
    finally:
        archive.unlink(missing_ok=True)
    if not path.exists():
        raise FileNotFoundError(f"ICT-FaceKit Light neutral mesh missing: {path}")
    return path.parent


def _obj(path: Path, *, faces: bool = True):
    pos, uv, tri, tri_uv = [], [], [], []
    with path.open() as fh:
        for line in fh:
            if line.startswith("v "):
                pos.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt ") and faces:
                uv.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f ") and faces:
                corners = [x.split("/") for x in line.split()[1:]]
                for i in range(1, len(corners) - 1):
                    part = [corners[j] for j in (0, i, i + 1)]
                    tri.append([int(x[0]) - 1 for x in part])
                    tri_uv.append([int(x[1]) - 1 for x in part])
    p = np.asarray(pos, np.float32)
    if not faces:
        return p
    t = np.asarray(tri, np.int32)
    u = np.asarray(uv, np.float32)[np.asarray(tri_uv, np.int32)]
    return p, t, u


def load(name: str, cache: Path | None = None) -> Source:
    if name not in SOURCES[:2]:
        raise ValueError(f"unknown external face model: {name}")
    cache = Path(cache) if cache else MODEL_CACHE
    if name == "gnm_v3":
        path = _gnm_path(cache)
        with np.load(path, allow_pickle=False) as z:
            names = z["vertex_group_names"].tolist()
            exterior = z["vertex_groups"][names.index("skin_exterior")] > .5
            skin = exterior
            ids = np.flatnonzero(skin)
            remap = np.full(len(skin), -1, np.int32)
            remap[ids] = np.arange(len(ids))
            all_tris = z["triangles"]
            valid = skin[all_tris].all(1)
            tris = remap[all_tris[valid]]
            uv = z["triangle_uvs"][valid].astype(np.float32)
            vertices = z["template_vertex_positions"][ids].astype(np.float32)
            ib = z["vertex_identity_basis"][:170, ids].astype(np.float32)
            eb = z["expression_basis"][:, ids].astype(np.float32)
            if len(eb) != len(z["expression_names"]):
                raise ValueError("GNM expression basis/name dimension mismatch")
            return Source(name, vertices, tris, uv, ib, eb, {}, exterior[ids],
                          {"model": name, "version": str(z["version"]), "sha256": GNM_SHA256,
                           "license": "Apache-2.0", "expression_dim": len(eb),
                           "identity_dim": len(z["vertex_identity_basis"]),
                           "source": "https://huggingface.co/google/gnm-v3"})
    folder = _ict_path(cache)
    vertices, tris, uv = _obj(folder / "generic_neutral_mesh.obj")
    if len(vertices) != 26719:
        raise ValueError("ICT-FaceKit Light vertex count mismatch")
    # Face and head/neck are the exterior. Socket/mouth components need their
    # own interior material and cannot be baked from the subject's skin photo.
    # M_Face and M_BackHead share the first 14,062 vertices; later vertices
    # are gums, teeth, eyes and tear/occlusion components.
    skin = np.arange(len(vertices)) < 14062
    exterior = skin.copy()
    ids = np.flatnonzero(skin)
    valid = skin[tris].all(1)
    tris, uv = tris[valid], uv[valid]
    # Light uses two horizontal UDIM tiles. Pack both into one atlas so the
    # existing glTF/USD single-texture material and baker retain every island.
    uv = uv.copy()
    uv[..., 0] *= .5
    neutral = vertices[ids] * .01  # ICT OBJ coordinates are centimetres.
    identity = np.stack([(_obj(folder / f"identity{i:03}.obj", faces=False)[ids] - vertices[ids]) * .01
                         for i in range(100)]).astype(np.float32)
    shapes = {}
    for path in sorted(folder.glob("*.obj")):
        if path.stem.startswith("identity") or path.stem == "generic_neutral_mesh":
            continue
        v = _obj(path, faces=False)
        if len(v) == len(vertices):
            shapes[path.stem] = ((v[ids] - vertices[ids]) * .01).astype(np.float32)
    return Source(name, neutral, tris, uv, identity, None, shapes, exterior[ids],
                  {"model": name, "revision": ICT_REVISION, "license": "MIT",
                   "identity_dim": len(identity), "authored_shapes": len(shapes),
                   "source": "https://github.com/USC-ICT/ICT-FaceKit"})


def _similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
    fix = np.diag([1., 1., np.linalg.det(u @ vt)])
    R = (u @ fix @ vt).T
    s = float(np.sum((a - ac) @ R.T * (b - bc)) / np.maximum(np.sum((a - ac) ** 2), 1e-12))
    return s, R, bc - s * R @ ac


def _closest_map(points: np.ndarray, surface: np.ndarray, tris: np.ndarray, k: int = 12):
    from scipy.spatial import cKDTree
    tri_pos = surface[tris]
    tree = cKDTree(tri_pos.mean(1))
    _, cand = tree.query(points, k=min(k, len(tris)))
    if cand.ndim == 1:
        cand = cand[:, None]
    best_t = np.zeros(len(points), np.int32)
    best_b = np.zeros((len(points), 3), np.float32)
    best_d = np.full(len(points), np.inf)
    for col in range(cand.shape[1]):
        tri_id = cand[:, col]
        corners = tri_pos[tri_id]
        q, bary = bake._closest_on_triangles(points, corners[:, 0], corners[:, 1], corners[:, 2])
        dist = np.linalg.norm(points - q, axis=1)
        take = dist < best_d
        best_d[take], best_t[take], best_b[take] = dist[take], tri_id[take], bary[take]
    return tris[best_t], best_b, best_d


def _map_values(values: np.ndarray, ids: np.ndarray, bary: np.ndarray) -> np.ndarray:
    return np.einsum("vj,vjc->vc", bary, values[ids])


def _fit_identity(src: Source, target: np.ndarray, *, iterations: int = 3):
    from scipy.spatial import cKDTree
    fit_ids = np.flatnonzero(src.fit_mask)
    fit_ids = fit_ids[::max(1, len(fit_ids) // 1800)]
    base = src.vertices.astype(np.float64)
    tgt = target.astype(np.float64)
    # Both source models and the vhuman head use +Y up, face +Z.
    sy = np.quantile(base[fit_ids, 1], [.05, .95])
    ty = np.quantile(tgt[:, 1], [.05, .95])
    scale = float(np.clip((ty[1] - ty[0]) / max(sy[1] - sy[0], 1e-6), .6, 1.6))
    R = np.eye(3)
    t = tgt.mean(0) - scale * base[fit_ids].mean(0)
    tree = cKDTree(tgt)
    coeff = np.zeros(0 if src.identity_basis is None else len(src.identity_basis))
    for _ in range(iterations):
        posed = base + (np.einsum("i,ivc->vc", coeff, src.identity_basis) if len(coeff) else 0)
        moving = scale * posed[fit_ids] @ R.T + t
        dist, nearest = tree.query(moving)
        keep = dist <= np.quantile(dist, .8)
        ds, dR, dt = _similarity(moving[keep], tgt[nearest[keep]])
        ds = float(np.clip(ds, .9, 1.1))
        t = ds * dR @ t + dt
        R = dR @ R
        scale *= ds
        if len(coeff):
            moving = scale * base[fit_ids] @ R.T + t
            _, nearest = tree.query(moving)
            B = np.einsum("ivc,dc->ivd", src.identity_basis[:, fit_ids], R) * scale
            A = B.transpose(1, 2, 0).reshape(-1, len(coeff))
            y = (tgt[nearest] - moving).reshape(-1)
            lam = max(float(np.median(np.sum(A * A, axis=0))) * .04, 1e-10)
            coeff = np.linalg.solve(A.T @ A + lam * np.eye(len(coeff)), A.T @ y)
            coeff = np.clip(coeff, -3., 3.)
    positioned = base + (np.einsum("i,ivc->vc", coeff, src.identity_basis) if len(coeff) else 0)
    positioned = scale * positioned @ R.T + t
    return positioned, scale, R, coeff


def _ict_name(name: str) -> list[str]:
    if name == "browInnerUp":
        return ["browInnerUp_L", "browInnerUp_R"]
    if name == "cheekPuff":
        return ["cheekPuff_L", "cheekPuff_R"]
    if name.endswith("Left"):
        return [name[:-4] + "_L"]
    if name.endswith("Right"):
        return [name[:-5] + "_R"]
    return [name]


def fit_and_transfer(src: Source, proc_pos: np.ndarray, proc_tris: np.ndarray,
                     proc_shapes: dict, proc_joints: np.ndarray, proc_weights: np.ndarray):
    from scipy.spatial import cKDTree
    pos, scale, R, identity = _fit_identity(src, proc_pos)
    # The statistical identity gives a plausible starting shape. Project the
    # visible skin onto the already fitted subject surface so the imported UV
    # atlas is baked from nearby source texels, then propagate that residual
    # gently into the socket/mouth interior.
    fit = np.flatnonzero(src.fit_mask)
    fit_tri, fit_bary, _ = _closest_map(pos[fit], proc_pos, proc_tris)
    disp = _map_values(proc_pos, fit_tri, fit_bary) - pos[fit]
    length = np.linalg.norm(disp, axis=1, keepdims=True)
    disp *= np.minimum(1., .025 / np.maximum(length, 1e-12))
    from scipy.spatial import cKDTree as Tree
    tree_fit = Tree(pos[fit])
    _, nearest_fit = tree_fit.query(pos, k=min(6, len(fit)))
    if nearest_fit.ndim == 1:
        nearest_fit = nearest_fit[:, None]
    correction = .75 * disp + .25 * disp[nearest_fit[fit]].mean(1)
    pos[fit] += correction
    interior = np.flatnonzero(~src.fit_mask)
    pos[interior] += .35 * disp[nearest_fit[interior]].mean(1)
    ids, bary, error = _closest_map(pos, proc_pos, proc_tris)
    shapes = {name: _map_values(delta, ids, bary).astype(np.float32)
              for name, delta in proc_shapes.items()}
    if src.name == "ict_facekit_light":
        for name, target in list(shapes.items()):
            native = [_name for _name in _ict_name(name) if _name in src.authored_shapes]
            if native and not name.startswith("ml_") and not name.startswith("corr_"):
                d = sum((src.authored_shapes[n] for n in native), np.zeros_like(pos))
                d = (scale * d @ R.T).astype(np.float32)
                size = np.sqrt(np.mean(d * d))
                target_size = np.sqrt(np.mean(target * target))
                if .2 * target_size < size < 4 * target_size:
                    shapes[name] = .5 * d + .5 * target
    else:
        basis = (scale * np.einsum("evc,dc->evd", src.expression_basis, R)).astype(np.float32)
        fit = np.flatnonzero(src.fit_mask)
        sample = fit[::max(1, len(fit) // 850)]
        A = basis[:, sample].reshape(len(basis), -1).T.astype(np.float64)
        names = [k for k in shapes if not k.startswith("ml_") and not k.startswith("corr_")]
        Y = np.stack([shapes[k][sample].reshape(-1) for k in names], 1).astype(np.float64)
        lam = max(float(np.median(np.sum(A * A, axis=0))) * .05, 1e-10)
        coeff = np.linalg.solve(A.T @ A + lam * np.eye(len(basis)), A.T @ Y)
        coeff = np.clip(coeff, -3., 3.)
        projected = np.einsum("ei,evc->ivc", coeff, basis)
        for i, name in enumerate(names):
            target = shapes[name]
            pred = projected[i]
            rms = np.sqrt(np.mean(target[sample] ** 2))
            mismatch = np.sqrt(np.mean((pred[sample] - target[sample]) ** 2))
            if rms > 1e-8 and mismatch < 1.5 * rms:
                shapes[name] = (.5 * pred + .5 * target).astype(np.float32)
    # Interpolate all influences, then keep the four largest per source vertex.
    jcat = proc_joints[ids].reshape(len(pos), -1)
    wcat = (proc_weights[ids] * bary[:, :, None]).reshape(len(pos), -1)
    order = np.argsort(-wcat, axis=1)[:, :4]
    J = np.take_along_axis(jcat, order, axis=1).astype(np.int32)
    W = np.take_along_axis(wcat, order, axis=1).astype(np.float32)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
    stats = dict(src.provenance, identity_coefficients=identity.tolist(),
                 median_surface_distance_mm=round(float(np.median(error)) * 1000, 3),
                 p95_surface_distance_mm=round(float(np.quantile(error, .95)) * 1000, 3),
                 vertices=int(len(pos)), triangles=int(len(src.triangles)))
    return pos.astype(np.float32), shapes, J, W, stats


def remap_contacts(viz: dict | None, proc_pos: np.ndarray, model_pos: np.ndarray,
                   model_tris: np.ndarray) -> dict | None:
    if not viz:
        return None
    from scipy.spatial import cKDTree
    from . import contacts
    _, near_proc = cKDTree(proc_pos).query(model_pos)
    _, near_model = cKDTree(model_pos).query(proc_pos)
    def region(old):
        return np.flatnonzero(np.isin(near_proc, np.asarray(old, np.int32))).tolist()
    out = {k: v for k, v in viz.items() if k not in
           ("eye", "lip_ids", "pairs_upper", "pairs_lower", "verts", "nbr_ptr", "nbr_idx")}
    out["eye"] = [dict(e, ids=region(e["ids"])) for e in viz["eye"]]
    out["lip_ids"] = region(viz["lip_ids"])
    pairs = [(int(near_model[u]), int(near_model[l]))
             for u, l in zip(viz["pairs_upper"], viz["pairs_lower"])]
    pairs = sorted({(u, l) for u, l in pairs if u != l})
    out["pairs_upper"] = [u for u, _ in pairs]
    out["pairs_lower"] = [l for _, l in pairs]
    out.update(contacts.graph(out, model_tris))
    return out


def as_template(src: Source) -> template.Template:
    n = len(src.vertices)
    uv = src.triangle_uvs.reshape(-1, 2).copy()
    return template.Template(np.zeros((n, 2), np.float32), np.zeros(n, np.int8),
                             np.full(n, -1, np.int8), np.zeros(n, np.int8),
                             np.zeros(n, np.int16), src.triangles.astype(np.int32),
                             np.zeros(len(src.triangles), np.int8), uv,
                             np.arange(len(uv), dtype=np.int32).reshape(-1, 3),
                             info=src.provenance.copy())
