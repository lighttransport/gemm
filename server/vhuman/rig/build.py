"""Rig a fitted head: template fit -> skeleton -> weights -> shapes -> exports.

    python -m server.vhuman.rig.build <head folder> [--out DIR] [--res 2048]

Needs numpy, Pillow, scipy and PyTorch (server/vhuman/requirements-rig.txt);
the vhuman server runs it in that interpreter as a job. Outputs (in
<head>/rig/ by default): rig.glb (skinned + morph targets, for the web viewer),
rig.usda + textures/ (UsdSkel for LightUSD's vchar and LightRig), rig.json
(controls, joints and the linear rig for any evaluator), rig_*.png maps,
preview.png and rig_report.json.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..eye.geometry import compute_tangents
from . import (attach, bake, expressions, features, fields, gltf, meshes, mouthparts, register, rigdef, skeleton,
               skinning, template, usd)
from .common import load_subject, normalize, vertex_normals

VERSION = 1


@dataclass
class ExportPart:
    name: str
    material: str
    positions: np.ndarray
    normals: np.ndarray
    uv: np.ndarray | None
    tangents: np.ndarray | None
    tris: np.ndarray
    joints: np.ndarray
    weights: np.ndarray
    shapes: dict = field(default_factory=dict)      # name -> (idx, dpos, dnrm | None)


@dataclass
class RigAsset:
    skeleton: dict
    parts: list
    materials: dict
    rig: dict
    info: dict


def _png(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _limit_png(source: bytes, size: int) -> bytes:
    """Use smaller atlases for the mobile rig without changing its UVs."""
    with Image.open(io.BytesIO(source)) as image:
        if max(image.size) <= size:
            return source
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, format="PNG", optimize=True)
        return out.getvalue()


def rig_definition(skel: dict, shape_names: list[str]) -> dict:
    controls = rigdef.control_table()
    cnames = {c["name"] for c in controls}
    corr = [{"name": n, "inputs": list(i), "weight": w} for n, i, w in rigdef.CORRECTIVES]
    inputs = cnames | {c["name"] for c in corr}
    blend = [{"name": n, "input": rigdef.SHAPE_INPUT.get(n, n)} for n in shape_names if n in inputs]
    jm = [{"input": i, "joint": j, "attr": a, "value": v} for i, j, a, v in rigdef.joint_matrix_entries(skel["scale"])]
    return {"format": "vhuman-rig", "version": VERSION, "namespace": "lr.face.v1",
            "controls": controls, "correctives": corr, "joints": skel["joints"], "joint_matrix": jm,
            "blendshapes": blend,
            "evaluation": "x=clamp(controls); c_p=min(1,w_p*prod clamp(x_i,0,1)); in=[x|c]; dJ=M in (per joint tx ty "
                          "tz rx ry rz, radians, local); local=T(rest_t+dt) R(rest_R Rz Ry Rx); b_k=in[src_k]; "
                          "v=rest+sum b_k d_k; LBS with world_j inv(bind_j)"}


def _shape_normals(pos, tris, shapes: dict, vmap: np.ndarray):
    """Per shape, normal deltas of a part (welded positions, part triangles in welded ids)."""
    n0 = vertex_normals(pos, tris)
    out = {}
    for name, d in shapes.items():
        n1 = vertex_normals(pos + d, tris)
        out[name] = (n1 - n0)[vmap]
    return out


def train_deformer(tmpl, pos, shapes, Jn, W, skel, feat, out, samples, log=print, progress=None):
    """The ML corrective deformer (mldeformer.py) on the welded template."""
    import torch
    from . import mldeformer, torchrig
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rig0 = rig_definition(skel, sorted(shapes))
    tr = torchrig.TorchRig(rig0, pos, shapes, Jn, W, device=dev)
    names = [j["name"] for j in skel["joints"]]
    jidx = {n: i for i, n in enumerate(names)}
    teeth = [(mouthparts.teeth(skel, up, skel["scale"], jidx)[0], "teeth_upper" if up else "teeth_lower")
             for up in (True, False)]
    tongue = mouthparts.tongue(skel, skel["scale"], jidx)
    contacts = mldeformer.Contacts(tmpl, feat, skel, teeth, dev, pos, tongue=tongue)
    stats = mldeformer.train(tr, tmpl, contacts, out, samples=samples, log=log, progress=progress)
    return mldeformer.MLDeformer(out), stats, contacts.export()


STAGES = {"features": .04, "register": .08, "fit_cache": .2, "skeleton_weights": .22, "shapes": .24,
          "deformer": .25, "bake": .5, "assemble": .72, "gltf": .76, "usd": .8, "preview": .86}


def head_parts(parts_t, pos, shapes, Jn, W) -> list[ExportPart]:
    """Skin and mouth-interior export parts of a (LOD) template with its shapes."""
    out = []
    for p in parts_t:
        if not len(p.tris):
            continue
        tris_w = p.vmap[p.tris]
        dn = _shape_normals(pos, tris_w, {k: v for k, v in shapes.items() if not k.startswith("ml_")}, p.vmap)
        sh = {}
        for name, d in shapes.items():
            dv = d[p.vmap]
            idx = np.flatnonzero(np.linalg.norm(dv, axis=1) > 2e-6)
            if len(idx):
                # ML components are sub-millimetre: their normal deltas are not worth the bytes
                sh[name] = (idx, dv[idx], None if name.startswith("ml_") else dn[name][idx])
        out.append(ExportPart(f"head_{p.name}", p.material, pos[p.vmap], p.normals, p.uv, p.tangents, p.tris,
                              Jn[p.vmap], W[p.vmap], sh))
    return out


def carried_materials(carried, subj) -> dict:
    materials = {}
    for c in carried:
        imgs = {}

        def grab(o):
            if isinstance(o, dict):
                for k2, v in o.items():
                    if k2.endswith("Texture") and isinstance(v, dict) and "index" in v:
                        src = subj.glb.doc["textures"][v["index"]]["source"]
                        view = subj.glb.doc["bufferViews"][subj.glb.doc["images"][src]["bufferView"]]
                        st = view.get("byteOffset", 0)
                        imgs[v["index"]] = bytes(subj.glb.bin[st:st + view["byteLength"]])
                    else:
                        grab(v)
            elif isinstance(o, list):
                for x in o:
                    grab(x)
        grab(c.material)
        materials[f"carried_{c.name}"] = {"gltf": c.material, "images": imgs}
    return materials


def carried_parts(carried, pos, skin_tris, Jn, W, shapes, jidx) -> list[ExportPart]:
    """Eyeballs rigid on the eye joints; eye-edge meshes wrap-bound to the skin."""
    out = []
    for c in carried:
        n = len(c.positions)
        sh = {}
        if c.joint:
            J = np.zeros((n, 4), np.int64)
            J[:, 0] = jidx[c.joint]
            Wc = np.zeros((n, 4))
            Wc[:, 0] = 1
        else:
            tb, bb, dist = attach.bind_wrap(c.positions, pos, skin_tris)
            J, Wc, dd = attach.wrap_data((tb, bb), skin_tris, Jn, W, shapes)
            for name, d in dd.items():
                idx = np.flatnonzero(np.linalg.norm(d, axis=1) > 2e-6)
                if len(idx):
                    sh[name] = (idx, d[idx], None)
        tan = compute_tangents(c.positions, c.normals, c.uv, c.tris).astype(np.float64) \
            if "normalTexture" in c.material else None
        out.append(ExportPart(c.name, f"carried_{c.name}", c.positions, c.normals, c.uv, tan, c.tris, J, Wc, sh))
    return out


def export_lods(levels, tmpl, pos, shapes, Jn, W, contacts_viz, mouth_parts, carried, jidx, asset, out, subj,
                cache_dir, log=print) -> dict:
    """LOD k assets from LOD0's rig data (lod.py): rig_lod<k>.glb, rig_lod<k>.usda,
    rig_deformer_lod<k>.safetensors, viz_lod<k>.json (rig.json is shared)."""
    from . import lod, native
    from .contacts import graph
    report = {}
    for lv in levels:
        t0 = time.perf_counter()
        tl = lod.get(lv, cache_dir)
        idx, w = lod.mapping(tmpl, tl, lv)
        tl = lod.snap_uvs(tmpl, tl, idx, w)
        pos_l = lod.apply(idx, w, pos)
        W_l = lod.apply(idx, w, W)
        W_l /= np.maximum(W_l.sum(1, keepdims=True), 1e-12)
        Jn_l = Jn[idx[:, 0]]
        shapes_l = {k: lod.apply(idx, w, d) for k, d in shapes.items()}
        parts_t = meshes.unweld(tl, pos_l)
        parts = head_parts(parts_t, pos_l, shapes_l, Jn_l, W_l) + list(mouth_parts)
        parts += carried_parts(carried, pos_l, tl.tris[tl.tri_mat == 0], Jn_l, W_l, shapes_l, jidx)
        materials = asset.materials
        mobile_modes = None
        if lv == 2:
            # The carried eye shells have many tiny ML targets. Keep their
            # highest-energy eight modes and all principal eye poses; the
            # facial skin retains the complete ML corrective basis.
            energy = {}
            for part in parts:
                if not part.name.startswith("eye_"):
                    continue
                for name, (_, delta, _) in part.shapes.items():
                    if name.startswith("ml_") and name != "ml_mean":
                        energy[name] = energy.get(name, 0.) + float(np.sum(delta * delta))
            mobile_modes = set(sorted(energy, key=energy.get, reverse=True)[:8])
            for part in parts:
                if part.name.startswith("eye_"):
                    part.shapes = {name: value for name, value in part.shapes.items()
                                   if not name.startswith("ml_") or name == "ml_mean" or name in mobile_modes}
            materials = {key: dict(spec, images={index: _limit_png(png, 1024 if key == "skin" else 512)
                                                  for index, png in spec.get("images", {}).items()})
                         for key, spec in asset.materials.items()}
        a = RigAsset(asset.skeleton, parts, materials, asset.rig, asset.info)
        cv = lod.contacts_viz(contacts_viz, idx, w, tl.tris) if contacts_viz else None
        viz = {"parts": {f"head_{p.name}": {"vmap": p.vmap.tolist()} for p in parts_t}, "contacts": cv,
               "welded_vertices": int(len(pos_l)), "lod": lv}
        (out / f"viz_lod{lv}.json").write_text(json.dumps(viz, separators=(",", ":"),
                                                          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        pk = native.write_package(out / f"rig_deformer_lod{lv}.safetensors", asset.rig, pos_l, shapes_l, Jn_l, W_l,
                                  asset.info.get("ml"), contacts_viz=cv)
        g = gltf.write(a, out / f"rig_lod{lv}.glb")
        u = usd.write(a, out, subj, name=f"rig_lod{lv}.usda")
        report[lv] = {"vertices": int(tl.n), "triangles": int(len(tl.tris)), "glb_bytes": g["bytes"],
                      "usd_bytes": u["bytes"], "package_bytes": pk["bytes"],
                      "seconds": round(time.perf_counter() - t0, 1)}
        if mobile_modes is not None:
            report[lv]["carried_ml_modes"] = sorted(mobile_modes)
        if log:
            log(f"LOD{lv}: {report[lv]}")
    del graph
    return report


def assemble(folder, out_dir=None, res: int = 2048, iters: int = 600, log=print, cache_dir=None,
             reuse_fit: bool = False, preview: bool = True, keep_asset: bool = False, progress=None,
             deformer_samples: int = 4096, lods=(1, 2), face_model: str = "gnm_v3", reconstruction=None) -> dict:
    from . import face_models
    if face_model not in face_models.SOURCES:
        raise ValueError(f"face_model must be one of {', '.join(face_models.SOURCES)}")
    t0 = time.perf_counter()
    folder = Path(folder)
    out = Path(out_dir) if out_dir else folder / "rig"
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    def lap(key, t):
        timings[key] = round(time.perf_counter() - t, 2)
        if progress:
            nxt = {"features": "register", "fit_cache": "skeleton_weights", "register": "skeleton_weights",
                   "skeleton_weights": "shapes", "shapes": "deformer" if deformer_samples else "bake",
                   "deformer": "bake", "bake": "assemble", "assemble": "gltf",
                   "gltf": "usd", "usd": "preview", "preview": None}.get(key)
            if nxt:
                progress(STAGES[nxt], {"register": "fitting the template (PyTorch)", "skeleton_weights": "skeleton and weights",
                                       "shapes": "expression shapes", "deformer": "ML deformer: ground truth",
                                       "bake": "transferring the skin maps",
                                       "assemble": "assembling meshes", "gltf": "writing rig.glb",
                                       "usd": "writing rig.usda", "preview": "rendering the pose sheet"}[nxt])
        return time.perf_counter()

    t = time.perf_counter()
    subj = load_subject(folder)
    fitted_source = None
    if reconstruction is not None:
        from ..reconstruction.pipeline import subject_override
        from ..reconstruction.fitting import topology_hash
        reconstruction = Path(reconstruction)
        manifest = json.loads((reconstruction / "manifest.json").read_text())
        from ..reconstruction.observations import sha256
        if manifest.get("source_head_sha256") != sha256(folder / "head_eyes.glb"):
            raise ValueError("candidate source head changed")
        if manifest.get("geometry_sha256") != sha256(reconstruction / "geometry.npz"):
            raise ValueError("candidate geometry hash mismatch")
        if manifest["face_model"] != face_model:
            raise ValueError("candidate face model mismatch")
        src = face_models.load(face_model)
        expected_source = manifest.get("source_loaded_sha256")
        if expected_source is not None and expected_source != face_models.fingerprint(src):
            raise ValueError("candidate source model weights changed")
        with np.load(reconstruction / "geometry.npz", allow_pickle=False) as z:
            if topology_hash(z["triangles"]) != topology_hash(src.triangles):
                raise ValueError("candidate topology mismatch")
            fitted_source = (z["neutral"], float(z["scale"]), z["rotation"])
        subj = subject_override(subj, reconstruction)
    tmpl = template.get(cache_dir)
    cache = out / "fit_cache.pkl"
    cached = None
    if reuse_fit and cache.exists():
        import pickle
        cached = pickle.loads(cache.read_bytes())
        if len(cached[1].get("positions", ())) != tmpl.n:
            cached = None
    if cached is not None:
        feat, fit = cached
        t = lap("fit_cache", t)
    else:
        feat = features.extract(subj)
        t = lap("features", t)
        fit = register.fit(tmpl, subj, feat, iters=iters, log=log)
        import pickle
        cache.write_bytes(pickle.dumps((feat, fit)))
    pos = fit["positions"]
    t = lap("register", t)
    skel = skeleton.place(feat)
    names = [j["name"] for j in skel["joints"]]
    jidx = {n: i for i, n in enumerate(names)}
    F = fields.compute(tmpl, pos, feat, skel["scale"])
    Jn, W, skin_info = skinning.weights(tmpl, pos, F, skel)
    t = lap("skeleton_weights", t)
    shapes = expressions.build(F, skel, (Jn, W))
    expr_dir = out / "expressions"
    expr_report = None
    if (expr_dir / "manifest.json").exists():             # the subject's own expressions (exprdata.py)
        from . import exprdata
        shapes, expr_report = exprdata.fit(tmpl, pos, subj, feat, shapes, rig_definition(skel, sorted(shapes)),
                                           skel, Jn, W, expr_dir, log=log)
    t = lap("shapes", t)
    ml, ml_stats, contacts_viz = None, None, None
    if deformer_samples:
        ml, ml_stats, contacts_viz = train_deformer(tmpl, pos, shapes, Jn, W, skel, feat, out, deformer_samples, log=log,
                                      progress=(lambda f, m: progress(STAGES["deformer"] + 0.2 * f, m))
                                      if progress else None)
        shapes = dict(shapes, **ml.target_deltas())
        t = lap("deformer", t)
    if reconstruction is not None and contacts_viz is None:
        from . import mldeformer
        teeth = [(mouthparts.teeth(skel, up, skel["scale"], jidx)[0], "teeth_upper" if up else "teeth_lower")
                 for up in (True, False)]
        tongue = mouthparts.tongue(skel, skel["scale"], jidx)
        contacts_viz = mldeformer.Contacts(tmpl, feat, skel, teeth, "cpu", pos, tongue=tongue).export()
    proc_pos, proc_shapes, proc_Jn, proc_W = pos, shapes, Jn, W
    proc_contacts = contacts_viz
    source_stats = None
    source = None
    if face_model != "procedural":
        source = face_models.load(face_model)
        pos, shapes, Jn, W, source_stats = face_models.fit_and_transfer(
            source, proc_pos, tmpl.tris[tmpl.tri_mat == 0], proc_shapes, proc_Jn, proc_W, fitted=fitted_source)
        contacts_viz = face_models.remap_contacts(proc_contacts, proc_pos, pos, source.triangles)
        active_tmpl = face_models.as_template(source)
    else:
        active_tmpl = tmpl
    parts_t = meshes.unweld(active_tmpl, pos)
    lining_rgb = tuple(subj.fit.get("fit", {}).get("lining", {}).get("srgb", (150, 100, 85)))
    if reconstruction is not None:
        from ..reconstruction.export import bake_model_atlas
        baked = bake_model_atlas(reconstruction, out, res)
    else:
        baked = bake.bake(active_tmpl, parts_t[0], pos, subj, out, res=res, lining_srgb=lining_rgb, log=log)
    wrinkle_maps = None
    if face_model == "procedural" and (expr_dir / "manifest.json").exists():
        from . import wrinkles
        brow_y = float(np.mean([b[:, 1].mean() for b in feat.brows]))
        wrinkle_maps = wrinkles.bake(tmpl, parts_t[0], pos, subj, expr_dir, out, res=min(res, 2048), log=log,
                                     brow_y=brow_y)
    t = lap("bake", t)
    # ---- parts ---------------------------------------------------------------------------
    parts: list[ExportPart] = head_parts(parts_t, pos, shapes, Jn, W)
    if source is not None:
        # Imported sources provide exterior skin only; preserve the fitted
        # procedural mouth bag as a separate interior material surface.
        proc_mouth = meshes.unweld(tmpl, proc_pos)[1]
        parts += head_parts([proc_mouth], proc_pos, proc_shapes, proc_Jn, proc_W)
    mouth_parts = []
    for pm in (*mouthparts.teeth(skel, True, skel["scale"], jidx), *mouthparts.teeth(skel, False, skel["scale"], jidx),
               mouthparts.tongue(skel, skel["scale"], jidx)):
        nrm = vertex_normals(pm.positions, pm.tris)
        uv = np.zeros((len(pm.positions), 2))
        mouth_parts.append(ExportPart(pm.name, pm.material, pm.positions, nrm, uv, None, pm.tris, pm.joints,
                                      pm.weights))
    parts += mouth_parts
    # carried eye meshes
    carried = attach.collect(subj.glb, subj.frame)
    materials = carried_materials(carried, subj)
    parts += carried_parts(carried, pos, active_tmpl.tris[active_tmpl.tri_mat == 0], Jn, W, shapes, jidx)
    # ---- materials ------------------------------------------------------------------------
    rd = {k: Path(v).read_bytes() for k, v in baked["files"].items()}
    materials["skin"] = {"gltf": {"name": "skin", "pbrMetallicRoughness": {
        "baseColorTexture": {"index": 0}, "metallicRoughnessTexture": {"index": 1}, "metallicFactor": 1.0,
        "roughnessFactor": 1.0}, "normalTexture": {"index": 2}},
        "images": {0: rd["rig_basecolor.png"], 1: rd["rig_orm.png"], 2: rd["rig_normal.png"]}}
    if reconstruction is not None:
        from ..reconstruction.export import apply_material
        apply_material(materials["skin"], baked, subj, out)
    mt = bake.mouth_texture()
    Image.fromarray(mt).save(out / "rig_mouth.png")
    materials["mouth"] = {"gltf": {"name": "mouth", "doubleSided": True, "pbrMetallicRoughness": {
        "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.35}},
        "images": {0: _png(mt)}}
    materials["teeth"] = {"gltf": {"name": "teeth", "pbrMetallicRoughness": {
        "baseColorFactor": [0.93, 0.89, 0.8, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.28}}}
    materials["gums"] = {"gltf": {"name": "gums", "pbrMetallicRoughness": {
        "baseColorFactor": [0.72, 0.33, 0.33, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.4}}}
    materials["tongue"] = {"gltf": {"name": "tongue", "pbrMetallicRoughness": {
        "baseColorFactor": [0.68, 0.28, 0.29, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.45}}}
    shape_names = sorted({n for p in parts for n in p.shapes})
    rig = rig_definition(skel, shape_names)
    rig["face_model"] = face_model
    if source_stats:
        rig["face_model_source"] = {k: v for k, v in source_stats.items() if k != "identity_coefficients"}
    if wrinkle_maps:
        rig["wrinkles"] = wrinkle_maps
    if ml is not None:
        rig["ml_deformer"] = {"model": "deformer.lrm", "basis": "deformer_basis.safetensors",
                              "targets": ml.target_names(), "inputs": ml.inputs,
                              "evaluation": "c = MLP2((x - input.mean) * input.scale) * output.scale + output.mean;"
                                            " weight(ml_mean) = 1, weight(ml_k) = c_k / output.scale_k"}
    asset = RigAsset(skel, parts, materials, rig, {"ml": ml})
    t = lap("assemble", t)
    if keep_asset:
        import pickle
        (out / "asset.pkl").write_bytes(pickle.dumps(asset))
    (out / "rig.json").write_text(json.dumps(rig, indent=1))
    if contacts_viz:
        from . import contacts as contacts_mod
        contacts_viz = dict(contacts_viz, **contacts_mod.graph(contacts_viz, active_tmpl.tris))
    viz_parts = {f"head_{p.name}": {"vmap": p.vmap.tolist()} for p in parts_t if len(p.tris)}
    from . import native
    native_pos, native_shapes, native_joints, native_weights = pos, shapes, Jn, W
    if source is not None:
        native_pos, native_shapes, native_joints, native_weights, mouth_map = native.append_surface(
            pos, shapes, Jn, W, proc_mouth.vmap, proc_pos, proc_shapes, proc_Jn, proc_W)
        viz_parts["head_mouth"] = {"vmap": mouth_map.tolist()}
    viz = {"parts": viz_parts, "contacts": contacts_viz,
           "welded_vertices": int(len(native_pos))}
    (out / "viz.json").write_text(json.dumps(viz, separators=(",", ":"),
                                             default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    native_stats = native.write_package(out / "rig_deformer.safetensors", rig, native_pos, native_shapes, native_joints, native_weights, ml,
                                        contacts_viz=contacts_viz)
    glb_stats = gltf.write(asset, out / "rig.glb")
    t = lap("gltf", t)
    usd_stats = usd.write(asset, out, subj)
    if lods and source is not None and reconstruction is not None:
        from . import source_lod
        lod_report = source_lod.export(lods, source, pos, shapes, Jn, W, contacts_viz, mouth_parts, carried, jidx,
                                       asset, out, subj, log)
    elif lods and source is not None:
        # LODs retain the established ring-safe low-resolution mesh. Re-bake
        # its own atlas because the imported model uses a different UV layout.
        lod_bake_dir = out / "lod_atlas"
        lod_bake_dir.mkdir(parents=True, exist_ok=True)
        proc_parts_t = meshes.unweld(tmpl, proc_pos)
        lod_baked = bake.bake(tmpl, proc_parts_t[0], proc_pos, subj, lod_bake_dir,
                              res=res, lining_srgb=lining_rgb, log=log)
        lod_images = {k: Path(v).read_bytes() for k, v in lod_baked["files"].items()}
        lod_materials = dict(materials)
        lod_materials["skin"] = dict(materials["skin"], images={
            0: lod_images["rig_basecolor.png"], 1: lod_images["rig_orm.png"],
            2: lod_images["rig_normal.png"]})
        lod_asset = RigAsset(skel, parts, lod_materials, rig, {"ml": ml})
        lod_report = export_lods(lods, tmpl, proc_pos, proc_shapes, proc_Jn, proc_W, proc_contacts,
                                 mouth_parts, carried, jidx, lod_asset, out, subj, cache_dir, log=log)
        for row in lod_report.values():
            row["topology"] = "procedural_lod"
    else:
        lod_report = export_lods(lods, tmpl, pos, shapes, Jn, W, contacts_viz, mouth_parts, carried, jidx, asset,
                                 out, subj, cache_dir, log=log) if lods else {}
    usd_stats["zip"] = usd.package(out)
    t = lap("usd", t)
    prev = preview_sheet(asset, out / "preview.png") if preview else []
    t = lap("preview", t)
    report = {"version": VERSION, "head": folder.name, "face_model": face_model,
              "face_model_fit": source_stats, "template": active_tmpl.info, "features": feat.info,
              "register": fit["stats"], "skin": skin_info, "bake": baked["stats"], "gltf": glb_stats,
              "usd": usd_stats, "shapes": len(rig["blendshapes"]), "controls": len(rig["controls"]),
              "deformer": ml_stats, "native_package": native_stats, "expressions": expr_report,
              "wrinkles": [m["name"] for m in (wrinkle_maps or {}).get("maps", [])], "lods": lod_report,
              "joints": names, "timings": timings, "seconds": round(time.perf_counter() - t0, 2),
              "preview": prev}
    (out / "rig_report.json").write_text(json.dumps(report, indent=1, default=float))
    (out / "features.json").write_text(json.dumps(feat.as_dict()))
    np.savez_compressed(out / "fit.npz", positions=pos,
                        init=fit["init"] if source is None else pos,
                        chart=fit["chart"] if source is None else np.zeros((len(pos), 2)))
    return report


POSES = [("neutral", {}), ("smile", {"mouthSmileLeft": 1, "mouthSmileRight": 1, "cheekSquintLeft": .4,
                                     "cheekSquintRight": .4}),
         ("jaw open", {"jawOpen": 1}), ("blink", {"eyeBlinkLeft": 1, "eyeBlinkRight": 1}),
         ("pucker", {"mouthPucker": 1}), ("brows up", {"browInnerUp": 1, "browOuterUpLeft": 1, "browOuterUpRight": 1,
                                                       "eyeWideLeft": .6, "eyeWideRight": .6}),
         ("frown", {"mouthFrownLeft": 1, "mouthFrownRight": 1, "browDownLeft": 1, "browDownRight": 1}),
         ("tongue out", {"jawOpen": .7, "tongueOut": 1})]


def pose_parts(asset: RigAsset, controls: dict) -> list[tuple]:
    R = rigdef.Rig(asset.rig, ml=asset.info.get("ml"))
    ev = R.evaluate(controls)
    wts = dict(zip(R.shape_names, ev["weights"]))
    out = []
    for p in asset.parts:
        shapes = [(idx, d) for n, (idx, d, _) in p.shapes.items()]
        sw = np.array([wts.get(n, 0.0) for n in p.shapes])
        P = rigdef.deform(p.positions, p.joints, p.weights / p.weights.sum(1, keepdims=True), ev["skin"], shapes, sw)
        out.append((p, P))
    return out


def _eye_proxies(asset: RigAsset) -> list[dict]:
    """Low-poly stand-ins for the preview: a sclera sphere and an iris disk
    per eye (the exported eyes are dense and refractive)."""
    from .common import rotation
    out = []
    for p in asset.parts:
        if not p.name.endswith("_iris"):
            continue
        j = int(p.joints[0, 0])
        c = p.positions.mean(0)
        bind = np.asarray(asset.skeleton["joints"][j]["bind"])
        centre, gaze = bind[:3, 3], bind[:3, 2]
        r_iris = float(np.percentile(np.linalg.norm((p.positions - c) - np.outer((p.positions - c) @ gaze, gaze),
                                                    axis=1), 98))
        R = float(np.linalg.norm(c - centre)) + 0.0025
        la, lo = np.meshgrid(np.linspace(0, np.pi, 14), np.linspace(0, 2 * np.pi, 24, endpoint=False), indexing="ij")
        sph = np.stack([np.sin(la) * np.cos(lo), np.cos(la), np.sin(la) * np.sin(lo)], -1).reshape(-1, 3)
        tris = []
        for i in range(13):
            for k in range(24):
                a, b = i * 24 + k, i * 24 + (k + 1) % 24
                tris += [(a, b, a + 24), (b, b + 24, a + 24)]
        sph_t = np.asarray(tris)[:, [0, 2, 1]]
        ex = normalize(np.cross([0, 1, 0], gaze))
        ey = np.cross(gaze, ex)
        rr, aa = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 2 * np.pi, 32, endpoint=False), indexing="ij")
        disk = c + 0.0006 * gaze + (rr * np.cos(aa)).reshape(-1, 1) * ex * r_iris + (rr * np.sin(aa)).reshape(-1, 1) * ey * r_iris
        dt = []
        for i in range(5):
            for k in range(32):
                a, b = i * 32 + k, i * 32 + (k + 1) % 32
                dt += [(a, a + 32, b), (b, a + 32, b + 32)]
        col = np.where(rr.reshape(-1, 1) < 0.35, 0.05, 0.3) * np.ones((1, 3))
        dt = np.asarray(dt)
        dt = np.concatenate([dt, dt[:, [0, 2, 1]]])        # either winding faces the viewer
        out.append({"joint": j, "meshes": [dict(pos=centre + sph * (R - 0.0025) * 0.995, tris=sph_t, color=(0.9, 0.9, 0.88)),
                                            dict(pos=disk, tris=dt, colors=col)]})
    return out


def preview_sheet(asset: RigAsset, path: Path, size: int = 300) -> list[str]:
    """Front views of a few poses (textured skin, flat-coloured mouth parts)."""
    from . import preview
    skin_tex = np.asarray(Image.open(io.BytesIO(asset.materials["skin"]["images"][0])))
    flat = {"teeth": (0.93, 0.9, 0.82), "gums": (0.75, 0.35, 0.35), "tongue": (0.7, 0.3, 0.3),
            "mouth": (0.42, 0.12, 0.12)}
    proxies = _eye_proxies(asset)
    R = rigdef.Rig(asset.rig, ml=asset.info.get("ml"))
    tiles = []
    for label, ctrl in POSES:
        scene = []
        for p, P in pose_parts(asset, ctrl):
            if p.name.startswith("eye_"):
                continue
            if p.material == "skin":
                scene.append(dict(pos=P, tris=p.tris, uv=p.uv, tri_uv=p.tris, texture=skin_tex))
            else:
                scene.append(dict(pos=P, tris=p.tris, colors=np.tile(flat.get(p.material, (0.92, 0.92, 0.9)),
                                                                      (len(P), 1))))
        M = R.evaluate(ctrl)["skin"]
        for e in proxies:
            m = M[e["joint"]]
            for mesh in e["meshes"]:
                P = mesh["pos"] @ m[:3, :3].T + m[:3, 3]
                cols = mesh.get("colors")
                if cols is None:
                    cols = np.tile(mesh["color"], (len(P), 1))
                scene.append(dict(pos=P, tris=mesh["tris"], colors=cols))
        tiles.append(preview.render_scene(scene, size, center=[0, -0.035, 0.02], extent=0.1))
    rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)]
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    return [label for label, _ in POSES]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("head", help="a fitted head folder (portrait.png, head_eyes.glb, fit.json, skin maps)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reconstruction", help="completed reconstruction candidate directory")
    ap.add_argument("--res", type=int, default=2048, help="skin atlas resolution")
    ap.add_argument("--iters", type=int, default=600, help="registration iterations")
    ap.add_argument("--cache", default="tmp/vhuman-rig/cache", help="template cache directory")
    ap.add_argument("--reuse-fit", action="store_true", help="reuse the features and registration of a previous run")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--keep-asset", action="store_true", help="also pickle the assembled asset (debugging)")
    ap.add_argument("--progress", action="store_true", help="print '@progress <fraction> <message>' lines")
    ap.add_argument("--lods", default="1,2", help="LOD levels to export besides LOD0 (comma list, '' for none)")
    ap.add_argument("--deformer-samples", type=int, default=4096,
                    help="ground-truth samples for the ML corrective deformer (0: no deformer)")
    ap.add_argument("--face-model", choices=("gnm_v3", "ict_facekit_light", "procedural"), default="gnm_v3")
    a = ap.parse_args(argv)
    prog = None
    if a.progress:
        def prog(f, m):
            print(f"@progress {f:.3f} {m}", flush=True)
        prog(0.01, "facial features")
    rep = assemble(a.head, a.out, a.res, a.iters, cache_dir=a.cache, reuse_fit=a.reuse_fit,
                   preview=not a.no_preview, keep_asset=a.keep_asset, progress=prog,
                   deformer_samples=a.deformer_samples,
                   face_model=a.face_model, reconstruction=a.reconstruction,
                   lods=tuple(int(x) for x in a.lods.split(",") if x.strip()),
                   log=(lambda m: print(m, flush=True)))
    print(json.dumps({k: rep[k] for k in ("seconds", "timings", "shapes", "controls", "register", "bake")},
                     default=float))


if __name__ == "__main__":
    main()
