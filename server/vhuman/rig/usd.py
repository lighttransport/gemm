"""UsdSkel export (.usda text) of an assembled rig, for LightUSD's vchar
viewer, LightRig and any UsdSkel consumer.

    /Character (SkelRoot, customData.vchar: facial controls -> blendshapes)
        Skeleton (joints, bind/rest transforms, animationSource -> ROM)
            ROM (SkelAnimation: a range-of-motion pass over every control,
                 joints and blendshape weights from the rig evaluator)
        Geom/<part> (Mesh + SkelBindingAPI, BlendShape children)
        Materials/<name> (UsdPreviewSurface, textures in textures/)
Controls that only drive blendshapes are listed in customData so vchar's
facial panel can pose them; joint-driven controls (jaw, gaze, tongue, head)
are exercised by the ROM animation and fully described in rig.json.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import rigdef, skeleton as SK
from .common import quat_from_matrix

FPS = 24
ROM_STEP = 12      # frames per control (0 -> 1 -> 0)


def _f(v):
    v = float(v)
    return "0" if abs(v) < 1e-12 else f"{v:.7g}"      # no subnormals: float parsers reject e.g. 1e-223


def _vec(a, n=3):
    return "[" + ", ".join("(" + ", ".join(_f(x) for x in row[:n]) + ")" for row in np.asarray(a)) + "]"


def _mat(m):
    m = np.asarray(m, np.float64).T            # USD: row vectors, translation in the last row
    return "( " + ", ".join("(" + ", ".join(_f(x) for x in r) + ")" for r in m) + " )"


def _ints(a):
    return "[" + ", ".join(str(int(x)) for x in np.asarray(a).reshape(-1)) + "]"


def _floats(a):
    return "[" + ", ".join(_f(x) for x in np.asarray(a).reshape(-1)) + "]"


def _ident(name: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return out if not out[0].isdigit() else "_" + out


def rom_frames(rig: rigdef.Rig):
    """(time, controls dict) samples: each control ramps 0 -> 1 -> 0
    (signed ones 0 -> 1 -> -1 -> 0)."""
    frames = [(0, {})]
    t = 0
    for c in rig.d["controls"]:
        n = c["name"]
        if c["min"] < 0:
            frames += [(t + 4, {n: 1.0}), (t + 8, {n: -1.0}), (t + ROM_STEP, {})]
        else:
            frames += [(t + ROM_STEP // 2, {n: 1.0}), (t + ROM_STEP, {})]
        t += ROM_STEP
    return frames


def _material_usd(name, spec, tex_dir: Path, rel: str) -> str:
    g = spec["gltf"]
    pbr = g.get("pbrMetallicRoughness", {})
    imgs = spec.get("images", {})
    lines = [f'        def Material "{name}"', "        {",
             f'            token outputs:surface.connect = </Character/Materials/{name}/Surface.outputs:surface>',
             '            def Shader "Surface"', "            {", '                uniform token info:id = "UsdPreviewSurface"']
    tex_nodes = []

    def tex(slot, key, out_type="float3", channel="rgb"):
        info = pbr.get(slot) if slot in pbr else g.get(slot)
        if not info or int(info["index"]) not in imgs:
            return None
        fn = f"{name}_{slot}.png"
        data = imgs[int(info["index"])]
        (tex_dir / fn).write_bytes(data if isinstance(data, (bytes, bytearray)) else b"")
        node = f"Tex_{slot}"
        tex_nodes.append((node, f"@{rel}/{fn}@", key == "normal"))
        return f"</Character/Materials/{name}/{node}.outputs:{channel}>"

    c = tex("baseColorTexture", "diffuse")
    if c:
        lines.append(f"                color3f inputs:diffuseColor.connect = {c}")
        if g.get("alphaMode") == "BLEND":            # e.g. the eyeshell: coverage in the texture's alpha
            lines.append(f"                float inputs:opacity.connect = {c.replace('outputs:rgb', 'outputs:a')}")
    else:
        bc = pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1])
        lines.append(f"                color3f inputs:diffuseColor = ({_f(bc[0])}, {_f(bc[1])}, {_f(bc[2])})")
        if bc[3] < 1:
            lines.append(f"                float inputs:opacity = {_f(bc[3])}")
    r = tex("metallicRoughnessTexture", "rough", channel="g")
    if r:
        lines.append(f"                float inputs:roughness.connect = {r}")
    else:
        lines.append(f"                float inputs:roughness = {_f(pbr.get('roughnessFactor', 0.5))}")
    lines.append(f"                float inputs:metallic = {_f(0.0)}")
    n = tex("normalTexture", "normal")
    if n:
        lines.append(f"                normal3f inputs:normal.connect = {n}")
    if "KHR_materials_transmission" in g.get("extensions", {}):
        lines.append("                float inputs:opacity = 0.15")
    lines += ["                token outputs:surface", "            }"]
    lines += ['            def Shader "Reader"', "            {", '                uniform token info:id = "UsdPrimvarReader_float2"',
              '                string inputs:varname = "st"', "                float2 outputs:result", "            }"]
    for node, path, is_normal in tex_nodes:
        lines += [f'            def Shader "{node}"', "            {", '                uniform token info:id = "UsdUVTexture"',
                  f"                asset inputs:file = {path}",
                  f"                float2 inputs:st.connect = </Character/Materials/{name}/Reader.outputs:result>",
                  f'                token inputs:sourceColorSpace = "{"raw" if is_normal else "auto"}"']
        if is_normal:
            lines += ["                float4 inputs:scale = (2, 2, 2, 1)", "                float4 inputs:bias = (-1, -1, -1, 0)"]
        lines += ['                token inputs:wrapS = "clamp"', '                token inputs:wrapT = "clamp"',
                  "                float3 outputs:rgb", "                float outputs:g", "                float outputs:a",
                  "            }"]
    lines.append("        }")
    return "\n".join(lines)


def write(asset, out: Path, subj=None, name: str = "rig.usda") -> dict:
    out = Path(out)
    tex_dir = out / "textures"
    tex_dir.mkdir(exist_ok=True)
    rig = rigdef.Rig(asset.rig, ml=asset.info.get("ml"))
    skel = asset.skeleton
    by_name = {j["name"]: j for j in skel["joints"]}
    def joint_path(name):
        chain = [name]
        while by_name[chain[-1]]["parent"] is not None:
            chain.append(by_name[chain[-1]]["parent"])
        return "/".join(reversed(chain))
    joints = [joint_path(j["name"]) for j in skel["joints"]]
    # vchar maps a control straight to a same-named blendshape: only shapes driven by their own control
    own = {b["name"] for b in asset.rig["blendshapes"] if b["input"] == b["name"]}
    blend_only = sorted(set(rig.shape_names) & set(rig.controls) & own
                        - {e["input"] for e in asset.rig["joint_matrix"]})
    frames = rom_frames(rig)
    end = frames[-1][0]
    L = ["#usda 1.0", "(", '    defaultPrim = "Character"', "    metersPerUnit = 1", '    upAxis = "Y"',
         f"    startTimeCode = 0", f"    endTimeCode = {end}", f"    timeCodesPerSecond = {FPS}", "    customLayerData = {",
         '        string generator = "vhuman rig (server/vhuman/rig)"', "    }", ")", ""]
    names_arr = "[" + ", ".join(f'"{n}"' for n in blend_only) + "]"
    L += ['def SkelRoot "Character" (', '    prepend apiSchemas = ["SkelBindingAPI"]', "    customData = {",
          "        dictionary vchar = {", f"            string[] controlNames = {names_arr}",
          f"            string[] controlMappings = {names_arr}",
          "            float2[] controlRanges = [" + ", ".join("(0, 1)" for _ in blend_only) + "]",
          "            float[] controlDefaults = [" + ", ".join("0" for _ in blend_only) + "]",
          f'            string rigDefinition = "{asset.info.get("rig_name", "rig.json")}"',
          '            string controlNamespace = "lr.face.v1"',
          "        }", "    }", ")", "{", "    rel skel:skeleton = </Character/Skeleton>"]
    L += ['    def Skeleton "Skeleton" (', '        prepend apiSchemas = ["SkelBindingAPI"]', "    )", "    {",
          "        uniform token[] joints = [" + ", ".join(f'"{p}"' for p in joints) + "]",
          "        uniform matrix4d[] bindTransforms = [" + ", ".join(_mat(j["bind"]) for j in skel["joints"]) + "]"]
    rest = []
    for j in skel["joints"]:
        m = np.eye(4)
        m[:3, :3] = np.asarray(j["rest_rotation"])
        m[:3, :3] *= j.get("rest_scale", 1.0)
        m[:3, 3] = j["rest_translation"]
        rest.append(m)
    L += ["        uniform matrix4d[] restTransforms = [" + ", ".join(_mat(m) for m in rest) + "]",
          "        rel skel:animationSource = </Character/Skeleton/ROM>"]
    # range-of-motion animation
    L += ['        def SkelAnimation "ROM"', "        {",
          "            uniform token[] joints = [" + ", ".join(f'"{p}"' for p in joints) + "]",
          "            uniform token[] blendShapes = [" + ", ".join(f'"{_ident(n)}"' for n in rig.shape_names) + "]"]
    tr, ro, bw = [], [], []
    for time_, ctrl in frames:
        ev = rig.evaluate(ctrl)
        loc = ev["local"]
        tr.append(f"                {time_}: " + _vec(loc[:, :3, 3]))
        qs = [quat_from_matrix(m[:3, :3] / skel["joints"][i].get("rest_scale", 1.0))
              for i, m in enumerate(loc)]
        ro.append(f"                {time_}: [" + ", ".join(f"({_f(q[3])}, {_f(q[0])}, {_f(q[1])}, {_f(q[2])})"
                                                           for q in qs) + "]")
        bw.append(f"                {time_}: " + _floats(ev["weights"]))
    L += ["            float3[] translations.timeSamples = {", ",\n".join(tr), "            }",
          "            quatf[] rotations.timeSamples = {", ",\n".join(ro), "            }",
          "            half3[] scales = [" + ", ".join("(" + ", ".join([_f(j.get("rest_scale", 1.0))] * 3) + ")"
                                                for j in skel["joints"]) + "]",
          "            float[] blendShapeWeights.timeSamples = {", ",\n".join(bw), "            }", "        }", "    }"]
    # geometry
    L += ['    def Scope "Geom"', "    {"]
    stats = {"meshes": 0, "blendshapes": 0}
    for p in asset.parts:
        nm = _ident(p.name)
        W = p.weights / np.maximum(p.weights.sum(1, keepdims=True), 1e-9)
        L += [f'        def Mesh "{nm}" (', '            prepend apiSchemas = ["SkelBindingAPI", "MaterialBindingAPI"]',
              "        )", "        {", '            uniform token subdivisionScheme = "none"',
              '            uniform token orientation = "rightHanded"',
              "            int[] faceVertexCounts = " + _ints(np.full(len(p.tris), 3)),
              "            int[] faceVertexIndices = " + _ints(p.tris),
              "            point3f[] points = " + _vec(p.positions),
              "            normal3f[] normals = " + _vec(p.normals) + ' (', '                interpolation = "vertex"', "            )"]
        if p.uv is not None:
            st = np.asarray(p.uv).copy()
            st[:, 1] = 1.0 - st[:, 1]                   # USD st: v up
            L += ["            texCoord2f[] primvars:st = " + _vec(st, 2) + " (", '                interpolation = "vertex"',
                  "            )"]
        L += ["            int[] primvars:skel:jointIndices = " + _ints(p.joints) + " (", "                elementSize = 4",
              '                interpolation = "vertex"', "            )",
              "            float[] primvars:skel:jointWeights = " + _floats(W) + " (", "                elementSize = 4",
              '                interpolation = "vertex"', "            )",
              "            matrix4d primvars:skel:geomBindTransform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )",
              "            rel skel:skeleton = </Character/Skeleton>",
              f"            rel material:binding = </Character/Materials/{_ident(p.material)}>"]
        if p.shapes:
            bs = list(p.shapes)
            L += ["            uniform token[] skel:blendShapes = [" + ", ".join(f'"{_ident(n)}"' for n in bs) + "]",
                  "            rel skel:blendShapeTargets = [" + ", ".join(f"</Character/Geom/{nm}/{_ident(n)}>"
                                                                     for n in bs) + "]"]
            for n in bs:
                idx, d, _ = p.shapes[n]
                L += [f'            def BlendShape "{_ident(n)}"', "            {",
                      "                uniform vector3f[] offsets = " + _vec(d),
                      "                uniform int[] pointIndices = " + _ints(idx), "            }"]
                stats["blendshapes"] += 1
        L += ["        }"]
        stats["meshes"] += 1
    L += ["    }", '    def Scope "Materials"', "    {"]
    for key, spec in asset.materials.items():
        L.append(_material_usd(_ident(key), spec, tex_dir, "textures"))
    L += ["    }", "}", ""]
    text = "\n".join(L)
    (out / name).write_text(text)
    stats.update({"bytes": len(text), "rom_frames": len(frames), "rom_end": end, "vchar_controls": len(blend_only)})
    return stats


def package(out: Path) -> str:
    """rig.usda + textures/ + rig.json as one zip (the layer's relative paths hold)."""
    import zipfile
    out = Path(out)
    z = out / "rig_usd.zip"
    partial = out / "rig_usd.zip.partial"
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as f:
        f.write(out / "rig.usda", "rig.usda")
        for extra in sorted(out.glob("rig_lod*.usda")):
            f.write(extra, extra.name)
        for extra in ("rig.json", "deformer.lrm", "deformer_basis.safetensors", "deformer.json"):
            if (out / extra).exists():
                f.write(out / extra, extra)
        for t in sorted((out / "textures").glob("*.png")):
            f.write(t, f"textures/{t.name}")
    partial.replace(z)
    return str(z)


def write_track(rig: rigdef.Rig, times, frames, out: Path, fps: float = 30.0, rig_layer: str | None = None) -> dict:
    """An animation layer over rig.usda: the rig evaluated per track frame
    (joints and blendshape weights), e.g. for a LightRig/MediaPipe capture."""
    out = Path(out)
    t0 = float(times[0])
    codes = [round((float(t) - t0) * fps, 4) for t in times]
    joints = [SK.usd_path(j["name"]) for j in rig.joints]
    tr, ro, bw = [], [], []
    for c, ctrl in zip(codes, frames):
        ev = rig.evaluate(ctrl)
        loc = ev["local"]
        tr.append(f"            {c}: " + _vec(loc[:, :3, 3]))
        qs = [quat_from_matrix(m[:3, :3]) for m in loc]
        ro.append(f"            {c}: [" + ", ".join(f"({_f(q[3])}, {_f(q[0])}, {_f(q[1])}, {_f(q[2])})" for q in qs) + "]")
        bw.append(f"            {c}: " + _floats(ev["weights"]))
    hdr = ["#usda 1.0", "(", '    defaultPrim = "Character"', "    metersPerUnit = 1", '    upAxis = "Y"',
           "    startTimeCode = 0", f"    endTimeCode = {codes[-1]}", f"    timeCodesPerSecond = {fps}"]
    if rig_layer:
        import os
        rel = os.path.relpath(Path(rig_layer).resolve(), out.resolve().parent)
        hdr.append(f"    subLayers = [@{rel}@]")
    L = hdr + [")", "",
         'over "Character"', "{", '    over "Skeleton"', "    {",
         "        rel skel:animationSource = </Character/Skeleton/Track>",
         '        def SkelAnimation "Track"', "        {",
         "            uniform token[] joints = [" + ", ".join(f'"{p}"' for p in joints) + "]",
         "            uniform token[] blendShapes = [" + ", ".join(f'"{_ident(n)}"' for n in rig.shape_names) + "]",
         "            float3[] translations.timeSamples = {", ",\n".join(tr), "            }",
         "            quatf[] rotations.timeSamples = {", ",\n".join(ro), "            }",
         "            half3[] scales = [" + ", ".join("(1, 1, 1)" for _ in joints) + "]",
         "            float[] blendShapeWeights.timeSamples = {", ",\n".join(bw), "            }",
         "        }", "    }", "}", ""]
    out.write_text("\n".join(L))
    return {"frames": len(codes), "end": codes[-1], "fps": fps, "out": str(out)}
