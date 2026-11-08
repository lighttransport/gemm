"""Blender side of peri_sweep: open one built scene (head.blend) and render many parameter hypotheses without rebuilding.

blender -b <head.blend> --python peri_sweep_blender.py -- <hyps.json> <out dir>
hyps.json: [{"name": ..., "subsurface_scale": s, "subsurface_weight": w, "eye_ao": a, "eye_back_delta": m}, ...]
Only knobs that are plain node inputs / object transforms in the built scene are supported (no relinking)."""
import json
import sys
import time
from pathlib import Path

import bpy


def skin_principled():
    mat = bpy.data.materials.get('skin')
    for n in mat.node_tree.nodes:
        if n.bl_idname == 'ShaderNodeBsdfPrincipled':
            return n
    raise RuntimeError('skin Principled BSDF not found')


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    hyps = json.loads(Path(argv[0]).read_text()); out = Path(argv[1]); out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'HIP'; prefs.get_devices()
    gpus = [d for d in prefs.devices if d.type == 'HIP']
    if not gpus:
        raise RuntimeError('no HIP device (set LD_LIBRARY_PATH to the ROCm libs)')
    for d in prefs.devices:
        d.use = d is gpus[0]
    scene.cycles.device = 'GPU'
    st = scene.render.image_settings
    st.file_format = 'OPEN_EXR'; st.color_depth = '32'; st.color_mode = 'RGBA'
    scene.render.use_persistent_data = True     # keep BVH / images on the GPU between hypotheses
    skin = skin_principled()
    base_samples = scene.cycles.samples
    base = dict(ss=skin.inputs['Subsurface Scale'].default_value, sw=skin.inputs['Subsurface Weight'].default_value)
    eye_objs = [o for o in bpy.data.objects if 'eye2_axis' in o.keys()]
    eye_loc = {o.name: tuple(o.location) for o in eye_objs}
    ao_nodes = [n for m in bpy.data.materials if m.node_tree for n in m.node_tree.nodes if n.name == 'peri_eye_ao']
    # defaults of every named knob node, restored before each hypothesis (no leakage between hypotheses)
    knob_defaults = {}
    for mat in bpy.data.materials:
        if mat.node_tree:
            for n in mat.node_tree.nodes:
                if n.name.startswith('peri_') and n.bl_idname in ('ShaderNodeRGB', 'ShaderNodeValue'):
                    knob_defaults[(mat.name, n.name)] = tuple(n.outputs[0].default_value) if n.bl_idname == 'ShaderNodeRGB' \
                        else float(n.outputs[0].default_value)
    base_method = skin.subsurface_method
    base_sss_ior = skin.inputs['Subsurface IOR'].default_value
    base_radius = tuple(skin.inputs['Subsurface Radius'].default_value)
    base_denoise = scene.cycles.use_denoising
    base_dicing = scene.cycles.dicing_rate
    ao_defaults = [n.inputs['To Min'].default_value for n in ao_nodes]
    known_knobs = {name for _, name in knob_defaults}
    for h in hyps:
        unknown = set(h.get('nodes', {})) - known_knobs
        if unknown:
            raise ValueError(f'unknown sweep knobs: {sorted(unknown)}')
    timings = {}
    for h in hyps:
        t0 = time.time()
        skin.subsurface_method = h.get('subsurface_method', base_method)
        for key, socket_name, default in (('subsurface_scale', 'Subsurface Scale', base['ss']),
                                          ('subsurface_weight', 'Subsurface Weight', base['sw'])):
            socket = skin.inputs[socket_name]
            if socket.is_linked:
                if key in h:
                    raise RuntimeError(f'{socket_name} is linked; use named regional controls')
            else:
                socket.default_value = float(h.get(key, default))
        sss_ior = skin.inputs.get('Subsurface IOR')
        if sss_ior is not None:
            sss_ior.default_value = float(h.get('subsurface_ior', base_sss_ior))
        elif 'subsurface_ior' in h:
            raise ValueError('selected subsurface method does not support a subsurface IOR')
        skin.inputs['Subsurface Radius'].default_value = tuple(float(x) * 1e-3 for x in h['subsurface_radius']) if 'subsurface_radius' in h else base_radius
        scene.cycles.use_denoising = bool(h.get('denoise', base_denoise))
        scene.cycles.dicing_rate = float(h.get('dicing_rate', base_dicing))
        for n, default in zip(ao_nodes, ao_defaults):
            n.inputs['To Min'].default_value = 1. - float(h['eye_ao']) if 'eye_ao' in h else default
        d = float(h.get('eye_back_delta', 0.))
        for o in eye_objs:
            ax = o['eye2_axis']; x, y, z = eye_loc[o.name]
            o.location = (x - ax[0] * d, y - ax[1] * d, z - ax[2] * d)
        for (mname, nname0), dv in knob_defaults.items():
            bpy.data.materials[mname].node_tree.nodes[nname0].outputs[0].default_value = dv
        for nname, value in h.get('nodes', {}).items():      # named knob nodes in any material (Value / RGB)
            for mat in bpy.data.materials:
                n = mat.node_tree.nodes.get(nname) if mat.node_tree else None
                if n is None:
                    continue
                if n.bl_idname == 'ShaderNodeRGB':
                    n.outputs[0].default_value = (*[float(x) for x in value], 1.)
                else:
                    n.outputs[0].default_value = float(value)
        scene.cycles.samples = int(h.get('samples', base_samples))
        if 'denoise' in h:
            scene.cycles.use_denoising = bool(h['denoise'])
        if 'dicing_rate' in h:
            scene.cycles.dicing_rate = float(h['dicing_rate'])
        scene.render.filepath = str(out / f"{h['name']}.exr")
        bpy.ops.render.render(write_still=True)
        timings[h['name']] = time.time() - t0
    (out / 'timings.json').write_text(json.dumps(timings, indent=1))


main()
