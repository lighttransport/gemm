"""Export Cycles lighting/AOV diagnostics from an existing built .blend (local GPU).

blender -b head.blend --python peri_passes_blender.py -- output.exr [native SSS weight]
This is a diagnostic capture, not a beauty post-process.
"""
import json
import hashlib
import sys
from pathlib import Path
import bpy


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    out = Path(argv[0]); out.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if 'cycles' not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module='cycles')
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'HIP'; prefs.get_devices()
    gpus = [d for d in prefs.devices if d.type == 'HIP']
    if not gpus:
        raise RuntimeError('no HIP GPU')
    for d in prefs.devices:
        d.use = d == gpus[0]
    scene.cycles.device = 'GPU'
    scene.cycles.use_auto_tile = True
    scene.cycles.tile_size = 256
    scene.compositing_node_group = None
    view = bpy.context.view_layer
    for name in ('diffuse_direct', 'diffuse_indirect', 'diffuse_color', 'glossy_direct', 'glossy_indirect', 'glossy_color', 'z', 'normal', 'position'):
        setattr(view, 'use_pass_' + name, True)
    view.cycles.denoising_store_passes = True
    mat = bpy.data.materials['skin']
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    skin = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
    if nodes.get('peri_k_lid_rough'):
        nodes['peri_k_lid_rough'].outputs[0].default_value = .8
    if len(argv) > 1:
        socket = skin.inputs['Subsurface Weight']
        for link in list(socket.links):
            links.remove(link)
        socket.default_value = float(argv[1])
    geom = nodes.new('ShaderNodeNewGeometry')
    for name, source, value in (
        ('skin_mask', None, 1.),
        ('skin_basecolor', skin.inputs['Base Color'].links[0].from_socket, None),
        ('skin_geometric_normal', geom.outputs['Normal'], None),
    ):
        aov = view.aovs.add(); aov.name = name; aov.type = 'COLOR'
        output = nodes.new('ShaderNodeOutputAOV'); output.aov_name = name
        if source is None:
            output.inputs['Color'].default_value = (value, value, value, 1.)
        else:
            links.new(source, output.inputs['Color'])
    # Blender 5 exports multilayer EXRs through compositor file outputs.
    group = bpy.data.node_groups.new('peri_pass_capture', 'CompositorNodeTree')
    scene.compositing_node_group = group
    group.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    layers = group.nodes.new('CompositorNodeRLayers')
    output = group.nodes.new('NodeGroupOutput')
    group.links.new(layers.outputs['Image'], output.inputs[0])
    files = group.nodes.new('CompositorNodeOutputFile')
    files.directory = str(out.parent); files.file_name = out.stem
    files.format.file_format = 'OPEN_EXR_MULTILAYER'; files.format.color_depth = '32'
    if not len(files.file_output_items):
        files.file_output_items.new('RGBA', 'Image')
    group.links.new(layers.outputs['Image'], files.inputs[0])
    print('AVAILABLE_PASSES', list(layers.outputs.keys()), flush=True)
    for name, candidates in (
        ('DiffDir', ('DiffDir', 'Diffuse Direct')),
        ('DiffInd', ('DiffInd', 'Diffuse Indirect')),
        ('DiffCol', ('DiffCol', 'Diffuse Color')),
        ('Position', ('Position',)),
        ('skin_mask', ('skin_mask',)),
        ('skin_geometric_normal', ('skin_geometric_normal',)),
    ):
        socket = next((layers.outputs.get(key) for key in candidates if layers.outputs.get(key) is not None), None)
        if socket is None:
            raise RuntimeError(f'missing {name} pass')
        files.file_output_items.new('RGBA', name)
        group.links.new(socket, files.inputs[name])
    print('CAPTURE_LINKS', [(s.name, s.is_linked) for s in files.inputs], flush=True)
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.filepath = str(out.with_name(out.stem + '_beauty.exr'))
    out.unlink(missing_ok=True)
    bpy.ops.render.render(write_still=True)
    if not out.is_file():
        matches = sorted(out.parent.glob(out.stem + '[0-9]*.exr'))
        if len(matches) != 1:
            raise RuntimeError(f'expected one multilayer capture: {matches}')
        matches[0].replace(out)
    camera = scene.camera.data
    record = dict(schema='vhuman.periocular_passes.v1',
                  samples=scene.cycles.samples, seed=scene.cycles.seed, dicing_rate=scene.cycles.dicing_rate,
                  tile_size=scene.cycles.tile_size,
                  device=gpus[0].name, blender_version=bpy.app.version_string, input_blend=bpy.data.filepath,
                  capture_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  resolution=[scene.render.resolution_x, scene.render.resolution_y],
                  lens_mm=camera.lens, sensor_height_mm=camera.sensor_height,
                  camera_origin=list(scene.camera.matrix_world.translation),
                  camera_forward=list(-scene.camera.matrix_world.col[2].xyz),
                  native_sss_weight=skin.inputs['Subsurface Weight'].default_value,
                  native_sss_radius=list(skin.inputs['Subsurface Radius'].default_value),
                  native_sss_scale=skin.inputs['Subsurface Scale'].default_value)
    out.with_suffix('.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
