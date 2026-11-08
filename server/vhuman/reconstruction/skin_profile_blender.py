"""Blender worker for the optional native-preserving skin-lighting profile.

Load the native head.blend, then invoke with -- request.json. Supports the
explicit backend/device of offline_render; no silent CPU fallback.
"""
import json
import hashlib
import time
import sys
from pathlib import Path
import bpy


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    started=time.monotonic()
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    request=json.loads(Path(argv[0]).read_text())
    out=Path(request['out'])/'skin_profile_capture.exr';out.parent.mkdir(parents=True,exist_ok=True)
    scene = bpy.context.scene
    if 'cycles' not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module='cycles')
    scene.render.engine = 'CYCLES'
    backend=request['device'].upper()
    device_name='CPU'
    if backend=='CPU':
        scene.cycles.device='CPU'
    else:
        prefs=bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type=backend;prefs.get_devices()
        devices=[d for d in prefs.devices if d.type==backend]
        index=int(request.get('gpu_index',0))
        if not 0<=index<len(devices):raise RuntimeError(f'no requested {backend} device {index}')
        for device in prefs.devices:device.use=device==devices[index]
        scene.cycles.device='GPU';device_name=devices[index].name
    scene.cycles.use_auto_tile = True
    scene.cycles.tile_size = 256
    scene.compositing_node_group = None
    view = bpy.context.view_layer
    for name in ('diffuse_direct', 'diffuse_indirect', 'diffuse_color', 'position'):
        setattr(view, 'use_pass_' + name, True)
    view.cycles.denoising_store_passes = True
    mat = bpy.data.materials['skin']
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    skin = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
    skin_obj=bpy.data.objects.get('skin')
    if skin_obj is None or 'skin_profile_weight' not in skin_obj.data.attributes:
        raise RuntimeError('skin profile requires staged anatomical weights; re-stage the candidate')
    socket=skin.inputs['Subsurface Weight']
    for link in list(socket.links):links.remove(link)
    socket.default_value=0.
    profile=nodes.new('ShaderNodeAttribute');profile.attribute_type='GEOMETRY';profile.attribute_name='skin_profile_weight'
    geom = nodes.new('ShaderNodeNewGeometry')
    for name, source, value in (
        ('skin_mask', None, 1.),
        ('skin_geometric_normal', geom.outputs['Normal'], None),
        ('skin_profile', profile.outputs['Fac'], None),
    ):
        aov = view.aovs.add(); aov.name = name; aov.type = 'COLOR'
        output = nodes.new('ShaderNodeOutputAOV'); output.aov_name = name
        if source is None:
            output.inputs['Color'].default_value = (value, value, value, 1.)
        else:
            links.new(source, output.inputs['Color'])
    eye_materials=set()
    for obj in bpy.data.objects:
        if obj.name.startswith('eye2_') or obj.name in {f'{side}_{part}' for side in ('left','right') for part in ('cornea','sclera','iris','pupil_cup')}:
            if obj.type=='MESH':eye_materials.update(m for m in obj.data.materials if m is not None)
    eye_aov=view.aovs.add();eye_aov.name='eye_mask';eye_aov.type='COLOR'
    for eye_material in eye_materials:
        if not eye_material.use_nodes:continue
        output=eye_material.node_tree.nodes.new('ShaderNodeOutputAOV');output.aov_name='eye_mask'
        output.inputs['Color'].default_value=(1.,1.,1.,1.)
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
        ('skin_profile', ('skin_profile',)),
        ('eye_mask', ('eye_mask',)),
    ):
        socket = next((layers.outputs.get(key) for key in candidates if layers.outputs.get(key) is not None), None)
        if socket is None:
            raise RuntimeError(f'missing {name} pass')
        files.file_output_items.new('RGBA', name)
        group.links.new(socket, files.inputs[name])
    overrides=request.get('material_overrides') or {}
    denoise_mode=float(overrides.get('skin_scatter_denoise',0.))
    if float(overrides.get('skin_indirect_firefly_guard',0.)) or denoise_mode:
        denoise=group.nodes.new('CompositorNodeDenoise')
        group.links.new(files.inputs['DiffInd'].links[0].from_socket,denoise.inputs['Image'])
        group.links.new(layers.outputs['Denoising Normal'],denoise.inputs['Normal'])
        group.links.new(layers.outputs['Denoising Albedo'],denoise.inputs['Albedo'])
        files.file_output_items.new('RGBA','DiffIndDenoised')
        group.links.new(denoise.outputs['Image'],files.inputs['DiffIndDenoised'])
    if denoise_mode==2:
        denoise=group.nodes.new('CompositorNodeDenoise')
        group.links.new(files.inputs['DiffDir'].links[0].from_socket,denoise.inputs['Image'])
        group.links.new(layers.outputs['Denoising Normal'],denoise.inputs['Normal'])
        group.links.new(layers.outputs['Denoising Albedo'],denoise.inputs['Albedo'])
        files.file_output_items.new('RGBA','DiffDirDenoised')
        group.links.new(denoise.outputs['Image'],files.inputs['DiffDirDenoised'])
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
                  device=device_name, backend=request['device'], seconds=time.monotonic()-started, blender_version=bpy.app.version_string, input_blend=bpy.data.filepath,
                  capture_source_sha256=source_hash,
                  resolution=[scene.render.resolution_x, scene.render.resolution_y],
                  lens_mm=camera.lens, sensor_height_mm=camera.sensor_height,
                  camera_origin=list(scene.camera.matrix_world.translation),
                  camera_forward=list(-scene.camera.matrix_world.col[2].xyz),
                  camera_rotation=[list(scene.camera.matrix_world.col[i].xyz) for i in range(3)],
                  native_sss_weight=skin.inputs['Subsurface Weight'].default_value,
                  native_sss_radius=list(skin.inputs['Subsurface Radius'].default_value),
                  native_sss_scale=skin.inputs['Subsurface Scale'].default_value)
    out.with_suffix('.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
