"""Capture signed world shading normals for comparison with UE SCS_Normal.

blender -b head.blend --python normal_passes_blender.py -- output.exr MODE
MODE is baseline, measured_only, or regional. HIP only; no CPU fallback.
"""
import hashlib
import json
from pathlib import Path
import sys
import bpy


def main():
    args=sys.argv[sys.argv.index('--')+1:]
    out=Path(args[0]);mode=args[1]
    if mode not in ('baseline','measured_only','regional'):
        raise ValueError('unknown normal audit mode')
    scene=bpy.context.scene
    if 'cycles' not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module='cycles')
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='HIP';prefs.get_devices()
    devices=[d for d in prefs.devices if d.type=='HIP']
    if not devices:raise RuntimeError('normal audit requires HIP')
    for device in prefs.devices:device.use=device==devices[0]
    scene.cycles.device='GPU';scene.cycles.samples=8;scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False
    scene.cycles.dicing_rate=2;scene.cycles.use_auto_tile=True;scene.cycles.tile_size=256
    scene.render.resolution_x=512;scene.render.resolution_y=512;scene.render.resolution_percentage=100
    mat=bpy.data.materials['skin'];nodes=mat.node_tree.nodes;links=mat.node_tree.links
    normal=next(n for n in nodes if n.type=='NORMAL_MAP')
    principled=next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
    for link in list(principled.inputs['Subsurface Weight'].links):links.remove(link)
    principled.inputs['Subsurface Weight'].default_value=0
    if nodes.get('peri_k_lid_rough'):nodes['peri_k_lid_rough'].outputs[0].default_value=.8
    if mode=='measured_only':
        textures=[n for n in nodes if n.type=='TEX_IMAGE' and n.image and Path(n.image.filepath).name=='skin_normal_map.png']
        if len(textures)!=1:raise RuntimeError('expected one measured skin normal texture')
        for link in list(normal.inputs['Color'].links):links.remove(link)
        links.new(textures[0].outputs['Color'],normal.inputs['Color'])
        normal.inputs['Strength'].default_value=1
    geom=nodes.new('ShaderNodeNewGeometry');uv=nodes.new('ShaderNodeUVMap');uv.uv_map='UVMap'
    view=bpy.context.view_layer;view.use_pass_position=True
    for name,source in [('shader_normal',normal.outputs['Normal']),('geometric_normal',geom.outputs['Normal']),('uv',uv.outputs['UV']),('skin_mask',None)]:
        aov=view.aovs.add();aov.name=name;aov.type='COLOR'
        node=nodes.new('ShaderNodeOutputAOV');node.aov_name=name
        if source is None:node.inputs['Color'].default_value=(1,1,1,1)
        else:links.new(source,node.inputs['Color'])
    group=bpy.data.node_groups.new('normal_audit_passes','CompositorNodeTree');scene.compositing_node_group=group
    group.interface.new_socket('Image',in_out='OUTPUT',socket_type='NodeSocketColor')
    layers=group.nodes.new('CompositorNodeRLayers');sink=group.nodes.new('NodeGroupOutput');group.links.new(layers.outputs['Image'],sink.inputs[0])
    files=group.nodes.new('CompositorNodeOutputFile');files.directory=str(out.parent);files.file_name=out.stem
    files.format.file_format='OPEN_EXR_MULTILAYER';files.format.color_depth='32'
    if not len(files.file_output_items):files.file_output_items.new('RGBA','Image')
    group.links.new(layers.outputs['Image'],files.inputs[0])
    for name in ['shader_normal','geometric_normal','skin_mask','uv','Position']:
        files.file_output_items.new('RGBA',name);group.links.new(layers.outputs[name],files.inputs[name])
    scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_depth='32';scene.render.image_settings.color_mode='RGBA'
    scene.render.filepath=str(out.with_name(out.stem+'_beauty.exr'));out.parent.mkdir(parents=True,exist_ok=True)
    out.unlink(missing_ok=True)
    bpy.ops.render.render(write_still=True)
    if not out.is_file():
        candidates=list(out.parent.glob(out.stem+'[0-9]*.exr'))
        if len(candidates)!=1:raise RuntimeError('normal capture output missing or ambiguous')
        candidates[0].replace(out)
    camera=scene.camera
    record=dict(schema='vhuman.normal_audit.v1',mode=mode,encoding='signed world xyz; no tone map',
                backend='hip',device=devices[0].name,blender_version=bpy.app.version_string,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_blend=bpy.data.filepath,
                samples=8,resolution=[512,512],dicing_rate=2,camera_origin=list(camera.matrix_world.translation),
                camera_forward=list(-camera.matrix_world.col[2].xyz),lens_mm=camera.data.lens,sensor_height_mm=camera.data.sensor_height)
    out.with_suffix('.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':main()
