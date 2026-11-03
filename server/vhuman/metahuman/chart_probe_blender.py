# Cycles side of the ColorChecker light calibration (light_calibration.py). Runs inside Blender:
#   blender -b --factory-startup --python chart_probe_blender.py -- plan.json out_dir
# Builds the vhuman studio rig exactly as cycles_scene.run does (STUDIO_LIGHTS x STUDIO_ENERGY_SCALE, disks aimed at the
# world origin, grey world), a matte 24-patch chart per probe (Principled, roughness 1, specular 0, as add_color_chart)
# and renders linear EXRs <condition>__<probe>.exr for conditions key / fill / rim / world (one source at a time) and all.
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reconstruction'))
import cycles_scene as cs  # noqa: E402

CONDITIONS = ('key', 'fill', 'rim', 'world', 'all')
SAMPLES = 256


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    plan = json.loads(Path(argv[0]).read_text()); out = Path(argv[1]); out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene; scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'
    scene.cycles.samples = SAMPLES; scene.cycles.use_denoising = False; scene.cycles.max_bounces = 12
    scene.render.film_transparent = True
    scene.view_settings.view_transform = 'Standard'; scene.view_settings.exposure = -1.5     # as cycles_scene (EXR stays linear)
    size = plan['size']; scene.render.resolution_x = scene.render.resolution_y = size; scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'OPEN_EXR'; scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '32'
    lights = {}
    for name, position, power, dia in cs.STUDIO_LIGHTS:
        light = bpy.data.lights.new(name, 'AREA'); light.energy = power * cs.STUDIO_ENERGY_SCALE; light.shape = 'DISK'; light.size = dia
        obj = bpy.data.objects.new(name, light); bpy.context.collection.objects.link(obj); obj.location = position
        obj.rotation_euler = (Vector((0, 0, 0)) - obj.location).to_track_quat('-Z', 'Y').to_euler()
        lights[name] = (light, light.energy)
    world = bpy.data.worlds.new('studio'); scene.world = world; world.use_nodes = True
    bg = world.node_tree.nodes['Background']; bg.inputs['Color'].default_value = (*(cs.STUDIO_WORLD[0],) * 3, 1)
    camera_data = bpy.data.cameras.new('camera'); camera = bpy.data.objects.new('camera', camera_data)
    bpy.context.collection.objects.link(camera); scene.camera = camera
    camera_data.clip_start = .01; camera_data.clip_end = 10
    mats = []
    for i, rgb in enumerate(plan['albedo_linear']):
        mat = bpy.data.materials.new(f'chart_{i:02d}'); mat.use_nodes = True; bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Base Color'].default_value = (*rgb, 1.); bsdf.inputs['Roughness'].default_value = 1.
        if 'Specular IOR Level' in bsdf.inputs:
            bsdf.inputs['Specular IOR Level'].default_value = 0.
        mats.append(mat)
    h = plan['patch_m'] / 2
    for probe, rec in plan['probes'].items():
        c = rec['cycles']; cam = c['camera']
        for o in [o for o in bpy.data.objects if o.name.startswith('patch_')]:
            bpy.data.objects.remove(o, do_unlink=True)
        right, up, normal = (Vector(v) for v in (c['right'], c['up'], c['normal']))
        rot = Matrix((right, up, normal)).transposed().to_4x4()
        for i, p in enumerate(c['patches']):
            mesh = bpy.data.meshes.new(f'patch_{i:02d}'); mesh.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
            mesh.materials.append(mats[i])
            obj = bpy.data.objects.new(f'patch_{i:02d}', mesh); bpy.context.collection.objects.link(obj)
            obj.matrix_world = Matrix.Translation(Vector(p)) @ rot
        camera_data.shift_x = camera_data.shift_y = 0; camera_data.lens = 36 * cam['focal_px'] / cam['size'][1]
        m = Matrix(cam['rotation']).transposed().to_4x4(); m.translation = Vector(cam['origin']); camera.matrix_world = m
        for cond in CONDITIONS:
            for name, (light, energy) in lights.items():
                light.energy = energy if cond in (name, 'all') else 0.
            bg.inputs['Strength'].default_value = cs.STUDIO_WORLD[1] if cond in ('world', 'all') else 0.
            scene.render.filepath = str(out / f'{cond}__{probe}.exr')
            bpy.ops.render.render(write_still=True)
    (out / 'done.json').write_text(json.dumps(dict(ok=True, conditions=CONDITIONS, probes=list(plan['probes']), samples=SAMPLES)) + '\n')


main()
