"""Blender worker for sampled anatomy animation with verified USD interchange.

Usage: blender -b --python usd_motion_worker.py -- --scene DIR --out DIR
Mesh animation is baked; material drivers are recorded as a portability limit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import bpy
from mathutils import Matrix
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from server.vhuman.reconstruction.usd_materials import capture, restore, verify


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_digest(sample):
    checksum = hashlib.sha256()
    for value, dtype in zip(sample, ('<f8', '<i4', '<f8')):
        array = np.asarray(value, dtype=dtype)
        checksum.update(str(array.shape).encode())
        checksum.update(array.tobytes())
    return checksum.hexdigest()


def mesh_sample(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = np.array([evaluated.matrix_world @ v.co for v in mesh.vertices])
        triangles = np.array([face.vertices[:] for face in mesh.loop_triangles], dtype=np.int32)
        uv = (np.array([[mesh.uv_layers.active.data[i].uv[:] for i in face.loops]
                        for face in mesh.loop_triangles]) if mesh.uv_layers.active
              else np.zeros((len(triangles), 3, 2)))
        return vertices, triangles, uv
    finally:
        evaluated.to_mesh_clear()


def retime_action(action, start, factor):
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for point in curve.keyframe_points:
                        point.co.x = 1 + (point.co.x - start) * factor
                        point.handle_left.x = 1 + (point.handle_left.x - start) * factor
                        point.handle_right.x = 1 + (point.handle_right.x - start) * factor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, help='offline_render output directory')
    parser.add_argument('--out', required=True)
    parser.add_argument('--samples-per-frame', type=int, default=2, choices=(1, 2, 4))
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    source, out = Path(args.scene).resolve(), Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory must be empty')
    config = json.loads((source / 'scene.json').read_text())
    request = json.loads((source / 'request.json').read_text())
    candidate = Path(config['candidate'])
    manifest = json.loads((candidate / 'manifest.json').read_text())
    if digest(candidate / 'geometry.npz') != manifest['geometry_sha256']:
        raise ValueError('candidate geometry hash mismatch')
    if 'motion' not in request:
        raise ValueError('an animated offline render is required')
    motion = Path(request['motion'])
    track = json.loads((motion / 'motion.json').read_text())
    if (track['candidate_geometry_sha256'] != manifest['geometry_sha256']
            or track.get('motion_sha256') != digest(motion / 'motion.npz')):
        raise ValueError('motion geometry or content hash mismatch')
    bpy.ops.wm.open_mainfile(filepath=str(source / 'head.blend'))
    scene = bpy.context.scene
    stored = json.loads(scene['motion_provenance'])
    if stored.get('motion_sha256') != track['motion_sha256']:
        raise ValueError('packed scene does not contain the requested motion')
    meshes = [bpy.data.objects[part['name']] for part in config['parts']]
    if len({obj.name for obj in meshes}) != len(meshes):
        raise ValueError('duplicate configured anatomy mesh')
    if any(obj.type != 'MESH' or obj.hide_render for obj in meshes):
        raise ValueError('configured anatomy must be visible meshes')
    start, end = scene.frame_start, scene.frame_end
    if end - start + 1 != track['frames']:
        raise ValueError('scene frame range differs from motion')
    fps = scene.render.fps / scene.render.fps_base
    if abs(fps - track['fps']) > 1e-6:
        raise ValueError('scene frame rate differs from motion')
    out.mkdir(parents=True, exist_ok=True)
    factor = args.samples_per_frame
    actions = {}
    for owner in [scene, *scene.objects, *[obj.data.shape_keys for obj in scene.objects
                                        if obj.type == 'MESH' and obj.data.shape_keys]]:
        animation = owner.animation_data
        if animation and animation.action:
            actions[animation.action.as_pointer()] = animation.action
    for action in actions.values():
        retime_action(action, start, factor)
    scene.frame_start, scene.frame_end = 1, (end - start) * factor + 1
    scene.render.fps *= factor
    # The source meshes use GNM Y-up coordinates. A common parent changes them
    # to Blender Z-up without overwriting animated object transforms.
    conversion = bpy.data.objects.new('anatomy_coordinate_conversion', None)
    scene.collection.objects.link(conversion)
    conversion.matrix_world = Matrix(((1, 0, 0, 0), (0, 0, -1, 0),
                                      (0, 1, 0, 0), (0, 0, 0, 1)))
    for obj in list(scene.objects):
        if obj.type == 'MESH' and obj.parent is None:
            obj.parent = conversion
    subdivision = {}
    for obj in meshes:
        subdivision[obj.name] = []
        for modifier in obj.modifiers:
            if modifier.type == 'SUBSURF':
                subdivision[obj.name].append(dict(name=modifier.name, type=modifier.type,
                    levels=modifier.levels, render_levels=modifier.render_levels,
                    show_viewport=modifier.show_viewport, show_render=modifier.show_render))
                modifier.show_viewport = modifier.show_render = False
            elif modifier.type != 'BOOLEAN':
                raise ValueError('unsupported baked modifier: ' + modifier.type)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    material_drivers = {mat.name: len(mat.node_tree.animation_data.drivers)
                        for obj in meshes for mat in obj.data.materials
                        if mat.node_tree and mat.node_tree.animation_data
                        and mat.node_tree.animation_data.drivers}
    capture(list({mat.name: mat for obj in meshes for mat in obj.data.materials}.values()), out)
    references = {}
    samples = scene.frame_end
    for frame in range(1, samples + 1):
        scene.frame_set(frame)
        graph = bpy.context.evaluated_depsgraph_get()
        for obj in meshes:
            references[(frame, obj.name)] = mesh_sample(obj, graph)
    names = [obj.name for obj in meshes]
    bpy.ops.object.select_all(action='DESELECT')
    conversion.select_set(True)
    for obj in meshes:
        obj.select_set(True)
    scene.frame_set(1)
    bpy.ops.wm.usd_export(filepath=str(out / 'head.usdc'), selected_objects_only=True,
        export_animation=True, export_shapekeys=False, export_materials=True,
        export_subdivision='IGNORE', export_textures_mode='NEW', relative_paths=True,
        convert_orientation=True, export_global_forward_selection='NEGATIVE_Z',
        export_global_up_selection='Y', generate_materialx_network=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = round(fps * factor)
    bpy.ops.wm.usd_import(filepath=str(out / 'head.usdc'), import_materials=True,
                         import_textures_mode='IMPORT_NONE')
    restore(out)
    material_verification = verify(out)
    imported = {obj.name: obj for obj in scene.objects if obj.type == 'MESH'}
    if set(imported) != set(names):
        raise ValueError('USD changed the anatomy mesh set')
    rows = []
    for frame in range(1, samples + 1):
        scene.frame_set(frame)
        graph = bpy.context.evaluated_depsgraph_get()
        for name, obj in imported.items():
            actual = mesh_sample(obj, graph)
            expected = references[(frame, name)]
            shapes = [a.shape == b.shape for a, b in zip(actual, expected)]
            row = dict(frame=frame, source_frame=start + (frame - 1) / factor,
                       mesh=name, vertices=len(actual[0]), triangles=len(actual[1]),
                       shape_match=all(shapes), topology_same=np.array_equal(actual[1], expected[1]))
            if shapes[0]:
                row['max_position_error_m'] = float(np.max(abs(actual[0] - expected[0])))
            if shapes[2]:
                row['max_uv_error'] = float(np.max(abs(actual[2] - expected[2])))
            row['passed'] = bool(row['shape_match'] and row['topology_same']
                and row.get('max_position_error_m', 1) < 1e-6
                and row.get('max_uv_error', 1) < 1e-6)
            row['imported_sample_sha256'] = sample_digest(actual)
            rows.append(row)
    passed = all(row['passed'] for row in rows)
    report = dict(schema='vhuman.animated_anatomy_usd.v1', passed=passed,
        source_scene_sha256=digest(source / 'head.blend'),
        candidate_geometry_sha256=manifest['geometry_sha256'],
        source_motion_sha256=track['motion_sha256'], source_fps=fps,
        exported_fps=fps * factor, samples_per_frame=factor, samples=samples,
        meshes=len(names), up_axis='Y', units='metres',
        material_capture_source_frame=start, material_drivers_not_exported=material_drivers,
        material_verification=material_verification, subdivision_modifiers=subdivision,
        material_sidecar_sha256=digest(out / 'blender_materials.json'),
        usd_sha256=digest(out / 'head.usdc'), checks=rows,
        limitations=['Finite sampled mesh animation, not an editable native rig',
                     'Material drivers/wrinkle animation are captured statically at the first frame',
                     'Interchange checks do not establish fit quality or collision freedom'])
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    if not passed:
        raise ValueError('animated USD roundtrip failed; inspect report.json')
    for name, specs in subdivision.items():
        for spec in specs:
            modifier = imported[name].modifiers.new(spec['name'], spec['type'])
            for key in ('levels', 'render_levels', 'show_viewport', 'show_render'):
                setattr(modifier, key, spec[key])
    scene.frame_start, scene.frame_end = 1, samples
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'reimported.blend'))
    shutil.copyfile(candidate / 'manifest.json', out / 'candidate_manifest.json')
    shutil.copyfile(motion / 'motion.json', out / 'source_motion.json')
    print(json.dumps(dict(passed=passed, meshes=len(names), samples=samples,
                          max_position_error_m=max(r['max_position_error_m'] for r in rows))), flush=True)


if __name__ == '__main__':
    main()
