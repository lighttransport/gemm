"""Fresh-process verifier/importer for a sampled anatomy USD bundle."""
import argparse
import json
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from server.vhuman.reconstruction.usd_materials import restore, verify
from server.vhuman.reconstruction import usd_material_animation
from server.vhuman.reconstruction.usd_motion_worker import digest, mesh_sample, sample_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--out', required=True, help='output .blend; must be inside the bundle')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    bundle, output = Path(args.bundle).resolve(), Path(args.out).resolve()
    if output.parent != bundle or output.exists():
        raise ValueError('use a new .blend filename directly inside the bundle')
    report = json.loads((bundle / 'report.json').read_text())
    if report.get('schema') != 'vhuman.animated_anatomy_usd.v1' or not report.get('passed'):
        raise ValueError('verified animated anatomy report required')
    hashes = [('head.usdc', report['usd_sha256']),
              ('blender_materials.json', report['material_sidecar_sha256'])]
    if 'material_animation_sidecar_sha256' in report:
        hashes.append((usd_material_animation.FILENAME, report['material_animation_sidecar_sha256']))
    for filename, expected in hashes:
        if digest(bundle / filename) != expected:
            raise ValueError('bundle hash mismatch: ' + filename)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = round(report['exported_fps'])
    scene.render.fps_base = scene.render.fps / report['exported_fps']
    bpy.ops.wm.usd_import(filepath=str(bundle / 'head.usdc'), import_materials=True,
                         import_textures_mode='IMPORT_NONE')
    restore(bundle)
    material_animation = None
    if 'material_animation_sidecar_sha256' in report:
        animation_record = usd_material_animation.restore(bundle, scene=scene)
        if animation_record['frames'] != list(range(1, report['samples'] + 1)):
            raise ValueError('material and mesh sample frames differ')
        material_animation = usd_material_animation.verify(bundle, scene=scene)
    materials = verify(bundle)
    objects = {obj.name: obj for obj in scene.objects if obj.type == 'MESH'}
    if set(objects) != {row['mesh'] for row in report['checks']}:
        raise ValueError('imported anatomy mesh set differs')
    by_frame = {}
    for row in report['checks']:
        by_frame.setdefault(row['frame'], []).append(row)
    if (len(objects) != report['meshes']
            or len(report['checks']) != report['samples'] * report['meshes']
            or set(by_frame) != set(range(1, report['samples'] + 1))
            or any(not row.get('passed') for row in report['checks'])
            or any(len(rows) != len(objects) or {row['mesh'] for row in rows} != set(objects)
                   for rows in by_frame.values())):
        raise ValueError('incomplete or inconsistent sample validation report')
    for frame, rows in sorted(by_frame.items()):
        scene.frame_set(frame)
        graph = bpy.context.evaluated_depsgraph_get()
        for row in rows:
            actual = sample_digest(mesh_sample(objects[row['mesh']], graph))
            if actual != row['imported_sample_sha256']:
                raise ValueError(f"sample changed: {row['mesh']} at frame {frame}")
    for name, specs in report['subdivision_modifiers'].items():
        for spec in specs:
            modifier = objects[name].modifiers.new(spec['name'], spec['type'])
            for key in ('levels', 'render_levels', 'show_viewport', 'show_render'):
                setattr(modifier, key, spec[key])
    scene.frame_start, scene.frame_end = 1, report['samples']
    scene.frame_set(1)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    # Resolve paths only after the .blend has its final base directory.
    caches = []
    for cache in bpy.data.cache_files:
        source = Path(bpy.path.abspath(cache.filepath)).resolve()
        if source != bundle / 'head.usdc':
            raise ValueError('unexpected external animation cache')
        cache.filepath = '//head.usdc'
        caches.append(cache.filepath)
    if not caches:
        raise ValueError('USD import did not retain an animation cache')
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    receipt = dict(passed=True, samples=len(by_frame), meshes=len(objects),
                   sample_checks=len(report['checks']), material_verification=materials,
                   material_animation_verification=material_animation,
                   relative_animation_caches=caches, usd_sha256=report['usd_sha256'])
    output.with_suffix('.validation.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
