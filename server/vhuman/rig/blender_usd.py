"""Import a rig USD/GLB into Blender while retaining signed corrective animation.

Run inside Blender: --python server/vhuman/rig/blender_usd.py -- --usd CLIP --out NEW.blend
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from server.vhuman.reconstruction.usd_material_animation import action_curves


def restore_shape_ranges(objects):
    """Expand slider bounds to the preserved curves, without changing their keys."""
    changes = []
    for obj in objects:
        keys = getattr(getattr(obj, 'data', None), 'shape_keys', None)
        if keys is None:
            continue
        for curve in action_curves(keys.animation_data):
            match = re.fullmatch(r'key_blocks\[("(?:\\.|[^"\\])*")\]\.value', curve.data_path)
            if not match:
                continue
            block = keys.key_blocks.get(json.loads(match[1]))
            if block is None or not curve.keyframe_points:
                continue
            values = [float(p.co.y) for p in curve.keyframe_points]
            # Cubic curves stay inside their control-point hull. USD's sampled
            # linear curves need only their actual key values.
            values += [float(h.y) for p in curve.keyframe_points if p.interpolation == 'BEZIER'
                       for h in (p.handle_left, p.handle_right)]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('nonfinite corrective curve: '+block.name)
            lo = min(block.slider_min, min(values)); hi = max(block.slider_max, max(values))
            if (lo < block.bl_rna.properties['slider_min'].hard_min
                    or hi > block.bl_rna.properties['slider_max'].hard_max):
                raise ValueError('corrective curve exceeds Blender shape-key limits: '+block.name)
            if lo < block.slider_min or hi > block.slider_max:
                before = [block.slider_min, block.slider_max]
                block.slider_min = lo; block.slider_max = hi
                changes.append(dict(mesh=obj.name, shape=block.name, before=before,
                                    after=[block.slider_min, block.slider_max]))
    return dict(changed_shapes=len(changes), changes=changes, animation_curves_unchanged=True)


def main():
    import bpy
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--usd', type=Path)
    source_group.add_argument('--gltf', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.out.exists() or args.out.suffix != '.blend':
        raise ValueError('use a new .blend output file')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    source = args.usd or args.gltf
    if args.usd:
        bpy.ops.wm.usd_import(filepath=str(source.resolve()))
    else:
        bpy.ops.import_scene.gltf(filepath=str(source.resolve()))
    report = restore_shape_ranges(bpy.context.scene.objects)
    intervals = [action.frame_range for action in bpy.data.actions if action.users]
    if intervals:
        bpy.context.scene.frame_start = math.floor(min(v[0] for v in intervals))
        bpy.context.scene.frame_end = math.ceil(max(v[1] for v in intervals))
    bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    for image in bpy.data.images:
        if image.source == 'FILE' and not image.packed_file:
            image.pack()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out.resolve()))
    report.update(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  frame_range=[bpy.context.scene.frame_start, bpy.context.scene.frame_end],
                  blender=bpy.app.version_string)
    args.out.with_suffix('.validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(changed_shapes=report['changed_shapes'], out=str(args.out))))


if __name__ == '__main__':
    main()
