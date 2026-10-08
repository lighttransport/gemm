"""Restore and validate a portable anatomy bundle in a fresh Blender process."""
import argparse
import json
import sys
from pathlib import Path
import bpy
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from server.vhuman.reconstruction.usd_materials import digest, restore, verify

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle', required=True)
parser.add_argument('--out-blend', required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
root = Path(args.bundle).resolve()
out = Path(args.out_blend).resolve()
if out.exists():
    raise ValueError('refusing to replace an existing Blender scene')
record = json.loads((root/'report.json').read_text())
if record.get('schema') != 'vhuman.full_anatomy_usd.v1':
    raise ValueError('unsupported anatomy bundle')
for name, key in [('head.usdc', 'usd_sha256'), ('blender_materials.json', 'material_sidecar_sha256'),
                  ('geometry.npz', 'candidate_geometry_sha256'), ('scene_assets.npz', 'scene_arrays_sha256')]:
    if digest(root/name) != record.get(key):
        raise ValueError('bundle hash mismatch: '+name)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.usd_import(filepath=str(root/'head.usdc'), import_materials=True, import_textures_mode='IMPORT_NONE')
restore(root)
material_report = verify(root)
arrays = np.load(root/'scene_assets.npz', allow_pickle=False)
axis = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if {obj.name for obj in meshes} != {row['name'] for row in record['meshes']}:
    raise ValueError('imported anatomy mesh names differ')
maximum = 0.
for obj in meshes:
    prefix = obj.name
    vertices = np.array([obj.matrix_world@v.co for v in obj.data.vertices])
    expected = arrays[prefix+'_positions']@axis.T
    triangles = np.array([face.vertices[:] for face in obj.data.polygons])
    uv = np.array([v.uv[:] for v in obj.data.uv_layers.active.data]).reshape(-1, 3, 2)
    expected_uv = arrays[prefix+'_uvs'].copy()
    expected_uv[..., 1] = 1-expected_uv[..., 1]
    if (vertices.shape != expected.shape or not np.array_equal(triangles, arrays[prefix+'_triangles'])
            or uv.shape != expected_uv.shape or not np.allclose(uv, expected_uv, atol=1e-6, rtol=0)):
        raise ValueError('imported anatomy topology/UV mismatch: '+prefix)
    error = float(np.max(abs(vertices-expected)))
    if error > 1e-6:
        raise ValueError('imported anatomy coordinates differ: '+prefix)
    maximum = max(maximum, error)
    for spec in record['subdivision_modifiers'][prefix]:
        modifier = obj.modifiers.new(spec['name'], spec['type'])
        modifier.levels = spec['levels']
        modifier.render_levels = spec['render_levels']
out.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(out))
report = dict(meshes=len(meshes), max_coordinate_error_m=maximum, materials=material_report,
              bundle=str(root), static=True, all_texture_data_packed=True)
out.with_suffix('.validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
