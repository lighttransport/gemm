# Smoke 03: spawn the preview actor, read vertices/triangles of every skeletal mesh component through
# get_mesh_data_for_conforming, dump them as raw binaries, export the archetype head skeletal mesh and a glTF of it (UVs),
# and record landmarks + component transforms so the host can determine the frame. Writes smoke/03_geometry.json.
import os
import struct

import unreal

import mh_common as mh

OUT = os.path.join(mh.OUT_ROOT, 'smoke', '03_geometry.json')
GEO_DIR = os.path.join(mh.OUT_ROOT, 'smoke', '03_geo')
EXPORT_PKG = '/Game/MHGen/Export'
MAGIC = b'VHMH'


def write_mesh_bin(path, verts, tris):
    """Raw mesh: magic, nverts, ntris (uint32), float32 xyz * nverts, int32 * 3 * ntris (UE units, cm)."""
    with open(path, 'wb') as f:
        f.write(MAGIC); f.write(struct.pack('<II', len(verts), len(tris) // 3))
        f.write(struct.pack(f'<{3 * len(verts)}f', *[c for v in verts for c in (v.x, v.y, v.z)]))
        f.write(struct.pack(f'<{len(tris)}i', *[int(t) for t in tris]))


def mesh_data(mesh):
    """get_mesh_data_for_conforming is static with out params; Python returns (ok, verts, tris) or (verts, tris)."""
    r = mh.subsystem().get_mesh_data_for_conforming(mesh)
    if isinstance(r, tuple) and len(r) == 3:
        ok, verts, tris = r
        if not ok:
            raise RuntimeError('get_mesh_data_for_conforming returned false')
        return verts, tris
    if isinstance(r, tuple) and len(r) == 2:
        return r
    raise RuntimeError(f'unexpected return {type(r)}')


def main():
    os.makedirs(GEO_DIR, exist_ok=True)
    out = dict(components=[], exports={})
    character = mh.create_character('/Game/MHGen/Smoke', 'Smoke03')
    sub = mh.subsystem()
    with mh.Editing(character):
        lm = sub.get_face_landmarks(character=character)
        out['landmarks'] = [[v.x, v.y, v.z] for v in lm]
        actor = sub.spawn_meta_human_actor(character=character)
        out['actor'] = actor.get_name() if actor else None
        if actor:
            out['actor_transform'] = str(actor.get_actor_transform())
            comps = actor.get_components_by_class(unreal.SkeletalMeshComponent)
            for comp in comps:
                rec = dict(name=comp.get_name(), cls=comp.get_class().get_name())
                for key, fn in (('relative_transform', 'get_relative_transform'), ('world_transform', 'get_world_transform')):
                    try:
                        rec[key] = str(getattr(comp, fn)())
                    except Exception as e:      # noqa: BLE001
                        rec[key] = f'err {e}'
                mesh = None
                for prop in ('skeletal_mesh_asset', 'skinned_asset'):
                    try:
                        mesh = comp.get_editor_property(prop)
                        if mesh is not None:
                            break
                    except Exception:      # noqa: BLE001
                        continue
                rec['mesh'] = mesh.get_path_name() if mesh else None
                if mesh is not None:
                    try:
                        verts, tris = mesh_data(mesh)
                        rec['n_vertices'] = len(verts); rec['n_triangles'] = len(tris) // 3
                        path = os.path.join(GEO_DIR, f'{comp.get_name()}.vhmh.bin')
                        write_mesh_bin(path, verts, tris)
                        rec['bin'] = path
                        xs = [v.x for v in verts]; ys = [v.y for v in verts]; zs = [v.z for v in verts]
                        rec['bbox'] = [[min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]]
                    except Exception as e:      # noqa: BLE001
                        rec['mesh_data_error'] = str(e)
                out['components'].append(rec)
        # geometry export (head skeletal mesh asset), then glTF for UVs/sections
        try:
            params = unreal.MetaHumanGeometryExportParams()
            params.project_path = EXPORT_PKG
            params.head_skeletal_mesh = True
            params.body_skeletal_mesh = False
            params.full_body_skeletal_mesh = False
            params.overwrite_existing_assets = True
            unreal.MetaHumanCharacterExportBlueprintLibrary.export_geometry(character, params)
            assets = unreal.EditorAssetLibrary.list_assets(EXPORT_PKG, recursive=True, include_folder=False)
            out['exports']['assets'] = assets
            for a in assets:
                obj = unreal.EditorAssetLibrary.load_asset(a)
                if isinstance(obj, unreal.SkeletalMesh):
                    gl = os.path.join(GEO_DIR, obj.get_name() + '.gltf')
                    opts = unreal.GLTFExportOptions()
                    for k, v in (('export_vertex_colors', False), ('export_vertex_skin_weights', False), ('texture_image_format', unreal.GLTFTextureImageFormat.NONE),
                                 ('export_materials', False)):
                        try:
                            opts.set_editor_property(k, v)
                        except Exception:      # noqa: BLE001
                            pass
                    ok = unreal.GLTFExporter.export_to_gltf(obj, gl, opts, set())
                    out['exports'][obj.get_name()] = dict(gltf=gl, ok=bool(ok))
                    try:
                        verts, tris = mesh_data(obj)
                        out['exports'][obj.get_name()].update(n_vertices=len(verts), n_triangles=len(tris) // 3)
                        write_mesh_bin(os.path.join(GEO_DIR, obj.get_name() + '.vhmh.bin'), verts, tris)
                    except Exception as e:      # noqa: BLE001
                        out['exports'][obj.get_name()]['mesh_data_error'] = str(e)
        except Exception as e:      # noqa: BLE001
            out['exports']['error'] = str(e)
    mh.delete_asset(character)
    return out


mh.guarded(main, OUT)
