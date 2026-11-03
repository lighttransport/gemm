# Dataset generator (runs inside UE): processes the job ids listed in a batch file, one MetaHumanCharacter per sample.
# usage (via launch.py): gen_batch.py <batch.json>   where batch.json = {"jobs_path": ..., "ids": [...]}
# Per sample writes samples/<id>/: params.json, landmarks.json, meshes/<component>.bin (VHMH raw, cm), tex/<type>.png
# (1024, exact bytes via ImageWriteBlueprintLibrary), renders/ (if the job has render=true and capture is available), done.json.
import hashlib
import json
import os
import struct
import time
import traceback

import unreal

import mh_common as mh

V1 = os.path.join(mh.OUT_ROOT, 'v1')
BATCH_PKG = '/Game/MHGen/Batch'
MAGIC = b'VHMH'
TEXTURE_TYPES = ('basecolor', 'normal', 'cavity')          # always; animated maps with ANIMATED=1
ANIMATED = os.environ.get('VHUMAN_MH_ANIMATED', '0') == '1'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def type_name(key):
    n = getattr(key, 'name', None) or str(key).split('.')[-1].split(':')[0]
    return n.strip().lower()


def export_texture(tex, path):
    opts = unreal.ImageWriteOptions()
    opts.format = unreal.DesiredImageFormat.PNG
    opts.compression_quality = 100
    opts.async_ = False
    unreal.ImageWriteBlueprintLibrary.export_to_disk(tex, path, opts)
    if not os.path.isfile(path):
        raise RuntimeError(f'texture export produced no file: {path}')


def write_mesh_bin(path, verts, tris):
    with open(path, 'wb') as f:
        f.write(MAGIC); f.write(struct.pack('<II', len(verts), len(tris) // 3))
        f.write(struct.pack(f'<{3 * len(verts)}f', *[c for v in verts for c in (v.x, v.y, v.z)]))
        f.write(struct.pack(f'<{len(tris)}i', *[int(t) for t in tris]))


def mesh_data(mesh):
    r = mh.subsystem().get_mesh_data_for_conforming(mesh)
    if isinstance(r, tuple) and len(r) == 3:
        if not r[0]:
            raise RuntimeError('get_mesh_data_for_conforming returned false')
        return r[1], r[2]
    return r


def skin_settings_from_job(skin):
    p = unreal.MetaHumanCharacterSkinProperties()
    p.u = float(skin['u']); p.v = float(skin['v'])
    p.face_texture_index = int(skin['face_texture_index'])
    p.body_texture_index = int(skin.get('body_texture_index', 0))
    p.roughness = float(skin.get('roughness', 1.0))
    p.show_top_underwear = True
    s = unreal.MetaHumanCharacterSkinSettings()
    s.skin = p
    fr = unreal.MetaHumanCharacterFrecklesProperties()
    f = skin.get('freckles')
    if f:
        fr.density = float(f['density']); fr.strength = float(f['strength']); fr.saturation = float(f['saturation']); fr.tone_shift = float(f['tone_shift'])
        fr.mask = getattr(unreal.MetaHumanCharacterFrecklesMask, f'TYPE{int(f["mask"])}')
    s.freckles = fr
    acc = unreal.MetaHumanCharacterAccentRegions()
    for region, vals in (skin.get('accents') or {}).items():
        r = unreal.MetaHumanCharacterAccentRegionProperties()
        r.redness = float(vals['redness']); r.saturation = float(vals['saturation']); r.lightness = float(vals['lightness'])
        setattr(acc, region, r)
    s.accents = acc
    try:
        s.texture_material_overrides.enable_texture_overrides = False
    except Exception:      # noqa: BLE001
        pass
    return s


def set_struct_from_dict(obj, d):
    """Assign plain values of a nested dict (from struct_to_dict) onto a UStruct, best effort."""
    for k, v in d.items():
        try:
            cur = obj.get_editor_property(k)
        except Exception:      # noqa: BLE001
            continue
        try:
            if isinstance(v, dict) and isinstance(cur, unreal.StructBase):
                set_struct_from_dict(cur, v); obj.set_editor_property(k, cur)
            elif isinstance(v, list) and isinstance(cur, unreal.LinearColor) and len(v) == 4:
                obj.set_editor_property(k, unreal.LinearColor(*v))
            elif isinstance(v, (int, float, bool)) and isinstance(cur, (int, float, bool)):
                obj.set_editor_property(k, type(cur)(v))
            elif isinstance(v, str) and isinstance(cur, unreal.EnumBase):
                name = v.split('.')[-1].split(':')[0].strip()
                obj.set_editor_property(k, getattr(type(cur), name))
        except Exception:      # noqa: BLE001
            continue


def apply_eyes(character, spec):
    eyes = character.get_editor_property('eyes_settings')
    settings = spec['settings']
    if spec['source'] == 'preset_jitter':
        set_struct_from_dict(eyes, settings)
    else:                                               # flat uniform sample
        for side in ('eye_left', 'eye_right'):
            e = getattr(eyes, side)
            iris, pupil, cornea, sclera = e.iris, e.pupil, e.cornea, e.sclera
            iris.pattern = getattr(unreal.MetaHumanCharacterEyesIrisPattern, f'IRIS{int(settings["iris_pattern"]):03d}')
            iris.rotation = float(settings['iris_rotation'])
            for k in ('primary_color_u', 'primary_color_v', 'secondary_color_u', 'secondary_color_v', 'color_blend', 'color_blend_softness',
                      'shadow_details', 'limbal_ring_size', 'limbal_ring_softness', 'global_saturation'):
                setattr(iris, k, float(settings[k]))
            iris.blend_method = getattr(unreal.MetaHumanCharacterEyesBlendMethod, settings['blend_method'].upper())
            pupil.dilation = float(settings['pupil_dilation']); pupil.feather = float(settings['pupil_feather'])
            cornea.size = float(settings['cornea_size']); cornea.limbus_softness = float(settings['cornea_limbus_softness'])
            sclera.transmission_spread = float(settings['sclera_transmission_spread'])
            sclera.vascularity_intensity = float(settings['sclera_vascularity_intensity']); sclera.vascularity_coverage = float(settings['sclera_vascularity_coverage'])
            e.iris, e.pupil, e.cornea, e.sclera = iris, pupil, cornea, sclera
            setattr(eyes, side, e)
    mh.subsystem().commit_eyes_settings(character=character, eyes_settings=eyes)
    return mh.struct_to_dict(character.get_editor_property('eyes_settings'))


def process(job, capture=None):
    sid = job['id']
    out = os.path.join(V1, 'samples', sid)
    os.makedirs(os.path.join(out, 'tex'), exist_ok=True); os.makedirs(os.path.join(out, 'meshes'), exist_ok=True)
    t0 = time.time(); files = {}
    sub = mh.subsystem()
    character = mh.create_character(BATCH_PKG, sid)
    try:
        with mh.Editing(character):
            sub.set_face_model_coefficients(character=character, coefficients=[float(x) for x in job['coefficients']])
            sub.commit_face_state(character=character)
            applied_coeffs = list(sub.get_face_model_coefficients(character=character))
            landmarks = [[v.x, v.y, v.z] for v in sub.get_face_landmarks(character=character)]
            character.preview_material_type = unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE
            sub.commit_skin_settings(character=character, skin_settings=skin_settings_from_job(job['skin']))
            applied_skin = mh.struct_to_dict(character.get_editor_property('skin_settings'))
            applied_eyes = apply_eyes(character, job['eyes'])
            texmap = character.get_editor_property('synthesized_face_textures')
            for key in texmap:
                name = type_name(key)
                if not ANIMATED and 'animated' in name:
                    continue
                rel = f'tex/{name}.png'
                export_texture(texmap[key], os.path.join(out, rel)); files[rel] = None
            actor = sub.spawn_meta_human_actor(character=character)
            comps = {}
            for comp in actor.get_components_by_class(unreal.SkeletalMeshComponent):
                mesh = None
                for prop in ('skeletal_mesh_asset', 'skinned_asset'):
                    try:
                        mesh = comp.get_editor_property(prop)
                        if mesh is not None:
                            break
                    except Exception:      # noqa: BLE001
                        continue
                if mesh is None:
                    continue
                verts, tris = mesh_data(mesh)
                rel = f'meshes/{comp.get_name()}.bin'
                write_mesh_bin(os.path.join(out, rel), verts, tris); files[rel] = None
                comps[comp.get_name()] = dict(n_vertices=len(verts), n_triangles=len(tris) // 3, mesh=mesh.get_path_name(),
                                              world_transform=str(comp.get_world_transform()))
            renders = None
            if job.get('render') and capture is not None:
                try:
                    renders = capture(actor, character, os.path.join(out, 'renders'))
                    for rel in renders.get('files', []):
                        files[rel] = None
                except Exception:      # noqa: BLE001
                    renders = dict(error=traceback.format_exc())
            actor.destroy_actor()
        params = dict(id=sid, kind=job['kind'], preset=job.get('preset'), seed=job['seed'], held_out=job['held_out'], render=job.get('render', False),
                      coefficients=applied_coeffs, coefficients_requested_sha256=hashlib.sha256(json.dumps(job['coefficients']).encode()).hexdigest(),
                      skin=job['skin'], skin_applied=applied_skin, eyes=job['eyes'], eyes_applied=applied_eyes, components=comps, renders=renders)
        mh.write_json(os.path.join(out, 'params.json'), params); files['params.json'] = None
        mh.write_json(os.path.join(out, 'landmarks.json'), dict(landmarks=landmarks)); files['landmarks.json'] = None
        for rel in files:
            files[rel] = sha256(os.path.join(out, rel))
        mh.write_json(os.path.join(out, 'done.json'), dict(ok=True, id=sid, seconds=time.time() - t0, sha256=files, finished=time.time()))
    finally:
        mh.delete_asset(character)


def main():
    batch = json.load(open(mh.script_args()[0]))
    jobs = {}
    with open(batch['jobs_path']) as f:
        for line in f:
            if line.strip():
                j = json.loads(line); jobs[j['id']] = j
    capture = None
    try:
        import capture as cap       # noqa: WPS433  (optional: present once smoke 05 is settled)
        capture = cap.make_capture()
    except Exception as e:      # noqa: BLE001
        mh.log(f'capture unavailable: {e}')
    results = {}
    for sid in batch['ids']:
        done = os.path.join(V1, 'samples', sid, 'done.json')
        if os.path.isfile(done):
            try:
                if json.load(open(done)).get('ok'):
                    results[sid] = 'skip'; continue
            except Exception:      # noqa: BLE001
                pass
        try:
            process(jobs[sid], capture)
            results[sid] = 'ok'
        except Exception:      # noqa: BLE001
            tb = traceback.format_exc(); mh.log(tb)
            mh.write_json(done, dict(ok=False, id=sid, error=tb))
            results[sid] = 'fail'
        mh.log(f'{sid}: {results[sid]}')
    return dict(results=results)


mh.guarded(main, os.path.join(V1, 'logs', f'batch-result-{int(time.time())}.json'))
