# Dump the skin shading the reference renders use (runs inside UE): the EDITABLE preview materials on a spawned
# character (instance chain, scalar/vector parameters, shading model, subsurface profile), the shipped subsurface
# profiles and the relevant renderer cvars. Derived parameters only. usage (via launch.py): dump_skin_material.py [<sid>]
import json
import os

import unreal

import mh_common as mh
import gen_batch as gb

V1 = os.path.join(mh.OUT_ROOT, 'v1')
OUT = os.path.join(V1, 'presets', 'skin_material.json')
PROFILES = ('/MetaHumanCharacter/Lookdev_UHM/Skin/Materials/SSP_skin_unified',
            '/MetaHumanCharacter/Lookdev_UHM/Eye/Materials/SSP_eye_unified',
            '/MetaHumanCharacter/Lookdev_UHM/Teeth/Materials/SSP_teeth_unified')
ASSETS = ('/MetaHumanCharacter/Lookdev_UHM/Skin/Materials/M_skin_unified',
          '/MetaHumanCharacter/Lookdev_UHM/Skin/Materials/M_skin_unified_UI',
          '/MetaHumanCharacter/Lookdev_UHM/Skin/Materials/MI_skin_unified',
          '/MetaHumanCharacter/Lookdev_UHM/Skin/Materials/MI_skin_unified_MH_preset',
          '/MetaHumanCharacter/Materials/MI_Skin_Head_UI_LOD0',
          '/MetaHumanCharacter/Materials/MI_Skin_Head_UI_LOD0_VT')
CVARS_INT = ('r.Substrate', 'r.SubsurfaceScattering', 'r.SSS.Burley.Quality', 'r.SSS.Quality', 'r.SSS.Checkerboard',
             'r.SSS.HalfRes', 'r.SSS.Burley.AlwaysUpdateParametersFromSeparable', 'r.SSS.Burley.EnableProfileIdCache',
             'r.SSS.SubSurfaceColorAsTansmittanceAtDistance', 'r.DynamicGlobalIlluminationMethod', 'r.ReflectionMethod',
             'r.Shadow.Virtual.Enable', 'r.RayTracing', 'r.Tonemapper.Quality', 'r.DefaultFeature.AutoExposure')
CVARS_FLOAT = ('r.SSS.Scale', 'r.SSS.Burley.MinGenerateMipsTileCount', 'r.SSS.SubSurfaceColorAsTansmittanceAtDistance')


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:      # noqa: BLE001
        return f'error: {e}'


def as_list(v):
    try:
        return [] if isinstance(v, str) else list(v)
    except TypeError:
        return []


def mid_get(m, kind, n):
    for attr in (f'k2_get_{kind}_parameter_value', f'get_{kind}_parameter_value'):
        if hasattr(m, attr):
            return safe(getattr(m, attr), n)
    return 'error: no getter'


def color(v):
    return [v.r, v.g, v.b, v.a] if isinstance(v, unreal.LinearColor) else str(v)


def material_record(m, depth=0):
    mel = unreal.MaterialEditingLibrary
    rec = dict(path=m.get_path_name(), cls=m.get_class().get_name())
    base = safe(m.get_base_material)
    if isinstance(base, unreal.Material):
        rec['base'] = base.get_path_name()
        for prop in ('shading_model', 'blend_mode', 'subsurface_profile', 'two_sided', 'use_material_attributes'):
            v = safe(base.get_editor_property, prop)
            rec[f'base_{prop}'] = v.get_path_name() if isinstance(v, unreal.Object) else str(v)
    if isinstance(m, unreal.MaterialInstance):
        sp = safe(m.get_editor_property, 'subsurface_profile')
        rec['instance_subsurface_profile'] = sp.get_path_name() if isinstance(sp, unreal.Object) else str(sp)
        for k in ('override_subsurface_profile',):
            rec[k] = str(safe(m.get_editor_property, k))
        names = safe(mel.get_scalar_parameter_names, m)
        rec['scalars'] = {str(n): safe(mel.get_material_instance_scalar_parameter_value, m, n) if isinstance(m, unreal.MaterialInstanceConstant)
                          else mid_get(m, 'scalar', n) for n in as_list(names)}
        names = safe(mel.get_vector_parameter_names, m)
        rec['vectors'] = {str(n): color(safe(mel.get_material_instance_vector_parameter_value, m, n) if isinstance(m, unreal.MaterialInstanceConstant)
                                        else mid_get(m, 'vector', n)) for n in as_list(names)}
        names = safe(mel.get_texture_parameter_names, m)
        tex = {}
        for n in as_list(names):
            t = safe(mel.get_material_instance_texture_parameter_value, m, n) if isinstance(m, unreal.MaterialInstanceConstant) else mid_get(m, 'texture', n)
            tex[str(n)] = t.get_path_name() if isinstance(t, unreal.Object) else str(t)
        rec['textures'] = tex
        names = safe(mel.get_static_switch_parameter_names, m)
        if as_list(names) and isinstance(m, unreal.MaterialInstanceConstant):
            rec['switches'] = {str(n): safe(mel.get_material_instance_static_switch_parameter_value, m, n) for n in names}
        parent = safe(m.get_editor_property, 'parent')
        if isinstance(parent, unreal.MaterialInterface) and depth < 6:
            rec['parent'] = material_record(parent, depth + 1)
    elif isinstance(m, unreal.Material):
        names = safe(mel.get_scalar_parameter_names, m)
        rec['scalars'] = {str(n): safe(mel.get_material_default_scalar_parameter_value, m, n) for n in as_list(names)}
        names = safe(mel.get_vector_parameter_names, m)
        rec['vectors'] = {str(n): color(safe(mel.get_material_default_vector_parameter_value, m, n)) for n in as_list(names)}
        names = safe(mel.get_static_switch_parameter_names, m)
        rec['switches'] = {str(n): safe(mel.get_material_default_static_switch_parameter_value, m, n) for n in as_list(names)}
    return rec


def main():
    args = mh.script_args()
    sid = args[0] if args else 'mh_000000'
    out = dict(sample=sid, profiles={}, assets={}, cvars={}, components=[])
    for p in PROFILES:
        a = unreal.EditorAssetLibrary.load_asset(p)
        out['profiles'][p] = mh.struct_to_dict(a.get_editor_property('settings')) if a else None
    for p in ASSETS:
        a = unreal.EditorAssetLibrary.load_asset(p)
        out['assets'][p] = material_record(a) if a else None
    for c in CVARS_INT:
        out['cvars'][c] = safe(unreal.SystemLibrary.get_console_variable_int_value, c)
    for c in CVARS_FLOAT:
        out['cvars'][c + ' (float)'] = safe(unreal.SystemLibrary.get_console_variable_float_value, c)
    params = json.load(open(os.path.join(V1, 'samples', sid, 'params.json')))
    sub = mh.subsystem()
    character = mh.create_character('/Game/MHGen/Dump', sid)
    character.preview_material_type = unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE
    try:
        with mh.Editing(character):
            sub.set_face_model_coefficients(character=character, coefficients=[float(x) for x in params['coefficients']])
            sub.commit_face_state(character=character)
            character.preview_material_type = unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE
            sub.commit_skin_settings(character=character, skin_settings=gb.skin_settings_from_job(params['skin']))
            gb.apply_eyes(character, params['eyes'])
            actor = sub.spawn_meta_human_actor(character=character)
            for comp in actor.get_components_by_class(unreal.MeshComponent):
                mats = safe(comp.get_materials)
                slots = safe(comp.get_material_slot_names)
                out['components'].append(dict(name=comp.get_name(), cls=comp.get_class().get_name(),
                                              slots=[str(s) for s in as_list(slots)],
                                              materials=[material_record(m) if m else None for m in as_list(mats)]))
            actor.destroy_actor()
    finally:
        mh.delete_asset(character)
    return out


mh.guarded(main, OUT)
