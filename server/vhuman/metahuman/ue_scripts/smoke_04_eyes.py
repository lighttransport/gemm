# Smoke 04: commit eye settings and read them back; dump the shipped eye presets (property access, else T3D export).
# Writes smoke/04_eyes.json. Runs inside UE.
import os

import unreal

import mh_common as mh

OUT = os.path.join(mh.OUT_ROOT, 'smoke', '04_eyes.json')
PRESETS = '/MetaHumanCharacter/Tools/EyePresets/EyePresets'
T3D_DIR = os.path.join(mh.OUT_ROOT, 'smoke', '04_presets_t3d')


struct_to_dict = mh.struct_to_dict


def main():
    out = {}
    character = mh.create_character('/Game/MHGen/Smoke', 'Smoke04')
    sub = mh.subsystem()
    with mh.Editing(character):
        eyes = character.get_editor_property('eyes_settings')
        out['default_eyes'] = struct_to_dict(eyes)
        iris = eyes.eye_left.iris
        iris.primary_color_u = 0.15; iris.primary_color_v = 0.6
        iris.limbal_ring_size = 0.8
        eyes.eye_left.iris = iris
        eyes.eye_right.iris = iris
        sub.commit_eyes_settings(character=character, eyes_settings=eyes)
        out['committed_eyes'] = struct_to_dict(character.get_editor_property('eyes_settings'))
    # presets
    asset = unreal.EditorAssetLibrary.load_asset(PRESETS)
    out['presets_asset'] = asset.get_path_name() if asset else None
    if asset:
        try:
            presets = asset.get_editor_property('presets')
            out['presets'] = [struct_to_dict(p) for p in presets]
            out['presets_route'] = 'property'
        except Exception as e:      # noqa: BLE001
            out['presets_property_error'] = str(e)
            try:
                os.makedirs(T3D_DIR, exist_ok=True)
                unreal.AssetToolsHelpers.get_asset_tools().export_assets([PRESETS], T3D_DIR)
                out['presets_route'] = 't3d'
                out['t3d_files'] = os.listdir(T3D_DIR)
            except Exception as e2:      # noqa: BLE001
                out['presets_t3d_error'] = str(e2)
    mh.delete_asset(character)
    return out


mh.guarded(main, OUT)
