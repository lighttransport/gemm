# Dump the face model coefficients, landmarks, skin and eye settings of every shipped preset character (derived
# parameters only). Writes v1/presets/face_presets.json. Runs inside UE.
import os

import unreal

import mh_common as mh

OUT = os.path.join(mh.OUT_ROOT, 'v1', 'presets', 'face_presets.json')
PRESETS = '/MetaHumanCharacter/Optional/Presets'


def main():
    sub = mh.subsystem()
    out = dict(presets={})
    for path in unreal.EditorAssetLibrary.list_assets(PRESETS, recursive=False, include_folder=False):
        asset = unreal.EditorAssetLibrary.load_asset(path)
        if not isinstance(asset, unreal.MetaHumanCharacter):
            continue
        rec = {}
        try:
            with mh.Editing(asset):
                rec['coefficients'] = list(sub.get_face_model_coefficients(character=asset))
                rec['landmarks'] = [[v.x, v.y, v.z] for v in sub.get_face_landmarks(character=asset)]
            rec['skin_settings'] = mh.struct_to_dict(asset.get_editor_property('skin_settings'))
            rec['eyes_settings'] = mh.struct_to_dict(asset.get_editor_property('eyes_settings'))
            try:
                rec['head_model_settings'] = mh.struct_to_dict(asset.get_editor_property('head_model_settings'))
            except Exception:      # noqa: BLE001
                pass
        except Exception as e:      # noqa: BLE001
            rec['error'] = str(e)
        out['presets'][path.split('/')[-1].split('.')[0]] = rec
        mh.log(f'preset {path}: {"ok" if "coefficients" in rec else rec.get("error")}')
    return out


mh.guarded(main, OUT)
