# Smoke 01: create a MetaHumanCharacter, open it for editing, read the face model coefficients and landmarks,
# and read the coefficients of the shipped presets. Runs inside UE. Writes smoke/01_create.json.
import os

import unreal

import mh_common as mh

OUT = os.path.join(mh.OUT_ROOT, 'smoke', '01_create.json')
PRESETS = '/MetaHumanCharacter/Optional/Presets'


def read_character(character):
    sub = mh.subsystem()
    with mh.Editing(character):
        coeffs = list(sub.get_face_model_coefficients(character=character))
        patches = mh.parse_coefficients(coeffs)
        lm = sub.get_face_landmarks(character=character)
        landmarks = [[v.x, v.y, v.z] for v in lm]
    return dict(n_coefficients=len(coeffs), n_patches=len(patches), n_pca_per_patch=[len(p['weights']) for p in patches],
                patch_scale=[p['scale'] for p in patches], patch_translation=[p['translation'] for p in patches],
                n_landmarks=len(landmarks), landmark_bbox=[[min(c[i] for c in landmarks) for i in range(3)],
                                                           [max(c[i] for c in landmarks) for i in range(3)]] if landmarks else None,
                coefficients=coeffs, landmarks=landmarks)


def main():
    out = {}
    character = mh.create_character('/Game/MHGen/Smoke', 'Smoke01')
    out['created'] = character.get_path_name()
    out['archetype'] = read_character(character)
    # presets
    names = [p for p in unreal.EditorAssetLibrary.list_assets(PRESETS, recursive=False, include_folder=False)]
    out['preset_assets'] = names
    presets = {}
    for path in names[:3]:          # a few only in the smoke test; gen reads all
        asset = unreal.EditorAssetLibrary.load_asset(path)
        if asset is None or not isinstance(asset, unreal.MetaHumanCharacter):
            presets[path] = dict(error='not a MetaHumanCharacter')
            continue
        try:
            d = read_character(asset)
            presets[path] = {k: v for k, v in d.items() if k not in ('coefficients', 'landmarks')}
            presets[path]['first_weights'] = d['coefficients'][:12]
        except Exception as e:      # noqa: BLE001
            presets[path] = dict(error=str(e))
    out['presets'] = presets
    # perturb one patch and commit (sculpt API round trip), then read back
    sub = mh.subsystem()
    with mh.Editing(character):
        c = list(sub.get_face_model_coefficients(character=character))
        p = mh.parse_coefficients(c)
        p[0]['weights'] = [w + 2.0 for w in p[0]['weights']]
        sub.set_face_model_coefficients(character=character, coefficients=mh.flatten_coefficients(p))
        sub.commit_face_state(character=character)
        c2 = list(sub.get_face_model_coefficients(character=character))
        lm2 = sub.get_face_landmarks(character=character)
    out['roundtrip'] = dict(changed=sum(1 for a, b in zip(c, c2) if abs(a - b) > 1e-6), n=len(c2),
                            landmark_delta_max=max(max(abs(v.x - w[0]), abs(v.y - w[1]), abs(v.z - w[2]))
                                                   for v, w in zip(lm2, out['archetype']['landmarks'])))
    mh.delete_asset(character)
    return out


mh.guarded(main, OUT)
