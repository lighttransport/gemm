# Runs inside the Unreal Editor's Python (3.11, no numpy). Shared helpers for the vhuman MetaHuman scripts.
# Patterns follow Engine/Plugins/MetaHuman/MetaHumanCharacter/Content/Python/metahuman_character_test_utils.py.
import json
import os
import sys
import time
import traceback

import unreal

OUT_ROOT = os.environ.get('VHUMAN_MH_OUT', '/mnt/nvme01/data/vhuman-mh')
SUBSYSTEM = unreal.MetaHumanCharacterEditorSubsystem


def log(msg):
    unreal.log(f'[vhuman-mh] {msg}')


def subsystem():
    return unreal.get_editor_subsystem(SUBSYSTEM)


def script_args():
    """Arguments after the script path (UnrealEditor-Cmd -ExecutePythonScript="script.py a b")."""
    return sys.argv[1:]


def _json_default(o):
    try:
        return [x for x in o]            # unreal.Array / Set / Map keys
    except TypeError:
        return str(o)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=1, sort_keys=True, default=_json_default)
    log(f'wrote {path}')


def finish(path, data, ok=True):
    """Write the result file (done.json semantics) and quit the editor. Never raises."""
    data = dict(data, ok=bool(ok), finished=time.time())
    try:
        write_json(path, data)
    finally:
        unreal.SystemLibrary.quit_editor()


def guarded(main, result_path):
    """Run main(); on exception write the traceback into the result file and quit."""
    try:
        data = main() or {}
        finish(result_path, data, ok=True)
    except Exception:            # noqa: BLE001
        tb = traceback.format_exc()
        log(tb)
        finish(result_path, dict(error=tb), ok=False)


def create_character(package_path, asset_name):
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    _, unique = tools.create_unique_asset_name(base_package_name=f'{package_path}/{asset_name}', suffix='')
    unique = unique.split('/')[-1]
    character = tools.create_asset(asset_name=unique, package_path=package_path, asset_class=unreal.MetaHumanCharacter,
                                   factory=unreal.new_object(type=unreal.MetaHumanCharacterFactoryNew))
    if character is None:
        raise RuntimeError(f'failed to create MetaHumanCharacter {package_path}/{unique}')
    return character


class Editing:
    """with Editing(character): ... -> try_add_object_to_edit / remove_object_to_edit."""

    def __init__(self, character):
        self.character = character

    def __enter__(self):
        if not subsystem().try_add_object_to_edit(character=self.character):
            raise RuntimeError(f'try_add_object_to_edit failed for {self.character.get_name()}')
        return self

    def __exit__(self, *exc):
        if subsystem().is_object_added_for_editing(character=self.character):
            subsystem().remove_object_to_edit(character=self.character)
        return False


def delete_asset(asset):
    path = asset.get_path_name()
    if not unreal.EditorAssetLibrary.delete_loaded_asset(asset):
        log(f'could not delete {path}')


def editor_world():
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()


def struct_to_dict(s, depth=0):
    """Recursively read a USTRUCT's editor properties into plain Python (best effort, derived parameters only)."""
    if depth > 4:
        return str(s)
    out = {}
    for name in [n for n in dir(s) if not n.startswith('_')]:
        try:
            v = s.get_editor_property(name)
        except Exception:      # noqa: BLE001
            continue
        if isinstance(v, (int, float, bool, str)):
            out[name] = v
        elif isinstance(v, unreal.LinearColor):
            out[name] = [v.r, v.g, v.b, v.a]
        elif isinstance(v, unreal.StructBase):
            out[name] = struct_to_dict(v, depth + 1)
        elif isinstance(v, unreal.EnumBase):
            out[name] = str(v)
        else:
            out[name] = str(v)
    return out


def parse_coefficients(coeffs):
    """[nPatches] then per patch: scale(1), quat(4), translation(3), nPCA(1), weights[nPCA] -> list of dicts."""
    c = list(coeffs)
    n = int(round(c[0]))
    i = 1
    patches = []
    for p in range(n):
        scale = c[i]; quat = c[i + 1:i + 5]; trans = c[i + 5:i + 8]; k = int(round(c[i + 8]))
        w = c[i + 9:i + 9 + k]
        patches.append(dict(scale=scale, quat=quat, translation=trans, weights=w))
        i += 9 + k
    if i != len(c):
        raise ValueError(f'coefficient layout mismatch: consumed {i} of {len(c)}')
    return patches


def flatten_coefficients(patches):
    out = [float(len(patches))]
    for p in patches:
        out += [p['scale']] + list(p['quat']) + list(p['translation']) + [float(len(p['weights']))] + list(p['weights'])
    return out
