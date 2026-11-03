# Dump the shipped eye presets as JSON (derived parameters only). Shares the reader with smoke_04. Runs inside UE.
import os

import unreal

import mh_common as mh
PRESETS = '/MetaHumanCharacter/Tools/EyePresets/EyePresets'
struct_to_dict = mh.struct_to_dict

OUT = os.path.join(mh.OUT_ROOT, 'v1', 'presets', 'eye_presets.json')


def main():
    asset = unreal.EditorAssetLibrary.load_asset(PRESETS)
    if asset is None:
        raise RuntimeError('eye presets asset not found')
    presets = asset.get_editor_property('presets')
    dicts = [struct_to_dict(p) for p in presets]
    out = dict(asset=asset.get_path_name(), presets=dicts, n=len(dicts))
    if not any(dicts):
        # FMetaHumanCharacterEyePreset is not BlueprintType: fall back to a T3D text export and parse it on the host
        t3d_dir = os.path.join(mh.OUT_ROOT, 'v1', 'presets', 'eye_presets_t3d')
        os.makedirs(t3d_dir, exist_ok=True)
        unreal.AssetToolsHelpers.get_asset_tools().export_assets([PRESETS], t3d_dir)
        out['t3d_dir'] = t3d_dir; out['t3d_files'] = os.listdir(t3d_dir)
        # also try the generic exporter task (COPY format) for completeness
        try:
            task = unreal.AssetExportTask()
            task.object = asset; task.filename = os.path.join(t3d_dir, 'EyePresets.copy'); task.automated = True; task.replace_identical = True; task.prompt = False
            unreal.Exporter.run_asset_export_task(task)
            out['copy_export'] = os.path.isfile(task.filename)
        except Exception as e:      # noqa: BLE001
            out['copy_export_error'] = str(e)
    return out


mh.guarded(main, OUT)
