"""Provenance for the MetaHuman-derived dataset and priors.

Dataset manifest format `vhuman.metahuman_dataset.v1`; prior/artifact report format `vhuman.metahuman_prior.v1`.
Records the UE build, hashes of the plugin model data the generator depends on, the licence basis (placeholders the
user fills from the EULA text), the sampling seed and held-out ids, and per-sample parameter vectors and file hashes.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

from . import DATASET_ROOT, MH_PLUGIN, MODELS_ROOT, UE_EDITOR_CMD, UE_ROOT

DATASET_FORMAT = 'vhuman.metahuman_dataset.v1'
PRIOR_FORMAT = 'vhuman.metahuman_prior.v1'
TS_DIR = MH_PLUGIN / 'Content/Optional/TextureSynthesis/TS-1.3-F_UE_res-1024_nchr-153'
IDENTITY_DIR = MH_PLUGIN / 'Content/Face/IdentityTemplate'
PLUGIN_FILES = {
    'metahuman_character_uplugin': MH_PLUGIN / 'MetaHumanCharacter.uplugin',
    'ts_params': TS_DIR / 'ts_params.json',
    'ts_texture_attributes': TS_DIR / 'texture_attributes.json',
    'ts_archive': TS_DIR / 'compressed.ar',
    'identity_geo_and_bindpose_pca': IDENTITY_DIR / 'rig_calibration_models/geo_and_bindpose.pca',
    'identity_hf_variants': IDENTITY_DIR / 'hf_variants.binary',
    'identity_masks_face': IDENTITY_DIR / 'masks_face.json',
    'identity_face_landmarks': IDENTITY_DIR / 'face_landmarks.json',
    'identity_symmetry': IDENTITY_DIR / 'symmetry.json',
    'eye_presets': MH_PLUGIN / 'Content/Tools/EyePresets/EyePresets.uasset',
}
LICENCE = dict(name='Unreal Engine EULA', revision='<placeholder: EULA revision date>', clause='<placeholder: section permitting AI training on MetaHuman-generated data>',
               url='<placeholder>', local_only=True, cloud_services_used=False, raw_assets_redistributed=False,
               note='Only statistics derived from locally generated characters leave the dataset root; raw Epic assets/model data are never copied into the repository or artifacts.')


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def code_sha256():
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        h.update(p.relative_to(root).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[3]).stdout.strip()
    except Exception:      # noqa: BLE001
        return None


def ue_record():
    return dict(root=str(UE_ROOT), build_version=json.loads((UE_ROOT / 'Engine/Build/Build.version').read_text()),
                editor_cmd_sha256=sha256(UE_EDITOR_CMD))


def plugin_content_record(hash_archive=False):
    rec = {}
    for key, path in PLUGIN_FILES.items():
        if not path.is_file():
            rec[key] = dict(path=str(path), present=False); continue
        entry = dict(path=str(path), size=path.stat().st_size)
        if key != 'ts_archive' or hash_archive:
            entry['sha256'] = sha256(path)
        rec[key] = entry
    return rec


def dataset_manifest(version='v1', *, sampling=None, frames=None, samples=None, hash_archive=False):
    return dict(format=DATASET_FORMAT, created=time.strftime('%Y-%m-%dT%H:%M:%S'), root=str(DATASET_ROOT / version),
                generator=dict(git_commit=git_commit(), code_sha256=code_sha256()), ue=ue_record(),
                plugin_content=plugin_content_record(hash_archive), licence=LICENCE, sampling=sampling or {}, frames=frames or {},
                samples=samples or [])


def prior_report(name, *, dataset_manifest_sha256, sample_ids_used, held_out_ids, weights_path, derived_from, limitations, extra=None):
    rep = dict(format=PRIOR_FORMAT, name=name, created=time.strftime('%Y-%m-%dT%H:%M:%S'), code_sha256=code_sha256(), git_commit=git_commit(),
               dataset_manifest_sha256=dataset_manifest_sha256, sample_ids_used=list(sample_ids_used), held_out_ids=list(held_out_ids),
               weights_sha256=sha256(weights_path), weights_path=str(weights_path), derived_from=derived_from, licence=LICENCE,
               limitations=list(limitations), default_enabled=False)
    if extra:
        rep.update(extra)
    return rep


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + '\n')
    return path


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Write the dataset manifest skeleton (no samples) or hash the plugin content.')
    ap.add_argument('--out', default=str(DATASET_ROOT / 'v1' / 'manifest.json'))
    ap.add_argument('--hash-archive', action='store_true', help='also sha256 the 985 MB texture archive')
    a = ap.parse_args()
    m = dataset_manifest(hash_archive=a.hash_archive)
    print(json.dumps({k: v for k, v in m.items() if k in ('ue', 'plugin_content', 'generator')}, indent=1)[:3000])
    write_json(a.out, m)


if __name__ == '__main__':
    main()
