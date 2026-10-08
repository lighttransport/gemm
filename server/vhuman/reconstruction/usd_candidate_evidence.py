"""Attach a validated reconstruction evidence subset to a sampled USD bundle.

Model weights are excluded. Source geometry, portrait, appearance and edit masks
remain original-resolution evidence, independently of the USD shader textures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

FILENAME = 'candidate_evidence.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(bundle):
    bundle = Path(bundle).resolve()
    report = json.loads((bundle/'report.json').read_text())
    if digest(bundle/'head.usdc') != report.get('usd_sha256'):
        raise ValueError('USD content hash mismatch')
    path = bundle/FILENAME
    if report.get('candidate_evidence_sha256') != digest(path):
        raise ValueError('candidate evidence receipt hash mismatch')
    record = json.loads(path.read_text())
    files = record.get('files', {})
    if (record.get('schema') != 'vhuman.usd_candidate_evidence.v1'
            or not isinstance(files, dict)
            or not {'manifest.json', 'geometry.npz', 'portrait.png'} <= files.keys()):
        raise ValueError('incomplete candidate evidence')
    root = bundle/'candidate_evidence'
    if root.is_symlink():raise ValueError('external candidate evidence directory')
    for name, spec in files.items():
        asset = root/name
        if (Path(name).name != name or '\\' in name or asset.is_symlink()
                or not asset.resolve().is_relative_to(root)
                or not asset.is_file() or asset.stat().st_size != spec['bytes']
                or digest(asset) != spec['sha256']):
            raise ValueError('candidate evidence file mismatch: '+name)
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest['geometry_sha256'] != report['candidate_geometry_sha256']
            or digest(root/'geometry.npz') != manifest['geometry_sha256']
            or digest(root/'portrait.png') != manifest['portrait_sha256']
            or digest(root/'manifest.json') != digest(bundle/'candidate_manifest.json')):
        raise ValueError('candidate evidence does not match USD source')
    return dict(passed=True, files=len(files), geometry_sha256=manifest['geometry_sha256'])


def attach(candidate, bundle):
    from .provenance import validate_candidate
    candidate, bundle = Path(candidate).resolve(), Path(bundle).resolve()
    manifest = validate_candidate(candidate)
    report = json.loads((bundle/'report.json').read_text())
    if (not report.get('passed') or report.get('candidate_geometry_sha256') != manifest['geometry_sha256']
            or digest(candidate/'manifest.json') != digest(bundle/'candidate_manifest.json')
            or digest(bundle/'head.usdc') != report.get('usd_sha256')):
        raise ValueError('candidate does not match verified USD bundle')
    root = bundle/'candidate_evidence'
    if root.exists():raise ValueError('candidate evidence already exists')
    root.mkdir()
    names = ['manifest.json', 'geometry.npz', 'portrait.png', 'generated_skin.json', 'skin_material.json',
             'ear_rebake_mask.png', 'local_color_edit_mask.png', 'skin_prior_transfer.png',
             'skin_source_core_mask_0.png', 'skin_strict_exclusion_0.png', 'local_color_report.json']
    names += [p.name for p in sorted(candidate.glob('skin_*.png'))]
    files = {}
    for name in sorted(set(names)):
        source = candidate/name
        if not source.is_file():continue
        shutil.copyfile(source, root/name)
        files[name] = dict(sha256=digest(root/name), bytes=(root/name).stat().st_size)
    validate_candidate(root)
    record = dict(schema='vhuman.usd_candidate_evidence.v1', files=files,
        limitation='Source evidence subset; inherited capture reports and model weights are excluded. Single-photo estimates and authored appearance remain priors; redistribution terms require separate receipts.')
    (bundle/FILENAME).write_text(json.dumps(record, indent=2)+'\n')
    report['candidate_evidence_sha256'] = digest(bundle/FILENAME)
    (bundle/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return verify(bundle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--bundle', required=True)
    args = parser.parse_args()
    print(json.dumps(attach(args.candidate, args.bundle)))
