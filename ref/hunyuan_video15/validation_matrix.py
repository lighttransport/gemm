"""Framework-free coverage audit for independent Hunyuan pipeline comparisons.

Create a matrix for three portraits, six expressions, two seeds, two profiles
and two lengths. Place each generation manifest and independent parity report
under runs/<case-id>/. Missing, stale, or wrong-precision evidence cannot pass.
This tool does not generate videos or treat numerical parity as visual quality.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

EXPRESSIONS = ('smile', 'laugh', 'surprise', 'sad', 'angry', 'blink')
COMPONENTS = ('qwen_hidden', 'siglip_hidden', 'vae_encoded', 'dit_first', 'latent_final', 'vae_decoded')
SCOPES = ('independent_official_component_chain_with_matched_noise',
          'independent_official_component_chain_with_reused_reference')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def matrix(portraits, seeds=(42, 43)):
    if len(portraits) != 3 or len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError('matrix requires three distinct portraits and two distinct seeds')
    images = [(str(Path(p).resolve()), digest(p)) for p in portraits]
    if len({value for _, value in images}) != 3:
        raise ValueError('portrait contents must differ')
    cases = []
    for (path, sha), expression, seed, preset, frames in itertools.product(
            images, EXPRESSIONS, seeds, ('fast12', 'quality'), (81, 121)):
        cases.append(dict(id=f'{sha[:12]}-{expression}-{seed}-{preset}-{frames}', image=path,
                          image_sha256=sha, expression=expression, seed=seed, preset=preset,
                          frames=frames, task='i2v', width=480, height=848))
    return {'format': 'vhuman.hv15_validation_matrix.v1', 'cases': cases}


def audit_case(case, root, encoder_dtype, backend):
    directory = root / case['id']
    manifest_path, report_path = directory / 'manifest.json', directory / 'parity.json'
    if not manifest_path.is_file() or not report_path.is_file():
        return {'id': case['id'], 'status': 'missing'}
    try:
        manifest = json.loads(manifest_path.read_text())
        report = json.loads(report_path.read_text())
        if report.get('scope') not in SCOPES or report.get('generation_manifest_sha256') != digest(manifest_path):
            raise ValueError('missing independent reference scope or stale generation receipt')
        if manifest.get('backend') != backend:
            raise ValueError('backend differs; legacy parity cannot certify the repository port')
        for key in ('task', 'preset', 'frames', 'seed', 'width', 'height', 'image_sha256'):
            if manifest.get(key) != case[key]:
                raise ValueError('generation differs: ' + key)
        # Expression tags live in the request; require explicit recorded intent.
        if manifest.get('request', {}).get('expression', manifest.get('expression')) != case['expression']:
            raise ValueError('expression intent is missing or differs')
        precision = report.get('reference_precision', {})
        for key in ('qwen', 'google_siglip'):
            if not str(precision.get(key, '')).startswith(encoder_dtype + '_'):
                raise ValueError('encoder precision differs: ' + key)
        components = report.get('results', {})
        if not all(components.get(name, {}).get('pass') is True for name in COMPONENTS):
            raise ValueError('missing or failed component comparison')
        frames = report.get('frame_errors', [])
        if (len(frames) != case['frames'] or [v.get('frame') for v in frames] != list(range(case['frames'])) or
                not all(v.get('pass') is True for v in frames) or report.get('pass') is not True):
            raise ValueError('missing or failed per-frame comparison')
        return {'id': case['id'], 'status': 'passed', 'report_sha256': digest(report_path)}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return {'id': case['id'], 'status': 'failed', 'reason': str(error)}


def audit(plan, root, encoder_dtype='float32', backend='hv15n_cuda_experimental'):
    if plan.get('format') != 'vhuman.hv15_validation_matrix.v1' or not plan.get('cases'):
        raise ValueError('invalid or empty matrix')
    cases = plan['cases']
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('duplicate case ids')
    portraits = {c['image_sha256'] for c in cases}
    seeds = {c['seed'] for c in cases}
    expected = set(itertools.product(portraits, EXPRESSIONS, seeds, ('fast12', 'quality'), (81, 121)))
    observed = {(c['image_sha256'], c['expression'], c['seed'], c['preset'], c['frames']) for c in cases}
    if len(portraits) != 3 or len(seeds) != 2 or len(cases) != 144 or observed != expected:
        raise ValueError('matrix must cover all 144 combinations')
    for case in cases:
        if Path(case['id']).name != case['id'] or case['id'] in ('.', '..'):
            raise ValueError('invalid case id')
    rows = [audit_case(case, Path(root), encoder_dtype, backend) for case in cases]
    counts = {state: sum(r['status'] == state for r in rows) for state in ('passed', 'failed', 'missing')}
    return dict(format='vhuman.hv15_validation_coverage.v1', backend=backend,
                encoder_dtype=encoder_dtype, required=len(rows), **counts, cases=rows,
                passed_all=counts['passed'] == len(rows), visual_quality='not_evaluated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    create.add_argument('--portraits', nargs=3, type=Path, required=True)
    create.add_argument('--seeds', nargs=2, type=int, default=(42, 43))
    create.add_argument('--out', type=Path, required=True)
    check = sub.add_parser('audit')
    check.add_argument('--matrix', type=Path, required=True)
    check.add_argument('--runs', type=Path, required=True)
    check.add_argument('--encoder-dtype', choices=('float32', 'float16'), default='float32')
    check.add_argument('--backend', default='hv15n_cuda_experimental')
    check.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = matrix(args.portraits, args.seeds) if args.command == 'create' else audit(
        json.loads(args.matrix.read_text()), args.runs, args.encoder_dtype, args.backend)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))
    return 0 if args.command == 'create' or result['passed_all'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
