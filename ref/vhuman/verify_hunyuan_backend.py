"""Compare a strict repository-GEMM run to a previously verified full reference.

Reuses independent reference tensors only after checking the complete recipe,
model receipts, prepared image/vision inputs and byte-identical input noise.
Both generation manifests remain in the audit trail. No Torch is imported.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cuda.hunyuan_video15_native.generate import digest, atomic_json
from ref.hunyuan_video15_native.verify import convert, pipeline_names, validate_pipeline_shapes
from ref.hunyuan_video15.compare import compare, compare_frames

RECIPE = ('task', 'preset', 'prompt', 'negative_prompt', 'frames', 'fps', 'width', 'height',
          'seed', 'steps', 'cfg', 'flow_shift', 'vision_profile', 'image_sha256',
          'upstream_reference_revision', 'model', 'verified_components')


def matching_recipe(candidate, baseline):
    for key in RECIPE:
        if key not in candidate or key not in baseline or candidate[key] != baseline[key]:
            raise ValueError('reference input/recipe differs: ' + key)
    metrics = candidate.get('metrics', {})
    if (candidate.get('gemm') != 'repo' or candidate.get('gemm_fallback') != 'error' or
            metrics.get('repo_gemm_calls', 0) <= 0 or metrics.get('cublas_gemm_calls') != 0 or
            metrics.get('fallback_gemm_calls') != 0):
        raise ValueError('candidate is not strict repository GEMM')


def verify(candidate_run, candidate_actual, baseline_run, baseline_actual, baseline_reference, output):
    candidate_run, candidate_actual, baseline_run, baseline_actual, baseline_reference, output = map(
        Path, (candidate_run, candidate_actual, baseline_run, baseline_actual, baseline_reference, output))
    candidate = json.loads((candidate_run/'manifest.json').read_text())
    baseline = json.loads((baseline_run/'manifest.json').read_text())
    matching_recipe(candidate, baseline)
    names = pipeline_names(candidate)
    if names != pipeline_names(baseline):
        raise ValueError('pipeline component sets differ')
    baseline_hash = digest(baseline_run/'manifest.json')
    accepted = json.loads((baseline_reference/'compare_parity.json').read_text())
    if (accepted.get('scope') != 'pipeline' or accepted.get('pass') is not True or
            accepted.get('generation_manifest_sha256') != baseline_hash or
            set(accepted.get('results', {})) != set(names) or
            not all(v.get('pass') is True for v in accepted['results'].values()) or
            len(accepted.get('decoded_frames') or []) != 81 or
            not all(v.get('pass') is True for v in accepted['decoded_frames'])):
        raise ValueError('baseline lacks accepted complete independent reference')
    reference = baseline_reference/'reference'
    receipts = json.loads((reference/'receipts.json').read_text())
    expected_weights = {k:v['sha256'] for k,v in accepted['reference_weights'].items()}
    for name in names:
        actual_hash = digest(baseline_actual/(name+'.f32'))
        ref_hash = digest(reference/(name+'.npy'))
        receipt = receipts.get(name, {})
        if (actual_hash != accepted['native_capture_sha256'].get(name) or
                ref_hash != accepted['reference_capture_sha256'].get(name) or
                ref_hash != receipt.get('sha256') or
                receipt.get('provenance', {}).get('generation_manifest_sha256') != baseline_hash or
                receipt.get('provenance', {}).get('weights') != expected_weights or
                receipt.get('provenance', {}).get('upstream_revision') != candidate['upstream_reference_revision']):
            raise ValueError('stale baseline capture/reference: ' + name)
    if digest(candidate_actual/'noise_input.f32') != digest(baseline_actual/'noise_input.f32'):
        raise ValueError('reference input noise differs')
    if candidate['task'] == 'i2v':
        for name in ('input.png', 'vision_pixels.f32'):
            if digest(candidate_run/name) != digest(baseline_run/name):
                raise ValueError('prepared reference image differs: ' + name)
    convert(candidate_actual)
    validate_pipeline_shapes(candidate_actual, names)
    validate_pipeline_shapes(reference, names)
    results = compare(reference, candidate_actual, names)
    frames = compare_frames(reference, candidate_actual)
    passed = all(v['pass'] for v in results.values()) and len(frames) == 81 and all(v['pass'] for v in frames)
    report = dict(scope='pipeline_reused_independent_reference', passed=passed, results=results,
                  decoded_frames=frames, reference_dtype=accepted['reference_dtype'],
                  candidate_manifest_sha256=digest(candidate_run/'manifest.json'),
                  baseline_manifest_sha256=baseline_hash,
                  baseline_acceptance_sha256=digest(baseline_reference/'compare_parity.json'),
                  verifier_sha256=digest(__file__), exact_input_noise=True, matched_recipe=list(RECIPE),
                  metrics=candidate['metrics'],
                  native_capture_sha256={n:digest(candidate_actual/(n+'.f32')) for n in names},
                  reference_capture_sha256={n:digest(reference/(n+'.npy')) for n in names})
    atomic_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-run', 'candidate-actual', 'baseline-run', 'baseline-actual', 'baseline-reference', 'output'):
        parser.add_argument('--'+name, required=True)
    result = verify(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
