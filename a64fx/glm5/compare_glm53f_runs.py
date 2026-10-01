#!/usr/bin/env python3
"""Compare completed resident runs; never treat verifier work as delivered tokens."""
import argparse
import hashlib
import json
import math
import pathlib
import statistics


def read_run(path):
    trials, complete = [], None
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith('GLM53F_BENCH_TRIAL '):
            trials.append(json.loads(line.split(' ', 1)[1]))
        elif line.startswith('GLM53F_BENCH_COMPLETE '):
            if complete is not None:
                raise ValueError('multiple completed runs in one log')
            complete = json.loads(line.split(' ', 1)[1])
    if not complete or complete.get('status') != 'PASS' or len(trials) < 3:
        raise ValueError('need one completed PASS run with at least three timed trials')
    if complete['repetitions'] != len(trials) or [t['trial'] for t in trials] != list(range(len(trials))):
        raise ValueError('missing, duplicated or out-of-order trials')
    shape = (trials[0]['prompt_tokens'], trials[0]['decode_transitions'])
    for t in trials:
        if not t.get('tokens_equal') or (t['prompt_tokens'], t['decode_transitions']) != shape:
            raise ValueError('token or shape mismatch within run')
        for key in ('prefill_seconds', 'decode_seconds', 'prefill_tok_s', 'decode_tok_s'):
            if not isinstance(t[key], (float, int)) or not math.isfinite(t[key]) or t[key] <= 0:
                raise ValueError(f'invalid {key}')
        for phase, tokens in zip(('prefill', 'decode'), shape):
            if not math.isclose(t[f'{phase}_tok_s'] * t[f'{phase}_seconds'], tokens, rel_tol=1e-5):
                raise ValueError(f'inconsistent {phase} token accounting')
        if t['minimum_available_kb'] < 2 * 1024 * 1024:
            raise ValueError('memory guard failed')
    summary = {'prompt_tokens': shape[0], 'decode_transitions': shape[1], 'trials': len(trials)}
    for phase in ('prefill', 'decode'):
        rates = [t[f'{phase}_tok_s'] for t in trials]
        summary[phase] = {'median_tok_s': statistics.median(rates),
                          'min_tok_s': min(rates), 'max_tok_s': max(rates)}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--baseline-ids', required=True)
    p.add_argument('--candidate-ids', required=True)
    p.add_argument('--minimum-improvement', type=float, default=0.05)
    p.add_argument('--output')
    args = p.parse_args()
    if not 0 < args.minimum_improvement < 1:
        p.error('minimum improvement must be between zero and one')
    try:
        baseline, candidate = read_run(args.baseline), read_run(args.candidate)
        if any(baseline[k] != candidate[k] for k in ('prompt_tokens', 'decode_transitions')):
            raise ValueError('baseline and candidate workloads differ')
        a = [int(s) for s in pathlib.Path(args.baseline_ids).read_text().split()]
        b = [int(s) for s in pathlib.Path(args.candidate_ids).read_text().split()]
        if a != b or len(a) != baseline['decode_transitions'] + 1 or any(i < 0 or i >= 154880 for i in a):
            raise ValueError('baseline and candidate generated IDs differ or are incomplete')
        ratio = {phase: candidate[phase]['median_tok_s'] / baseline[phase]['median_tok_s']
                 for phase in ('prefill', 'decode')}
        report = {'baseline': baseline, 'candidate': candidate, 'speedup': ratio,
                  'tokens_equal': True,
                  'token_sha256': hashlib.sha256(('\n'.join(map(str, a)) + '\n').encode()).hexdigest(),
                  'promotion_candidate': max(ratio.values()) >= 1 + args.minimum_improvement and min(ratio.values()) >= 0.98,
                  'target_met': candidate['prefill']['median_tok_s'] >= 2000 and candidate['decode']['median_tok_s'] >= 100}
        result = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.output:
            with open(args.output, 'x') as f:
                f.write(result)
        print(result, end='')
    except (ValueError, KeyError, OSError, TypeError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
