#!/usr/bin/env python3
"""Report resident MTP sweeps only after complete greedy-reference checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_ids(path):
    ids = [int(s) for s in Path(path).read_text().split()]
    if not ids or any(i < 0 or i >= 154880 for i in ids):
        raise ValueError('invalid or empty token IDs')
    return ids


def report(log, prefix, cycles, depths, reference_path=None):
    reference = read_ids(reference_path or str(prefix) + '.greedy')
    groups, complete = {}, None
    for line in Path(log).read_text().splitlines():
        if line.startswith('GLM53F_SPEC_COMPLETE '):
            if complete is not None:
                raise ValueError('multiple completed runs')
            complete = json.loads(line.split(' ', 1)[1])
        if not line.startswith('GLM53F_SPEC_TRIAL '):
            continue
        t = json.loads(line.split(' ', 1)[1])
        for key in ('variant', 'trial', 'drafts', 'delivered', 'accepted', 'proposed', 'reference_checked', 'minimum_available_kb'):
            if type(t[key]) is not int:
                raise ValueError(f'invalid integer {key}')
        depth = t['drafts']
        if t['status'] != 'PASS' or t['variant'] < 0 or depth not in depths or t['trial'] < -1:
            raise ValueError('failed trial or invalid variant/depth')
        if t['warmup'] is not (t['trial'] == -1):
            raise ValueError('invalid warmup accounting')
        if t['proposed'] != cycles * depth or not 0 <= t['accepted'] <= t['proposed'] or t['delivered'] != 2 * cycles + t['accepted']:
            raise ValueError('incomplete cycles or inconsistent delivered-token accounting; use --ignore-eos')
        if t['reference_checked'] != t['delivered'] or t['minimum_available_kb'] < 2 * 1024 * 1024:
            raise ValueError('incomplete reference check or insufficient memory')
        for key in ('seconds', 'tok_s'):
            if type(t[key]) not in (int, float) or not math.isfinite(t[key]) or t[key] <= 0:
                raise ValueError(f'invalid {key}')
        if not math.isclose(t['seconds'] * t['tok_s'], t['delivered'], rel_tol=1e-5):
            raise ValueError('inconsistent elapsed time or rate')
        path = f'{prefix}.v{t["variant"]}.d{depth}.trial{t["trial"]}'
        ids = read_ids(path)
        if len(ids) != t['delivered'] or ids != reference[:len(ids)]:
            raise ValueError('generated IDs differ from greedy reference or are incomplete')
        groups.setdefault((t['variant'], depth), []).append(t)
    if not groups or {d for _, d in groups} != set(depths):
        raise ValueError('missing draft depths')
    if not complete or complete['status'] != 'PASS' or complete['cycles'] != cycles or complete['variants'] != len(groups):
        raise ValueError('missing or inconsistent completion record')
    if len({v for v, _ in groups}) != len(groups):
        raise ValueError('variant changed draft depth')
    summaries = []
    for (variant, depth), trials in sorted(groups.items()):
        if len(trials) < 4 or len(trials) != complete['repetitions'] + 1 or [t['trial'] for t in trials] != list(range(-1, len(trials) - 1)):
            raise ValueError('require one warmup and at least three ordered timed trials per variant')
        timed = trials[1:]
        rates = [t['tok_s'] for t in timed]
        summaries.append({'variant': variant, 'depth': depth, 'timed_trials': len(timed),
            'median_tok_s': statistics.median(rates), 'min_tok_s': min(rates), 'max_tok_s': max(rates),
            'accepted': sum(t['accepted'] for t in timed), 'proposed': sum(t['proposed'] for t in timed),
            'minimum_available_kb': min(t['minimum_available_kb'] for t in trials),
            'decode_target_met': statistics.median(rates) >= 100})
    return {'cycles': cycles, 'greedy_exact': True,
        'reference_sha256': hashlib.sha256(('\n'.join(map(str, reference)) + '\n').encode()).hexdigest(),
        'variants': summaries}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', required=True)
    p.add_argument('--ids-prefix', required=True)
    p.add_argument('--reference-ids')
    p.add_argument('--cycles', type=int, default=128)
    p.add_argument('--depths', type=int, nargs='+', default=[1, 2, 3, 4])
    p.add_argument('--output')
    a = p.parse_args()
    if not 1 <= a.cycles <= 32768 or not set(a.depths) <= {1, 2, 3, 4}:
        p.error('invalid cycles or depths')
    try:
        result = json.dumps(report(a.log, a.ids_prefix, a.cycles, a.depths, a.reference_ids), indent=2, allow_nan=False) + '\n'
        if a.output:
            with open(a.output, 'x') as f:
                f.write(result)
        print(result, end='')
    except (ValueError, KeyError, OSError, TypeError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
