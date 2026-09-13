#!/usr/bin/env python3
"""Attribute buffered verifier timings to each layer's dense owner.

Inactive ranks can wait for owner attention inside their broadcast spans.
Summing those rank maxima would count the same work twice. Owner durations
give an approximate serial path; the difference from measured verification
time exposes unaccounted transport and rank skew. Expert rank maxima are
reported separately and must not be added to this path.
"""
import argparse
import json
import math
import re
from pathlib import Path

from spec_report import report as spec_report


PHASES = ('pre', 'attention', 'gate', 'broadcast', 'experts', 'shared_reduce', 'post')


def records(path, rank):
    summary, layers = {}, {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] not in ('VERIFY_TIMING', 'VERIFY_LAYER'):
            raise ValueError('unexpected timing record in ' + str(path))
        row = dict(field.split('=', 1) for field in fields[1:])
        integers = ('rank', 'pos', 'inputs') + (('layer', 'owner') if fields[0] == 'VERIFY_LAYER' else ())
        expected = set(integers) | set(PHASES)
        if fields[0] == 'VERIFY_TIMING':
            expected |= {'begin', 'head'}
        if set(row) != expected or len(fields) != len(expected)+1:
            raise ValueError('invalid timing fields')
        row = {key: int(value) if key in integers else float(value) for key, value in row.items()}
        if row['rank'] != rank or row['pos'] < 0 or not 1 <= row['inputs'] <= 6:
            raise ValueError('invalid timing identity')
        if any(not math.isfinite(value) or value < 0 for key, value in row.items() if key not in integers):
            raise ValueError('invalid timing duration')
        if fields[0] == 'VERIFY_LAYER':
            if not 0 <= row['layer'] < 40 or row['owner'] != row['layer'] % 12:
                raise ValueError('invalid layer owner')
            key, target = (row['pos'], row['layer']), layers
        else:
            key, target = row['pos'], summary
        if key in target:
            raise ValueError('duplicate timing record')
        target[key] = row
    return summary, layers


def report(directory, start, stop):
    measured = spec_report(directory, start, stop)
    data = [records(directory / f'verify-timing.rank{rank:02d}.log', rank) for rank in range(12)]
    pattern = r'SPEC_CYCLE pos=(\d+) drafted=\d+ verified=(\d+) accepted=\d+ emitted=(\d+) '
    selected = [(int(pos), int(verified), int(emitted)) for pos, verified, emitted in
                re.findall(pattern, (directory / 'inference.rank00.log').read_text())
                if int(pos) >= start and int(pos)+int(emitted) <= stop]
    if len(selected) != measured['cycles']:
        raise ValueError('inconsistent selected cycle count')
    sums = dict.fromkeys(('begin',)+PHASES+('head',), 0.0)
    per_layer = [dict.fromkeys(PHASES, 0.0) for _ in range(40)]
    maximum_experts = [0.0]*40
    for pos, verified, _ in selected:
        for summaries, layers in data:
            if pos not in summaries or summaries[pos]['inputs'] != verified:
                raise ValueError('requested cycle missing from bounded timing buffer')
            for layer in range(40):
                if (pos, layer) not in layers or layers[pos, layer]['inputs'] != verified:
                    raise ValueError('incomplete layer timing')
            # Rounded text records can differ by at most 40 half-microseconds.
            for phase in PHASES:
                if abs(sum(layers[pos, layer][phase] for layer in range(40))-summaries[pos][phase]) > 0.000025:
                    raise ValueError('layer timings disagree with rank summary')
        sums['begin'] += data[0][0][pos]['begin']
        sums['head'] += data[11][0][pos]['head']
        for layer in range(40):
            row = data[layer % 12][1][pos, layer]
            for phase in PHASES:
                sums[phase] += row[phase]
                per_layer[layer][phase] += row[phase]
            maximum_experts[layer] += max(item[1][pos, layer]['experts'] for item in data)
    scale = 1000/measured['emitted']
    phases = {key: value*scale for key, value in sums.items()}
    path_ms = sum(phases.values())
    return dict(directory=str(directory), cycles=measured['cycles'], emitted=measured['emitted'],
                first_position=measured['first_position'], last_position=measured['last_position'],
                tokens_per_second=measured['tokens_per_second'],
                measured_phases_ms_per_emitted=measured['phases_ms_per_emitted'],
                owner_path_ms_per_emitted=phases,
                owner_path_total_ms_per_emitted=path_ms,
                verify_minus_owner_path_ms_per_emitted=measured['phases_ms_per_emitted']['verify']-path_ms,
                owner_ffn_envelope_ms_per_emitted=sum(phases[key] for key in ('broadcast', 'experts', 'shared_reduce')),
                nested_max_experts_ms_per_emitted=sum(maximum_experts)*scale,
                layers=[dict(layer=layer, owner=layer % 12,
                             owner_ms_per_emitted={key: value*scale for key, value in per_layer[layer].items()},
                             nested_max_experts_ms_per_emitted=maximum_experts[layer]*scale)
                        for layer in range(40)],
                caveat='Owner path is approximate. Shared/reduce includes routed-expert wait. '
                       'Nested maximum expert durations overlap the owner FFN envelope; do not add them. '
                       'Pre includes Engram and attention mHC; gate includes post-attention and FFN mHC. '
                       'Post includes residual handoff. Durations are per actual emitted token.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--start', type=int, default=1000)
    parser.add_argument('--stop', type=int, default=1105)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.start < 0 or args.stop <= args.start:
        parser.error('invalid position range')
    encoded = json.dumps(report(args.directory, args.start, args.stop), indent=2, allow_nan=False)+'\n'
    if args.json:
        args.json.write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
