#!/usr/bin/env python3
"""Measure ordinary decode over an explicit history range on all twelve ranks.

Optional baseline comparison checks every input/next-token triple. It is a
regression check, not a numerical logit comparison or checkpoint validation.
Use spec_report.py for actual emitted-token speculative throughput.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from spec_report import percentile


def triples(text):
    return re.findall(r'TOKEN pos=(\d+) input=(\d+) next=(\d+)', text)


def report(directory, start, stop, baseline=None):
    text = (directory/'inference.rank00.log').read_text()
    if 'SPEC_CYCLE' in text or 'SPEC_FINISHED' in text:
        raise ValueError('use spec_report.py for speculative decode')
    if 'VERIFY_READY' in text:
        raise ValueError('batched fixed-input replay is not ordinary decode throughput')
    if (directory/'args.json').exists():
        args = json.loads((directory/'args.json').read_text())
        for flag in ('--speculate', '--verify-replay-batch'):
            values = [args[i+1] for i, value in enumerate(args[:-1]) if value == flag]
            if values and values[-1] != '0':
                raise ValueError('speculation and fixed-input batches need separate throughput accounting')
    if (directory/'exit-code').exists() and (directory/'exit-code').read_text().strip() != '0':
        raise ValueError('run did not exit successfully')
    sequence = triples(text)
    if not sequence or [int(row[0]) for row in sequence] != list(range(len(sequence))):
        raise ValueError('incomplete or noncontiguous token trace')
    if baseline is not None and sequence != triples((baseline/'inference.rank00.log').read_text()):
        raise ValueError('token regression differs from baseline')
    selected = [(int(pos), float(seconds)*1000) for pos, seconds in
                re.findall(r'TOKEN pos=(\d+)[^\n]* seconds=([\d.]+)', text)
                if start <= int(pos) < stop]
    if [row[0] for row in selected] != list(range(start, stop)):
        raise ValueError('requested history range is incomplete')
    times = [row[1] for row in selected]
    if any(not math.isfinite(value) or value <= 0 for value in times):
        raise ValueError('invalid token duration')
    memory, finished_shape = [], None
    for rank in range(12):
        match = re.search(r'INFERENCE_FINISHED rank=(\d+) prompt=(\d+) generated=(\d+) '
                          r'seconds=[\d.]+ available=(\d+)',
                          (directory/f'inference.rank{rank:02d}.log').read_text())
        if not match or int(match[1]) != rank or int(match[2])+int(match[3])-1 != len(sequence):
            raise ValueError('rank did not finish the full token sequence')
        shape = (int(match[2]), int(match[3]))
        if finished_shape is not None and shape != finished_shape:
            raise ValueError('ranks disagree on prompt/output counts')
        finished_shape = shape
        if start < shape[0]:
            raise ValueError('ordinary decode timing must begin after prompt processing')
        memory.append(int(match[4]))
    mean = sum(times)/len(times)
    return dict(directory=str(directory), first_position=start, last_position=stop-1, samples=len(times),
                token_triples=len(sequence), base=str(baseline) if baseline is not None else None,
                all_token_triples_exact=True if baseline is not None else None,
                instrumented=(directory/'profile.rank00.bin').exists(),
                mean_ms=mean, p95_ms=percentile(times, .95), tokens_per_second=1000/mean,
                min_final_memavailable_bytes=min(memory),
                binary_sha256=hashlib.sha256((directory/'ds41f_run').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--start', type=int, default=1000)
    parser.add_argument('--stop', type=int, default=1105)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.start < 0 or args.stop <= args.start:
        parser.error('invalid position range')
    encoded = json.dumps(report(args.directory, args.start, args.stop, args.baseline), indent=2, allow_nan=False)+'\n'
    if args.json:
        args.json.write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
