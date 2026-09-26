#!/usr/bin/env python3
"""Compare complete greedy continuations and summarize independent phase timers.
Usage: check_handoff.py RUN_DIR [--reference LOG] [--require-exact]
TP1 state-import decode can serve as a resharding reference (HANDOFF_TPS='1 4 2').
A separate full-prefill reference is needed to validate producer numerics.
"""
import argparse
import pathlib
import re
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=pathlib.Path)
parser.add_argument('--reference', type=pathlib.Path)
parser.add_argument('--require-exact', action='store_true')
args = parser.parse_args()
logs = sorted(args.run.glob('decode.tp*.rank0.log'))
if not logs:
    parser.error('no decode rank0 logs')


def tokens(path):
    text = path.read_text()
    result = re.search(r'RESULT .*gen=(\d+) decode_tok_s=([0-9.]+)', text)
    pairs = [(int(n), int(tok)) for n, tok in re.findall(r'q38d: token n=(\d+) pos=\d+ id=(\d+)', text)]
    if not result or [n for n, _ in pairs] != list(range(int(result[1]))):
        raise ValueError('incomplete decode: ' + str(path))
    return [tok for _, tok in pairs], float(result[2]), text


reference = args.reference or (args.run / 'decode.tp1.rank0.log')
if not reference.exists():
    if args.reference:
        parser.error('reference log does not exist: ' + str(reference))
    reference = logs[0]
ref, _, ref_text = tokens(reference)
print('reference:', reference)
exact = True
for path in logs:
    values, rate, text = tokens(path)
    matches = sum(a == b for a, b in zip(ref, values))
    first_diff = next((i for i, (a, b) in enumerate(zip(ref, values)) if a != b), None)
    equal = len(values) == len(ref) and values == ref
    exact &= equal
    print(f'{path.name}: decode={rate:.3f} tok/s match={matches}/{max(len(values), len(ref))}'
          f' exact={equal} first_difference={first_diff}')
    if equal:
        ref_logits = [float.fromhex(v) for v in re.findall(r'token n=\d+ pos=\d+ id=\d+ logit=(\S+)', ref_text)]
        logits = [float.fromhex(v) for v in re.findall(r'token n=\d+ pos=\d+ id=\d+ logit=(\S+)', text)]
        if values and len(logits) == len(ref_logits) == len(values):
            print('  max_selected_logit_abs_difference=%.9g' % max(abs(a - b) for a, b in zip(logits, ref_logits)))
    for line in text.splitlines():
        if line.startswith('q38d_state:'):
            print(' ', line)
for line in (args.run / 'prefill.rank0.log').read_text().splitlines():
    if line.startswith(('q38p_pp: nodes=', 'q38p_prefill:', 'q38d_state:')):
        print(line)
if args.require_exact and not exact:
    sys.exit(1)
