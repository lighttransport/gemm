#!/usr/bin/env python3
"""Measure actual emitted tokens per complete speculative cycle near 1K.

Only complete cycles wholly within the requested position range are included.
Amortized time per token is separate from burst-to-burst cycle latency.
Forced rejection diagnostics are deliberately rejected as speed measurements.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    index = fraction*(len(values)-1)
    low = int(index)
    return values[low]+(index-low)*(values[min(low+1,len(values)-1)]-values[low])


def report(directory, start, stop):
    text = (directory/'inference.rank00.log').read_text()
    pattern = (r'SPEC_CYCLE pos=(\d+) drafted=(\d+) verified=(\d+) accepted=(\d+) emitted=(\d+) '
               r'draft_seconds=([\d.]+) verify_seconds=([\d.]+) commit_seconds=([\d.]+) '
               r'seconds=([\d.]+) forced=(-?\d+)')
    cycles = []
    for match in re.finditer(pattern,text):
        row = dict(zip(('position','drafted','verified','accepted','emitted'),map(int,match.groups()[:5])))
        row.update(zip(('draft_seconds','verify_seconds','commit_seconds','seconds'),map(float,match.groups()[5:9])))
        if int(match[10]) != -1:
            raise ValueError('forced-rejection diagnostics are not speed measurements')
        if not (0<=row['accepted']<=row['drafted']<=5 and row['verified']==row['drafted']+1
                and 1<=row['emitted']<=row['accepted']+1
                and all(math.isfinite(row[x]) and row[x]>=0 for x in ('draft_seconds','verify_seconds','commit_seconds'))
                and math.isfinite(row['seconds']) and row['seconds']>0):
            raise ValueError('invalid cycle record')
        if cycles and row['position'] != cycles[-1]['position']+cycles[-1]['emitted']:
            raise ValueError('noncontiguous speculative cycles')
        cycles.append(row)
    finished = re.search(r'SPEC_FINISHED cycles=(\d+) accepted=(\d+) verified=(\d+) emitted=(\d+) verifier=(\w+)',text)
    if not finished or list(map(int,finished.groups()[:4])) != [len(cycles)]+[
            sum(row[key] for row in cycles) for key in ('accepted','verified','emitted')]:
        raise ValueError('incomplete or inconsistent speculation summary')
    selected = [row for row in cycles if row['position']>=start and row['position']+row['emitted']<=stop]
    if not selected:
        raise ValueError('no complete cycles in requested range')
    memory = []
    for rank in range(12):
        match = re.search(r'INFERENCE_FINISHED.*available=(\d+)',(directory/f'inference.rank{rank:02d}.log').read_text())
        if not match:
            raise ValueError('rank did not finish')
        memory.append(int(match[1]))
    seconds = sum(row['seconds'] for row in selected)
    emitted = sum(row['emitted'] for row in selected)
    per_token = [row['seconds']*1000/row['emitted'] for row in selected for _ in range(row['emitted'])]
    return dict(directory=str(directory),verifier=finished[5],first_position=selected[0]['position'],
                last_position=selected[-1]['position']+selected[-1]['emitted']-1,
                cycles=len(selected),emitted=emitted,seconds=seconds,tokens_per_second=emitted/seconds,
                mean_amortized_ms_per_token=seconds*1000/emitted,
                p95_amortized_ms_per_token=percentile(per_token,.95),
                p95_cycle_ms=percentile([row['seconds']*1000 for row in selected],.95),
                mean_emitted_per_cycle=emitted/len(selected),
                verified_per_emitted=sum(row['verified'] for row in selected)/emitted,
                phases_ms_per_emitted={key:sum(row[key+'_seconds'] for row in selected)*1000/emitted
                                      for key in ('draft','verify','commit')},
                min_final_memavailable_bytes=min(memory),
                binary_sha256=hashlib.sha256((directory/'ds41f_run').read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--start',type=int,default=1000)
    ap.add_argument('--stop',type=int,default=1105)
    ap.add_argument('--json',type=Path)
    args = ap.parse_args()
    if args.start<0 or args.stop<=args.start:
        ap.error('invalid position range')
    result = report(args.directory,args.start,args.stop)
    encoded = json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.json:
        args.json.write_text(encoded)
    print(encoded,end='')


if __name__ == '__main__':
    main()
