#!/usr/bin/env python3
"""Exact-ELF decomposition and region-scoped raw PMU diagnostics.

Native captures require a compute allocation. Simulator repetitions each use a
fresh process; no warmup/pass-count substitution or sample removal is allowed.
The existing driver's untimed warmup is retained by both paths.
Raw A64FX event 0x017 needs the 0x325/0x326 erratum correction before it
represents physical HBM refill bytes. These measurements alone do not qualify
the simulator's physical bandwidth model.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(args):
    manifest = json.loads(args.manifest.read_text())
    binary = args.binary.resolve()
    cases = manifest['cases']
    if args.only:
        selected = set(args.only.split(','))
        if selected - {c['id'] for c in cases}:
            raise ValueError('unknown case selected')
        cases = [c for c in cases if c['id'] in selected]
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('nonempty unique case set required')
    native = args.qlair is None
    if native and (platform.machine() != 'aarch64' or not os.environ.get('PJM_JOBID')):
        raise ValueError('native collection requires an A64FX compute allocation')
    if native:
        cpu_hz = int(Path('/sys/devices/system/cpu/cpu12/cpufreq/scaling_cur_freq').read_text()) * 1000
        if cpu_hz != 2000000000:
            raise ValueError('normal 2 GHz mode required')
    else:
        cpu_hz = 2000000000
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=2, qualified=False, physical_bandwidth_validated=False,
                  observation_scope='driver timer and optional region PMU; existing untimed warmup',
                  kind='native' if native else 'simulated', timing_backend=None if native else 'event',
                  job_id=os.environ.get('PJM_JOBID') if native else None, host=platform.node(),
                  cpu_hz=cpu_hz, page_size=os.sysconf('SC_PAGE_SIZE') if native else None,
                  binary_sha256=sha(binary), manifest_sha256=sha(args.manifest),
                  simulator_sha256=sha(args.qlair) if args.qlair else None, cases={})
    env = {k: v for k, v in os.environ.items() if not k.startswith('QLAIR_SIM_')}
    if native:
        # The same ELF must use the same allocator. Record inherited settings;
        # never silently apply LD_PRELOAD to one side of an exact-ELF comparison.
        result['allocator_environment'] = {k: v for k, v in env.items()
                                           if k.startswith('XOS_') or k == 'LD_PRELOAD'}
    for case in cases:
        record = dict(case=case, complete=False, samples=[], pmu_samples=[], commands=[], profiles=[])
        for repeat in range(1 if native else args.repeats):
            if sha(binary) != result['binary_sha256'] or (args.qlair and sha(args.qlair) != result['simulator_sha256']):
                raise ValueError('executable changed during campaign')
            stem = args.output / (case['id'] + '-' + str(repeat))
            cmd = [str(binary), case['variant'], str(case['ring']), str(case['calls']),
                   str(case['passes']), str(args.samples if native else 1)]
            if args.qlair:
                cmd = [str(args.qlair.resolve()), '--a64fx-backend', 'event', '--profile-markers',
                       '--profile-report', str(stem.resolve()) + '.profile.json',
                       '--profile-format', 'json', '-n', args.budget, cmd[0], '--'] + cmd[1:]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, env=env, timeout=args.timeout)
            stem.with_suffix('.log').write_text(p.stdout)
            rows = [dict(x.split('=', 1) for x in line.split()[1:])
                    for line in p.stdout.splitlines() if line.startswith('##DECOMP ')]
            pmu = [dict(x.split('=', 1) for x in line.split()[1:])
                   for line in p.stdout.splitlines() if line.startswith('##REGION_PMU ')]
            if pmu and len(pmu) == len(rows):
                for timer, counter in zip(rows, pmu):
                    if any(timer.get(k) != counter.get(k)
                           for k in ('variant', 'ring', 'calls', 'passes', 'sample')):
                        raise ValueError('timer/PMU sample scope mismatch: ' + case['id'])
                    raw = {k: int(counter[k]) for k in ('r011', 'r003', 'r017', 'r018',
                                                         'r325', 'r326', 'r184')}
                    if raw['r011'] <= 0 or raw['r017'] < raw['r325'] + raw['r326']:
                        raise ValueError('invalid PMU group or HBM correction: ' + case['id'])
                    counter['l2_read_bytes'] = raw['r003'] * 256
                    counter['hbm_read_bytes_corrected'] = (
                        raw['r017'] - raw['r325'] - raw['r326']) * 256
                    counter['hbm_write_bytes'] = raw['r018'] * 256
            elif pmu or args.require_region_pmu:
                raise ValueError('missing/incomplete region PMU group: ' + case['id'])
            good = (p.returncode == 0 and len(rows) == (args.samples if native else 1) and
                    'Maximum instruction count reached' not in p.stdout and
                    all(row['correct'] == '1' and int(row['ticks']) > 0 and int(row['freq']) > 0 and
                        all(str(case[k]) == row[k] for k in ('variant', 'ring', 'calls', 'passes'))
                        for row in rows))
            record['commands'].append(cmd)
            record['samples'].extend(rows)
            record['pmu_samples'].extend(pmu)
            if args.qlair:
                profile = Path(str(stem) + '.profile.json')
                if not profile.is_file(): good = False
                else: record['profiles'].append(json.loads(profile.read_text()))
            if not good:
                result['cases'][case['id']] = record
                (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
                raise ValueError('incomplete or incorrect case: ' + case['id'])
        values = [int(r['ticks']) * cpu_hz / int(r['freq']) / case['passes'] / case['calls']
                  for r in record['samples']]
        record.update(complete=True, cycles_per_call=statistics.median(values),
                      deterministic=len(set(values)) == 1 if not native else None)
        if record['pmu_samples']:
            record['region_pmu_median_per_call'] = {
                metric: statistics.median(int(row[metric]) / case['passes'] / case['calls']
                                          for row in record['pmu_samples'])
                for metric in ('r011', 'r184', 'l2_read_bytes',
                               'hbm_read_bytes_corrected', 'hbm_write_bytes')}
        result['cases'][case['id']] = record
        (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        print(case['id'], round(record['cycles_per_call'], 6), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--manifest', type=Path, default=Path(__file__).with_name('cases.json'))
    p.add_argument('--qlair', type=Path)
    p.add_argument('--only')
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--budget', default='200M')
    p.add_argument('--require-region-pmu', action='store_true')
    a = p.parse_args()
    if not 1 <= a.samples <= 31 or a.repeats < 1:
        p.error('samples must be 1..31; repeats must be positive')
    return capture(a)


if __name__ == '__main__':
    raise SystemExit(main())
