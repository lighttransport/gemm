"""Batch driver: runs ue_scripts/gen_batch.py over jobs.jsonl in editor sessions of --batch jobs each, with resume.

usage: python -m server.vhuman.metahuman.generate [--jobs PATH] [--batch 25] [--limit N] [--only id,id] [--verify] [--dry-run]
A sample is done when samples/<id>/done.json exists with ok=true. Each editor session processes the next --batch pending
jobs; a crashed session is retried once for the same slice before moving on. --verify recomputes the sha256 of every file
listed in done.json.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from . import DATASET_ROOT
from .launch import run

V1 = DATASET_ROOT / 'v1'


def load_jobs(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def is_done(sample_dir):
    d = sample_dir / 'done.json'
    if not d.is_file():
        return False
    try:
        return bool(json.loads(d.read_text()).get('ok'))
    except Exception:      # noqa: BLE001
        return False


def pending(jobs, only=None):
    out = []
    for j in jobs:
        if only and j['id'] not in only:
            continue
        if not is_done(V1 / 'samples' / j['id']):
            out.append(j)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(jobs):
    bad = 0
    for j in jobs:
        d = V1 / 'samples' / j['id'] / 'done.json'
        if not d.is_file():
            continue
        rec = json.loads(d.read_text())
        for rel, digest in rec.get('sha256', {}).items():
            p = V1 / 'samples' / j['id'] / rel
            if not p.is_file() or sha256(p) != digest:
                print('MISMATCH', j['id'], rel); bad += 1
    print('verify: mismatches', bad)
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--jobs', default=str(V1 / 'jobs.jsonl')); ap.add_argument('--batch', type=int, default=25)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--only', default='')
    ap.add_argument('--verify', action='store_true'); ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--timeout', type=float, default=5400.)
    a = ap.parse_args()
    jobs = load_jobs(a.jobs)
    if a.verify:
        sys.exit(0 if verify(jobs) else 1)
    only = set(a.only.split(',')) if a.only else None
    todo = pending(jobs, only)
    if a.limit:
        todo = todo[:a.limit]
    print(f'{len(jobs)} jobs, {len(todo)} pending', flush=True)
    if a.dry_run:
        for j in todo[:10]:
            print(j['id'], j['kind'], j['preset'], 'render' if j['render'] else '', 'held-out' if j['held_out'] else '')
        return
    t0 = time.time(); done = 0
    while todo:
        slice_ = todo[:a.batch]
        ids = ','.join(j['id'] for j in slice_)
        batch_file = V1 / 'logs' / f'batch-{int(time.time())}.json'
        batch_file.parent.mkdir(parents=True, exist_ok=True)
        batch_file.write_text(json.dumps(dict(jobs_path=str(a.jobs), ids=[j['id'] for j in slice_])))
        for attempt in (1, 2):
            run('gen_batch.py', [str(batch_file)], timeout=a.timeout, result=None)
            remaining = [j for j in slice_ if not is_done(V1 / 'samples' / j['id'])]
            if not remaining:
                break
            print(f'attempt {attempt}: {len(remaining)} of {len(slice_)} not done ({ids[:80]}...)', flush=True)
            slice_ = remaining
            batch_file.write_text(json.dumps(dict(jobs_path=str(a.jobs), ids=[j['id'] for j in slice_])))
        finished = [j for j in todo[:a.batch] if is_done(V1 / 'samples' / j['id'])]
        done += len(finished)
        todo = pending(todo[a.batch:], only)
        rate = (time.time() - t0) / max(done, 1)
        print(f'done {done}, pending {len(todo)}, {rate:.0f} s/sample, eta {rate * len(todo) / 3600:.1f} h', flush=True)


if __name__ == '__main__':
    main()
