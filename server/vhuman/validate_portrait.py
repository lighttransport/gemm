"""Exercise the real HTTP portrait workflow on an idle 8 GiB CUDA device.

    . tmp/vhuman-dev/activate.sh
    "$PYTHON" -m server.vhuman.validate_portrait

This never stops other GPU processes. Reports and generated images use repo tmp/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from server.model_paths import MODELS_ROOT
import sys
import threading
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from . import gpu, qwen
from .app import App, make_handler
from .service import ROOT
from .rig.benchmark import Telemetry

SUBJECTS = (
    'a 30-year-old woman with light olive skin, dark brown hair tied back, hazel eyes',
    'a 45-year-old man with deep brown skin, short curly black hair, brown eyes, clean shaven',
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=ROOT/'tmp/portrait-validation'/datetime.now().strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--models-root', default=str(MODELS_ROOT))
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--images-only', action='store_true', help='skip reconstruction; report generation coverage only')
    parser.add_argument('--cancel-only', action='store_true', help='cancel a real image job after model allocation and verify cleanup')
    parser.add_argument('--resume', action='store_true', help='reuse successful jobs from this work directory; retry rejected reconstruction')
    parser.add_argument('--job-timeout', type=int, default=7200)
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    report = dict(format='vhuman.portrait_8gb_validation.v1', utc=datetime.now(timezone.utc).isoformat(),
                  status='starting', jobs=[], visual_review='pending; inspect review.html before claiming visual quality')
    previous = args.work/'validation.json'
    if args.resume and previous.is_file():
        report['jobs'] = json.loads(previous.read_text()).get('jobs', [])
    def save():
        (args.work/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    with gpu.execution('cuda', args.device, args.models_root):
        status = gpu.gpu_status()
        ready = qwen.availability(python=sys.executable)
    report.update(gpu=status, qwen=ready)
    if not ready['available'] or status is None or status['free_mib'] < gpu.QWEN_LOW8_MIN_FREE_MIB:
        report.update(status='blocked', reason=ready.get('reason') or 'free at least 7680 MiB before running; existing GPU work was left running')
        save()
        print(json.dumps(report, indent=2))
        return 2
    if args.check_only:
        report['status'] = 'ready_for_execution'
        save()
        print(json.dumps(report, indent=2))
        return 0
    config = SimpleNamespace(work=str(args.work/'work'), mock=False, qwen_python=sys.executable,
                             rig_python=sys.executable, inference_backend='cuda', device=args.device,
                             models_root=args.models_root)
    from .runtime import configure_args
    configure_args(config)
    app = App(config)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(app, quiet=True))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def request(path, body=None):
        with urlopen(Request(base+path, data=json.dumps(body).encode() if body is not None else None,
                             headers={'Content-Type': 'application/json'}), timeout=30) as response:
            return json.load(response)
    def job(body, force=False):
        if args.resume and not force:
            for old in reversed(report['jobs']):
                if old['request'] == body and old['state'] == 'done':
                    result = old['result']
                    if body['kind'] == 'portrait_reconstruct' and result.get('multiview', {}).get('status') not in ('accepted', 'front_only'):
                        continue
                    print(body['kind']+': reusing completed validation job '+old['job_id'], flush=True)
                    return result
        started = time.monotonic()
        with Telemetry(args.device) as telemetry:
            jid = request('/v1/jobs', body)['id']
            last = None
            while time.monotonic()-started < args.job_timeout:
                current = request('/v1/jobs/'+jid)
                if current['message'] != last:
                    print(body['kind']+': '+current['message'], flush=True)
                    last = current['message']
                if current['state'] in ('done', 'failed', 'cancelled'):
                    break
                time.sleep(1)
            else:
                request('/v1/jobs/'+jid+'/cancel', {})
                raise RuntimeError('validation job timed out: '+jid)
        memory = telemetry.report()
        memory['physical_peak_bytes'] = max(s['memory_used_bytes']+s['memory_reserved_bytes'] for s in telemetry.samples)
        memory['physical_scope'] = 'whole-device NVML used plus driver/firmware reserved memory'
        memory.pop('samples', None)
        row = dict(kind=body['kind'], request=body, job_id=jid, state=current['state'],
                   seconds=round(time.monotonic()-started, 2), memory=memory,
                   result=current.get('result'), error=current.get('error'))
        report['jobs'].append(row)
        save()
        if current['state'] != 'done':
            raise RuntimeError(current.get('error') or current['state'])
        if memory['physical_peak_bytes'] >= 7680*1024**2:
            raise RuntimeError(f"whole-device peak exceeded 7680 MiB: {memory['physical_peak_bytes']/1024**2:.0f} MiB")
        if memory['final_bytes'] > memory['idle_bytes']+256*1024**2:
            raise RuntimeError('device memory did not return within 256 MiB of its initial use')
        return current['result']
    def approve(doc, ids=()):
        return request(f"/v1/portrait-sessions/{doc['id']}/review",
                       dict(revision=doc['active_revision'], portrait_reviewed=True, view_ids=list(ids)))
    def cancellation():
        from .portrait import listing
        before = {d['id'] for d in listing(app.service)}
        native_before = set((ROOT/'tmp').glob('qimg21-native-*'))
        body = dict(kind='portrait_generate', subject=SUBJECTS[0], seed=77, steps=100, qwen_preset='low8')
        started = time.monotonic()
        cancelled_at = None
        stopped_at = None
        with Telemetry(args.device) as telemetry:
            jid = request('/v1/jobs', body)['id']
            while time.monotonic()-started < args.job_timeout:
                current = request('/v1/jobs/'+jid)
                if cancelled_at is None and any(s['memory_used_bytes'] >= 6*1024**3 for s in telemetry.samples):
                    print('portrait_generate: cancelling after GPU allocation', flush=True)
                    request('/v1/jobs/'+jid+'/cancel', {})
                    cancelled_at = time.monotonic()
                if current['state'] in ('done', 'failed', 'cancelled'):
                    stopped_at = time.monotonic()
                    break
                if cancelled_at is not None and time.monotonic()-cancelled_at > 15:
                    raise RuntimeError('GPU job did not stop within 15 seconds of cancellation')
                time.sleep(.2)
            else:
                request('/v1/jobs/'+jid+'/cancel', {})
                raise RuntimeError('cancellation validation timed out')
            # NVML can report the old context briefly after the killed worker
            # exits. Measure a bounded return to idle rather than sampling only
            # the instant the HTTP job changes state.
            release_deadline = time.monotonic()+5
            while time.monotonic() < release_deadline:
                if telemetry.sample()['memory_used_bytes'] <= telemetry.samples[0]['memory_used_bytes']+64*1024**2:
                    break
                time.sleep(.1)
            released_at = time.monotonic()
        memory = telemetry.report()
        memory['physical_peak_bytes'] = max(s['memory_used_bytes']+s['memory_reserved_bytes'] for s in telemetry.samples)
        memory.pop('samples', None)
        report['jobs'].append(dict(kind='portrait_cancel', request=body, job_id=jid, state=current['state'],
                                   seconds=round(time.monotonic()-started, 2), memory=memory,
                                   cancellation_seconds=round(stopped_at-cancelled_at, 2) if cancelled_at else None,
                                   memory_return_seconds=round(released_at-cancelled_at, 2) if cancelled_at else None))
        save()
        if cancelled_at is None or current['state'] != 'cancelled':
            raise RuntimeError('real GPU generation did not cancel: '+str(current.get('error') or current['state']))
        if memory['physical_peak_bytes'] >= 7680*1024**2:
            raise RuntimeError('cancellation run exceeded the 7680 MiB whole-device ceiling')
        if memory['final_bytes'] > memory['idle_bytes']+64*1024**2:
            raise RuntimeError('cancelled worker retained device memory')
        if {d['id'] for d in listing(app.service)} != before:
            raise RuntimeError('cancelled generation published a portrait')
        if set((ROOT/'tmp').glob('qimg21-native-*'))-native_before:
            raise RuntimeError('cancelled native worker retained temporary files')
        if list((app.service.work/'portrait_sessions').glob('*/portraits/.*.partial')):
            raise RuntimeError('cancelled generation retained partial images')
    documents = []
    try:
        report['status'] = 'running'
        save()
        for identity, subject in enumerate(() if args.cancel_only else SUBJECTS):
            doc = job(dict(kind='portrait_generate', subject=subject, seed=11+identity*101,
                           steps=24, qwen_preset='low8'))['session']
            sid, revision = doc['id'], doc['active_revision']
            # A generation execution gate, not a substitute for human quality review.
            from .reconstruction.observations import observe
            from .portrait import artifact
            observe(artifact(app.service, sid, doc['portraits'][0]['file']))
            approve(doc)
            for method in ('individual', 'sheet'):
                doc = job(dict(kind='portrait_views', session_id=sid, revision=revision,
                               seed=31+identity*101, steps=24, qwen_preset='low8', method=method))['session']
                ids = [v['id'] for v in doc['view_runs'][-1]['views']]
                approve(doc, ids)
                if not args.images_only:
                    result = job(dict(kind='portrait_reconstruct', session_id=sid, revision=revision,
                                      subject=subject, view_ids=ids, face_model='gnm_v3', res=512,
                                      texture_res=1024, iterations=80, build_rig=True))
                    if result['multiview']['status'] != 'accepted':
                        raise RuntimeError('multiview reconstruction rejected; inspect '+result['multiview'].get('reason', 'view checks'))
            documents.append(doc)
        if args.cancel_only:
            cancellation()
            report.update(status='cancellation_passed', coverage='real_gpu_cancellation')
        else:
            repeated = job(dict(kind='portrait_generate', subject=SUBJECTS[0], seed=11, steps=24, qwen_preset='low8'), force=True)['session']
            report['warm_repeat_identical'] = repeated['portraits'][0]['sha256'] == documents[0]['portraits'][0]['sha256']
            if not report['warm_repeat_identical']:
                raise RuntimeError('same-seed warm generation differed from the cold output')
            report.update(status='execution_passed_visual_review_pending',
                          coverage='images_only' if args.images_only else 'generation_and_reconstruction')
    except Exception as exc:
        report.update(status='failed', error=str(exc))
    finally:
        # Recover even partially completed sessions for inspection after a failure.
        from .portrait import listing
        documents = listing(app.service)
        panels = []
        for doc in documents:
            rows = doc['portraits']+[v for r in doc['view_runs'] for v in r['views']]
            for row in rows:
                source = f"work/portrait_sessions/{doc['id']}/{row['file']}"
                panels.append(f'<figure><img src="{html.escape(source)}"><figcaption>{html.escape(doc["subject"])} · '
                              f'{html.escape(row.get("label", "Front"))} · {row["size"]}</figcaption></figure>')
        (args.work/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Portrait validation review</title>'
            '<style>body{background:#111;color:#eee;font:14px system-ui}main{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}'
            'figure{margin:0}img{width:100%}figcaption{padding:8px}</style><h1>Generated input review</h1>'
            '<p>Review identity, pose, expression, cropping, eyes and texture consistency. Execution success does not certify visual quality.</p><main>'
            +''.join(panels)+'</main>')
        server.shutdown()
        server.server_close()
        thread.join()
        save()
    print(json.dumps({k: v for k, v in report.items() if k != 'jobs'}, indent=2))
    print('Report: '+str(args.work/'validation.json'))
    return 0 if report['status'] in ('execution_passed_visual_review_pending', 'cancellation_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
