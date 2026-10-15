"""Sustained local WebGL playback of hash-bound GLBs on NVIDIA hardware.

Measures exported joint/morph animation and PBR rendering, independently of
the native exact-contact CUDA deformer. Uses local Three.js; no CDN requests.
"""
from __future__ import annotations

import argparse
import base64
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from urllib.request import Request, urlopen

from ..test_browser import Cdp, find_chrome
from .benchmark import ROOT, Telemetry, machine, receipt


def run(assets, three, out, seconds=60., warmup=5., turntables=()):
    out, three = Path(out).absolute(), Path(three).absolute()
    if out.exists():
        raise FileExistsError(f'refusing to overwrite baseline: {out}')
    if not math.isfinite(seconds) or not math.isfinite(warmup) or seconds <= 0 or warmup < 0:
        raise ValueError('positive duration and nonnegative warmup required')
    chrome_path = find_chrome()
    if not chrome_path:
        raise RuntimeError('Chromium/Chrome required')
    entries = []
    for name, source in assets:
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('asset labels must be simple letters, numbers, underscores or hyphens')
        entries.append(dict(name=name, source=Path(source).absolute()))
    if not entries or len({entry['name'] for entry in entries}) != len(entries):
        raise ValueError('nonempty assets with unique names required')
    out.mkdir(parents=True)
    player = out/'player'
    player.mkdir()
    (player/'three').symlink_to(three, target_is_directory=True)
    for entry in entries:
        entry['receipt'] = receipt(entry['source'])
        (player/(entry['name']+'.glb')).symlink_to(entry['source'])
    script = Path(__file__).with_suffix('.js')
    shutil.copyfile(script, player/'benchmark.js')
    (player/'favicon.ico').write_bytes(b'')
    (player/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>vHuman playback baseline</title>
<style>body{margin:0;background:#242830;overflow:hidden}canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"./three/build/three.module.js","three/addons/":"./three/examples/jsm/"}}</script>
<script type="module" src="./benchmark.js"></script>''')
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(player)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    report = dict(machine=machine(), browser=subprocess.check_output([chrome_path, '--version'], text=True).strip(),
                  three=receipt(three/'package.json'), three_module=receipt(three/'build/three.module.js'),
                  harness=receipt(script), runner=receipt(Path(__file__)),
                  configuration=dict(seconds_per_asset=seconds, warmup_seconds=warmup,
                                     native_exact_contact_projection=False,
                                     scope='exported GLB animation and PBR rendering; desktop headless Chromium, not a physical mobile device'),
                  assets=[])
    cdp = chrome = None
    try:
        with Telemetry() as telemetry, tempfile.TemporaryDirectory(prefix='vpb-', dir=ROOT/'tmp') as scratch:
            profile = out/'chrome-profile'
            profile.mkdir()
            flags = ['--headless=new', '--ozone-platform=headless', '--no-sandbox', '--no-first-run',
                     '--no-default-browser-check', '--ignore-gpu-blocklist', '--use-gl=angle', '--use-angle=vulkan',
                     '--enable-features=Vulkan', '--disable-vulkan-surface', '--window-size=1280,720',
                     '--remote-debugging-port=0', f'--user-data-dir={profile}', 'about:blank']
            report['browser_flags'] = flags
            environment = dict(os.environ, TMPDIR=scratch)
            environment.pop('DISPLAY', None)
            try:
                with (out/'chrome.log').open('w') as log:
                    chrome = subprocess.Popen([chrome_path, *flags], env=environment, stdout=log, stderr=subprocess.STDOUT)
                    active_port = profile/'DevToolsActivePort'
                    deadline = time.monotonic()+30
                    while not active_port.exists():
                        if time.monotonic()>deadline or chrome.poll() is not None:
                            raise RuntimeError((out/'chrome.log').read_text()[-3000:])
                        time.sleep(.1)
                    port = active_port.read_text().splitlines()[0]
                    target = json.loads(urlopen(Request(f'http://127.0.0.1:{port}/json/new', method='PUT')).read())
                    cdp = Cdp(target['webSocketDebuggerUrl'])
                    for domain in ('Runtime.enable', 'Log.enable', 'Page.enable'):
                        cdp.call(domain)
                    cdp.call('Emulation.setDeviceMetricsOverride', dict(width=1280, height=720, deviceScaleFactor=1, mobile=False))
                    cdp.call('Page.bringToFront')
                    cdp.call('Page.navigate', {'url': f'http://127.0.0.1:{server.server_port}/'})
                    cdp.wait_for('window.bench', timeout=30)
                    # ANGLE/Vulkan may replace its initial command buffer while the
                    # GPU process starts. Retry the empty page once, before loading
                    # any measured asset; never erase errors during a measurement.
                    time.sleep(1)
                    startup_errors = cdp.evaluate('bench.errors')
                    report['startup_retry_errors'] = startup_errors
                    if startup_errors:
                        cdp.call('Page.reload')
                        time.sleep(.3)
                        cdp.wait_for('window.bench', timeout=30)
                        time.sleep(1)
                    if cdp.evaluate('bench.contextLost || bench.errors.length'):
                        raise RuntimeError(f'WebGL initialization failed: {cdp.evaluate("bench.errors")}')
                    report['driver'] = cdp.evaluate('bench.driver')
                    driver = report['driver']['renderer'].lower()
                    if 'nvidia' not in driver or any(term in driver for term in ('swiftshader', 'llvmpipe', 'lavapipe')):
                        raise RuntimeError(f'hardware NVIDIA rendering required: {driver}')
                    for entry in entries:
                        started = time.time()
                        name = entry['name']
                        loaded = cdp.evaluate(f'bench.load({json.dumps(name+".glb")},{json.dumps(name in turntables)})')
                        warmup_until = time.monotonic()+warmup
                        while time.monotonic()<warmup_until:
                            time.sleep(min(.2, max(0, warmup_until-time.monotonic())))
                        cdp.evaluate(f'bench.begin({seconds})')
                        print(f'{name}: started {seconds:g}s playback; {loaded["vertices"]} vertices', flush=True)
                        deadline, progress = time.monotonic()+seconds+30, time.monotonic()+10
                        while cdp.evaluate('bench.active'):
                            if time.monotonic()>deadline:
                                raise RuntimeError(f'playback stalled: {name}')
                            if time.monotonic()>progress:
                                print(f'{name}: playback continues', flush=True)
                                progress += 10
                            time.sleep(.5)
                        # Let asynchronous GPU queries complete before collecting their tagged result.
                        time.sleep(.3)
                        result = cdp.evaluate('bench.summary().at(-1)')
                        errors = cdp.evaluate('bench.errors') + cdp.console_errors()
                        if errors or result['disjoint'] or not result['frames'] or result['fps'] <= 0:
                            raise RuntimeError(f'invalid playback measurement: {errors}, disjoint={result["disjoint"]}')
                        if loaded['clip'] and result['animation_time_range'][1]-result['animation_time_range'][0] < min(seconds*.5, loaded['clip_duration']*.5):
                            raise RuntimeError('animation did not advance during playback')
                        png = cdp.call('Page.captureScreenshot', {'format': 'png'})['data']
                        (out/(name+'.png')).write_bytes(base64.b64decode(png))
                        ended = time.time()
                        samples = [s for s in telemetry.samples if started <= s['time'] <= ended]
                        asset = dict(name=name, asset=entry['receipt'], load=loaded, playback=result,
                                     peak_device_memory_bytes=max(s['memory_used_bytes'] for s in samples))
                        asset['after_unload'] = cdp.evaluate('bench.unload()')
                        cdp.call('HeapProfiler.collectGarbage')
                        report['assets'].append(asset)
                        print(f'{name}: {result["fps"]:.2f} FPS, interval p95={result["frame_interval_ms"]["p95"]:.2f} ms', flush=True)
            finally:
                if cdp:
                    cdp.close()
                if chrome:
                    chrome.terminate()
                    try:
                        chrome.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        chrome.kill()
                        chrome.wait(timeout=10)
                time.sleep(1)
        report['telemetry'] = telemetry.report()
        report['passed'] = True
        (out/'baseline.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
    except Exception as exc:
        report['passed'] = False
        report['failure'] = str(exc)
        if 'telemetry' in locals():
            report['telemetry'] = telemetry.report()
        (out/'baseline.json').write_text(json.dumps(report, indent=2)+'\n')
        raise
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset', action='append', nargs=2, metavar=('LABEL', 'GLB'), required=True)
    p.add_argument('--turntable', action='append', default=[])
    p.add_argument('--three', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--warmup', type=float, default=5)
    args = p.parse_args()
    run(args.asset, args.three, args.out, args.seconds, args.warmup, args.turntable)


if __name__ == '__main__':
    main()
