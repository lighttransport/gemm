"""Check the local face-review page, responsive controls and download integrity."""
import argparse
import base64
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import zipfile

from ..test_browser import Cdp, find_chrome
from .quality_review import digest


def verify(work, out):
    work, out = Path(work).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    review = work / 'review'
    manifest = json.loads((review / 'manifest.json').read_text())
    archives = []
    for bundle in manifest['bundles']:
        path = review / bundle['href']
        if digest(path) != bundle['sha256']:
            raise ValueError('review ZIP hash differs')
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError('review ZIP integrity failure')
            names = archive.namelist()
            if any(Path(name).is_absolute() or '..' in Path(name).parts for name in names):
                raise ValueError('unsafe review archive member')
            if not any(name.endswith('/portable.blend') for name in names):
                raise ValueError('portable Blender scene missing')
            evidence_names=[name for name in names if name.endswith('/candidate_evidence.json')]
            evidence=None
            if evidence_names:
                if len(evidence_names)!=1:raise ValueError('ambiguous candidate evidence')
                from .usd_candidate_evidence import verify as verify_evidence
                prefix=str(Path(evidence_names[0]).parent)+'/'
                with tempfile.TemporaryDirectory(prefix='evidence-',dir=out) as directory:
                    required=[prefix+name for name in ('head.usdc','report.json','candidate_manifest.json','candidate_evidence.json')]
                    required.extend(name for name in names if name.startswith(prefix+'candidate_evidence/'))
                    for name in required:archive.extract(name,directory)
                    evidence=verify_evidence(Path(directory)/Path(prefix))
            archives.append(dict(name=path.name, files=len(names), bytes=path.stat().st_size))
            if evidence is not None:archives[-1]['candidate_evidence']=evidence
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(work.parent)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    chrome = cdp = runtime = None
    try:
        executable = find_chrome()
        if not executable:
            raise RuntimeError('Chromium is required for review verification')
        profile = out / 'chrome-profile'
        profile.mkdir(exist_ok=True)
        active = profile / 'DevToolsActivePort'
        active.unlink(missing_ok=True)
        # Chromium creates a Unix socket below TMPDIR; deep report paths can
        # exceed its platform path limit even though ordinary files work.
        runtime = tempfile.TemporaryDirectory(prefix='qrv-', dir=work.parent)
        environment = dict(os.environ, TMPDIR=runtime.name)
        environment.pop('DISPLAY', None)
        with (out / 'chrome.log').open('w') as log:
            chrome = subprocess.Popen([executable, '--headless=new', '--ozone-platform=headless',
                '--no-sandbox', '--no-first-run', '--no-default-browser-check', '--disable-gpu',
                '--window-size=1440,1100', '--remote-debugging-port=0',
                '--user-data-dir=' + str(profile), 'about:blank'],
                stdout=log, stderr=subprocess.STDOUT, env=environment)
            deadline = time.monotonic() + 30
            while not active.exists():
                if chrome.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError((out / 'chrome.log').read_text()[-3000:])
                time.sleep(.1)
            port = active.read_text().splitlines()[0]
            target = json.loads(urlopen(Request(f'http://127.0.0.1:{port}/json/new', method='PUT')).read())
            cdp = Cdp(target['webSocketDebuggerUrl'])
            for domain in ('Runtime.enable', 'Log.enable', 'Page.enable'):
                cdp.call(domain)
            url = f'http://127.0.0.1:{server.server_port}/{work.name}/review/'
            cdp.call('Page.navigate', {'url': url})
            cdp.wait_for('window.vhumanQualityReview?.ready', timeout=30)
            info = cdp.evaluate('vhumanQualityReview')
            if info['cards'] != manifest['comparisons'] or info['bundles'] != len(archives):
                raise ValueError('review panel counts differ')
            data = cdp.evaluate('JSON.parse(document.getElementById("review-data").textContent)')
            image_checks = 0
            for card in data['cards']:
                identifier = json.dumps(card['id'])
                for index in range(len(card['views'])):
                    cdp.evaluate(f'(()=>{{const s=document.getElementById({identifier}).querySelector("select");s.value={index};s.dispatchEvent(new Event("change"));}})()')
                    cdp.evaluate(f'Promise.all([...document.getElementById({identifier}).querySelectorAll("img")].map(i=>i.decode()))')
                    if not cdp.evaluate(f'[...document.getElementById({identifier}).querySelectorAll("img")].every(i=>i.complete&&i.naturalWidth>0)'):
                        raise ValueError('review comparison image failed to load')
                    image_checks += 2
                cdp.evaluate(f'(()=>{{const s=document.getElementById({identifier}).querySelector("select");s.value=0;s.dispatchEvent(new Event("change"));}})()')
            cdp.evaluate('Promise.all([...document.images].map(i=>i.decode()))')
            links = cdp.evaluate('[...document.querySelectorAll("a[href]")].map(a=>a.getAttribute("href")).filter(x=>x&&!x.startsWith("#"))')
            for href in set(links):
                with urlopen(Request(urljoin(url, href), method='HEAD')) as response:
                    if response.status != 200:
                        raise ValueError('broken review link: ' + href)
            selected=json.dumps(data['cards'][0]['id'])
            cdp.evaluate(f'document.getElementById({selected}).scrollIntoView({{block:"center"}})')
            screenshots = []
            for button, label in ((0, 'earlier'), (2, 'candidate')):
                cdp.evaluate(f'document.getElementById({selected}).querySelectorAll("button")[{button}].click()')
                cdp.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                value = cdp.evaluate(f'document.getElementById({selected}).dataset.split')
                if value != ('100' if button == 0 else '0'):
                    raise ValueError('comparison control did not change split')
                shot = base64.b64decode(cdp.call('Page.captureScreenshot', {'format': 'png'})['data'])
                (out / f'comparison_{label}.png').write_bytes(shot)
                screenshots.append(shot)
            if screenshots[0] == screenshots[1]:
                raise ValueError('comparison controls did not change visible output')
            cdp.evaluate(f'document.getElementById({selected}).querySelectorAll("button")[1].click();scrollTo(0,0)')
            desktop = cdp.call('Page.captureScreenshot', {'format': 'png'})
            (out / 'desktop.png').write_bytes(base64.b64decode(desktop['data']))
            cdp.call('Emulation.setDeviceMetricsOverride', dict(width=390, height=844,
                     deviceScaleFactor=1, mobile=True))
            cdp.evaluate('scrollTo(0,0)')
            cdp.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
            overflow = cdp.evaluate('document.documentElement.scrollWidth>innerWidth+1')
            if overflow:
                raise ValueError('review has horizontal overflow on narrow viewport')
            mobile = cdp.call('Page.captureScreenshot', {'format': 'png'})
            (out / 'narrow_viewport.png').write_bytes(base64.b64decode(mobile['data']))
            errors = [error for error in cdp.console_errors() if 'favicon.ico' not in error]
            if errors:
                raise ValueError(errors)
            result = dict(passed=True, panels=info['cards'], image_checks=image_checks,
                reachable_links=len(set(links)), archives=archives,
                comparison_controls_change_pixels=True, narrow_viewport_no_horizontal_overflow=True,
                console_errors=[], physical_mobile_test=False,
                limitation='Static review UI verification, not live-avatar rendering or physical-device performance')
            (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
            return result
    finally:
        if cdp:
            cdp.close()
        if chrome:
            chrome.terminate()
            try:
                chrome.wait(timeout=5)
            except subprocess.TimeoutExpired:
                chrome.kill()
                chrome.wait()
        if runtime:
            runtime.cleanup()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True)
    parser.add_argument('--out', required=True)
    print(json.dumps(verify(**vars(parser.parse_args())), indent=2))
