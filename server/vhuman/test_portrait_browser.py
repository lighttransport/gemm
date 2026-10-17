"""Drive the actual portrait workspace through Chrome/CDP, with mock generation."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from .app import App, make_handler
from .service import ROOT
from .test_browser import Cdp, find_chrome

SCRATCH = ROOT/'tmp/test-portrait-browser'


@unittest.skipUnless(find_chrome(), 'Chrome/Chromium missing')
class PortraitBrowserTests(unittest.TestCase):
    def test_review_both_methods_reload_revision_and_mobile_layout(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            work = Path(temp)
            app = App(SimpleNamespace(work=str(work/'work'), mock=True, qwen_python=sys.executable,
                       rig_python=sys.executable, inference_backend='cpu', device=0, models_root='/mnt/disk1/models'))
            app.health = lambda: dict(qwen=dict(available=True, backend='mock'), gpu=None, video={}, pixal3d={},
                portrait_generators={m: dict(available=True, backend='mock') for m in ('qwen-image21', 'klein3')})
            handler = make_handler(app, quiet=True)
            three = Path(os.environ.get('VHUMAN_THREE', ROOT/'tmp/vhuman-dev/tools/three/package'))
            class LocalHandler(handler):
                def _send(self, status, body, ctype, extra=None):
                    if ctype.startswith('text/html') and three.is_dir():
                        body = body.replace(b'https://unpkg.com/three@0.163.0/', b'/test-three/')
                    return super()._send(status, body, ctype, extra)

                def do_GET(self):
                    if self.path.startswith('/test-three/'):
                        path = (three/self.path[len('/test-three/'):]).resolve()
                        if path.is_relative_to(three.resolve()) and path.suffix == '.js' and path.is_file():
                            return self._send(200, path.read_bytes(), 'text/javascript')
                        return self._error(404, 'no such Three.js file')
                    return super().do_GET()

            server = ThreadingHTTPServer(('127.0.0.1', 0), LocalHandler)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            profile = work/'chrome'
            log = (SCRATCH/'chrome.log').open('w')
            chrome = subprocess.Popen([find_chrome(), '--headless=new', '--no-sandbox', '--no-first-run',
                '--no-default-browser-check', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                '--window-size=1400,1000', '--remote-debugging-port=0', '--user-data-dir='+str(profile), 'about:blank'],
                stdout=log, stderr=subprocess.STDOUT)
            cdp = None
            try:
                active = profile/'DevToolsActivePort'
                deadline = time.monotonic()+40
                while not active.is_file() and chrome.poll() is None and time.monotonic() < deadline:
                    time.sleep(.1)
                self.assertTrue(active.is_file(), (SCRATCH/'chrome.log').read_text()[-3000:])
                port = active.read_text().splitlines()[0]
                target = json.loads(urlopen(Request(f'http://127.0.0.1:{port}/json/new', method='PUT')).read())
                cdp = Cdp(target['webSocketDebuggerUrl'])
                for domain in ('Page.enable', 'Runtime.enable', 'Network.enable', 'Log.enable'):
                    cdp.call(domain)
                url = f'http://127.0.0.1:{server.server_port}/head'
                cdp.call('Page.navigate', dict(url=url))
                cdp.wait_for('window.__portraitReady === true')
                cdp.wait_for("!document.getElementById('portrait-generate').disabled")
                click = lambda selector: cdp.evaluate(f'document.querySelector({json.dumps(selector)}).click()')
                self.assertEqual(cdp.evaluate("document.getElementById('portrait-model').value"), 'qwen-image21')
                cdp.evaluate("document.getElementById('portrait-model').value='klein3'; document.getElementById('portrait-model').dispatchEvent(new Event('change'))")
                cdp.wait_for("document.getElementById('portrait-steps').value === '16'")
                self.assertEqual(cdp.evaluate("document.getElementById('portrait-view-steps').value"), '24')
                click('#portrait-generate')
                cdp.wait_for("!document.getElementById('portrait-image').hidden && !document.getElementById('portrait-approved').disabled")
                self.assertTrue(cdp.evaluate("document.getElementById('portrait-generate-views').disabled"))
                click('#portrait-approved')
                cdp.wait_for("!document.getElementById('portrait-generate-views').disabled")
                click('#portrait-generate-views')
                cdp.wait_for("document.querySelectorAll('#portrait-views img').length === 4 && !document.getElementById('portrait-approved').disabled")
                click('#portrait-views input[type=checkbox]')
                cdp.wait_for("!document.getElementById('portrait-reconstruct-views').disabled")
                click('#portrait-views img')
                self.assertTrue(cdp.evaluate("document.getElementById('portrait-lightbox').open"))
                click('#portrait-close-compare')
                click('[name=portrait-method][value=sheet]')
                click('#portrait-generate-views')
                cdp.wait_for("document.querySelectorAll('#portrait-sheets button').length === 1 && !document.getElementById('portrait-approved').disabled")
                # Reload restores review and generated views from the server.
                cdp.call('Page.navigate', dict(url=url))
                cdp.wait_for('window.__portraitReady === true')
                self.assertTrue(cdp.evaluate("document.getElementById('portrait-approved').checked"))
                self.assertEqual(cdp.evaluate("document.getElementById('portrait-model').value"), 'klein3')
                self.assertEqual(cdp.evaluate("document.getElementById('portrait-steps').value"), '16')
                self.assertEqual(cdp.evaluate("document.querySelectorAll('#portrait-views img').length"), 4)
                screenshot = cdp.call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': True})
                (SCRATCH/'portrait-editor-desktop.png').write_bytes(base64.b64decode(screenshot['data']))
                cdp.call('Emulation.setDeviceMetricsOverride', dict(width=390, height=844, deviceScaleFactor=1, mobile=True))
                self.assertTrue(cdp.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
                screenshot = cdp.call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': True})
                (SCRATCH/'portrait-editor-mobile.png').write_bytes(base64.b64decode(screenshot['data']))
                # Switching generators must invalidate approval just like a new seed.
                cdp.evaluate("document.getElementById('portrait-model').value='qwen-image21'; document.getElementById('portrait-model').dispatchEvent(new Event('change'))")
                cdp.wait_for("document.getElementById('portrait-steps').value === '24' && !document.getElementById('portrait-generate').disabled")
                click('#portrait-generate')
                cdp.wait_for("document.querySelectorAll('#portrait-history button').length === 2 && !document.getElementById('portrait-approved').disabled")
                self.assertFalse(cdp.evaluate("document.getElementById('portrait-approved').checked"))
                self.assertEqual(cdp.evaluate("document.querySelectorAll('#portrait-views img').length"), 0)
                self.assertTrue(cdp.evaluate("document.getElementById('portrait-reconstruct-views').disabled"))
                errors = cdp.console_errors()
                self.assertFalse(errors, errors)
            finally:
                if cdp:
                    cdp.close()
                chrome.terminate()
                try:
                    chrome.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    chrome.kill()
                    chrome.wait()
                log.close()
                server.shutdown()
                server.server_close()
                server_thread.join()


if __name__ == '__main__':
    unittest.main()
