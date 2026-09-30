#!/usr/bin/env python3
"""Headless-Chrome test for the virtual-human eye demo (web/vhuman_eye.html).

Starts the demo server (--mock, a scratch work dir under tmp/vhuman/), drives
Chrome over the DevTools protocol (WebGL through SwiftShader) and checks:
- the page renders its first textured frame (window.__eyeReady);
- no console errors or uncaught exceptions;
- a live slider (pupil dilation) changes uniforms only: no texture request;
- a structure change (seed) requests new textures;
- the GLB export returns a glTF binary;
- the head page (a mock head job) draws its analytic eyes: a shader proxy
  per eye, the GLB's eye meshes hidden, the eye drawn on the canvas.

    python3 -m unittest server.vhuman.test_browser -v

Skips when Chrome is missing or unpkg.com (three.js) is unreachable.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "tmp" / "vhuman" / "test-browser"
THREE_URL = "https://unpkg.com/three@0.163.0/build/three.module.js"


def find_chrome() -> str | None:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return None


def unpkg_reachable() -> bool:
    try:
        with urlopen(Request(THREE_URL, method="HEAD"), timeout=8) as r:
            return r.status == 200
    except (URLError, OSError):
        return False


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Cdp:
    """Minimal stdlib WebSocket client for the DevTools protocol (after
    server/pixal3d/test_browser.py); events are buffered in self.events."""

    def __init__(self, url: str):
        host, rest = url.removeprefix("ws://").split("/", 1)
        hostname, port = host.split(":")
        self.sock = socket.create_connection((hostname, int(port)), timeout=30)
        key = base64.b64encode(b"vhuman-browser-test!").decode()
        self.sock.sendall((f"GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                           "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        response = b""
        while b"\r\n\r\n" not in response:
            response += self.sock.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(response.decode(errors="replace"))
        self.next_id = 1
        self.events: list[dict] = []

    def _exact(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise EOFError("DevTools socket closed")
            data += chunk
        return data

    def _read(self) -> dict:
        first, second = self._exact(2)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._exact(8))[0]
        data = self._exact(length)
        if (first & 0x0F) == 8:
            raise EOFError("Chrome closed the DevTools socket")
        return json.loads(data)

    def call(self, method: str, params: dict | None = None) -> dict:
        call_id = self.next_id
        self.next_id += 1
        payload = json.dumps({"id": call_id, "method": method, "params": params or {}}).encode()
        mask = b"VHEM"
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if len(payload) < 126:
            header = bytes([0x81, 0x80 | len(payload)])
        elif len(payload) < 65536:
            header = bytes([0x81, 0x80 | 126]) + struct.pack("!H", len(payload))
        else:
            header = bytes([0x81, 0x80 | 127]) + struct.pack("!Q", len(payload))
        self.sock.sendall(header + mask + body)
        while True:
            message = self._read()
            if message.get("id") == call_id:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})
            if "method" in message:
                self.events.append(message)

    def evaluate(self, expression: str):
        result = self.call("Runtime.evaluate", {"expression": expression, "awaitPromise": True,
                                                "returnByValue": True})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    def wait_for(self, expression: str, timeout: float = 60.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.evaluate(expression)
            if value:
                return value
            time.sleep(0.2)
        raise AssertionError(f"timed out waiting for: {expression}")

    def console_errors(self) -> list[str]:
        out = []
        for e in self.events:
            m, p = e["method"], e.get("params", {})
            if m == "Runtime.consoleAPICalled" and p.get("type") == "error":
                out.append(" ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args", [])))
            elif m == "Runtime.exceptionThrown":
                out.append(json.dumps(p.get("exceptionDetails", {}))[:400])
            elif m == "Log.entryAdded" and p.get("entry", {}).get("level") == "error":
                out.append(p["entry"].get("text", "") + " " + p["entry"].get("url", ""))
        return out

    def requests_to(self, fragment: str) -> int:
        return sum(1 for e in self.events if e["method"] == "Network.requestWillBeSent"
                   and fragment in e["params"]["request"]["url"])

    def close(self) -> None:
        self.sock.close()


@unittest.skipUnless(find_chrome(), "Chrome/Chromium is not installed")
class EyePageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not unpkg_reachable():
            raise unittest.SkipTest("unpkg.com (three.js) is unreachable")
        SCRATCH.mkdir(parents=True, exist_ok=True)
        cls.work = Path(tempfile.mkdtemp(prefix="run-", dir=SCRATCH))
        cls.port = free_port()
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
        cls.log = (cls.work / "server.log").open("w")
        cls.server = subprocess.Popen(
            [sys.executable, "-m", "server.vhuman.app", "--port", str(cls.port), "--mock",
             "--work", str(cls.work / "work")],
            cwd=ROOT, env=env, stdout=cls.log, stderr=subprocess.STDOUT)
        cls.base = f"http://127.0.0.1:{cls.port}"
        deadline = time.monotonic() + 30
        while True:
            try:
                urlopen(cls.base + "/health", timeout=2).read()
                break
            except (URLError, OSError):
                if time.monotonic() > deadline or cls.server.poll() is not None:
                    cls.tearDownClass()
                    raise RuntimeError("the demo server did not start: "
                                       + (cls.work / "server.log").read_text()[-2000:])
                time.sleep(0.2)
        profile = cls.work / "chrome-profile"
        cls.chrome = subprocess.Popen(
            [find_chrome(), "--headless=new", "--no-sandbox", "--use-angle=swiftshader",
             "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--window-size=1280,900",
             "--remote-debugging-port=0", f"--user-data-dir={profile}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        active = profile / "DevToolsActivePort"
        deadline = time.monotonic() + 20
        while not active.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        port = int(active.read_text().splitlines()[0])
        target = json.loads(urlopen(Request(f"http://127.0.0.1:{port}/json/new", method="PUT"), timeout=5).read())
        cls.cdp = Cdp(target["webSocketDebuggerUrl"])
        for domain in ("Runtime.enable", "Log.enable", "Network.enable", "Page.enable"):
            cls.cdp.call(domain)
        try:
            cls.cdp.call("Page.setDownloadBehavior", {"behavior": "deny"})
        except RuntimeError:
            pass
        cls.cdp.call("Page.navigate", {"url": cls.base + "/"})

    @classmethod
    def tearDownClass(cls):
        for name in ("cdp",):
            obj = getattr(cls, name, None)
            if obj:
                obj.close()
        for name in ("chrome", "server"):
            proc = getattr(cls, name, None)
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if getattr(cls, "log", None):
            cls.log.close()
        if getattr(cls, "work", None):
            shutil.rmtree(cls.work, ignore_errors=True)

    def test_skin_diffusion_composition(self):
        """Float passes preserve constant diffuse and mask opaque non-skin."""
        self.cdp.call("Page.navigate", {"url": self.base + "/rig"})
        self.cdp.wait_for("!!(document.querySelector('#reconstruction') && window.__skinRenderer)")
        result = self.cdp.evaluate("""(async()=>{
          const THREE=await import('three'), {SkinRenderer}=await import('/vhuman_skin_shader.js');
          const renderer=new THREE.WebGLRenderer({preserveDrawingBuffer:true});renderer.setSize(64,64);
          renderer.toneMapping=THREE.NeutralToneMapping;renderer.outputColorSpace=THREE.SRGBColorSpace;
          const skin=new SkinRenderer(renderer);if(!skin.supported){renderer.dispose();return {skip:true};}
          const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(40,1,.01,10);camera.position.z=1;
          const material=new THREE.MeshStandardMaterial({name:'skin',color:0xcc8877,roughness:.5});
          scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),material));
          const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(.2,.4,1);scene.add(light);
          const blocker=new THREE.Mesh(new THREE.SphereGeometry(.05,16,16),new THREE.MeshBasicMaterial({color:0}));
          blocker.position.set(.12,0,.1);scene.add(blocker);
          skin.patch(scene);
          function read(){const bytes=new Uint8Array(64*64*4);renderer.getContext().readPixels(0,0,64,64,renderer.getContext().RGBA,renderer.getContext().UNSIGNED_BYTE,bytes);return bytes;}
          skin.render(scene,camera);const before=read();skin.enabled=true;skin.render(scene,camera);const after=read();
          const point=(b,x,y)=>Array.from(b.slice((y*64+x)*4,(y*64+x)*4+4));
          const result={before:point(before,24,32),after:point(after,24,32),blocked:point(after,44,32)};
          skin.dispose();renderer.dispose();return result;
        })()""")
        if result.get("skip"):
            self.skipTest("float WebGL2 targets unavailable")
        self.assertGreater(sum(result["after"][:3]), 100)
        for a, b in zip(result["before"], result["after"]):
            self.assertLessEqual(abs(a-b), 3)
        self.assertLess(sum(result["blocked"][:3]), 10)

    def test_page(self):
        cdp = self.cdp
        cdp.wait_for("window.__eyeReady === true", timeout=180)
        state = cdp.evaluate("({key: __eyeState.texKey, n: __eyeState.textureRequests})")
        self.assertTrue(state["key"])
        self.assertGreaterEqual(cdp.requests_to("/v1/eye/textures"), 1)
        # The shader actually drew the eye: the canvas centre is not transparent.
        drawn = cdp.evaluate("""(() => { const c = document.querySelector('#gl-pane canvas');
            const g = c.getContext('webgl2'); const px = new Uint8Array(4);
            g.readPixels(c.width >> 1, c.height >> 1, 1, 1, g.RGBA, g.UNSIGNED_BYTE, px); return Array.from(px); })()""")
        self.assertEqual(drawn[3], 255, drawn)

        # A live slider: uniforms only.
        before_net, before_state = cdp.requests_to("/v1/eye/textures"), state["n"]
        cdp.evaluate("(() => { const d = document.getElementById('f-pupil-dilation'); d.value = 1.15;"
                     " d.dispatchEvent(new Event('input')); })()")
        time.sleep(1.5)
        cdp.evaluate("1")                                  # drain events
        self.assertEqual(cdp.requests_to("/v1/eye/textures"), before_net)
        self.assertEqual(cdp.evaluate("__eyeState.textureRequests"), before_state)
        self.assertAlmostEqual(cdp.evaluate("__eyeState.params.pupil.dilation"), 1.15, places=3)   # slider step

        # A structure change: new textures.
        cdp.evaluate("(() => { const s = document.getElementById('f-structure-seed'); s.value = 4242;"
                     " s.dispatchEvent(new Event('change')); })()")
        cdp.wait_for(f"__eyeState.textureRequests > {before_state} && __eyeState.texKey !== '{state['key']}'"
                     " && !document.getElementById('gl-pane').classList.contains('busy')", timeout=120)
        self.assertGreater(cdp.requests_to("/v1/eye/textures"), before_net)
        self.assertEqual(cdp.evaluate("__eyeState.params.structure.seed"), 4242)

        # GLB export.
        cdp.evaluate("document.getElementById('dl-glb').click()")
        url = cdp.wait_for("__eyeState.lastExport && __eyeState.lastExport.urls.glb", timeout=120)
        self.assertTrue(url.endswith("/eye.glb"), url)
        magic = cdp.evaluate(f"fetch({json.dumps(url)}).then(r => r.arrayBuffer())"
                             ".then(b => String.fromCharCode(...new Uint8Array(b.slice(0, 4))))")
        self.assertEqual(magic, "glTF")

        cdp.evaluate("1")
        self.assertEqual(cdp.console_errors(), [])
        self.assertEqual(cdp.evaluate("__eyeState.errors"), [])


    def test_page_head(self):
        """After test_page (alphabetical): a mock head, then the head page."""
        cdp = self.cdp
        job = cdp.evaluate("fetch('/v1/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'},"
                           " body: JSON.stringify({kind: 'head', subject: 'test person', seed: 4, quality: 'preview'})})"
                           ".then(r => r.json())")
        cdp.wait_for(f"fetch('/v1/jobs/{job['id']}').then(r => r.json()).then(j => j.state === 'done')", timeout=300)
        cdp.call("Page.navigate", {"url": self.base + "/head"})
        cdp.wait_for("window.__headReady === true", timeout=180)
        state = cdp.evaluate("""(() => { const s = __headState; const out = {analytic: s.eyesAnalytic, proxies: [], hidden: []};
            s.model.traverse(o => { if (o.userData.analyticEye) out.proxies.push([o.name, o.visible]);
              if (/^eye_(left|right)_(shell|iris)$/.test(o.name)) out.hidden.push(!o.visible); });
            return out; })()""")
        self.assertTrue(state["analytic"])
        self.assertEqual(sorted(n for n, _ in state["proxies"]), ["eye_left_analytic", "eye_right_analytic"])
        self.assertTrue(all(v for _, v in state["proxies"]))
        self.assertTrue(state["hidden"] and all(state["hidden"]))
        skin = cdp.evaluate("""(() => { let out; __headState.model.traverse(o => {
            if (o.isMesh && o.material.name === 'head') out = {
                normal: !!o.material.normalMap, roughness: !!o.material.roughnessMap,
                tangents: !!o.geometry.attributes.tangent}; });
            return {...out, downloads: document.querySelectorAll('#skin-downloads a').length}; })()""")
        self.assertTrue(skin["normal"] and skin["roughness"] and skin["tangents"])
        self.assertEqual(skin["downloads"], 5)
        # Wet margins remain visible with either analytic or portable eyes.
        for analytic in (False, True):
            wet = cdp.evaluate("""(() => {
                const t = document.getElementById('analytic'); t.checked = %s;
                t.dispatchEvent(new Event('change'));
                const out = []; __headState.model.traverse(o => {
                    if (/^eye_(left|right)_(tearline|caruncle)$/.test(o.name))
                        out.push({name: o.name, visible: o.visible, roughness: o.material.roughness});
                }); return out;
            })()""" % ("true" if analytic else "false"))
            self.assertEqual(len(wet), 4)
            self.assertTrue(all(m["visible"] and m["roughness"] < 0.3 for m in wet))
        # Shadow preview changes rendered pixels in both modes, without new
        # textures or changing wet-margin opacity. Repeated input is absolute.
        requests = cdp.requests_to('/v1/eye/textures')
        for analytic in (False, True):
            shadow = cdp.evaluate("""(() => {
                document.querySelector('[data-view=eyes]').click();
                const t = document.getElementById('analytic'); t.checked = %s;
                t.dispatchEvent(new Event('change'));
                const s = __headState, slider = document.getElementById('lid-shadow');
                const canvas = document.querySelector('#viewer canvas'), gl = canvas.getContext('webgl2');
                const wet = () => { const a = []; s.model.traverse(o => {
                    if (/_(tearline|caruncle)$/.test(o.name)) a.push(o.material.opacity);
                }); return a; };
                const before = wet();
                const set = v => { slider.value = v; slider.dispatchEvent(new Event('input')); };
                const read = () => { s.render(); const a = new Uint8Array(canvas.width * canvas.height * 4);
                    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, a); return a; };
                set(0); const off = read();
                set(.5); set(.5);
                const half = ['left', 'right'].map(side => s.model.getObjectByName(`eye_${side}_eyeshell`).material.opacity);
                set(1); const on = read();
                let changed = 0, brightened = 0;
                for (let i = 0; i < on.length; i += 4) {
                    if (on[i] !== off[i] || on[i + 1] !== off[i + 1] || on[i + 2] !== off[i + 2]) changed++;
                    if (on[i] > off[i] + 1 || on[i + 1] > off[i + 1] + 1 || on[i + 2] > off[i + 2] + 1) brightened++;
                }
                return {changed, brightened, half, before, after: wet(), value: document.getElementById('lid-shadow-value').value};
            })()""" % ("true" if analytic else "false"))
            self.assertGreater(shadow['changed'], 10, shadow)
            self.assertEqual(shadow['brightened'], 0, shadow)
            self.assertEqual(shadow['half'], [0.5, 0.5])
            self.assertEqual(shadow['before'], shadow['after'])
            self.assertEqual(shadow['value'], '1.00')
        self.assertEqual(cdp.requests_to('/v1/eye/textures'), requests)
        # the eyes close-up: at each eye centre's projection, the analytic
        # shader draws (the pixel changes when its proxies are hidden)
        drawn = cdp.evaluate("""(() => { document.querySelector('[data-view=eyes]').click();
            const s = __headState, c = document.querySelector('#viewer canvas'), g = c.getContext('webgl2');
            const read = () => { s.render(); return s.model.getObjectByName('eye_right').children.length &&
              ['eye_right', 'eye_left'].map(n => { const p = s.model.getObjectByName(n).getWorldPosition(s.camera.position.clone())
                .project(s.camera); const px = new Uint8Array(4);
                g.readPixels(Math.round((p.x * 0.5 + 0.5) * c.width), Math.round((p.y * 0.5 + 0.5) * c.height), 1, 1,
                             g.RGBA, g.UNSIGNED_BYTE, px); return Array.from(px); }); };
            const on = read(); s.model.traverse(o => { if (o.userData.analyticEye) o.visible = false; });
            const off = read(); s.model.traverse(o => { if (o.userData.analyticEye) o.visible = true; }); s.render();
            return {on, off}; })()""")
        for on, off in zip(drawn["on"], drawn["off"]):
            self.assertEqual(on[3], 255, drawn)
            self.assertNotEqual(on[:3], off[:3], drawn)
        # The illumination control must reach the CPU variant job and its
        # persisted metadata, including a safe unsupported-fit result.
        original = cdp.evaluate("__headState.current.id")
        cdp.evaluate("document.getElementById('skin-delight').value = .7; document.getElementById('apply-skin').click()")
        cdp.wait_for(f"window.__headReady && __headState.current.id !== {json.dumps(original)}", timeout=180)
        variant = cdp.evaluate("({id: __headState.current.id, skin: __headState.current.skin})")
        persisted = json.loads((self.work / 'work' / 'heads' / variant['id'] / 'head.json').read_text())
        self.assertEqual(persisted['source_head'], original)
        self.assertEqual(variant['skin']['params']['delight_strength'], .7)
        self.assertNotEqual(variant['skin']['illumination']['status'], 'disabled')
        # Force an older GLB (then an older fit) to finish after another
        # selection. It must neither add a second head nor attach wrong eyes.
        for delayed in ('head_eyes', 'fit'):
            cdp.evaluate("""(() => {
                const first = __headState.heads.find(h => h.id === %s);
                const second = __headState.heads.find(h => h.id !== first.id);
                const key = %s, old = first.urls[key];
                first.urls[key] = old + '?race=' + key;
                const fetchOriginal = window.fetch;
                window.__raceHeld = false;
                window.fetch = async function(input, options) {
                    const url = typeof input === 'string' ? input : input.url;
                    if (url.includes('?race=' + key)) {
                        window.__raceHeld = true;
                        await new Promise(resolve => { window.__raceRelease = resolve; });
                    }
                    return fetchOriginal.call(this, input, options);
                };
                window.__raceRestore = () => { window.fetch = fetchOriginal; first.urls[key] = old; };
                window.__raceSecond = second.id;
                document.querySelector('.head[data-id="' + first.id + '"]').click();
            })()""" % (json.dumps(original), json.dumps(delayed)))
            cdp.wait_for("window.__raceHeld", timeout=30)
            cdp.evaluate("document.querySelector('.head[data-id=\"' + __raceSecond + '\"]').click()")
            cdp.wait_for("window.__headReady && __headState.current.id === __raceSecond", timeout=180)
            cdp.evaluate("window.__raceModel = __headState.model; __raceRelease(); __raceRestore()")
            time.sleep(2)
            race = cdp.evaluate("""(() => {
                const s = __headState; let proxies = 0;
                s.model.traverse(o => { if (o.userData.analyticEye) proxies++; });
                return {same: s.model === __raceModel, ready: __headReady, proxies,
                    groups: s.model.parent.children.filter(o => o.type === 'Group').length};
            })()""")
            self.assertEqual(race, {'same': True, 'ready': True, 'proxies': 2, 'groups': 1})
        self.assertEqual(cdp.console_errors(), [])


if __name__ == "__main__":
    unittest.main()
