"""HTTP tests for the eye demo server (mock Qwen and Pixal3D, no GPU).

    python3 -m unittest server.vhuman.test_app -v
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path

from .app import App, make_handler
from .service import ROOT

TMP = ROOT / "tmp" / "vhuman" / "test-app"


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TMP.mkdir(parents=True, exist_ok=True)
        cls.work = Path(tempfile.mkdtemp(prefix="work-", dir=TMP))
        args = argparse.Namespace(work=str(cls.work), qwen_python=None, mock=True, host="127.0.0.1", port=0)
        cls.app = App(args)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.app, quiet=True))
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        shutil.rmtree(cls.work, ignore_errors=True)

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=60) as r:
            return r.status, r.headers.get("Content-Type"), r.read()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, r.headers.get("Content-Type"), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type"), e.read()

    def status_of(self, path):
        try:
            return self.get(path)[0]
        except urllib.error.HTTPError as e:
            return e.code

    def wait_job(self, job_id, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = json.loads(self.get(f"/v1/jobs/{job_id}")[2])
            if job["state"] in ("done", "failed", "cancelled"):
                return job
            time.sleep(0.2)
        self.fail("job did not finish")

    def test_rig_page_and_routes(self):
        status, ctype, body = self.get("/rig")
        self.assertEqual(status, 200)
        self.assertIn(b"class LinearRig", body)
        status, _, body = self.get("/rig?head=abc&take=123")
        self.assertEqual(status, 200)
        self.assertIn(b"params.get('take')", body)
        self.assertIn("rig", json.loads(self.get("/health")[2]))
        for bad in ("/v1/heads/abc/rig/rig.glb", "/v1/heads/abc/rig/../fit.json", "/v1/heads/abc/rig/fit_cache.pkl"):
            self.assertEqual(self.status_of(bad), 404)
        status, _, body = self.post("/v1/jobs", {"kind": "rig", "head_id": "missing"})
        self.assertEqual(status, 202)
        job = self.wait_job(json.loads(body)["id"])
        self.assertEqual(job["state"], "failed")

    def test_speech_take_job_and_routes(self):
        from .test_rig import _toy_rig
        from .test_speech import _aux
        head_id = "speecht1"
        rig = self.work / "heads" / head_id / "rig"
        source = rig / "takes" / ("a" * 12)
        source.mkdir(parents=True, exist_ok=True)
        (rig / "rig.json").write_text(json.dumps(_toy_rig()))
        (rig / "rig.usda").write_text("#usda 1.0\n")
        (source / "manifest.json").write_text(json.dumps({"id": source.name, "head_id": head_id, "duration": .5}))
        (source / "align.json").write_text(json.dumps(_aux()))
        with wave.open(str(source / "audio.wav"), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
            wav.writeframes(bytes(24000))
        status, _, raw = self.post("/v1/jobs", {"kind": "rig_speech", "head_id": head_id,
                                                   "source_take": source.name})
        self.assertEqual(status, 202)
        job = self.wait_job(json.loads(raw)["id"])
        self.assertEqual(job["state"], "done", job.get("error"))
        take = job["result"]
        listing = json.loads(self.get(f"/v1/heads/{head_id}/rig/takes")[2])["takes"]
        self.assertTrue(any(t["id"] == take["id"] for t in listing))
        self.assertEqual(self.status_of(take["urls"]["audio"]), 200)
        self.assertEqual(self.status_of(take["urls"]["animation"]), 200)
        result_dir = rig / "takes" / take["id"]
        (result_dir / "soft_tissue.usda").write_text("#usda 1.0\n")
        (result_dir / "soft_tissue_report.json").write_text('{"format":"vhuman.soft_tissue.v1"}')
        updated = self.app.service.take_summary(head_id, take["id"])
        self.assertEqual(self.status_of(updated["urls"]["soft_tissue"]), 200)
        self.assertEqual(self.status_of(updated["urls"]["soft_tissue_report"]), 200)
        request = urllib.request.Request(self.base + take["urls"]["audio"], headers={"Range": "bytes=0-9"})
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), (source / "audio.wav").read_bytes()[:10])
        self.assertEqual(self.status_of(f"/v1/heads/{head_id}/rig/takes/{take['id']}/../rig.json"), 404)
        raw = self.post("/v1/jobs", {"kind": "rig_speech", "head_id": head_id,
                                     "wav": str(source / "audio.wav")})[2]
        self.assertEqual(self.wait_job(json.loads(raw)["id"])["state"], "failed")

    def test_page_health_schema(self):
        status, ctype, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        status, ctype, body = self.get("/vhuman_eye_shader.js")          # shared by both pages
        self.assertEqual(status, 200)
        self.assertIn("javascript", ctype)
        self.assertIn(b"export const EYE_FRAG", body)
        health = json.loads(self.get("/health")[2])
        self.assertTrue(health["ok"])
        self.assertTrue(health["qwen"]["available"])
        self.assertIn("rig_speech", health)
        self.assertIn("rig_soft_tissue", health)
        self.assertIn("rig_emotion", health)
        schema = json.loads(self.get("/v1/eye/schema")[2])
        self.assertEqual(len(schema["presets"]), 12)

    def test_textures_cached_and_served(self):
        body = {"params": {"iris": {"pattern": "pattern_4"}, "structure": {"seed": 77}}, "res": 512}
        status, _, raw = self.post("/v1/eye/textures", body)
        self.assertEqual(status, 200)
        first = json.loads(raw)
        self.assertEqual(set(first["urls"]), {"iris_masks", "iris_normal", "sclera_masks", "sclera_normal",
                                              "iris_color_chart"})
        self.assertEqual(first["uniforms"]["uvMapping"], "angular-two-segment-v1")
        t = time.perf_counter()
        second = json.loads(self.post("/v1/eye/textures", body)[2])
        self.assertLess(time.perf_counter() - t, 0.5)
        self.assertEqual(first["key"], second["key"])
        # a live parameter does not change the texture key
        live = json.loads(self.post("/v1/eye/textures", {**body, "params": {**body["params"],
                                                                           "pupil": {"dilation": 1.1}}})[2])
        self.assertEqual(live["key"], first["key"])
        status, ctype, png = self.get(first["urls"]["iris_masks"])
        self.assertEqual((status, ctype, png[:4]), (200, "image/png", b"\x89PNG"))

    def test_export_and_render(self):
        status, _, raw = self.post("/v1/eye/export", {"params": {}, "res": 512, "formats": ["glb", "textures"]})
        self.assertEqual(status, 200)
        urls = json.loads(raw)["urls"]
        status, ctype, glb = self.get(urls["glb"])
        self.assertEqual((ctype, glb[:4]), ("model/gltf-binary", b"glTF"))
        self.assertEqual(self.get(urls["textures_zip"])[2][:2], b"PK")
        status, ctype, png = self.post("/v1/eye/render", {"params": {}, "size": 64, "spp": 1})
        self.assertEqual((status, ctype), (200, "image/png"))

    def test_errors_and_traversal(self):
        self.assertEqual(self.post("/v1/eye/textures", {"params": {"iris": {"bogus": 1}}})[0], 400)
        self.assertEqual(self.post("/v1/eye/textures", {"res": 999})[0], 400)
        self.assertEqual(self.post("/v1/eye/export", {"formats": ["obj"]})[0], 400)
        self.assertEqual(self.post("/v1/jobs", {"kind": "mine-bitcoin"})[0], 400)
        self.assertEqual(self.status_of("/v1/eye/files/aaaaaaaaaaaaaaaaaaaaaaaa/..%2F..%2Fjobs"), 404)
        self.assertIn(self.status_of("/v1/eye/files/../../etc/passwd"), (400, 404))
        self.assertEqual(self.status_of("/v1/plates/nope/thumb.png"), 404)
        self.assertEqual(self.status_of("/v1/baselines/abc/../../x"), 404)
        self.assertEqual(self.status_of("/v1/jobs/unknown"), 404)
        req = urllib.request.Request(self.base + "/v1/eye/textures", b"x" * (2 << 20),
                                     {"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=30)
        self.assertEqual(ctx.exception.code, 413)

    def test_plate_job_then_plate_detail(self):
        status, _, raw = self.post("/v1/jobs", {"kind": "plates", "count": 2, "seed": 4, "color": "green"})
        self.assertEqual(status, 202)
        job = self.wait_job(json.loads(raw)["id"])
        self.assertEqual(job["state"], "done", job.get("error"))
        plates = json.loads(self.get("/v1/plates")[2])["plates"]
        self.assertTrue(plates)
        pid = plates[0]["id"]
        self.assertEqual(self.get(plates[0]["thumb_url"])[1], "image/png")
        tex = json.loads(self.post("/v1/eye/textures", {"params": {}, "res": 512,
                                                        "detail": {"source": "plate", "id": pid}})[2])
        self.assertIn("iris_photo", tex["urls"])
        exp = json.loads(self.post("/v1/eye/export", {"params": {}, "res": 512, "formats": ["glb"],
                                                      "detail": {"source": "plate", "id": pid}})[2])
        self.assertIn("glb", exp["urls"])

    def test_baseline_job(self):
        raw = self.post("/v1/jobs", {"kind": "baseline", "source": "analytic"})[2]
        job = self.wait_job(json.loads(raw)["id"])
        self.assertEqual(job["state"], "done", job.get("error"))
        listing = json.loads(self.get("/v1/baselines")[2])["baselines"]
        self.assertTrue(listing)
        self.assertEqual(self.get(listing[0]["glb_url"])[2][:4], b"glTF")

    def test_head_job_and_routes(self):
        status, ctype, _ = self.get("/head")
        self.assertEqual(status, 200)
        raw = self.post("/v1/jobs", {"kind": "head", "subject": "a test person", "seed": 5, "quality": "preview"})[2]
        job = self.wait_job(json.loads(raw)["id"], timeout=180)
        self.assertEqual(job["state"], "done", job.get("error"))
        heads = json.loads(self.get("/v1/heads")[2])["heads"]
        self.assertTrue(heads)
        urls = heads[0]["urls"]
        self.assertEqual(self.get(urls["head_eyes"])[2][:4], b"glTF")
        self.assertEqual(self.get(urls["portrait"])[1], "image/png")
        self.assertEqual(self.status_of(f"/v1/heads/{heads[0]['id']}/secret.txt"), 404)
        bad = self.post("/v1/jobs", {"kind": "head", "subject": "x", "fov": 200})[2]
        self.assertEqual(self.wait_job(json.loads(bad)["id"])["state"], "failed")

    def test_cancel_queued_job(self):
        ids = [json.loads(self.post("/v1/jobs", {"kind": "plates", "count": 1, "seed": 50 + k})[2])["id"]
               for k in range(3)]
        cancelled = json.loads(self.post(f"/v1/jobs/{ids[-1]}/cancel", {})[2])
        self.assertIn(cancelled["state"], ("cancelled", "running", "done"))
        for job_id in ids:
            self.assertIn(self.wait_job(job_id)["state"], ("done", "cancelled"))


if __name__ == "__main__":
    unittest.main()
