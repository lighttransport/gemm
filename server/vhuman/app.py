#!/usr/bin/env python3
"""Virtual-human eye demo server (standalone, stdlib only).

    python3 -m server.vhuman.app [--port 8790] [--qwen-python tmp/qimg21-ref-venv/bin/python] [--mock]

CPU endpoints answer synchronously (procedural textures take well under a
second and are cached); GPU work (Qwen iris plates, the Pixal3D baseline)
runs as jobs on one worker, under the backend/device lock shared with the
Pixal3D demo server.

    GET  /                          the page (web/vhuman_eye.html)
    GET  /health
    GET  /v1/eye/schema             parameters, groups, presets
    POST /v1/eye/textures           {params, res, detail} -> texture URLs + shader uniforms
    POST /v1/eye/export             {params, res, formats: [glb, ue], pair, detail} -> download URLs
    POST /v1/eye/render             {params, size, yaw, pitch, fov, detail} -> PNG (CPU ray tracer)
    GET  /v1/eye/files/<key>/<name>
    GET  /v1/plates                 the iris plate library
    GET  /v1/plates/<id>/<file>
    GET  /v1/baselines, /v1/baselines/<id>/<file>   Pixal3D baseline runs
    GET  /head                      the head page (web/vhuman_head.html)
    GET  /vhuman_eye_shader.js      the analytic eye shader, shared by both pages
    GET  /v1/heads, /v1/heads/<id>/<file>           Qwen portrait -> Pixal3D head -> fitted eyes
    GET  /rig                       the facial rig page (web/vhuman_rig.html)
    GET  /v1/heads/<id>/rig/<file>  rig outputs
    GET  /v1/heads/<id>/body/<file>  combined avatar outputs
    GET  /v1/heads/<id>/body/motions[...]
    POST /v1/body/uploads           raw image/video bytes -> upload id
    GET  /v1/heads/<id>/rig/takes, /v1/heads/<id>/rig/takes/<take>/<file>
    POST /v1/jobs                   {kind: plates|baseline|head|head_skin|expressions|rig|body|rig_speech, ...} -> {id}
    GET  /v1/jobs, /v1/jobs/<id>    POST /v1/jobs/<id>/cancel
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import gpu
from .eye import params as P
from .service import ROOT, WORK, EyeService, ServiceError

PAGE = ROOT / "web" / "vhuman_eye.html"
HEAD_PAGE = ROOT / "web" / "vhuman_head.html"
RIG_PAGE = ROOT / "web" / "vhuman_rig.html"
EYE_SHADER = ROOT / "web" / "vhuman_eye_shader.js"      # shared by both pages
MAX_BODY = 1 << 20
MAX_JOBS_QUEUED = 8
KEEP_JOBS = 32


class JobCancelled(Exception):
    pass


class TooLarge(ValueError):
    pass


class Jobs:
    """One worker thread; jobs persist as tmp/vhuman/jobs/<id>/job.json."""

    def __init__(self, root: Path, handlers: dict):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.handlers = handlers
        self.jobs: "OrderedDict[str, dict]" = OrderedDict()
        self.cancels: dict[str, threading.Event] = {}
        self.cond = threading.Condition()
        self._restore()
        threading.Thread(target=self._worker, name="vhuman-jobs", daemon=True).start()

    def _restore(self) -> None:
        for path in sorted(self.root.glob("*/job.json"), key=lambda p: p.stat().st_mtime):
            try:
                job = json.loads(path.read_text())
            except ValueError:
                continue
            if job.get("state") in ("queued", "running"):
                job.update(state="failed", error="server restarted")
            self.jobs[job["id"]] = job

    def _save(self, job: dict) -> None:
        folder = self.root / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        partial = folder / "job.json.partial"
        partial.write_text(json.dumps(job, default=str))
        partial.replace(folder / "job.json")

    def submit(self, kind: str, request: dict) -> dict:
        if kind not in self.handlers:
            raise ServiceError(f"kind must be one of {', '.join(self.handlers)}")
        with self.cond:
            if sum(j["state"] == "queued" for j in self.jobs.values()) >= MAX_JOBS_QUEUED:
                raise ServiceError("the job queue is full")
            job = {"id": uuid.uuid4().hex, "kind": kind, "request": request, "state": "queued",
                   "progress": 0.0, "message": "queued", "log": [], "created": time.time()}
            self.jobs[job["id"]] = job
            self.cancels[job["id"]] = threading.Event()
            self._save(job)
            while len(self.jobs) > KEEP_JOBS:
                old_id, old = next(iter(self.jobs.items()))
                if old["state"] in ("queued", "running"):
                    break
                self.jobs.pop(old_id)
            self.cond.notify()
            return dict(job)

    def get(self, job_id: str) -> dict:
        with self.cond:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return json.loads(json.dumps(self.jobs[job_id], default=str))

    def list(self) -> list[dict]:
        with self.cond:
            return [{k: v for k, v in j.items() if k != "log"} for j in reversed(self.jobs.values())]

    def cancel(self, job_id: str) -> dict:
        with self.cond:
            job = self.jobs[job_id]
            if job["state"] == "queued":
                job.update(state="cancelled", message="cancelled")
                self._save(job)
            elif job["state"] == "running":
                self.cancels[job_id].set()
                job["message"] = "cancelling"
            return dict(job)

    def _worker(self) -> None:
        while True:
            with self.cond:
                while not any(j["state"] == "queued" for j in self.jobs.values()):
                    self.cond.wait()
                job = next(j for j in self.jobs.values() if j["state"] == "queued")
                job.update(state="running", started=time.time(), message="starting")
                cancel = self.cancels.setdefault(job["id"], threading.Event())
                self._save(job)

            def progress(fraction: float, message: str, _job=job) -> None:
                if cancel.is_set():
                    raise JobCancelled("cancelled")
                with self.cond:
                    _job["progress"] = round(float(fraction), 4)
                    _job["message"] = message
                    _job["log"] = (_job["log"] + [f"{time.strftime('%H:%M:%S')} {message}"])[-200:]
                    self._save(_job)

            try:
                result = self.handlers[job["kind"]](job["request"], progress, cancel)
                state, extra = "done", {"result": result, "progress": 1.0, "message": "done"}
            except (JobCancelled, gpu.Cancelled):
                state, extra = "cancelled", {"message": "cancelled"}
            except Exception as exc:  # the job fails; the worker lives on
                traceback.print_exc()
                state, extra = "failed", {"error": str(exc), "message": "failed"}
            with self.cond:
                job.update(state=state, finished=time.time(), **extra)
                self._save(job)


class App:
    def __init__(self, args):
        self.args = args
        self.service = EyeService(Path(args.work))
        from . import qwen, baseline
        from .head import pipeline as head_pipeline
        from .rig import emotion as rig_emotion, exprdata, job as rig_job, speech as rig_speech, soft_tissue, soft_deformer, video_fit
        from .body import job as body_job, motion as body_motion
        from .reconstruction.job import reconstruction_job
        from .reconstruction.upload import server_request
        self.rig_job = rig_job
        self.rig_speech = rig_speech
        self.soft_tissue = soft_tissue
        self.rig_emotion = rig_emotion
        self.body_job = body_job
        self.gpu = gpu
        self.qwen_opts = {"python": args.qwen_python, "mock": args.mock}
        handlers = {
            "plates": lambda req, prog, cancel: qwen.plates_job(self.service, req, prog, cancel, **self.qwen_opts),
            "baseline": lambda req, prog, cancel: baseline.baseline_job(self.service, req, prog, cancel,
                                                                        mock=args.mock, python=args.qwen_python),
            "head": lambda req, prog, cancel: head_pipeline.head_job(self.service, req, prog, cancel,
                                                                     python=args.qwen_python, mock=args.mock),
            "head_skin": lambda req, prog, cancel: head_pipeline.skin_job(self.service, req, prog, cancel),
            "expressions": lambda req, prog, cancel: exprdata.expressions_job(self.service, req, prog, cancel,
                                                                              python=args.qwen_python, mock=args.mock),
            "rig_refine_portrait": lambda req, prog, cancel: reconstruction_job(self.service, server_request(self.service, req), prog, cancel,
                python=getattr(args, "rig_python", None), mock=args.mock),
            "portrait_reconstruct": lambda req, prog, cancel: reconstruction_job(self.service, server_request(self.service, req, direct=True), prog, cancel,
                python=getattr(args, "rig_python", None), mock=args.mock, direct=True),
            "rig": lambda req, prog, cancel: rig_job.rig_job(self.service, req, prog, cancel,
                                                             python=getattr(args, "rig_python", None), mock=args.mock),
            "body": lambda req, prog, cancel: body_job.body_job(self.service, req, prog, cancel,
                        python=args.qwen_python, rig_python=getattr(args, "rig_python", None),
                        model_dir=getattr(args, "sam3d_body_model", body_job.MODEL_DIR),
                        sam3_model=getattr(args, "sam3_model", body_job.SAM3_MODEL),
                        clip_bpe=getattr(args, "clip_bpe", body_job.CLIP_BPE), mock=args.mock),
            "body_motion": lambda req, prog, cancel: body_motion.fit(self.service, req, prog, cancel,
                        rig_python=getattr(args, "rig_python", None) or body_job.DEFAULT_RIG_PYTHON,
                        model_dir=getattr(args, "sam3d_body_model", body_job.MODEL_DIR), mock=args.mock),
            "rig_speech": lambda req, prog, cancel: rig_speech.speech_job(
                self.service, req, prog, cancel, model=getattr(args, "tts_model", None) or rig_speech.DEFAULT_MODEL,
                aligner=getattr(args, "aligner", None) or rig_speech.DEFAULT_ALIGNER,
                backend=getattr(args, "tts_backend", "auto"),
                emotion_runner=getattr(args, "emotion_runner", None) or rig_emotion.DEFAULT_RUNNER,
                emotion_model=getattr(args, "emotion_model", None) or rig_emotion.DEFAULT_MODEL),
            "rig_soft_tissue": lambda req, prog, cancel: soft_tissue.soft_tissue_job(
                self.service, req, prog, cancel, python=getattr(args, "rig_python", None)),
            "rig_train_deformer": lambda req, prog, cancel: soft_deformer.train_job(
                self.service, req, prog, cancel, python=getattr(args, "rig_python", None)),
            "rig_fit_video": lambda req, prog, cancel: video_fit.fit_job(
                self.service, req, prog, cancel, python=getattr(args, "rig_python", None)),
        }
        def configured(handler):
            def run(request, progress, cancel):
                with gpu.execution(getattr(args, "inference_backend", "auto"),
                                   getattr(args, "device", 0), getattr(args, "models_root", "/mnt/disk1/models")) as selected:
                    result = handler(request, progress, cancel)
                    result.setdefault("backend", "mock" if args.mock else selected)
                    result.setdefault("device", gpu.device_index())
                    return result
            return run
        self.jobs = Jobs(Path(args.work) / "jobs", {name: configured(fn) for name, fn in handlers.items()})

    def health(self) -> dict:
        with gpu.execution(getattr(self.args, "inference_backend", "auto"),
                           getattr(self.args, "device", 0), getattr(self.args, "models_root", "/mnt/disk1/models")):
            return self._health()

    def _health(self) -> dict:
        from . import baseline, qwen
        return {"ok": True, "backend": gpu.backend(), "device": gpu.device_index(), "gpu": self.gpu.gpu_status(), "qwen": qwen.availability(**self.qwen_opts),
                "pixal3d": baseline.availability(mock=self.args.mock), "plates": len(self.service.list_plates()),
                "rig": self.rig_job.availability(getattr(self.args, "rig_python", None)),
                "body": self.body_job.availability(getattr(self.args, "sam3d_body_model", self.body_job.MODEL_DIR),
                                                    getattr(self.args, "rig_python", None), self.args.mock),
                "rig_speech": self.rig_speech.availability(
                    getattr(self.args, "tts_model", None) or self.rig_speech.DEFAULT_MODEL,
                    getattr(self.args, "aligner", None) or self.rig_speech.DEFAULT_ALIGNER,
                    getattr(self.args, "tts_backend", "auto")),
                "rig_soft_tissue": self.soft_tissue.availability(python=getattr(self.args, "rig_python", None)),
                "rig_emotion": self.rig_emotion.availability(
                    getattr(self.args, "emotion_runner", None) or self.rig_emotion.DEFAULT_RUNNER,
                    getattr(self.args, "emotion_model", None) or self.rig_emotion.DEFAULT_MODEL),
                "algo_version": P.ALGO_VERSION}


def make_handler(app: App, quiet: bool = False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "vhuman/1"

        def log_message(self, fmt, *args):
            if quiet or (self.path.startswith("/v1/jobs/") and self.command == "GET"):
                return
            super().log_message(fmt, *args)

        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload, default=str).encode(), "application/json")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def _file(self, path: Path, immutable: bool = False) -> None:
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix == ".glb":
                ctype = "model/gltf-binary"
            headers = {"Cache-Control": "public, max-age=31536000, immutable" if immutable else "no-cache"}
            if path.suffix in (".glb", ".zip"):
                headers["Content-Disposition"] = f'attachment; filename="{path.name}"'
            data = path.read_bytes()
            if path.suffix == ".wav":
                ctype = "audio/wav"
                headers["Accept-Ranges"] = "bytes"
                request_range = self.headers.get("Range")
                if request_range:
                    match = re.fullmatch(r"bytes=(\d+)-(\d*)", request_range)
                    if not match:
                        return self._send(416, b"", ctype, {"Content-Range": f"bytes */{len(data)}"})
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else len(data) - 1
                    if start >= len(data) or end < start:
                        return self._send(416, b"", ctype, {"Content-Range": f"bytes */{len(data)}"})
                    end = min(end, len(data) - 1)
                    headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
                    return self._send(206, data[start:end + 1], ctype, headers)
            self._send(200, data, ctype, headers)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                # Drain (bounded) so the client reads the 413 instead of a reset.
                remaining = min(length, 64 * MAX_BODY)
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 1 << 16))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                self.close_connection = True
                raise TooLarge("request body too large")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                raise ServiceError("body must be JSON") from None
            if not isinstance(body, dict):
                raise ServiceError("body must be a JSON object")
            return body

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html", "/vhuman_eye"):
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8",
                                      {"Cache-Control": "no-cache"})
                if path == "/vhuman_eye_shader.js":
                    return self._send(200, EYE_SHADER.read_bytes(), "text/javascript; charset=utf-8",
                                      {"Cache-Control": "no-cache"})
                if path in ("/head", "/vhuman_head"):
                    return self._send(200, HEAD_PAGE.read_bytes(), "text/html; charset=utf-8",
                                      {"Cache-Control": "no-cache"})
                if path in ("/rig", "/vhuman_rig"):
                    return self._send(200, RIG_PAGE.read_bytes(), "text/html; charset=utf-8",
                                      {"Cache-Control": "no-cache"})
                if path == "/v1/heads":
                    return self._json(200, {"heads": app.service.list_heads()})
                if path in ("/vhuman_skin_shader.js", "/vhuman_gaussian.js"):
                    return self._file(ROOT / "web" / path[1:])
                if path.startswith("/v1/heads/") and "/reconstruction" in path:
                    hid, _, tail = path[len("/v1/heads/"):].partition("/reconstruction")
                    if tail in ("", "/"):
                        return self._json(200, {"candidates": app.service.list_reconstructions(hid)})
                    rid, sep, name = tail.lstrip("/").partition("/")
                    if not sep:
                        return self._error(404, "no such file")
                    return self._file(app.service.reconstruction_file(hid, rid, name))
                if path.startswith("/v1/heads/") and "/rig/takes" in path:
                    hid, _, tail = path[len("/v1/heads/"):].partition("/rig/takes")
                    if tail == "":
                        return self._json(200, {"takes": app.service.list_takes(hid)})
                    parts = tail.lstrip("/").split("/")
                    if len(parts) == 2 and tail.startswith("/"):
                        return self._file(app.service.take_file(hid, parts[0], parts[1]))
                    return self._error(404, "no such file")
                if path.startswith("/v1/heads/") and "/rig/" in path:
                    hid, _, name = path[len("/v1/heads/"):].partition("/rig/")
                    return self._file(app.service.rig_file(hid, name))
                if path.startswith("/v1/heads/") and "/body/" in path:
                    hid, _, name = path[len("/v1/heads/"):].partition("/body/")
                    if name == "motions":
                        return self._json(200, {"motions": app.service.list_motions(hid)})
                    return self._file(app.service.body_file(hid, name))
                if path.startswith("/v1/heads/"):
                    return self._file(app.service.head_file(*path[len("/v1/heads/"):].split("/", 1)))
                if path == "/health":
                    return self._json(200, app.health())
                if path == "/v1/eye/schema":
                    return self._json(200, P.schema())
                if path.startswith("/v1/eye/files/"):
                    key, _, name = path[len("/v1/eye/files/"):].partition("/")
                    return self._file(app.service.file(key, name), immutable=True)
                if path == "/v1/plates":
                    return self._json(200, {"plates": app.service.list_plates()})
                if path.startswith("/v1/plates/"):
                    plate_id, _, name = path[len("/v1/plates/"):].partition("/")
                    app.service.plate(plate_id)
                    base = (app.service.plates / plate_id).resolve()
                    target = (base / name).resolve()
                    if base not in target.parents or not target.is_file():
                        return self._error(404, "no such file")
                    return self._file(target, immutable=True)
                if path == "/v1/baselines":
                    return self._json(200, {"baselines": app.service.list_baselines()})
                if path.startswith("/v1/baselines/"):
                    return self._file(app.service.baseline_file(*path[len("/v1/baselines/"):].split("/", 1)),
                                      immutable=True)
                if path == "/v1/jobs":
                    return self._json(200, {"jobs": app.jobs.list()})
                if path.startswith("/v1/jobs/"):
                    return self._json(200, app.jobs.get(path[len("/v1/jobs/"):]))
                return self._error(404, "not found")
            except KeyError:
                return self._error(404, "no such job")
            except ServiceError as exc:
                return self._error(404 if "no such" in str(exc) or "unknown" in str(exc) else 400, str(exc))

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            try:
                if path == "/v1/portrait/uploads":
                    from .reconstruction.upload import upload, MAX_UPLOAD
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > MAX_UPLOAD:
                        self.close_connection = True
                        raise TooLarge("portrait upload too large")
                    return self._json(201, upload(app.service, self.rfile, length,
                                                 self.headers.get("Content-Type", "")))
                if path == "/v1/body/uploads":
                    from .body.motion import upload, MAX_UPLOAD
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > MAX_UPLOAD:
                        self.close_connection = True
                        raise TooLarge("body media upload is too large")
                    return self._json(201, upload(app.service, self.rfile, length,
                                                  self.headers.get("Content-Type", "")))
                if path == "/v1/rig/uploads":
                    from .rig.video_fit import upload, MAX_UPLOAD
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > MAX_UPLOAD:
                        self.close_connection = True
                        raise TooLarge("face video upload is too large")
                    return self._json(201, upload(app.service, self.rfile, length,
                                                  self.headers.get("Content-Type", "")))
                body = self._body()
                if path == "/v1/eye/textures":
                    return self._json(200, app.service.textures(body.get("params") or {}, body.get("res", 1024),
                                                                body.get("detail")))
                if path == "/v1/eye/export":
                    return self._json(200, app.service.export(body.get("params") or {}, body.get("res", 2048),
                                                              tuple(body.get("formats") or ("glb",)),
                                                              bool(body.get("pair", False)), body.get("detail")))
                if path == "/v1/eye/render":
                    png = app.service.render(body.get("params") or {}, body.get("size", 384), body.get("yaw", 0.0),
                                             body.get("pitch", 0.0), body.get("fov", 0.0), body.get("spp", 4),
                                             body.get("detail"))
                    return self._send(200, png, "image/png")
                if path == "/v1/jobs":
                    kind = body.pop("kind", None)
                    return self._json(202, app.jobs.submit(kind, body))
                if path.startswith("/v1/jobs/") and path.endswith("/cancel"):
                    return self._json(200, app.jobs.cancel(path[len("/v1/jobs/"):-len("/cancel")]))
                return self._error(404, "not found")
            except KeyError:
                return self._error(404, "no such job")
            except TooLarge as exc:
                return self._error(413, str(exc))
            except (ServiceError, P.ParamError) as exc:
                return self._error(400, str(exc))
            except Exception as exc:
                traceback.print_exc()
                return self._error(500, str(exc))

    return Handler


def main(argv=None) -> int:
    from .body import job as body_job
    ap = argparse.ArgumentParser(description="Virtual-human eye demo server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    from . import runtime
    runtime.add_arguments(ap)
    ap.add_argument("--work", default=str(WORK))
    default_py = ROOT / "tmp/qimg21-ref-venv/bin/python"
    ap.add_argument("--qwen-python", default=None,
                    help="interpreter for cuda/qimg21/native_generate.py (needs torch)")
    ap.add_argument("--mock", action="store_true", help="mock Qwen and Pixal3D (no GPU)")
    default_rig = ROOT / "tmp/vhuman-rig-venv/bin/python"
    ap.add_argument("--rig-python", default=None,
                    help="interpreter for the facial rig builder (numpy, scipy, torch)")
    ap.add_argument("--sam3d-body-model", default=None,
                    help="local SAM 3D Body checkpoint directory")
    ap.add_argument("--sam3-model", default=None,
                    help="optional SAM 3 garment segmentation checkpoint")
    ap.add_argument("--clip-bpe", default=None,
                    help="directory containing garment tokenizer vocab.json and merges.txt")
    ap.add_argument("--tts-model", default=None, help="Qwen3-TTS model directory for rig_speech jobs")
    ap.add_argument("--aligner", default=None, help="ja_align.safetensors for rig_speech jobs")
    ap.add_argument("--tts-backend", choices=("auto", "cpu", "cuda", "rocm"), default="auto")
    ap.add_argument("--emotion-runner", default=None, help="SenseVoiceSmall GGUF runtime executable")
    ap.add_argument("--emotion-model", default=None, help="SenseVoiceSmall GGUF model path")
    args = ap.parse_args(argv)
    from . import runtime
    runtime.configure_args(args)
    app = App(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.daemon_threads = True
    print(f"vhuman eye demo on http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
