"""Text/Image -> 3D studio for the Pixal3D demo: Qwen-Image 2.1 + Pixal3D.

A session is one object's workflow, kept on disk under <root>/<id>/:

    images/NNN.png       the object after each step (RGBA, 512^2); the last one
                         in `history` is the current object; undo drops it
    images/NNN_raw.png   the text-to-image output, before framing
    views/NNN/           a view set of image NNN (qimg21_i23d dataset layout)
    models/NNN/          Pixal3D reconstructions (<runner>.glb, reconstruction.json)
    session.json         the record the page renders

Stages (each one job of kind "i23d" on the server's single worker, so they
never share the GPU with Pixal3D or the Qwen tab):

    text         prompt -> Qwen-Image 2.1 (transparent background by default)
                 -> framed RGBA object
    upload       photo -> object extraction (qwen | rmbg | alpha) -> framed RGBA
    edit         object-preserving edit of the current object, optionally
                 limited to a rectangle (latent blending + pixel paste-back)
    views        a turntable of the current object (2D preview; posed
                 multiview reconstruction input)
    turnaround   a character turnaround sheet (front / left / back [/ right])
                 of the current object in ONE image, split into consistently
                 framed posed views; its front panel becomes the current
                 object. Far more consistent than separate views, so it is
                 the view set to use for multiview reconstruction
    reconstruct  Pixal3D native runner and/or PyTorch reference -> GLB; single
                 view from the current object by default (measurably better
                 than posing generated views, see cuda/qimg21/IMAGE_TO_3D.md)

The image model (qimg21_i23d.native.NativeBackend) stays resident between
stages, so edits and views take seconds; it is released before Pixal3D runs
and whenever another kind of job needs the device.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cuda/qimg21"))

from qimg21_i23d import imageops, ops, reconstruct  # noqa: E402

SIZE = 512
STAGES = ("text", "upload", "edit", "views", "turnaround", "reconstruct")
SESSION_ID = re.compile(r"[0-9a-f]{32}")
SERVED = (".png", ".glb", ".json", ".log")
MAX_PIXELS = 40_000_000


class StudioError(ValueError):
    pass


class StudioCancelled(Exception):
    """A cancel was requested; raised between steps of a stage."""


def _int(request, key, default, lo, hi):
    value = request.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or not lo <= value <= hi:
        raise StudioError(f"{key} must be an integer in [{lo}, {hi}]")
    return int(value)


def _float(request, key, default, lo, hi):
    value = request.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= float(value) <= hi:
        raise StudioError(f"{key} must be a number in [{lo}, {hi}]")
    return float(value)


def _text(request, key, required=True, limit=20000):
    # Long prompts are fine: the image backend checks them against the text
    # encoder's token limit and says so. This only bounds the request size.
    value = request.get(key, "")
    if not isinstance(value, str) or (required and not value.strip()) or len(value) > limit:
        raise StudioError(f"{key} must be {'a non-empty ' if required else 'a '}string of at most {limit} characters")
    return value.strip()


class Studio:
    def __init__(self, root: Path, *, backend_factory=None, native_options: dict | None = None,
                 reference_options: dict | None = None, runner_factory=None, ttl: float = 24 * 3600,
                 image_limit: int = 32 << 20):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl, self.image_limit = ttl, image_limit
        self.native_options = native_options or {}
        self.reference_options = reference_options or {}
        self._backend_factory = backend_factory or self._native_backend
        self._runner_factory = runner_factory or self._pixal3d_runner
        self._backend = None
        self.lock = threading.RLock()
        self.busy: set[str] = set()     # sessions with a stage running
        self.last_used = time.monotonic()
        self.image_model_path = None     # for health(); the backend factory decides the real one

    # ---- image model ------------------------------------------------------

    @staticmethod
    def _native_backend():
        from qimg21_i23d.native import NativeBackend
        return NativeBackend()

    def _pixal3d_runner(self, name: str, settings):
        if name == "native":
            return reconstruct.Pixal3DNative("cuda", settings=settings, **self.native_options)
        return reconstruct.Pixal3DReference("cuda", settings=settings, **self.reference_options)

    def backend(self):
        with self.lock:
            if self._backend is None:
                self._backend = self._backend_factory()
            return self._backend

    def release(self) -> None:
        """Stop the image model's resident processes (device memory back)."""
        with self.lock:
            if self._backend is not None:
                self._backend.close()

    def release_if_idle(self, idle: float = 600.0) -> bool:
        """Release the resident image model after `idle` seconds without a
        stage, so an idle studio doesn't hold GPU memory indefinitely."""
        with self.lock:
            if self.busy or self._backend is None or time.monotonic() - self.last_used < idle:
                return False
            fast = getattr(self._backend, "_fast", None)
            if fast is None or not fast.alive():
                return False
            self._backend.close()
            return True

    # ---- sessions ---------------------------------------------------------

    def _dir(self, sid: str) -> Path:
        if not isinstance(sid, str) or not SESSION_ID.fullmatch(sid):
            raise KeyError(sid)
        path = self.root / sid
        if not (path / "session.json").is_file():
            raise KeyError(sid)
        return path

    def _load(self, sid: str) -> dict:
        return json.loads((self._dir(sid) / "session.json").read_text())

    def _save(self, record: dict) -> dict:
        record["updated_at"] = time.time()
        path = self.root / record["id"] / "session.json"
        partial = path.with_suffix(".json.partial")
        partial.write_text(json.dumps(record, indent=1, default=str))
        partial.replace(path)
        return self.public(record)

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl
        for path in self.root.iterdir():
            if path.name in self.busy:
                continue
            try:
                if SESSION_ID.fullmatch(path.name) and json.loads(
                        (path / "session.json").read_text()).get("updated_at", 0) < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
            except (OSError, ValueError):
                shutil.rmtree(path, ignore_errors=True)

    def create(self) -> dict:
        with self.lock:
            self._expire()
            sid = uuid.uuid4().hex
            (self.root / sid / "images").mkdir(parents=True)
            return self._save({"id": sid, "created_at": time.time(), "history": [], "views": None,
                               "models": []})

    def public(self, record: dict) -> dict:
        """The session with file references turned into URLs."""
        base = f"/v1/i23d/sessions/{record['id']}/files/"
        folder = self.root / record["id"]

        def url(relative):
            # Paths get reused (views regenerated, a turnaround after undo):
            # the file's mtime in the URL keeps browsers from showing a cached copy.
            try:
                version = (folder / relative).stat().st_mtime_ns
            except OSError:
                version = 0
            return f"{base}{relative}?v={version}"

        out = json.loads(json.dumps(record))
        for step in out["history"]:
            step["url"] = url(step["file"])
            if step.get("raw"):
                step["raw_url"] = url(step["raw"])
        if out.get("views"):
            out["views"]["urls"] = [url(f) for f in out["views"]["files"]]
            if out["views"].get("sheet"):
                out["views"]["sheet_url"] = url(out["views"]["sheet"])
        for model in out["models"]:
            for run in model["runs"]:
                run["url"] = url(run["file"])
        out["current"] = out["history"][-1] if out["history"] else None
        return out

    def state(self, sid: str) -> dict:
        with self.lock:
            return self.public(self._load(sid))

    def undo(self, sid: str) -> dict:
        with self.lock:
            record = self._load(sid)
            if sid in self.busy:
                raise StudioError("a stage is running on this session")
            if len(record["history"]) < 2:
                raise StudioError("nothing to undo")
            record["history"].pop()
            record["views"] = None      # views belong to the image they were made from
            return self._save(record)

    def file(self, sid: str, relative: str) -> Path:
        base = self._dir(sid).resolve()
        path = (base / relative).resolve()
        if base not in path.parents or path.suffix not in SERVED or not path.is_file():
            raise KeyError(relative)
        return path

    def health(self) -> dict:
        from qimg21_i23d.native import NativeBackend
        native = reconstruct.Pixal3DNative("cuda", **self.native_options).available()
        reference = reconstruct.Pixal3DReference("cuda", **self.reference_options).available()
        model = self.image_model_path or NativeBackend.__init__.__defaults__[0]
        return {"image_model_ready": NativeBackend.available(model), "pixal3d_native_ready": native[0],
                "pixal3d_reference_ready": reference[0], "moge_ready": reconstruct.MOGE.exists(),
                "image_model_resident": bool(self._backend is not None and
                                             getattr(self._backend, "_fast", None) and
                                             self._backend._fast.alive())}

    # ---- stages -----------------------------------------------------------

    def run(self, request: dict, cancel=None, progress=None) -> dict:
        stage = request.get("stage")
        if stage not in STAGES:
            raise StudioError(f"stage must be one of {', '.join(STAGES)}")
        sid = request.get("session")
        text_turnaround = stage == "turnaround" and request.get("use_reference") is False
        if not sid and stage not in ("text", "upload") and not text_turnaround:
            raise StudioError("this stage needs a session with an object")
        if not sid:
            sid = self.create()["id"]
        # Load and mark busy in one step: an undo or expiry can't slip between.
        with self.lock:
            try:
                record = self._load(sid)
            except KeyError:
                raise StudioError("session not found (it may have expired); start a new one") from None
            if sid in self.busy:
                raise StudioError("a stage is already running on this session")
            self.busy.add(sid)
            self.last_used = time.monotonic()
            self._save(record)                  # touch: a running session never expires
        try:
            return self._run_stage(stage, record, request, cancel, progress, text_turnaround)
        finally:
            with self.lock:
                self.busy.discard(sid)
                self.last_used = time.monotonic()

    def _run_stage(self, stage, record, request, cancel, progress, text_turnaround) -> dict:
        self._cancel = cancel
        if stage in ("edit", "views", "turnaround", "reconstruct") and not record["history"] and not text_turnaround:
            raise StudioError("the session has no object yet; run text or upload first")

        def report(phase, percent):
            if cancel is not None and cancel.is_set():
                raise StudioCancelled("job cancelled")
            if progress:
                progress(phase, percent)

        started = time.perf_counter()
        before = len(record["history"])
        details = getattr(self, "_" + stage)(record, request, report)
        # A cancel that arrived while the backend was busy: discard the
        # stage's result rather than save it under a "cancelled" job.
        if cancel is not None and cancel.is_set():
            raise StudioCancelled("job cancelled")
        details["seconds"] = round(time.perf_counter() - started, 3)
        if len(record["history"]) > before:
            record["history"][-1]["seconds"] = details["seconds"]
        if stage in ("views", "turnaround"):
            record["views"]["seconds"] = details["seconds"]
        with self.lock:
            state = self._save(record)
        return {"kind": "i23d", "stage": stage, "session": record["id"], "details": details, "state": state}

    def _next_image(self, record: dict) -> tuple[str, Path]:
        index = len(record["history"])
        while (self.root / record["id"] / f"images/{index:03d}.png").exists():
            index += 1
        name = f"images/{index:03d}.png"
        return name, self.root / record["id"] / name

    def _push(self, record: dict, name: str, stage: str, label: str, details: dict, raw: str | None = None):
        record["history"].append({"file": name, "stage": stage, "label": label, "raw": raw,
                                  "seconds": None, "at": time.time()})
        record["views"] = None

    def _text(self, record, request, report):
        prompt = _text(request, "prompt")
        steps, seed = _int(request, "steps", 16, 1, 50), _int(request, "seed", 0, 0, 2**31 - 1)
        width, height = _int(request, "width", 512, 256, 1024), _int(request, "height", 512, 256, 1024)
        if width % 32 or height % 32:
            raise StudioError("width and height must be multiples of 32")
        transparent = request.get("transparent", True) is not False
        name, path = self._next_image(record)
        report("Qwen-Image 2.1: text to object", 10)
        info = ops.generate_object(prompt, path, self.backend(), width=width, height=height, size=(SIZE, SIZE),
                                   transparent=transparent, steps=steps, seed=seed)
        details = {"prompt": info["prompt"], "generation_seconds": info["seconds"], "extraction": info["extraction"],
                   "timings": info["timings"]}
        raw = Path(info["raw"])
        self._push(record, name, "text", prompt, details, raw=raw.relative_to(self.root / record["id"]).as_posix())
        return details

    def _upload(self, record, request, report):
        source = request.get("image_b64")
        method = request.get("method", "qwen")
        if method not in ("qwen", "rmbg", "alpha"):
            raise StudioError("method must be qwen, rmbg or alpha")
        steps, seed = _int(request, "steps", 16, 1, 50), _int(request, "seed", 0, 0, 2**31 - 1)
        work = self.root / record["id"] / "uploads"
        work.mkdir(exist_ok=True)
        photo = work / f"{uuid.uuid4().hex}.img"
        if isinstance(source, Path):
            shutil.copyfile(source, photo)
        else:
            import base64
            if not isinstance(source, str) or not source:
                raise StudioError("upload needs an image")
            photo.write_bytes(base64.b64decode(source.split(",", 1)[-1], validate=True))
        try:
            if photo.stat().st_size > self.image_limit:
                raise StudioError("image too large")
            from PIL import Image, UnidentifiedImageError
            try:
                with Image.open(photo) as im:
                    width, height = im.size
            except (UnidentifiedImageError, OSError):
                raise StudioError("the upload is not a readable image") from None
            if width * height > MAX_PIXELS:
                raise StudioError(f"the image is {width}x{height}; at most {MAX_PIXELS // 1_000_000} megapixels")
            name, path = self._next_image(record)
            report(f"extracting the object ({method})", 10)
            try:
                info = ops.preprocess_object(photo, path, self.backend() if method == "qwen" else None,
                                             method=method, size=(SIZE, SIZE), steps=steps, seed=seed)
            except imageops.MaskError as exc:
                raise StudioError(str(exc)) from None
        finally:
            photo.unlink(missing_ok=True)
        details = {"method": method, "alignment": info.get("alignment"), "warnings": info.get("warnings")}
        self._push(record, name, "upload", f"photo ({method})", details)
        return details

    def _edit(self, record, request, report):
        instruction = _text(request, "instruction")
        strength = _float(request, "strength", 1.0, 0.05, 1.0)
        steps, seed = _int(request, "steps", 16, 1, 50), _int(request, "seed", 0, 0, 2**31 - 1)
        rect = request.get("rect")
        if rect is not None:
            if not (isinstance(rect, list) and len(rect) == 4 and all(isinstance(v, (int, float)) for v in rect)):
                raise StudioError("rect must be [x, y, w, h] in pixels of the current object")
            rect = tuple(int(round(v)) for v in rect)
        current = self.root / record["id"] / record["history"][-1]["file"]
        name, path = self._next_image(record)
        report("Qwen-Image 2.1: editing", 10)
        try:
            info = ops.edit_object(current, instruction, path, self.backend(), strength=strength, rect=rect,
                                   mask_feather=_int(request, "feather", 4, 0, 64) if rect else 0,
                                   transparent=True, steps=steps, seed=seed)
        except imageops.MaskError as exc:
            raise StudioError(str(exc)) from None
        details = {"instruction": instruction, "strength": strength, "rect": rect,
                   "generation_seconds": info.get("seconds")}
        self._push(record, name, "edit", instruction, details)
        return details

    def _views(self, record, request, report):
        count = _int(request, "count", 8, 2, 24)
        elevation = _float(request, "elevation", 0.0, -60.0, 60.0)
        steps, seed = _int(request, "steps", 16, 1, 50), _int(request, "seed", 0, 0, 2**31 - 1)
        current = record["history"][-1]["file"]
        final = self.root / record["id"] / "views" / Path(current).stem
        # Built next to the old set and swapped in only on success, so a
        # failed or cancelled run leaves the recorded views intact.
        root = final.with_name(final.name + ".new")
        shutil.rmtree(root, ignore_errors=True)
        report("Qwen-Image 2.1: views", 5)
        done = []

        def on_view(view):
            done.append(view)
            report(f"view {len(done)} of {count}", 5 + int(90 * len(done) / count))

        summary = ops.generate_turntable([self.root / record["id"] / current], root, self.backend(), views=count,
                                         elevation_deg=elevation,
                                         params=ops.ViewParams(steps=steps, seed=seed), on_view=on_view)
        report("saving the views", 98)
        shutil.rmtree(final, ignore_errors=True)
        root.rename(final)
        root = final
        meta = json.loads((root / "metadata.json").read_text())
        base = root.relative_to(self.root / record["id"]).as_posix()
        record["views"] = {"of": current, "dir": base, "count": count, "elevation": elevation,
                           "files": [f"{base}/{v['file']}" for v in meta["views"]],
                           "azimuths": [v["azimuth_deg"] for v in meta["views"]],
                           "validation": summary.get("validation")}
        return {"count": count, "elevation": elevation, "seconds_per_view":
                round(sum(v["seconds"] for v in meta["views"]) / max(1, len(meta["views"])), 3)}

    def _turnaround(self, record, request, report):
        count = _int(request, "count", 4, 3, 4)
        steps, seed = _int(request, "steps", 20, 1, 50), _int(request, "seed", 0, 0, 2**31 - 1)
        subject = _text(request, "prompt", required=False) or None
        # use_reference false: a sheet from text alone (a new character); the
        # default draws the current object as a character.
        use_reference = request.get("use_reference", True) is not False and bool(record["history"])
        if not use_reference and not subject:
            raise StudioError("a turnaround from text needs a prompt")
        reference = self.root / record["id"] / record["history"][-1]["file"] if use_reference else None
        root = self.root / record["id"] / "views" / f"turnaround_{len(record['history']):03d}"
        shutil.rmtree(root, ignore_errors=True)
        report("Qwen-Image 2.1: turnaround sheet", 10)
        try:
            summary = ops.generate_turnaround(root, self.backend(), prompt=subject, reference=reference,
                                              views=count, size=SIZE, steps=steps, seed=seed)
        except imageops.MaskError as exc:
            raise StudioError(str(exc)) from None
        report("framing the views", 90)
        base = root.relative_to(self.root / record["id"]).as_posix()
        name, path = self._next_image(record)
        shutil.copyfile(root / summary["views"][0]["file"], path)
        self._push(record, name, "turnaround", (subject or "") if not use_reference else f"turnaround ({count} views)",
                   {}, raw=f"{base}/sheet.png")
        record["views"] = {"of": name, "dir": base, "count": count, "elevation": 0.0, "kind": "turnaround",
                           "files": [f"{base}/{v['file']}" for v in summary["views"]],
                           "azimuths": [v["azimuth_deg"] for v in summary["views"]],
                           "sheet": f"{base}/sheet.png", "validation": summary.get("validation")}
        return {"count": count, "views": [v["view"] for v in summary["views"]]}

    def _reconstruct(self, record, request, report):
        cancel_event = getattr(self, "_cancel", None)
        which = request.get("runner", "native")
        mode = request.get("mode", "single")
        if which not in ("native", "reference", "both"):
            raise StudioError("runner must be native, reference or both")
        if mode not in ("single", "multiview"):
            raise StudioError("mode must be single or multiview")
        current = record["history"][-1]["file"]
        if mode == "multiview" and not (record.get("views") and record["views"]["of"] == current):
            raise StudioError("multiview needs views of the current object; run the views stage first")
        fov = request.get("fov")
        if fov is not None:
            fov = _float(request, "fov", 20.0, 5.0, 120.0)
        quality = request.get("quality", "standard")
        try:
            settings = reconstruct.ReconSettings.preset(
                quality, seed=_int(request, "seed", 42, 0, 2**32 - 1),
                texture_size=_int(request, "texture_size", None, 1024, 4096) if "texture_size" in request else None,
                triangle_target=_int(request, "triangle_target", None, 10_000, 5_000_000)
                if "triangle_target" in request else None)
        except ValueError as exc:
            raise StudioError(str(exc)) from None
        runners = [self._runner_factory(name, settings) for name in ("native", "reference")
                   if which in (name, "both")]
        for runner in runners:
            ok, missing = runner.available()
            if not ok:
                raise StudioError(f"Pixal3D {runner.name} is not ready: missing {', '.join(missing)}")
        report("releasing the image model", 3)
        self.release()
        index = len(record["models"])
        out = self.root / record["id"] / "models" / f"{index:03d}"
        shutil.rmtree(out, ignore_errors=True)
        report(f"Pixal3D {' + '.join(r.name for r in runners)} ({mode})", 8)
        source = (self.root / record["id"] / record["views"]["dir"]) if mode == "multiview" else \
            self.root / record["id"] / current
        # The session's view set is one ring (any elevation): use all of it.
        settings.cancel = cancel_event
        try:
            result = ops.reconstruct_3d(source, out, runners, mode=mode, fov_deg=fov, elevations=None)
        except reconstruct.ReconstructionCancelled:
            shutil.rmtree(out, ignore_errors=True)
            raise StudioCancelled("job cancelled") from None
        base = out.relative_to(self.root / record["id"]).as_posix()
        model = {"of": current, "mode": mode, "quality": quality, "camera": result.get("camera"),
                 "comparison": result.get("comparison"), "at": time.time(),
                 "runs": [{"runner": r["runner"], "file": f"{base}/{Path(r['output']).name}",
                           "seconds": r["seconds"], "mesh": r["mesh"]} for r in result["runs"]]}
        record["models"].append(model)
        return {"mode": mode, "camera": model["camera"], "runs": model["runs"], "comparison": model["comparison"]}
