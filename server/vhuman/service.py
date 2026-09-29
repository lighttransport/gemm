"""Eye generation shared by the CLI and the web server.

Outputs live in a content-addressed cache under tmp/vhuman/cache/<key>/:
the key hashes everything that determines the files, so a repeated request
is a directory lookup and every URL is immutable.
"""
from __future__ import annotations

import io
import hashlib
import json
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .eye import assets, iris, optics, render
from .eye import params as P

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "tmp" / "vhuman-independent"
RESOLUTIONS = (512, 1024, 2048)
CACHE_LIMIT = 256            # cache entries kept (oldest removed first)


class ServiceError(ValueError):
    pass


def _key(*parts) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


class EyeService:
    def __init__(self, work: Path = WORK):
        self.work = Path(work)
        self.cache = self.work / "cache"
        self.plates = self.work / "plates"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.plates.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def _prune(self) -> None:
        entries = sorted((d for d in self.cache.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
        for d in entries[:-CACHE_LIMIT]:
            shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def resolution(res) -> int:
        if res not in RESOLUTIONS:
            raise ServiceError(f"res must be one of {RESOLUTIONS}")
        return int(res)

    # ---- detail source (procedural or an iris plate) ---------------------------

    def plate(self, plate_id: str) -> dict:
        if not isinstance(plate_id, str) or not plate_id.isalnum() or len(plate_id) > 64:
            raise ServiceError("bad plate id")
        meta = self.plates / plate_id / "plate.json"
        if not meta.is_file():
            raise ServiceError(f"unknown plate {plate_id}")
        return json.loads(meta.read_text())

    def list_plates(self) -> list[dict]:
        out = []
        for meta in sorted(self.plates.glob("*/plate.json"), key=lambda m: m.stat().st_mtime, reverse=True):
            try:
                rec = json.loads(meta.read_text())
            except ValueError:
                continue
            rec["thumb_url"] = f"/v1/plates/{rec['id']}/thumb.png"
            out.append(rec)
        return out

    def _plate_textures(self, plate_id: str, res: int) -> tuple[np.ndarray, np.ndarray]:
        """A plate's structure masks and photo colours at `res` (the plate is
        stored as its normalised square iris at 1024)."""
        folder = self.plates / plate_id
        masks = np.asarray(Image.open(folder / "iris_masks.png"), np.float32) / 255.0
        photo = np.asarray(Image.open(folder / "iris_photo.png").convert("RGB"), np.float32) / 255.0
        if masks.shape[0] != res:
            masks = np.asarray(Image.fromarray((masks * 255).astype(np.uint8)).resize((res, res), Image.BICUBIC),
                               np.float32) / 255.0
            photo = np.asarray(Image.fromarray((photo * 255).astype(np.uint8)).resize((res, res), Image.BICUBIC),
                               np.float32) / 255.0
        return masks, photo

    def _detail(self, detail) -> tuple[str, str | None]:
        detail = detail or {"source": "procedural"}
        source = detail.get("source", "procedural")
        if source == "procedural":
            return "procedural", None
        if source == "plate":
            plate_id = detail.get("id")
            self.plate(plate_id)
            return "plate", plate_id
        raise ServiceError("detail.source must be procedural or plate")

    # ---- the live texture set --------------------------------------------------

    def textures(self, params: dict, res: int = 1024, detail=None) -> dict:
        p = P.validate(params)
        res = self.resolution(res)
        source, plate_id = self._detail(detail)
        key = _key("live", P.structure_key(p, res), source, plate_id)
        folder = self.cache / key
        started = time.perf_counter()
        with self._lock(key):
            if not (folder / "done").is_file():
                folder.mkdir(parents=True, exist_ok=True)
                textures, timings = assets.live_textures(p, res)
                if plate_id:
                    tex = self.plate_iris(plate_id, res)
                    textures["iris_masks"] = assets.to_u8(tex.masks)
                    textures["iris_normal"] = assets.normal_u8(tex.normal)
                    textures["iris_photo"] = assets.to_u8(tex.photo)
                with ThreadPoolExecutor(4) as pool:     # PNG encoding releases the GIL
                    list(pool.map(lambda kv: Image.fromarray(kv[1]).save(folder / f"{kv[0]}.png", compress_level=1),
                                  textures.items()))
                (folder / "timings.json").write_text(json.dumps(timings))
                (folder / "done").write_text("")
                self._prune()
            else:
                folder.touch()
        timings = json.loads((folder / "timings.json").read_text())
        names = sorted(f.stem for f in folder.glob("*.png"))
        return {"key": key, "res": res, "detail": {"source": source, "id": plate_id},
                "urls": {n: f"/v1/eye/files/{key}/{n}.png" for n in names},
                "uniforms": optics.uniforms(p), "timings": timings,
                "seconds": round(time.perf_counter() - started, 4)}

    # ---- exports -----------------------------------------------------------------

    def export(self, params: dict, res: int = 2048, formats=("glb",), pair: bool = False, detail=None) -> dict:
        p = P.validate(params)
        res = self.resolution(res)
        unknown = set(formats) - {"glb", "textures"}
        if unknown or not formats:
            raise ServiceError("formats must be a non-empty subset of glb, textures")
        source, plate_id = self._detail(detail)
        key = _key("export", P.bake_key(p, res, source + str(plate_id)), sorted(formats), pair)
        folder = self.cache / key
        started = time.perf_counter()
        with self._lock(key):
            if not (folder / "done").is_file():
                folder.mkdir(parents=True, exist_ok=True)
                iris_tex = self.plate_iris(plate_id, res) if plate_id else None
                (folder / "params.json").write_text(json.dumps(p, indent=1))
                if "glb" in formats:
                    assets.export_glb(p, folder / "eye.glb", res, pair=pair, iris_tex=iris_tex)
                if "textures" in formats:
                    assets.export_textures(p, folder / "textures", res, iris_tex=iris_tex)
                    with zipfile.ZipFile(folder / "eye_textures.zip", "w", zipfile.ZIP_STORED) as z:
                        for f in sorted((folder / "textures").iterdir()):
                            z.write(f, f"eye_textures/{f.name}")
                (folder / "done").write_text("")
                self._prune()
        urls = {"params": f"/v1/eye/files/{key}/params.json"}
        if (folder / "eye.glb").is_file():
            urls["glb"] = f"/v1/eye/files/{key}/eye.glb"
        if (folder / "eye_textures.zip").is_file():
            urls["textures_zip"] = f"/v1/eye/files/{key}/eye_textures.zip"
        return {"key": key, "urls": urls, "dir": str(folder), "seconds": round(time.perf_counter() - started, 3)}

    def plate_iris(self, plate_id: str, res: int) -> iris.IrisTextures:
        """An iris plate as iris textures (structure masks + photo colours)."""
        masks, photo = self._plate_textures(plate_id, res)
        return iris.IrisTextures(res, masks, iris.noise.normal_map(masks[..., 0] * 0.6, strength=res / 64.0),
                                 None, 0.0, photo)

    def render(self, params: dict, size: int = 384, yaw: float = 0.0, pitch: float = 0.0, fov: float = 0.0,
               spp: int = 4, detail=None) -> bytes:
        if not 64 <= int(size) <= 512:
            raise ServiceError("size must be in [64, 512]")
        _, plate_id = self._detail(detail)
        cam = render.Camera(fov_deg=float(fov), yaw_deg=float(yaw), pitch_deg=float(pitch))
        img = render.render(P.validate(params), int(size), cam, spp=int(spp),
                            iris_tex=self.plate_iris(plate_id, 1024) if plate_id else None)
        buf = io.BytesIO()
        Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(buf, "PNG")
        return buf.getvalue()

    def list_baselines(self) -> list[dict]:
        out = []
        for meta in sorted((self.work / "baselines").glob("*/baseline.json"), key=lambda m: m.stat().st_mtime,
                           reverse=True):
            try:
                rec = json.loads(meta.read_text())
            except ValueError:
                continue
            bid = rec["id"]
            out.append({"id": bid, "source": rec["source"], "quality": rec.get("quality"), "metrics": rec["metrics"],
                        "seconds": rec.get("seconds"), "created": rec.get("created"),
                        "glb_url": f"/v1/baselines/{bid}/pixal3d.glb", "input_url": f"/v1/baselines/{bid}/input.png"})
        return out

    def baseline_file(self, bid: str, name: str = "") -> Path:
        if not (isinstance(bid, str) and bid.isalnum() and len(bid) <= 32) or name not in (
                "pixal3d.glb", "input.png", "baseline.json"):
            raise ServiceError("no such file")
        path = self.work / "baselines" / bid / name
        if not path.is_file():
            raise ServiceError("no such file")
        return path

    def list_heads(self) -> list[dict]:
        out = []
        for meta in sorted((self.work / "heads").glob("*/head.json"), key=lambda m: m.stat().st_mtime,
                           reverse=True):
            try:
                rec = json.loads(meta.read_text())
            except ValueError:
                continue
            hid = rec["id"]
            base = f"/v1/heads/{hid}/"
            out.append({"id": hid, "subject": rec.get("subject"), "seed": rec.get("seed_used", rec.get("seed")),
                        "quality": rec.get("quality"), "seconds": rec.get("seconds"), "created": rec.get("created"),
                        "skin": rec.get("skin"), "iris": rec.get("iris"), "fit": (rec.get("fit") or {}).get("fit"), "plate": (rec.get("fit") or {}).get("plate"),
                        "urls": {n.split(".")[0]: base + n for n in ("portrait.png", "landmarks.png", "pixal3d.glb",
                                                                    "head_eyes.glb", "fit.json", "skin_basecolor.png", "skin_normal.png",
                                                                    "skin_orm.png", "skin_mask.png", "skin.json")
                                 if (meta.parent / n).is_file()},
                        "rig": self._rig_summary(meta.parent, base),
                        "body": self._body_summary(meta.parent, base)})
        return out

    @staticmethod
    def _rig_summary(folder: Path, base: str) -> dict | None:
        rep = folder / "rig" / "rig_report.json"
        if not rep.is_file():
            return None
        try:
            report = json.loads(rep.read_text())
        except ValueError:
            return None
        keys = {"glb": "rig.glb", "json": "rig.json", "usda": "rig.usda", "usd_zip": "rig_usd.zip",
                "preview": "preview.png", "report": "rig_report.json"}
        urls = {k: f"{base}rig/{n}" for k, n in keys.items() if (folder / "rig" / n).is_file()}
        brief = {k: report.get(k) for k in ("version", "template", "register", "bake", "shapes", "controls",
                                             "joints", "seconds", "deformer", "expressions", "wrinkles")}
        brief["lods"] = sorted(int(k) for k in (report.get("lods") or {}))
        return {"urls": urls, "report": brief, "lods": brief["lods"]}

    def rig_file(self, hid: str, name: str) -> Path:
        """A rig output: the fixed names of rig/job.py or textures/<name>.png."""
        from .rig.job import RIG_FILES
        if not (isinstance(hid, str) and hid.isalnum() and len(hid) <= 32):
            raise ServiceError("no such file")
        folder, _, base = name.partition("/")
        ok = name in RIG_FILES or (folder in ("textures", "expressions") and base.endswith(".png")
                                   and "/" not in base and ".." not in base
                                   and all(c.isalnum() or c in "_-." for c in base))
        if not ok:
            raise ServiceError("no such file")
        path = self.work / "heads" / hid / "rig" / name
        if not path.is_file():
            raise ServiceError("no such file")
        return path

    @staticmethod
    def _body_summary(folder: Path, base: str) -> dict | None:
        report_path = folder / "body" / "body_report.json"
        if not report_path.is_file() or not (folder / "body" / "avatar.glb").is_file():
            return None
        try:
            report = json.loads(report_path.read_text())
        except ValueError:
            return None
        keys = {"glb": "avatar.glb", "usda": "avatar.usda", "usd_zip": "avatar_usd.zip",
                "json": "avatar.json", "image": "body_image.png", "report": "body_report.json"}
        return {"urls": {k: f"{base}body/{n}" for k, n in keys.items()
                         if (folder / "body" / n).is_file()},
                "report": {k: report.get(k) for k in ("body_vertices", "body_triangles", "body_joints",
                                                       "face_joints", "landmark_error_m", "garments", "total_seconds")}}

    def body_file(self, hid: str, name: str) -> Path:
        from .body.job import FILES
        from .body.motion import MOTION_FILES
        if not (isinstance(hid, str) and hid.isalnum() and len(hid) <= 32):
            raise ServiceError("no such file")
        folder, _, base = name.partition("/")
        allowed = name in FILES or (folder == "textures" and base.endswith(".png") and "/" not in base
                                    and all(c.isalnum() or c in "_-." for c in base))
        if folder == "motions":
            take, sep, artifact = base.partition("/")
            allowed = bool(sep and len(take) == 12 and all(c in "0123456789abcdef" for c in take)
                           and artifact in MOTION_FILES)
        if not allowed:
            raise ServiceError("no such file")
        path = self.work / "heads" / hid / "body" / name
        if not path.is_file():
            raise ServiceError("no such file")
        return path

    def list_motions(self, hid: str) -> list[dict]:
        self.body_file(hid, "avatar.json")
        root = self.work / "heads" / hid / "body" / "motions"
        out = []
        for path in sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                take = path.parent.name
                manifest = json.loads(self.body_file(hid, f"motions/{take}/manifest.json").read_text())
                base = f"/v1/heads/{hid}/body/motions/{take}/"
                manifest["urls"] = {"motion": base + "motion.json", "glb": base + "motion.glb"}
                out.append(manifest)
            except (ValueError, ServiceError):
                continue
        return out

    def take_file(self, hid: str, take_id: str, name: str) -> Path:
        """Only public, fixed-name artifacts from a completed speech take."""
        from .rig.speech import TAKE_FILES
        if not isinstance(take_id, str) or len(take_id) != 12 or any(c not in "0123456789abcdef" for c in take_id):
            raise ServiceError("no such take")
        if name not in TAKE_FILES:
            raise ServiceError("no such file")
        base = self.rig_file(hid, "rig.json").parent / "takes" / take_id
        if not (base / "manifest.json").is_file():
            raise ServiceError("no such take")
        path = base / name
        if not path.is_file():
            raise ServiceError("no such file")
        return path

    def take_summary(self, hid: str, take_id: str) -> dict:
        manifest = json.loads(self.take_file(hid, take_id, "manifest.json").read_text())
        current_rig = hashlib.sha256(self.rig_file(hid, "rig.json").read_bytes()).hexdigest()[:16]
        manifest["rig_stale"] = bool(manifest.get("rig_sha256") and manifest["rig_sha256"] != current_rig)
        base = f"/v1/heads/{hid}/rig/takes/{take_id}/"
        manifest["urls"] = {key: base + name for key, name in {
            "manifest": "manifest.json", "audio": "audio.wav", "align": "align.json",
            "animation": "animation.json", "usd": "animation.usda", "lightrig": "lightrig.txt",
            "emotion": "emotion.json", "soft_tissue": "soft_tissue.usda",
            "soft_tissue_report": "soft_tissue_report.json", "fit_report": "fit_report.json",
        }.items() if (self.work / "heads" / hid / "rig" / "takes" / take_id / name).is_file()}
        return manifest

    def list_takes(self, hid: str) -> list[dict]:
        root = self.rig_file(hid, "rig.json").parent / "takes"
        if not root.is_dir():
            return []
        out = []
        for path in sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                out.append(self.take_summary(hid, path.parent.name))
            except (ValueError, ServiceError):
                continue
        return out

    def head_file(self, hid: str, name: str = "") -> Path:
        from .head.pipeline import FILES
        if not (isinstance(hid, str) and hid.isalnum() and len(hid) <= 32) or name not in FILES:
            raise ServiceError("no such file")
        path = self.work / "heads" / hid / name
        if not path.is_file():
            raise ServiceError("no such file")
        return path

    def file(self, key: str, name: str) -> Path:
        """A cache file, guarded against path traversal."""
        if not (isinstance(key, str) and key.isalnum() and len(key) == 24):
            raise ServiceError("bad key")
        base = (self.cache / key).resolve()
        path = (base / name).resolve()
        if base not in path.parents or not path.is_file():
            raise ServiceError("no such file")
        return path
