"""Prepared observation ZIP uploads; no arbitrary remote filesystem paths."""
import hashlib
import errno
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from ..service import ServiceError
from .observations import load

MAX_UPLOAD = 64 << 20
MAX_EXPANDED = 256 << 20
MAX_ENTRIES = 64


def upload(service, stream, length, content_type):
    if not 0 < length <= MAX_UPLOAD or content_type.split(";")[0] not in ("application/zip", "application/x-zip-compressed"):
        raise ServiceError("expected prepared observation ZIP, up to 64 MiB")
    data = stream.read(length)
    if len(data) != length:
        raise ServiceError("truncated GenIA upload")
    uid = hashlib.sha256(data).hexdigest()[:24]
    parent = service.work / "genia_uploads"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="upload-", dir=parent) as temporary:
        root = Path(temporary)
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = archive.infolist()
                if len(entries) > MAX_ENTRIES or sum(x.file_size for x in entries) > MAX_EXPANDED:
                    raise ValueError("observation ZIP exceeds expansion limits")
                names = set()
                for entry in entries:
                    parts = PurePosixPath(entry.filename)
                    name = entry.filename.rstrip("/")
                    if (not name or parts.is_absolute() or "\\" in name or any(p in ("", ".", "..") for p in name.split("/"))
                            or stat.S_ISLNK(entry.external_attr >> 16) or name in names
                            or entry.flag_bits & 1):
                        raise ValueError("unsafe or duplicate ZIP entry")
                    names.add(name)
                    if entry.is_dir():
                        (root / name).mkdir(parents=True, exist_ok=True)
                        continue
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry) as source, target.open("wb") as out:
                        shutil.copyfileobj(source, out, 1 << 20)
            observations = load(root / "scene.json")
            document = json.loads((root / "scene.json").read_text())
            expected = {"scene.json"}
            for view in document["views"]:
                expected.update(view[key] for key in ("image", "mask", "depth"))
            actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
            if actual != expected:
                raise ValueError("ZIP may contain only scene.json and its referenced assets")
            destination = parent / uid
            # Publish only fully validated bundles. Equal content hashes identify
            # concurrent duplicate uploads; reuse their already-complete directory.
            if not destination.exists():
                try:
                    root.rename(destination)
                except OSError as exc:
                    if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not destination.is_dir():
                        raise
            return {"upload_id": uid, "views": len(observations.views), "sha256": hashlib.sha256(data).hexdigest()}
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
            raise ServiceError("invalid GenIA observation bundle: " + str(exc)) from exc


def resolve(service, uid):
    if not isinstance(uid, str) or len(uid) != 24 or any(c not in "0123456789abcdef" for c in uid):
        raise ServiceError("valid GenIA upload_id required")
    root = service.work / "genia_uploads" / uid
    if not root.resolve().is_relative_to((service.work / "genia_uploads").resolve()) or not (root / "scene.json").is_file():
        raise ServiceError("no such GenIA upload")
    return root / "scene.json"
