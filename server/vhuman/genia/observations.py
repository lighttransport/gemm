"""Bounded static RGB/mask/Z-depth observations in explicit R3 camera space."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from . import FORMAT

MAX_VIEWS = 16
MAX_PIXELS = 4_000_000
MAX_MANIFEST = 1 << 20
UNIT_SCALE = {"m": 1., "cm": .01, "mm": .001}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def asset(root, name):
    if not isinstance(name, str) or "\\" in name:
        raise ValueError("observation assets must use relative POSIX paths")
    parts = PurePosixPath(name)
    if parts.is_absolute() or not parts.parts or any(p in (".", "..") for p in name.split("/")):
        raise ValueError("observation assets must remain inside the bundle")
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("observation asset missing or outside bundle: " + name)
    return path


def matrix(value, shape, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid " + name) from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("expected finite " + name + " matrix")
    return result


@dataclass(frozen=True)
class View:
    image: np.ndarray
    mask: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    c2w: np.ndarray
    hashes: dict

    def pointmap(self):
        # Match upstream pinhole_xy: integer pixel centers, not u+.5/v+.5.
        h, w = self.depth.shape
        y, x = np.indices((h, w), dtype=np.float32)
        points = np.stack(((x-self.K[0, 2])*self.depth/self.K[0, 0],
                           (y-self.K[1, 2])*self.depth/self.K[1, 1], self.depth), axis=-1)
        points[~(np.isfinite(self.depth) & (self.depth > 0))] = np.nan
        return np.ascontiguousarray(points, dtype=np.float32)


@dataclass(frozen=True)
class Observations:
    views: tuple[View, ...]
    source_hash: str
    source: Path
    units: str

    def provenance(self):
        return {"format": FORMAT, "sha256": self.source_hash, "units": "m",
                "source_units": self.units, "camera_convention": "R3",
                "views": [{"size": list(v.image.shape[1::-1]), "assets": v.hashes,
                           "K": v.K.tolist(), "c2w": v.c2w.tolist()} for v in self.views]}


def load(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("no such observation manifest")
    if path.stat().st_size > MAX_MANIFEST:
        raise ValueError("observation manifest exceeds 1 MiB")
    try:
        doc = json.loads(path.read_text())
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid observation manifest JSON") from exc
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise ValueError("expected " + FORMAT)
    units = doc.get("units")
    if not isinstance(units, str) or units not in UNIT_SCALE:
        raise ValueError("units must be m, cm or mm")
    records = doc.get("views")
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_VIEWS:
        raise ValueError("expected 1..16 static views")
    views = []
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError("invalid observation view")
        try:
            paths = {key: asset(path.parent, rec[key]) for key in ("image", "mask", "depth")}
        except KeyError as exc:
            raise ValueError("each view needs image, mask and depth") from exc
        with Image.open(paths["image"]) as image:
            w, h = image.size
            if min(w, h) < 1 or w*h > MAX_PIXELS:
                raise ValueError("observation image exceeds 4M pixels")
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(paths["mask"]) as mask_image:
            if mask_image.size != (w, h):
                raise ValueError("mask dimensions differ from image")
            mask = np.asarray(mask_image.convert("L")) > 127
        if not mask.any():
            raise ValueError("foreground mask is empty")
        # mmap parses the bounded .npy header without allocating an untrusted
        # declared tensor. Shape/dtype are checked before copying its pixels.
        try:
            raw_depth = np.load(paths["depth"], mmap_mode="r", allow_pickle=False)
            if not isinstance(raw_depth, np.ndarray):
                raw_depth.close()
                raise ValueError("depth must be a single .npy array")
            if raw_depth.shape != (h, w) or raw_depth.dtype.kind != "f" or raw_depth.dtype.itemsize != 4:
                raise ValueError("depth must be float32 .npy matching the image")
            depth = np.array(raw_depth, dtype=np.float32) * np.float32(UNIT_SCALE[units])
            del raw_depth
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError("invalid float32 Z-depth .npy") from exc
        if not np.isfinite(depth[mask]).all() or not (depth[mask] > 0).all():
            raise ValueError("foreground Z-depth must be positive and finite")
        k = matrix(rec.get("K"), (3, 3), "pixel-space K")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-7, rtol=0) or abs(k[0, 1])+abs(k[1, 0]) > 1e-7:
            raise ValueError("K must be undistorted pinhole intrinsics without skew")
        c2w = matrix(rec.get("c2w"), (4, 4), "camera-to-world")
        if (not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
                or not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                or abs(np.linalg.det(c2w[:3, :3])-1) > 1e-5):
            raise ValueError("c2w must be a rigid, right-handed camera-to-world transform")
        c2w[:3, 3] *= UNIT_SCALE[units]
        limit = np.finfo(np.float32).max
        if np.any(np.abs(k) > limit) or np.any(np.abs(c2w) > limit):
            raise ValueError("camera values exceed float32 range")
        k = k.astype(np.float32)
        c2w = c2w.astype(np.float32)
        if k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("focal lengths underflow float32 range")
        hashes = {key: digest(p) for key, p in paths.items()}
        for key, value in hashes.items():
            expected = rec.get(key + "_sha256")
            if expected is not None and expected != value:
                raise ValueError(key + " checksum mismatch")
        views.append(View(pixels, mask, depth, k, c2w, hashes))
    return Observations(tuple(views), digest(path), path, units)
