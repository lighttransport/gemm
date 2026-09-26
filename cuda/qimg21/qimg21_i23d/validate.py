"""2D sanity checks for a generated dataset (validation.json).

These catch broken outputs -- missing or unreadable files, wrong sizes, empty
or clipped foregrounds, blank frames -- so a downstream reconstruction system
can drop them. They say nothing about geometric correctness: agreement between
views cannot be established from per-image statistics, and nothing here claims
it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import imageops

DEFAULTS = {
    "min_foreground": 0.01,      # at least 1% of the frame is object
    "max_foreground": 0.98,      # and it is not the whole frame
    "edge_margin": 2,            # bbox touching the border by <= this counts as clipped
    "blank_std": 2.0,            # composited RGB std below this is a blank frame
}


def check_image(path: Path, *, width: int | None, height: int | None, expect_alpha: bool,
                limits: dict = DEFAULTS) -> dict:
    report = {"file": str(path), "ok": True, "errors": [], "warnings": []}

    def fail(message):
        report["ok"] = False
        report["errors"].append(message)

    if not path.is_file():
        fail("missing")
        return report
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            report.update(format=image.format, mode=image.mode, width=image.width, height=image.height)
            rgba = np.asarray(image.convert("RGBA"))
            has_alpha = "A" in image.getbands()
    except (OSError, ValueError) as exc:
        fail(f"unreadable image: {exc}")
        return report
    if report["format"] != "PNG" and path.suffix.lower() == ".png":
        fail(f"not a PNG ({report['format']})")
    if width and height and (report["width"], report["height"]) != (width, height):
        fail(f"size {report['width']}x{report['height']}, expected {width}x{height}")
    if expect_alpha and not has_alpha:
        fail("no alpha channel")
    alpha = rgba[..., 3]
    coverage = float((alpha > 127).mean())
    report["foreground_fraction"] = round(coverage, 5)
    if expect_alpha:
        if coverage < limits["min_foreground"]:
            fail(f"foreground covers {coverage:.2%} of the frame")
        elif coverage > limits["max_foreground"]:
            fail(f"foreground covers {coverage:.2%}: the background was not removed")
        bbox = imageops.alpha_bbox(rgba, 127)
        if bbox:
            x0, y0, x1, y1 = bbox
            report["foreground_bbox"] = [x0, y0, x1, y1]
            h, w = alpha.shape
            margin = limits["edge_margin"]
            clipped = [side for side, value in (("left", x0), ("top", y0), ("right", w - x1), ("bottom", h - y1))
                       if value <= margin]
            if clipped:
                report["warnings"].append(f"foreground touches the {', '.join(clipped)} edge: possibly cropped")
            report["foreground_center"] = [round((x0 + x1) / 2 / w, 4), round((y0 + y1) / 2 / h, 4)]
    rgb = imageops.composite(rgba, (127, 127, 127)).astype(np.float32)
    std = float(rgb.std())
    report["rgb_std"] = round(std, 3)
    if std < limits["blank_std"]:
        mean = float(rgb.mean())
        fail(f"blank frame (mean {mean:.1f}, std {std:.2f})")
    return report


def validate_dataset(root, files: list[str], *, width: int | None = None, height: int | None = None,
                     expect_alpha: bool = True, limits: dict | None = None, write: bool = True) -> dict:
    root = Path(root)
    merged = {**DEFAULTS, **(limits or {})}
    images = [check_image(root / name, width=width, height=height, expect_alpha=expect_alpha, limits=merged)
              for name in files]
    modes = sorted({r.get("mode") for r in images if r.get("mode")})
    sizes = sorted({(r["width"], r["height"]) for r in images if "width" in r})
    fractions = [r["foreground_fraction"] for r in images if "foreground_fraction" in r]
    summary = {
        "images": len(images),
        "passed": sum(r["ok"] for r in images),
        "failed": [r["file"] for r in images if not r["ok"]],
        "consistent_mode": len(modes) <= 1,
        "modes": modes,
        "consistent_size": len(sizes) <= 1,
        "sizes": [list(s) for s in sizes],
        "foreground_fraction": ({"min": min(fractions), "max": max(fractions),
                                 "mean": round(float(np.mean(fractions)), 5)} if fractions else None),
        "note": "2D checks only; they do not establish geometric consistency between views",
    }
    report = {"summary": summary, "limits": merged, "images": images}
    if write:
        (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
