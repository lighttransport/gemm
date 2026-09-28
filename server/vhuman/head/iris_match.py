"""Portrait-specific Qwen iris candidates, quality-gated and colour matched."""
from __future__ import annotations

import json
import numpy as np

from .. import gpu, qwen
from ..eye import chart


def target_color(eyes):
    """Use the same detected iris colour as the head fitter."""
    return next((eye.color for eye in eyes if eye.color), {})


def color_distance(target, color):
    """Squared log-linear RGB distance, shared with library selection."""
    a, b = np.asarray(target, float), np.asarray(color, float)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("inf")
    if (a < 0).any() or (b < 0).any():
        return float("inf")
    return float(np.sum((np.log(a + .01) - np.log(b + .01)) ** 2))


def describe_color(color):
    """Original chart hue, with brightness described independently.

    Comparing preset RGB values directly confounds hue and brightness (a
    dark grey portrait could otherwise choose a darker green preset).
    """
    target = color.get("median_linear", [])
    if not np.isfinite(color_distance(target, target)):
        return None
    suggested = color.get("suggested") or {}
    if "primary_color_u" in suggested and "primary_color_v" in suggested:
        u, v = suggested["primary_color_u"], suggested["primary_color_v"]
    else:
        uu, vv = np.meshgrid(np.linspace(0, 1, 65), np.linspace(0, 1, 65))
        rgb = chart.chart_color(uu, vv)
        error = ((np.log(rgb + .01) - np.log(np.asarray(target) + .01)) ** 2).sum(-1)
        index = np.unravel_index(np.argmin(error), error.shape)
        u, v = uu[index], vv[index]
    names = ("brown", "brown", "amber", "hazel", "green", "gray_green", "gray", "blue_gray", "blue", "light_blue")
    index = int(np.argmin(np.abs(np.array([x[0] for x in chart.ANCHORS]) - u)))
    text = qwen.color_words(names[index])
    if v < .28 and not text.startswith("dark"):
        text = "dark " + text
    elif v > .72 and not text.startswith("light"):
        text = "light " + text
    return text


def generate(service, eyes, backend, *, head_id, seed, progress, cancel):
    """Generate two candidates on an already-owned Qwen session.

    The caller owns the GPU lock and backend lifetime, allowing head_job to
    reuse the portrait's resident Qwen model. A failed quality gate falls
    back to the library; inference/cancellation errors propagate normally.
    """
    color = target_color(eyes)
    description = describe_color(color)
    info = {"requested": "portrait", "target_linear": color.get("median_linear"),
            "color_prompt": description, "candidates": [], "rejected": 0,
            "qwen_preset": getattr(backend, "preset", "mock")}
    if description is None:
        return service.list_plates(), dict(info, source="library", fallback="iris colour unavailable")
    rejected = service.work / "plates-rejected"
    rejected.mkdir(parents=True, exist_ok=True)

    def step(fraction, message):
        if cancel.is_set():
            raise gpu.Cancelled("cancelled")
        progress(fraction, message)

    result = qwen.generate_plates(backend, service.plates, rejected, count=2,
                                  seed=(int(seed) + 100003) % (2 ** 31), color=description,
                                  style="macro", steps=20, progress=step)
    step(1., "iris candidates ready")
    kept = result["kept"]
    for rec in kept:
        rec.update(source_head=head_id, portrait_target_linear=info["target_linear"])
        (service.plates / rec["id"] / "plate.json").write_text(json.dumps(rec, indent=1))
    info.update(candidates=[p["id"] for p in kept], rejected=len(result["rejected"]))
    if kept:
        return kept, dict(info, source="portrait")
    return service.list_plates(), dict(info, source="library", fallback="no candidate passed the quality gate")


def prepare(service, eyes, *, head_id, seed, source, progress, cancel, python=None, mock=False, preset="fast12"):
    """Standalone preparation for head-fit (which otherwise stays CPU-only)."""
    if source == "library":
        return service.list_plates(), {"requested": "library", "source": "library"}
    if source != "portrait":
        raise ValueError("iris_source must be portrait or library")
    if preset not in ("fast12", "low8"):
        raise ValueError("qwen_preset must be fast12 or low8")
    lock = service.work / "mock-gpu.lock" if mock else gpu.LOCK_PATH
    with gpu.device_session(8192 if preset == "low8" else gpu.QWEN_MIN_FREE_MIB, cancel, lock_path=lock, check_memory=not mock):
        backend = qwen.make_backend(python, mock, preset=preset)
        try:
            return generate(service, eyes, backend, head_id=head_id, seed=seed, progress=progress, cancel=cancel)
        finally:
            backend.close()


def color_controls(target, source):
    """Fit the existing shader's saturation and RGB tint to measured colour.

    Prefer the least change among equally accurate fits. Tint is bounded
    by the eye schema (0..1), so dark generated plates cannot always reach
    brighter targets; the caller records the remaining error explicitly.
    Both the portable bake and analytic shader use these same controls.
    """
    if not np.isfinite(color_distance(target, source)):
        return {"global_saturation": 1., "global_tint": [1., 1., 1.]}
    target, source = np.asarray(target), np.asarray(source)
    saturation = np.linspace(0, 2, 401)
    luma = source @ np.array([.2126, .7152, .0722])
    adjusted = np.maximum(luma + (source - luma) * saturation[:, None], 0)
    tint = np.clip(target / np.maximum(adjusted, 1e-5), 0, 1)
    result = adjusted * tint
    error = ((np.log(result + .01) - np.log(target + .01)) ** 2).sum(1)
    regularizer = .0001 * ((saturation - 1) ** 2 + ((tint - 1) ** 2).sum(1))
    i = int(np.argmin(error + regularizer))
    return {"global_saturation": float(saturation[i]), "global_tint": tint[i].tolist()}


def controlled_color(source, params):
    source = np.asarray(source)
    luma = source @ np.array([.2126, .7152, .0722])
    return np.maximum(luma + (source - luma) * params["global_saturation"], 0) * params["global_tint"]
