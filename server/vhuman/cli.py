"""Virtual-human eyes from the command line.

    python3 -m server.vhuman.cli eye --preset hazel --res 2048 --out tmp/vhuman/hazel [--pair] [--plate ID]
    python3 -m server.vhuman.cli sheet --out tmp/vhuman/presets.png [--size 256]
    python3 -m server.vhuman.cli render --preset blue --yaw 20 --out tmp/vhuman/blue.png
    python3 -m server.vhuman.cli plates --count 8 --seed 1 --color hazel [--style macro] [--mode refine]
    python3 -m server.vhuman.cli study --seeds 8 [--styles macro,clinical,ocularist,minimal]
    python3 -m server.vhuman.cli extract photo.png --out tmp/vhuman/extracted [--add]
    python3 -m server.vhuman.cli replate
    python3 -m server.vhuman.cli head --subject "a 30-year-old woman with short dark hair" [--quality standard]
    python3 -m server.vhuman.cli head-fit --portrait P.png --glb pixal3d.glb
    python3 -m server.vhuman.cli rig --head ID [--res 2048]      (template fit, skeleton, shapes, GLB + USD)
    python3 -m server.vhuman.cli rig-track --head ID --track T.txt --out anim.usda   (LightRig track -> UsdSkel)
    python3 -m server.vhuman.cli baseline --source analytic|qwen [--quality preview]
    python3 -m server.vhuman.cli bench

Every command prints a JSON summary on stdout. GPU commands take the CUDA
lock shared with the Pixal3D demo server and need --qwen-python (default
tmp/qimg21-ref-venv/bin/python) for Qwen-Image.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .eye import assets, extract, iris, render, sclera
from .eye import params as P
from .service import ROOT, WORK, EyeService

DEFAULT_QWEN_PY = ROOT / "tmp/qimg21-ref-venv/bin/python"


def _params(args) -> dict:
    p = P.preset(args.preset) if getattr(args, "preset", None) else P.defaults()
    if getattr(args, "params", None):
        overrides = json.loads(Path(args.params[1:]).read_text() if args.params.startswith("@") else args.params)
        for group, values in overrides.items():
            p.setdefault(group, {}).update(values)
    if getattr(args, "seed", None) is not None and "structure" in p:
        p["structure"]["seed"] = args.seed
    return P.validate(p)


def _progress(fraction, message):
    print(f"[{fraction * 100:5.1f}%] {message}", file=sys.stderr, flush=True)


def cmd_eye(args) -> dict:
    svc = EyeService(Path(args.work))
    p = _params(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    iris_tex = svc.plate_iris(args.plate, args.res) if args.plate else None
    glb = assets.export_glb(p, out / "eye.glb", args.res, pair=args.pair, iris_tex=iris_tex)
    textures = assets.export_textures(p, out / "textures", args.res, iris_tex=iris_tex)
    live = assets.write_live(p, out / "live", min(args.res, 1024))
    (out / "params.json").write_text(json.dumps(p, indent=1))
    img = render.render(p, 512, render.Camera(yaw_deg=15, pitch_deg=-5), spp=4, iris_tex=iris_tex)
    render.save(img, out / "preview.png")
    return {"out": str(out), "glb": {k: glb[k] for k in ("bytes", "seconds", "res")}, "textures": textures["files"],
            "live": live["timings"], "seconds": round(time.perf_counter() - started, 2)}


def cmd_sheet(args) -> dict:
    size = args.size
    names = list(P.PRESETS)
    cols = 4
    rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * (size + 22)), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)
    started = time.perf_counter()
    for k, name in enumerate(names):
        img = render.render(P.preset(name), size, render.Camera(yaw_deg=args.yaw, pitch_deg=args.pitch), spp=4)
        x, y = (k % cols) * size, (k // cols) * (size + 22)
        sheet.paste(Image.fromarray((img[..., :3] * 255 + 0.5).astype(np.uint8)), (x, y))
        draw.text((x + 8, y + size + 5), name.replace("_", " "), fill=(220, 224, 230))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    return {"out": args.out, "presets": names, "seconds": round(time.perf_counter() - started, 2)}


def cmd_render(args) -> dict:
    p = _params(args)
    started = time.perf_counter()
    cam = render.Camera(fov_deg=args.fov, yaw_deg=args.yaw, pitch_deg=args.pitch)
    iris_tex = EyeService(Path(args.work)).plate_iris(args.plate, 1024) if args.plate else None
    img = render.render(p, args.size, cam, spp=args.spp, iris_tex=iris_tex)
    render.save(img, args.out)
    return {"out": args.out, "seconds": round(time.perf_counter() - started, 2)}


def cmd_plates(args) -> dict:
    from . import qwen
    svc = EyeService(Path(args.work))
    req = {"count": args.count, "seed": args.seed_start, "color": args.color, "style": args.style,
           "mode": args.mode, "strength": args.strength, "steps": args.steps}
    if args.mode == "refine":
        req["params"] = _params(args)
    return qwen.plates_job(svc, req, _progress, threading.Event(), python=args.qwen_python, mock=args.mock)


class _GpuSampler:
    """Peak device memory used while a block runs (nvidia-smi, 1 Hz)."""

    def __init__(self):
        from . import gpu
        self.gpu, self.peak, self._stop = gpu, 0, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            s = self.gpu.gpu_status()
            if s:
                self.peak = max(self.peak, s["total_mib"] - s["free_mib"])
            self._stop.wait(1.0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def cmd_study(args) -> dict:
    """The prompt study: each style x seeds, colours cycling through the
    presets; pass rate, failure taxonomy, timings and peak GPU memory."""
    from . import gpu, qwen
    svc = EyeService(Path(args.work))
    styles = args.styles.split(",")
    colors = list(qwen.COLORS)
    out = Path(args.out or svc.work / "study" / time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    rejected = out / "rejected"
    rejected.mkdir(exist_ok=True)
    results = {"styles": {}, "steps": args.steps, "seeds": args.seeds, "size": 1024}
    started = time.perf_counter()
    base = gpu.gpu_status() or {}
    with _GpuSampler() as sampler, gpu.device_session(gpu.QWEN_MIN_FREE_MIB, check_memory=not args.mock,
                                                      lock_path=(svc.work / "mock-gpu.lock") if args.mock
                                                      else gpu.LOCK_PATH):
        backend = qwen.make_backend(args.qwen_python, args.mock)
        try:
            for style in styles:
                recs = {"kept": [], "rejected": []}
                for k in range(args.seeds):
                    color = colors[(k + styles.index(style) * 3) % len(colors)]
                    r = qwen.generate_plates(backend, svc.plates, rejected, count=1, seed=args.seed_start + k,
                                             color=color, style=style, steps=args.steps,
                                             progress=lambda f, m, s=style: _progress(f, f"{s}: {m}"))
                    recs["kept"] += r["kept"]
                    recs["rejected"] += r["rejected"]
                fails: dict[str, int] = {}
                for rec in recs["rejected"]:
                    for f in (rec.get("quality", {}).get("failures") or [rec.get("error", "error")]):
                        fails[f] = fails.get(f, 0) + 1
                times = [r["generate_seconds"] for r in recs["kept"] + recs["rejected"]]
                results["styles"][style] = {
                    "pass_rate": round(len(recs["kept"]) / args.seeds, 3), "kept": [r["id"] for r in recs["kept"]],
                    "failures": fails, "median_generate_s": round(float(np.median(times)), 2) if times else None,
                    "median_effective_resolution": int(np.median([r["quality"]["effective_resolution"]
                                                                  for r in recs["kept"]])) if recs["kept"] else None}
        finally:
            backend.close()
    results["peak_gpu_mib"] = sampler.peak
    results["gpu_used_before_mib"] = (base.get("total_mib", 0) - base.get("free_mib", 0)) if base else None
    results["seconds"] = round(time.perf_counter() - started, 1)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    return {"out": str(out), **results}


def cmd_extract(args) -> dict:
    plate = extract.extract(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray((plate.masks * 255 + 0.5).astype(np.uint8)).save(out / "iris_masks.png")
    Image.fromarray((plate.photo * 255 + 0.5).astype(np.uint8)).save(out / "iris_photo.png")
    result = {"out": str(out), "quality": plate.quality, "colors": plate.colors,
              "limbus": plate.limbus.as_list(), "pupil": plate.pupil.as_list()}
    if args.add:
        from . import qwen
        svc = EyeService(Path(args.work))
        rec = qwen.save_plate(svc.plates, plate, Path(args.image), {"mode": "photo", "source": str(args.image)})
        result["plate_id"] = rec["id"]
    return result


def _skin_params(args):
    from .head import skin
    raw = args.skin
    return skin.validate(json.loads(Path(raw[1:]).read_text() if raw.startswith("@") else raw)) if raw else skin.validate()


def cmd_head(args) -> dict:
    from .head import pipeline
    svc = EyeService(Path(args.work))
    req = {"subject": args.subject, "seed": args.seed_start, "quality": args.quality, "fov": args.fov,
           "steps": args.steps, "iris_source": args.iris_source, "skin": _skin_params(args), "qwen_preset": args.qwen_preset}
    return pipeline.head_job(svc, req, _progress, threading.Event(), python=args.qwen_python, mock=args.mock)


def cmd_head_fit(args) -> dict:
    """Fit eyes to an existing portrait + Pixal3D GLB, as a head record."""
    import shutil
    import uuid
    from .head import fit, iris_match, landmarks
    from . import qwen
    skin_params = _skin_params(args)
    started = time.perf_counter()
    svc = EyeService(Path(args.work))
    folder = svc.work / "heads" / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    shutil.copy(args.portrait, folder / "portrait.png")
    shutil.copy(args.glb, folder / "pixal3d.glb")
    plates, iris_info = iris_match.prepare(svc, landmarks.find_eyes(folder / "portrait.png"),
        head_id=folder.name, seed=args.seed_start, source=args.iris_source, progress=_progress,
        cancel=threading.Event(), python=args.qwen_python, mock=args.mock, preset=args.qwen_preset)
    rec = fit.fit_head(folder / "portrait.png", folder / "pixal3d.glb", folder, fov_deg=args.fov,
                       plates=plates, plate_loader=svc.plate_iris, res=1024, iris_info=iris_info, skin_params=skin_params)
    head = {"id": folder.name, "subject": args.subject, "quality": "imported", "fov": args.fov,
            "created": time.time(), "fit": {k: rec[k] for k in ("fit", "plate", "eyes", "seconds")},
            "export": rec["export"], "iris": rec["iris"], "skin": rec["skin"], "seconds": round(time.perf_counter() - started, 2)}
    if rec["iris"]["source"] == "portrait":
        head["license"] = qwen.LICENSE
    (folder / "head.json").write_text(json.dumps(head, indent=1, default=float))
    return {"id": folder.name, "fit": rec["fit"], "plate": rec["plate"], "iris": rec["iris"], "skin": rec["skin"], "seconds": head["seconds"]}


def cmd_replate(args) -> dict:
    from . import qwen
    return qwen.replate(EyeService(Path(args.work)).plates)


def cmd_baseline(args) -> dict:
    from . import baseline
    svc = EyeService(Path(args.work))
    req = {"source": args.source, "quality": args.quality, "color": args.color, "seed": args.seed_start,
           "params": _params(args)}
    return baseline.baseline_job(svc, req, _progress, threading.Event(), mock=args.mock, python=args.qwen_python)



def cmd_rig(args) -> dict:
    """The rig job without the server: a subprocess in the rig interpreter."""
    from .rig import job as rig_job
    svc = EyeService(Path(args.work))
    return rig_job.rig_job(svc, {"head_id": args.head, "res": args.res, "iters": args.iters}, _progress,
                           threading.Event(), python=args.rig_python)


def cmd_rig_track(args) -> dict:
    """Evaluate a LightRig face track on a built rig -> a UsdSkel animation layer."""
    from .rig import rigdef, usd
    svc = EyeService(Path(args.work))
    rig_json = svc.rig_file(args.head, "rig.json")
    rig = rigdef.Rig(json.loads(rig_json.read_text()))
    times, frames = rigdef.read_track(args.track)
    return usd.write_track(rig, times, frames, Path(args.out), fps=args.fps,
                           rig_layer=str(rig_json.parent / "rig.usda"))


def cmd_bench(args) -> dict:
    """Timing targets (cold caches): procedural textures and renders."""
    out = {}
    for res in (1024, 2048):
        p = P.preset("hazel")
        p["structure"]["seed"] = int(time.time()) % 100000      # cold structure cache
        t = time.perf_counter(); it = iris.build(p, res); t_iris = time.perf_counter() - t
        t = time.perf_counter(); sclera.build(p, res); t_sclera = time.perf_counter() - t
        t = time.perf_counter(); iris.bake_color(it, p); t_bake = time.perf_counter() - t
        p["structure"]["seed"] += 1
        t = time.perf_counter(); assets.live_textures(p, res); t_set = time.perf_counter() - t
        svc = EyeService(Path(args.work) / "bench")
        p["structure"]["seed"] += 1
        t = time.perf_counter(); svc.textures(p, res); t_endpoint = time.perf_counter() - t
        out[str(res)] = {"iris_structure_s": round(t_iris, 3), "sclera_s": round(t_sclera, 3),
                         "texture_set_s": round(t_set, 3), "textures_endpoint_s": round(t_endpoint, 3),
                         "iris_bake_s": round(t_bake, 3)}
    t = time.perf_counter(); render.render(P.preset("hazel"), 512, spp=4); out["render_512_spp4_s"] = round(time.perf_counter() - t, 3)
    img, _, _ = extract.synthetic_eye(P.preset("blue"), 1024, 3)
    t = time.perf_counter(); extract.extract(img); out["extract_1024_s"] = round(time.perf_counter() - t, 3)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--qwen-python", default=str(DEFAULT_QWEN_PY) if DEFAULT_QWEN_PY.exists() else None)
    ap.add_argument("--mock", action="store_true", help="mock Qwen and Pixal3D (no GPU)")
    default_rig = ROOT / "tmp/vhuman-rig-venv/bin/python"
    ap.add_argument("--rig-python", default=str(default_rig) if default_rig.exists() else None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_params(sp):
        sp.add_argument("--preset", choices=list(P.PRESETS))
        sp.add_argument("--params", help='JSON overrides, e.g. {"pupil": {"dilation": 1.1}}, or @file.json')
        sp.add_argument("--seed", type=int, help="structure seed")
        return sp

    sp = with_params(sub.add_parser("eye", help="textures, GLB, texture set and a preview"))
    sp.add_argument("--res", type=int, choices=(512, 1024, 2048), default=2048)
    sp.add_argument("--out", required=True)
    sp.add_argument("--pair", action="store_true")
    sp.add_argument("--plate", help="an iris plate id as the detail source")
    sp.set_defaults(fn=cmd_eye)
    sp = sub.add_parser("sheet", help="contact sheet of the presets")
    sp.add_argument("--out", required=True)
    sp.add_argument("--size", type=int, default=256)
    sp.add_argument("--yaw", type=float, default=0.0)
    sp.add_argument("--pitch", type=float, default=0.0)
    sp.set_defaults(fn=cmd_sheet)
    sp = with_params(sub.add_parser("render", help="CPU render"))
    sp.add_argument("--out", required=True)
    sp.add_argument("--size", type=int, default=512)
    sp.add_argument("--spp", type=int, default=4)
    sp.add_argument("--yaw", type=float, default=0.0)
    sp.add_argument("--pitch", type=float, default=0.0)
    sp.add_argument("--fov", type=float, default=0.0)
    sp.add_argument("--plate")
    sp.set_defaults(fn=cmd_render)
    sp = with_params(sub.add_parser("plates", help="Qwen iris plates into the library"))
    sp.add_argument("--count", type=int, default=4)
    sp.add_argument("--seed-start", type=int, default=1)
    sp.add_argument("--color", default="hazel")
    sp.add_argument("--style", default="macro")
    sp.add_argument("--mode", choices=("text", "refine"), default="text")
    sp.add_argument("--strength", type=float, default=0.5)
    sp.add_argument("--steps", type=int, default=20)
    sp.set_defaults(fn=cmd_plates)
    sp = sub.add_parser("study", help="prompt study for iris plates")
    sp.add_argument("--styles", default="macro,clinical,ocularist,minimal")
    sp.add_argument("--seeds", type=int, default=8)
    sp.add_argument("--seed-start", type=int, default=100)
    sp.add_argument("--steps", type=int, default=20)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_study)
    sp = sub.add_parser("extract", help="normalise an iris photo into a plate")
    sp.add_argument("image")
    sp.add_argument("--out", required=True)
    sp.add_argument("--add", action="store_true", help="also add it to the plate library")
    sp.set_defaults(fn=cmd_extract)
    sp = with_params(sub.add_parser("baseline", help="Pixal3D baseline"))
    sp.add_argument("--source", choices=("analytic", "qwen"), default="analytic")
    sp.add_argument("--quality", choices=("preview", "standard", "high"), default="preview")
    sp.add_argument("--color", default="hazel")
    sp.add_argument("--seed-start", type=int, default=7)
    sp.set_defaults(fn=cmd_baseline)
    sp = sub.add_parser("head", help="text -> Qwen portrait -> Pixal3D head -> fitted eyes (GPU)")
    sp.add_argument("--subject", required=True)
    sp.add_argument("--seed-start", type=int, default=11)
    sp.add_argument("--quality", choices=("preview", "standard", "high"), default="standard")
    sp.add_argument("--fov", type=float, default=20.0)
    sp.add_argument("--steps", type=int, default=24)
    sp.add_argument("--iris-source", choices=("portrait", "library"), default="portrait")
    sp.add_argument("--qwen-preset", choices=("fast12", "low8"), default="fast12")
    sp.add_argument("--skin", help="Skin controls as JSON or @file.json")
    sp.set_defaults(fn=cmd_head)
    sp = sub.add_parser("head-fit", help="fit eyes to an existing portrait + Pixal3D GLB")
    sp.add_argument("--portrait", required=True)
    sp.add_argument("--glb", required=True)
    sp.add_argument("--fov", type=float, default=20.0)
    sp.add_argument("--subject", default="imported")
    sp.add_argument("--iris-source", choices=("portrait", "library"), default="library")
    sp.add_argument("--seed-start", type=int, default=11)
    sp.add_argument("--qwen-preset", choices=("fast12", "low8"), default="fast12")
    sp.add_argument("--skin", help="Skin controls as JSON or @file.json")
    sp.set_defaults(fn=cmd_head_fit)
    sp = sub.add_parser("rig", help="facial rig of a fitted head (needs --rig-python: numpy, scipy, torch)")
    sp.add_argument("--head", required=True)
    sp.add_argument("--res", type=int, choices=(1024, 2048, 4096), default=2048)
    sp.add_argument("--iters", type=int, default=600)
    sp.set_defaults(fn=cmd_rig)
    sp = sub.add_parser("rig-track", help="LightRig face track (timestamp + 52 controls per line) -> USD animation")
    sp.add_argument("--head", required=True)
    sp.add_argument("--track", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--fps", type=float, default=30.0)
    sp.set_defaults(fn=cmd_rig_track)
    sub.add_parser("replate", help="re-extract the plate library from its source images").set_defaults(fn=cmd_replate)
    sub.add_parser("bench", help="timing targets").set_defaults(fn=cmd_bench)
    args = ap.parse_args(argv)
    try:
        result = args.fn(args)
    except (P.ParamError, ValueError, extract.ExtractError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
