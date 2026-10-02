"""Standalone native video publication and reviewed synthetic candidate training.

No shared server files or live rigs are changed by candidate training.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
import threading
import uuid
from generate import ROOT, RUNNER, DEFAULT_MODEL, atomic_json, digest, generate, run_process

sys.path.insert(0, str(ROOT))
REVIEWS = ("identity", "expression", "camera", "visibility", "artifacts")


def reviewed_dataset(path):
    path = Path(path).resolve()
    dataset = json.loads(path.read_text())
    if dataset.get("schema") != "hv15n.synthetic_dataset.v1":
        raise ValueError("unsupported reviewed dataset")
    clips, seen, seeds = [], set(), {}
    for entry in dataset.get("clips", []):
        if not all(entry.get("review", {}).get(key) is True for key in REVIEWS):
            raise ValueError("every clip needs explicit identity/expression/camera/visibility/artifact review")
        folder = (path.parent / entry["clip"]).resolve()
        manifest = json.loads((folder / "manifest.json").read_text())
        if entry.get("generation_manifest_sha256") != digest(folder / "manifest.json"):
            raise ValueError("reviewed generation manifest hash is missing or stale")
        if manifest.get("task") != "i2v" or manifest.get("backend") != "hv15n_cuda_experimental":
            raise ValueError("candidate training requires native I2V provenance")
        sha = digest(folder / "clip.mp4")
        if entry.get("clip_sha256") != sha or sha in seen:
            raise ValueError("reviewed clip hash is missing, stale or duplicated")
        seen.add(sha)
        split = entry.get("split")
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test")
        seed = manifest["seed"]
        if seed in seeds and seeds[seed] != split:
            raise ValueError("all clips with the same seed must stay in the same partition")
        seeds[seed] = split
        if split != "train" and entry.get("expression_frames"):
            raise ValueError("expression refinement can use training clips only")
        clips.append(dict(entry, folder=folder, manifest=manifest, sha256=sha))
    if not clips or not any(c["split"] == "train" for c in clips) or not any(c["split"] == "test" for c in clips):
        raise ValueError("provide separate training and independent test clips")
    if sum(c["split"] == "validation" for c in clips) != 1:
        raise ValueError("the existing modal trainer supports exactly one validation clip")
    if len(clips) > 9:
        raise ValueError("limit candidate datasets to eight train/validation takes and one test take")
    if len({c["manifest"]["image_sha256"] for c in clips}) != 1:
        raise ValueError("all clips must use the same reviewed portrait")
    return dataset, clips


@contextmanager
def candidate_stage(out, cancel):
    try:
        yield
    except BaseException as error:
        path = out / "candidate.json"
        receipt = json.loads(path.read_text())
        receipt.update(state="cancelled" if cancel.is_set() else "failed", error=str(error))
        atomic_json(path, receipt)
        raise


def publish(*, work, head, cancel=None, **options):
    from server.vhuman import gpu
    from server.vhuman.service import EyeService
    service = EyeService(Path(work))
    portrait = service.head_file(head, "portrait.png")
    run_id = uuid.uuid4().hex
    videos = portrait.parent / "videos"
    videos.mkdir(exist_ok=True)
    stage = videos / (".partial-" + run_id)
    budget = options.get("vram_budget_mib", 14336)
    device = options.get("device", 0)
    try:
        with gpu.execution("cuda", device), gpu.device_session(budget - 2048, cancel):
            result = generate(out=stage, image=portrait, task="i2v", cancel=cancel, **options)
        result.update(id=run_id, head_id=head, request={"head_id": head, "prompt": result["prompt"],
                      "preset": result["preset"], "seed": result["seed"]}, synthetic=True)
        atomic_json(stage / "manifest.json", result)
        stage.rename(videos / run_id)
        return {"id": run_id, "folder": str(videos / run_id), "manifest": result}
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def fit_clip(service, head, entry, cancel, python):
    from server.vhuman.rig import video_fit
    clip = entry.get("fit_clip", entry["folder"] / "clip.mp4")
    with clip.open("rb") as stream:
        upload = video_fit.upload(service, stream, clip.stat().st_size, "video/mp4")
    result = video_fit.fit_job(service, {"head_id": head, "upload_id": upload["id"]},
                              lambda f, m: print(m, flush=True), cancel, python=python)
    folder = service.take_file(head, result["id"], "animation.json").parent
    report = json.loads((folder / "fit_report.json").read_text())
    if report.get("valid_frames", 0) / max(1, report["frames"]) < .95:
        raise ValueError("candidate fit has too many frames without a visible face")
    if report.get("landmark_error_after_normalized", float("inf")) > .03:
        raise ValueError("candidate normalized landmark fit error exceeds 0.03")
    meta = json.loads((folder / "manifest.json").read_text())
    meta.update(source="synthetic_i2v", synthetic=True, synthetic_clip_sha256=entry["sha256"],
                observation_clip_sha256=digest(clip),
                synthetic_seed=entry["manifest"]["seed"], dataset_split=entry["split"],
                generation_manifest_sha256=digest(entry["folder"] / "manifest.json"),
                review=entry["review"])
    atomic_json(folder / "manifest.json", meta)
    return folder


def restore_clip(entry, portrait, out, cancel=None):
    """Fit in the original portrait coordinates, undoing the video center crop."""
    from PIL import Image, ImageOps
    out.mkdir(parents=True, exist_ok=True)
    with Image.open(portrait) as opened:
        rgba = ImageOps.exif_transpose(opened).convert("RGBA")
    base = Image.new("RGBA", rgba.size, (96, 96, 96, 255))
    base.alpha_composite(rgba)
    scale = max(480 / base.width, 848 / base.height)
    resized = base.convert("RGB").resize((round(base.width * scale), round(base.height * scale)), Image.Resampling.LANCZOS)
    left, top = round((resized.width - 480) / 2), round((resized.height - 848) / 2)
    backdrop = out / "backdrop.png"
    resized.save(backdrop)
    factor = min(1, 768 / max(base.size))
    width, height = max(2, round(base.width * factor / 2) * 2), max(2, round(base.height * factor / 2) * 2)
    result = out / "observations.mp4"
    run_process(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-loop", "1", "-framerate", "24", "-i", backdrop,
                 "-i", entry["folder"] / "clip.mp4", "-filter_complex",
                 f"[0:v][1:v]overlay={left}:{top}:shortest=1,scale={width}:{height}:flags=lanczos",
                 "-r", "24", "-frames:v", "81", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", result], cancel=cancel)
    atomic_json(out / "transform.json", {"synthetic": True, "source_clip_sha256": entry["sha256"],
        "observation_clip_sha256": digest(result), "portrait_sha256": digest(portrait),
        "crop_origin": [left, top], "resized_portrait": list(resized.size), "observation_size": [width, height]})
    return result


def expression_frames(clips, takes, portrait, out):
    """Restore the model's center crop; use fitted controls at the selected time."""
    import numpy as np
    from PIL import Image, ImageOps
    from server.vhuman.rig.exprdata import EXPRESSIONS
    rgba = ImageOps.exif_transpose(Image.open(portrait)).convert("RGBA")
    base = Image.new("RGBA", rgba.size, (96, 96, 96, 255))
    base.alpha_composite(rgba)
    base = base.convert("RGB")
    out.mkdir(parents=True, exist_ok=True)
    base.save(out / "ref.png")
    manifest = {"ref": "ref.png", "size": list(base.size), "expressions": {}, "synthetic": True,
                "source": "reviewed_native_i2v", "controls": "fitted_at_selected_time"}
    scale = max(480 / base.width, 848 / base.height)
    resized = base.resize((round(base.width * scale), round(base.height * scale)), Image.Resampling.LANCZOS)
    left, top = round((resized.width - 480) / 2), round((resized.height - 848) / 2)
    for clip, take in zip(clips, takes):
        track = json.loads((take / "animation.json").read_text())
        frames = track["frames"]
        for name, index in clip.get("expression_frames", {}).items():
            if name not in EXPRESSIONS or name in manifest["expressions"] or type(index) is not int or not 0 <= index < 81:
                raise ValueError("expression selection needs a unique known name and frame index 0..80")
            selected = out / (name + "_crop.png")
            run_process(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i",
                         clip["folder"] / "clip.mp4", "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", selected])
            with Image.open(selected) as image:
                restored = resized.copy()
                restored.paste(image.convert("RGB"), (left, top))
                restored.resize(base.size, Image.Resampling.LANCZOS).save(out / (name + ".png"))
            selected.unlink()
            sample = min(frames, key=lambda f: abs(f["t"] - index / 24))
            controls = {k: float(v) for k, v in sample["v"].items() if np.isfinite(v) and abs(v) > .001}
            if not any(abs(controls.get(k, 0)) >= .15 for k in EXPRESSIONS[name][1]):
                raise ValueError(f"selected {name} frame has insufficient fitted expression motion")
            manifest["expressions"][name] = {"file": name + ".png", "prompt": clip["manifest"]["prompt"],
                "controls": controls, "wrinkle_group": EXPRESSIONS[name][2],
                "clip_sha256": clip["sha256"], "source_frame": index, "source_fps": 24}
    if not manifest["expressions"]:
        raise ValueError("select at least one training expression frame")
    atomic_json(out / "manifest.json", manifest)


def test_candidate(rig_dir, takes):
    import numpy as np
    from server.vhuman.rig import rigdef, safetensors as st, soft_deformer as sd
    definition = json.loads((rig_dir / "rig.json").read_text())
    package, _ = st.load(rig_dir / "rig_deformer.safetensors")
    rest = package["rest"].astype(np.float32)
    viz = json.loads((rig_dir / "viz.json").read_text())
    geometry = digest(rig_dir / "rig_deformer.safetensors")[:16]
    model, meta = st.load(rig_dir / "soft_deformer.safetensors")
    if meta["geometry_sha256"] != geometry:
        raise ValueError("candidate geometry signature mismatch")
    results = []
    for take in takes:
        x, y, fps = sd._load_take(rig_dir, take, rigdef.Rig(definition), package, rest, viz, geometry)
        if abs(fps - float(meta["fps"])) > .001:
            raise ValueError("test frame rate differs from training")
        prediction = sd.predict_sequence(x, model["weight"], tuple(model["recurrence"])) @ model["basis"]
        active = np.any(np.abs(y) > 1.e-8, axis=0)
        if not active.any():
            raise ValueError("independent test teacher has no soft tissue motion")
        baseline = float(np.sqrt(np.mean(y[:, active] ** 2)) * 1000)
        error = float(np.sqrt(np.mean((prediction[:, active] - y[:, active]) ** 2)) * 1000)
        results.append({"take": take.name, "baseline_rmse_mm": baseline, "model_rmse_mm": error,
                        "pass": bool(np.isfinite(error) and error < baseline)})
    return {"geometry_sha256": geometry, "independent_test": results,
            "pass": bool(results and all(r["pass"] for r in results))}


def train_candidate(*, work, head, dataset, out, python, device=0, cancel=None):
    from server.vhuman import gpu
    from server.vhuman.service import EyeService
    from server.vhuman.rig import soft_tissue, soft_deformer
    cancel = cancel or threading.Event()
    _, clips = reviewed_dataset(dataset)
    source = EyeService(Path(work)).head_file(head, "head.json").parent
    if digest(source / "portrait.png") != clips[0]["manifest"]["image_sha256"]:
        raise ValueError("reviewed dataset portrait differs from the selected head")
    out = Path(out).resolve()
    work_path = Path(work).resolve()
    if out.is_relative_to(work_path) or work_path.is_relative_to(out):
        raise ValueError("candidate output must be separate from the live work directory")
    out.mkdir(parents=True, exist_ok=False)
    candidate = out / "work/heads" / head
    shutil.copytree(source, candidate, ignore=shutil.ignore_patterns("rig", "videos", "candidates"))
    rig = candidate / "rig"
    shutil.copytree(source / "rig", rig, ignore=shutil.ignore_patterns("takes", "expressions", "soft_deformer*"))
    definition = json.loads((rig / "rig.json").read_text())
    definition.pop("soft_deformer", None)
    atomic_json(rig / "rig.json", definition)
    service = EyeService(out / "work")
    original_hash = digest(source / "rig/rig_deformer.safetensors")
    atomic_json(out / "candidate.json", {"schema": "hv15n.rig_candidate.v1", "state": "building",
        "synthetic": True, "live_head": str(source), "original_geometry_sha256": original_hash,
        "dataset_sha256": digest(dataset), "clips": [{"sha256": c["sha256"], "seed": c["manifest"]["seed"],
                                                     "split": c["split"]} for c in clips]})
    with candidate_stage(out, cancel), gpu.execution("cuda", device):
        for i, clip in enumerate(clips):
            clip["fit_clip"] = restore_clip(clip, source / "portrait.png", out / "observations" / str(i), cancel)
        initial = [fit_clip(service, head, c, cancel, python) for c in clips]
        expression_frames(clips, initial, source / "portrait.png", rig / "expressions")
        # Rebuild shapes before fitting the final controls/physics teacher.
        with gpu.device_session(2048, cancel):
            run_process([python, "-m", "server.vhuman.rig.build", candidate, "--out", rig,
                         "--reuse-fit", "--no-preview", "--deformer-samples", "0",
                         "--cache", out / "cache"], cancel=cancel)
        shutil.rmtree(rig / "takes")
        final = [fit_clip(service, head, c, cancel, python) for c in clips]
        for take in final:
            soft_tissue.soft_tissue_job(service, {"head_id": head, "take_id": take.name},
                                       lambda f, m: print(m, flush=True), cancel, python=python)
        train = [t.name for c, t in zip(clips, final) if c["split"] == "train"]
        validation = [t.name for c, t in zip(clips, final) if c["split"] == "validation"]
        soft_deformer.train_job(service, {"head_id": head, "take_ids": train + validation},
                                lambda f, m: print(m, flush=True), cancel, python=python)
        report = test_candidate(rig, [t for c, t in zip(clips, final) if c["split"] == "test"])
    receipt = json.loads((out / "candidate.json").read_text())
    receipt.update(state="reviewable" if report["pass"] else "test_failed", report=report,
                   candidate_geometry_sha256=digest(rig / "rig_deformer.safetensors"),
                   expression_manifest_sha256=digest(rig / "expressions/manifest.json"))
    atomic_json(out / "candidate.json", receipt)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--model", default=str(DEFAULT_MODEL), help="model package directory (default: %(default)s)")
    for key in ("work", "head", "prompt"):
        gen.add_argument("--" + key, required=True)
    gen.add_argument("--preset", choices=("quality", "fast12"), default="quality")
    gen.add_argument("--negative-prompt", default="")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--device", type=int, default=0)
    gen.add_argument("--vram-budget-mib", type=int, default=14336)
    gen.add_argument("--gemm", choices=("repo", "cublas"), default="repo")
    gen.add_argument("--gemm-fallback", choices=("cublas", "error"), default="cublas")
    gen.add_argument("--runner", default=str(RUNNER))
    gen.add_argument("--allow-experimental", action="store_true")
    train = commands.add_parser("train-candidate")
    for key in ("work", "head", "dataset", "out"):
        train.add_argument("--" + key, required=True)
    train.add_argument("--python", default=str(ROOT / "tmp/vhuman-rig-venv/bin/python"))
    train.add_argument("--device", type=int, default=0)
    args = vars(ap.parse_args())
    command = args.pop("command")
    result = publish(**args) if command == "generate" else train_candidate(**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
