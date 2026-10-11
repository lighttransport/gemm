"""Job orchestration for offline MuscleMimic. No JAX/MuJoCo imports here."""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import uuid

from .. import gpu
from . import job, motion
from .musclemimic_assets import read_installation, reference, sha256


def add_arguments(parser):
    parser.add_argument("--musclemimic-python", help="isolated Python interpreter with the pinned offline stack")
    parser.add_argument("--musclemimic-assets", help="directory containing registered installation.json")
    parser.add_argument("--musclemimic-device", choices=("cpu", "cuda"), default="cpu",
                        help="policy inference device; physics uses CPU MuJoCo")


def catalog(assets, python=None):
    if assets is None:
        return {"available": False, "references": []}
    try:
        data = read_installation(assets)
        ready = bool(python and Path(python).is_file())
        return {"available": ready, "research_only": True, "references": sorted(data["references"])}
    except (OSError, ValueError, KeyError):
        return {"available": False, "references": [], "reason": "invalid offline asset registration"}


def validate_request(request):
    if bool(request.get("upload_id")) == bool(request.get("reference_motion")):
        raise ValueError("select exactly one uploaded video or reference motion")
    if request.get("upload_id") and request.get("fixed_camera") is not True:
        raise ValueError("confirm fixed_camera=true: one fully visible person on a flat floor")


def fit(service, request, progress, cancel, *, model_dir=None, python=None, assets=None, device="cpu"):
    validate_request(request)
    installation = read_installation(assets)
    if not python or not Path(python).is_file():
        raise ValueError("configure --musclemimic-python with the isolated offline interpreter")
    if device not in ("cpu", "cuda"):
        raise ValueError("MuscleMimic policy device must be cpu or cuda")
    if device == "cuda" and gpu.backend() != "cuda":
        raise ValueError("CUDA policy inference requires the pipeline's --backend cuda")
    if request.get("reference_motion"):
        reference(installation, request["reference_motion"])
    hid = request.get("head_id")
    avatar = service.body_file(hid, "avatar.glb")
    avatar_json = service.body_file(hid, "avatar.json")
    body_meta = service.body_file(hid, "body_mhr.glb.json")
    model_dir = Path(model_dir or job.MODEL_DIR)
    from .mhr import ASSET_FILES
    if any(not (model_dir / "safetensors" / n).is_file() for n in ASSET_FILES):
        raise ValueError("native MHR assets are missing")
    take = uuid.uuid4().hex[:12]
    root = avatar.parent / "motions"
    stage = root / ("." + take)
    stage.mkdir(parents=True)
    manifest = {"id": take, "head_id": hid, "kind": "musclemimic", "backend": "musclemimic",
                "status": "rejected", "accepted": False, "research_only": True,
                "avatar_sha256": sha256(avatar_json), "avatar_glb_sha256": sha256(avatar),
                "frames": 0, "duration": 0, "fps": 30}
    published = False
    try:
        inputs = stage / "avatar-input"
        inputs.mkdir()
        for path in (avatar, avatar_json, body_meta):
            shutil.copyfile(path, inputs / path.name)
        worker_request = {"avatar": str((inputs / avatar.name).resolve()), "avatar_json": str((inputs / avatar_json.name).resolve()),
                          "body_meta": str((inputs / body_meta.name).resolve()), "mhr_assets": str((model_dir / "safetensors").resolve())}
        if request.get("reference_motion"):
            worker_request["reference_motion"] = request["reference_motion"]
            manifest["reference_motion"] = request["reference_motion"]
        else:
            source, upload = motion.source_file(service, request["upload_id"])
            if upload["kind"] != "video":
                raise ValueError("physical motion refinement needs video, not a still image")
            progress(.03, "sampling timestamped body motion")
            # Extraction/camera checks run in the isolated interpreter, since
            # the server itself does not require SciPy or OpenCV.
            prep_request = stage / "prepare-request.json"
            prep_request.write_text(json.dumps({"source": str(source.resolve())}))
            job._run([str(python), "-m", "server.vhuman.body.musclemimic", "prepare-video",
                      "--request", str(prep_request.resolve()), "--work", str(stage.resolve())], cancel, timeout=300)
            prepared = json.loads((stage / "prepared.json").read_text())
            sidecars = []
            for i, (frame, box) in enumerate(zip(prepared["frames"], prepared["boxes"])):
                if cancel.is_set():
                    raise gpu.Cancelled("cancelled")
                progress(.08 + .37*i/len(prepared["frames"]), f"dense body pose {i+1}/{len(prepared['frames'])}")
                sidecars.append(str(motion._sam_frame(Path(frame), stage / f"pose_{i:05d}.glb",
                                                     tuple(box), model_dir, cancel, False).resolve()))
            worker_request.update(sidecars=sidecars, times=prepared["times"])
            manifest.update(source_upload_id=request["upload_id"], source_sha256=upload["sha256"],
                            camera=prepared["camera"], root_motion="estimated")
        request_path = stage / "request.json"
        request_path.write_text(json.dumps(worker_request))
        progress(.48, "simulating muscle-driven motion and fitting MHR avatar")
        session = gpu.device_session(2048, cancel) if device == "cuda" else nullcontext()
        try:
            with session:
                job._run([str(python), "-m", "server.vhuman.body.musclemimic_worker", "--request", str(request_path.resolve()),
                          "--assets", str(Path(assets).resolve()), "--work", str(stage.resolve()), "--policy-device", device,
                          "--policy-device-index", str(gpu.device_index())],
                         cancel, timeout=7200)
        except (RuntimeError, ValueError):
            if cancel.is_set():
                raise gpu.Cancelled("cancelled")
            if not (stage / "simulation.json").exists():
                raise
        if cancel.is_set():
            raise gpu.Cancelled("cancelled")
        if sha256(avatar_json) != manifest["avatar_sha256"] or sha256(avatar) != manifest["avatar_glb_sha256"]:
            for name in ("baseline.json", "baseline.glb"):
                if (stage / name).exists():
                    (stage / name).rename(stage / ("stale-" + name))
            raise ValueError("avatar changed during refinement; retry with the rebuilt avatar")
        report = json.loads((stage / "simulation.json").read_text())
        accepted = bool(report.get("completed") and report.get("avatar", {}).get("accepted"))
        manifest.update(accepted=accepted, status="accepted" if accepted else "rejected",
                        error=report.get("error"), provenance=report.get("provenance", {}))
        track = stage / ("motion.json" if accepted else "baseline.json")
        if track.exists():
            data = json.loads(track.read_text())
            manifest.update(frames=len(data["frames"]), duration=data["duration"])
        (stage / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        # Candidate files and simulator arrays remain private diagnostics.
        if not accepted:
            for name in ("motion.json", "motion.glb"):
                if (stage / name).exists():
                    (stage / name).rename(stage / ("rejected-" + name))
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
        stage.rename(root / take)
        published = True
        progress(.99, "physical motion ready for review" if accepted else "physical refinement rejected; diagnostics retained")
        base = f"/v1/heads/{hid}/body/motions/{take}/"
        return {**manifest, "motion_url": base + "motion.json" if accepted else None,
                "glb_url": base + "motion.glb" if accepted else None, "evaluation_url": base + "evaluation.json"}
    except Exception as exc:
        # Retain failed fitting evidence, without publishing partial tracks.
        if not cancel.is_set() and not published:
            (stage / "evaluation.json").write_text(json.dumps({"completed": False, "error": str(exc)}))
            manifest["error"] = str(exc)
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
            for name in ("motion.json", "motion.glb"):
                if (stage / name).exists():
                    (stage / name).rename(stage / ("rejected-" + name))
            stage.rename(root / take)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    import argparse
    import threading
    from .musclemimic_worker import configure_process
    from .musclemimic_video import sample_video, check_camera, person_boxes
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare-video",))
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    configure_process(args.work, "cpu")
    req = json.loads(args.request.read_text())
    frames, times = sample_video(Path(req["source"]), args.work, threading.Event())
    boxes = person_boxes(frames)
    camera = check_camera(frames, boxes)
    (args.work / "prepared.json").write_text(json.dumps({"frames": list(map(str, frames)), "boxes": boxes,
                                                        "times": times.tolist(), "camera": camera}))


if __name__ == "__main__":
    main()
