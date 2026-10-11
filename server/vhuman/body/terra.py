"""Terra job orchestration. Numerical dependencies stay in the isolated worker."""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import uuid

from .. import gpu
from . import job, motion
from .musclemimic_assets import sha256
from .terra_assets import read_installation, reference
from .terra_terrain import validate_spec


def add_arguments(parser):
    parser.add_argument("--terra-python", help="isolated Python interpreter for the pinned Terra/MHR stack")
    parser.add_argument("--terra-assets", help="registered Terra installation directory")
    parser.add_argument("--terra-device", choices=("cpu", "cuda"), default="cpu", help="policy inference device; physics uses CPU MuJoCo")


def catalog(assets, python=None):
    if assets is None:
        return {"available": False, "references": [], "backend": "terra"}
    try:
        data = read_installation(assets)
        return {"available": bool(python and Path(python).is_file()), "research_only": True,
                "backend": "terra", "references": sorted(data["references"])}
    except (OSError, ValueError, KeyError):
        return {"available": False, "references": [], "backend": "terra", "reason": "invalid Terra registration"}


def validate_request(request):
    if bool(request.get("upload_id")) == bool(request.get("reference_motion")):
        raise ValueError("select exactly one uploaded video or MHR reference")
    if request.get("upload_id") and request.get("fixed_camera") is not True:
        raise ValueError("confirm fixed_camera=true: one fully visible upright person and a fixed camera")
    terrain = request.get("terrain", "auto")
    if isinstance(terrain, str):
        if terrain not in ("auto", "flat"):
            raise ValueError("terrain must be auto, flat or a TerrainSpec object; server paths are not accepted")
    else:
        validate_spec(terrain)
    seed = request.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Terra seed must be an unsigned 32-bit integer")
    if request.get("policy_actions", "sampled") not in ("sampled", "mean"):
        raise ValueError("policy_actions must be sampled or mean")


def fit(service, request, progress, cancel, *, model_dir=None, python=None, assets=None, device="cpu"):
    validate_request(request)
    installation = read_installation(assets)
    if not python or not Path(python).is_file():
        raise ValueError("configure --terra-python with the isolated interpreter")
    if device not in ("cpu", "cuda") or device == "cuda" and gpu.backend() != "cuda":
        raise ValueError("CUDA Terra policy requires the pipeline's CUDA backend")
    if request.get("reference_motion"):
        reference(installation, request["reference_motion"])
    hid = request.get("head_id")
    avatar, definition = service.body_file(hid, "avatar.glb"), service.body_file(hid, "avatar.json")
    body_meta = service.body_file(hid, "body_mhr.glb.json")
    model_dir = Path(model_dir or job.MODEL_DIR)
    from .mhr import ASSET_FILES
    if any(not (model_dir / "safetensors" / n).is_file() for n in ASSET_FILES):
        raise ValueError("native MHR assets are missing")
    take = uuid.uuid4().hex[:12]
    root = avatar.parent / "motions"; stage = root / ("." + take); stage.mkdir(parents=True)
    manifest = {"id": take, "head_id": hid, "kind": "terra", "backend": "terra", "status": "rejected",
                "accepted": False, "research_only": True, "frames": 0, "duration": 0., "fps": 30,
                "avatar_sha256": sha256(definition), "avatar_glb_sha256": sha256(avatar), "body_meta_sha256": sha256(body_meta)}
    published = False
    try:
        inputs = stage / "avatar-input"; inputs.mkdir()
        for path in (avatar, definition, body_meta):
            shutil.copyfile(path, inputs / path.name)
        worker = {"avatar": str((inputs / avatar.name).resolve()), "avatar_json": str((inputs / definition.name).resolve()),
                  "body_meta": str((inputs / body_meta.name).resolve()), "mhr_assets": str((model_dir / "safetensors").resolve()),
                  "terrain": request.get("terrain", "auto"), "seed": request.get("seed", 0),
                  "policy_actions": request.get("policy_actions", "sampled")}
        if request.get("reference_motion"):
            worker["reference_motion"] = request["reference_motion"]; manifest["reference_motion"] = request["reference_motion"]
        else:
            source, upload = motion.source_file(service, request["upload_id"])
            if upload["kind"] != "video":
                raise ValueError("Terra requires motion video, not a still image")
            prep = stage / "prepare-request.json"; prep.write_text(json.dumps({"source": str(source.resolve())}))
            progress(.03, "checking fixed-camera video and preserving timestamps")
            job._run([str(python), "-m", "server.vhuman.body.musclemimic", "prepare-video", "--request", str(prep.resolve()),
                      "--work", str(stage.resolve())], cancel, timeout=300)
            prepared = json.loads((stage / "prepared.json").read_text())
            sidecars = []
            for i, (frame, box) in enumerate(zip(prepared["frames"], prepared["boxes"])):
                if cancel.is_set():
                    raise gpu.Cancelled("cancelled")
                progress(.08 + .37*i/len(prepared["frames"]), f"dense MHR pose {i+1}/{len(prepared['frames'])}")
                sidecars.append(str(motion._sam_frame(Path(frame), stage / f"pose_{i:05d}.glb", tuple(box), model_dir, cancel, False).resolve()))
            worker.update(sidecars=sidecars, times=prepared["times"], camera=prepared["camera"])
            manifest.update(source_upload_id=request["upload_id"], source_sha256=upload["sha256"], root_motion="estimated")
        request_path = stage / "request.json"; request_path.write_text(json.dumps(worker, allow_nan=False))
        progress(.48, "reconstructing terrain, retargeting and simulating motion")
        session = gpu.device_session(2048, cancel) if device == "cuda" else nullcontext()
        try:
            with session:
                job._run([str(python), "-m", "server.vhuman.body.terra_worker", "--request", str(request_path.resolve()),
                          "--assets", str(Path(assets).resolve()), "--work", str(stage.resolve()), "--policy-device", device,
                          "--policy-device-index", str(gpu.device_index())], cancel, timeout=7200)
        except (RuntimeError, ValueError):
            if cancel.is_set():
                raise gpu.Cancelled("cancelled")
            if not (stage / "simulation.json").exists():
                raise
        if cancel.is_set():
            raise gpu.Cancelled("cancelled")
        if (sha256(definition) != manifest["avatar_sha256"] or sha256(avatar) != manifest["avatar_glb_sha256"]
                or sha256(body_meta) != manifest["body_meta_sha256"]):
            for name in ("baseline.json", "baseline.glb"):
                if (stage / name).exists():
                    (stage / name).rename(stage / ("stale-" + name))
            raise ValueError("avatar changed during Terra fitting; retry with the rebuilt avatar")
        report = json.loads((stage / "simulation.json").read_text())
        accepted = bool(report.get("completed") and report.get("avatar", {}).get("accepted"))
        manifest.update(accepted=accepted, status="accepted" if accepted else "rejected", error=report.get("error"),
                        provenance=report.get("provenance", {}), terrain_url="terrain.json" if (stage / "terrain.json").exists() else None)
        track = stage / ("motion.json" if accepted else "baseline.json")
        if track.exists():
            data = json.loads(track.read_text()); manifest.update(frames=len(data["frames"]), duration=data["duration"])
        (stage / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        if not accepted:
            for name in ("motion.json", "motion.glb"):
                if (stage / name).exists():
                    (stage / name).rename(stage / ("rejected-" + name))
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
        stage.rename(root / take); published = True
        base = f"/v1/heads/{hid}/body/motions/{take}/"
        progress(.99, "Terra motion ready for review" if accepted else "Terra motion rejected; diagnostics retained")
        return {**manifest, "motion_url": base + "motion.json" if accepted else None,
                "glb_url": base + "motion.glb" if accepted else None,
                "terrain_url": base + "terrain.json" if manifest["terrain_url"] else None, "evaluation_url": base + "evaluation.json"}
    except Exception as exc:
        if not cancel.is_set() and not published:
            (stage / "evaluation.json").write_text(json.dumps({"completed": False, "error": str(exc)}))
            manifest["error"] = str(exc); (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
            for name in ("motion.json", "motion.glb"):
                if (stage / name).exists():
                    (stage / name).rename(stage / ("rejected-" + name))
            stage.rename(root / take)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
