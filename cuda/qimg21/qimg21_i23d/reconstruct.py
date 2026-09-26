"""Pixal3D reconstruction: RGBA object or posed view set -> textured GLB.

Two runners, one interface:

- Pixal3DNative: the repository's native runner (cpu/pixal3d/pixal3d) on
  CUDA, ROCm (rdna4/pixal3d/libpixal3d_rocm.so) or CPU.
- Pixal3DReference: the pinned upstream PyTorch pipeline
  (ref/pixal3d/run_reference_sv.py / run_reference_mv.py), CUDA through
  ref/pixal3d/.venv-reference-cuda310, ROCm through ref/pixal3d/.venv-rocm.

Both take the same inputs the Pixal3D demo server gives them:
- single view: an RGBA object image and its camera (FOV, and for the native
  runner the distance 0.5 / (tan(FOV / 2) * mesh_scale) that
  ref/pixal3d/prepare_input.py derives);
- multiview: a directory with RGBA frames and a NeRF/Blender transforms.json
  (at most 16 frames; frame 0 defines the output orientation).

Generated views carry requested cameras, not calibrated ones (see
dataset.CAMERA_NOTE): Pixal3D is told they are posed, and how well the
reconstruction agrees with them is what the result has to show.
"""
from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from . import imageops

ROOT = Path(__file__).resolve().parents[3]
PIXAL3D = ROOT / "ref/pixal3d"
MODEL_DIR = Path("/mnt/disk2/models/Pixal3D")
DINOV3 = Path("/mnt/disk2/models/dinov3-vitl16/model.safetensors")
NAF = PIXAL3D / "weights/naf_release.safetensors"
MOGE = Path("/mnt/disk2/models/moge-2-vitl/model.pt")
MAX_FRAMES = 16
BACKENDS = ("cuda", "rocm", "cpu")


class ReconstructionError(RuntimeError):
    pass


@dataclass
class ReconSettings:
    seed: int = 42
    texture_size: int = 4096            # 1024, 2048 or 4096
    triangle_target: int = 1_000_000
    gpu_execution: str = "resident"     # native: legacy | resident
    gpu_kernels: str = "auto"           # native: auto | blas | mma
    flow_precision: str = "mixed"       # native: bf16 | fp32 | mixed
    vram_budget_mib: int | None = None  # native
    resolution: int = 1024              # reference
    low_vram: bool = True               # reference
    timeout: float = 10800.0

    def validate(self) -> "ReconSettings":
        if self.texture_size not in (1024, 2048, 4096):
            raise ValueError("texture_size must be 1024, 2048 or 4096")
        if not 10_000 <= self.triangle_target <= 5_000_000:
            raise ValueError("triangle_target must be in [10000, 5000000]")
        if self.gpu_execution not in ("legacy", "resident") or self.gpu_kernels not in ("auto", "blas", "mma") \
                or self.flow_precision not in ("bf16", "fp32", "mixed"):
            raise ValueError("invalid native GPU execution, kernel or flow-precision choice")
        if self.vram_budget_mib is not None and not 513 <= self.vram_budget_mib <= 14336:
            raise ValueError("vram_budget_mib must be in [513, 14336]")
        return self


def camera_distance(fov_rad: float, mesh_scale: float = 1.0) -> float:
    """The distance at which a unit object fills the frame (prepare_input.py)."""
    return 0.5 / (math.tan(fov_rad * 0.5) * mesh_scale)


def glb_summary(path: Path) -> dict:
    """Vertex/triangle counts and bounds of a GLB's first primitive."""
    raw = Path(path).read_bytes()
    magic, version, total = struct.unpack_from("<III", raw)
    if magic != 0x46546C67 or version != 2 or total != len(raw):
        raise ReconstructionError(f"{path} is not a valid GLB")
    size, kind = struct.unpack_from("<II", raw, 12)
    if kind != 0x4E4F534A:
        raise ReconstructionError(f"{path} has no GLB JSON chunk")
    scene = json.loads(raw[20:20 + size])
    primitive = scene["meshes"][0]["primitives"][0]
    position = scene["accessors"][primitive["attributes"]["POSITION"]]
    indices = scene["accessors"][primitive["indices"]]
    return {"bytes": len(raw), "vertices": position["count"], "triangles": indices["count"] // 3,
            "bounds": [position.get("min"), position.get("max")]}


def _run(cmd: list[str], log: Path, timeout: float) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        stream.write("+ " + " ".join(cmd) + "\n")
        stream.flush()
        try:
            code = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            raise ReconstructionError(f"{Path(cmd[0]).name} exceeded {timeout:g} s; see {log}") from None
    text = log.read_text(errors="replace")
    if code:
        raise ReconstructionError(f"{Path(cmd[0]).name} failed ({code}); see {log}:\n{text[-3000:]}")
    return text


def _launcher(backend: str) -> Path:
    """The Pixal3D Python environment: the pinned reference for CUDA, the
    project environment (ref/pixal3d/run.sh) otherwise."""
    return PIXAL3D / ("run_reference_cuda310.sh" if backend == "cuda" else "run.sh")


def estimate_camera(rgba_path: Path, work: Path, backend: str = "cuda", mesh_scale: float = 1.0,
                    moge: Path = MOGE) -> dict:
    """MoGe-2 FOV for an RGBA object, through ref/pixal3d/prepare_input.py
    (as the demo server's auto_camera does). Returns its metadata:
    fov (radians), distance, mesh_scale, camera_source."""
    if not Path(moge).exists():
        raise ReconstructionError(f"MoGe-2 checkpoint missing: {moge}; pass a FOV instead")
    env = "cpu" if backend == "cpu" else backend
    work = Path(work).resolve()
    prepared, metadata = work / "camera_prepared.png", work / "camera.json"
    _run([str(PIXAL3D / "run.sh"), env, str(PIXAL3D / "prepare_input.py"), "--input", str(Path(rgba_path).resolve()),
          "--output", str(prepared), "--metadata", str(metadata), "--moge-model", str(moge),
          "--mesh-scale", str(mesh_scale), "--device", "cpu" if backend == "cpu" else "cuda"],
         work / "camera.log", 1800)
    return json.loads(metadata.read_text())


def select_frames(transforms: dict, max_frames: int = MAX_FRAMES, elevations=(0.0,)) -> list[dict]:
    """Frames for Pixal3D: the reference (frame 0, which defines the output
    orientation) plus generated views, at most max_frames in all.

    Frames of this package's datasets carry their requested angles: they are
    kept when their elevation is in `elevations` (None: any), except a
    generated copy of the reference view (same azimuth and elevation).
    Frames without angles (another tool's transforms.json) are all kept. If
    too many remain, an evenly spaced subset in the given order is kept."""
    frames = transforms["frames"]
    if not frames:
        raise ReconstructionError("transforms.json has no frames")
    reference = next((f for f in frames if f.get("generated") is False), frames[0])

    def angles(frame):
        return frame.get("azimuth_deg", 0.0) % 360.0, frame.get("elevation_deg", 0.0)

    rest = []
    for frame in frames:
        if frame is reference:
            continue
        if "azimuth_deg" not in frame or "elevation_deg" not in frame:
            rest.append(frame)
            continue
        azimuth, elevation = angles(frame)
        if elevations is not None and not any(abs(elevation - e) < 1e-6 for e in elevations):
            continue
        if abs(azimuth - angles(reference)[0]) < 1e-6 and abs(elevation - angles(reference)[1]) < 1e-6:
            continue
        rest.append(frame)
    room = max_frames - 1
    if len(rest) > room:
        rest = [rest[i * len(rest) // room] for i in range(room)]
    return [reference] + rest


def stage_views(dataset: Path, work: Path, max_frames: int = MAX_FRAMES, elevations=(0.0,)) -> tuple[Path, list[str]]:
    """A flat Pixal3D views directory (viewNN.png + transforms.json) from a
    dataset directory; returns it and the source file of each frame."""
    transforms = json.loads((dataset / "transforms.json").read_text())
    frames = select_frames(transforms, max_frames, elevations)
    views = work / "views"
    if views.exists():
        shutil.rmtree(views)
    views.mkdir(parents=True)
    staged, sources = [], []
    for i, frame in enumerate(frames):
        source = dataset / frame["file_path"]
        rgba = imageops.load_rgba(source)
        if not (rgba[..., 3] < 255).any():
            raise ReconstructionError(f"{source} has no alpha; Pixal3D multiview needs RGBA frames")
        name = f"view{i:02d}.png"
        shutil.copyfile(source, views / name)
        entry = {"file_path": name, "transform_matrix": frame["transform_matrix"]}
        if "camera_angle_x" in frame:
            entry["camera_angle_x"] = frame["camera_angle_x"]
        staged.append(entry)
        sources.append(frame["file_path"])
    (views / "transforms.json").write_text(json.dumps(
        {"camera_angle_x": transforms["camera_angle_x"], "mesh_scale": transforms.get("mesh_scale", 1.0),
         "frames": staged, "generated_views": transforms.get("generated_views", True),
         "camera_parameters": transforms.get("camera_parameters")}, indent=1))
    return views, sources


class Pixal3DNative:
    """cpu/pixal3d/pixal3d on CUDA, ROCm or CPU."""
    name = "native"

    def __init__(self, backend: str = "cuda", binary=ROOT / "cpu/pixal3d/pixal3d", model_dir=MODEL_DIR,
                 dinov3=DINOV3, naf=NAF, settings: ReconSettings | None = None, device: int | None = None,
                 threads: int = 0):
        if backend not in BACKENDS:
            raise ValueError(f"Pixal3D backend must be one of {', '.join(BACKENDS)}, got {backend!r}")
        self.backend, self.binary = backend, Path(binary)
        self.model_dir, self.dinov3, self.naf = Path(model_dir), Path(dinov3), Path(naf)
        self.settings = (settings or ReconSettings()).validate()
        self.device, self.threads = device, threads

    def available(self) -> tuple[bool, list[str]]:
        missing = [str(p) for p in (self.binary, self.model_dir, self.dinov3, self.naf) if not p.exists()]
        if self.backend == "rocm" and not (ROOT / "rdna4/pixal3d/libpixal3d_rocm.so").is_file():
            missing.append("rdna4/pixal3d/libpixal3d_rocm.so")
        return not missing, missing

    def _common(self, out: Path, profile: Path) -> list[str]:
        s = self.settings
        execution = "legacy" if self.backend == "cpu" else s.gpu_execution
        cmd = ["--output", str(out), "--seed", str(s.seed), "--model-dir", str(self.model_dir),
               "--dinov3", str(self.dinov3), "--naf", str(self.naf), "--gpu-execution", execution,
               "--gpu-kernels", s.gpu_kernels, "--gpu-flow-precision", s.flow_precision,
               "--profile-json", str(profile), "--texture-size", str(s.texture_size),
               "--triangle-target", str(s.triangle_target)]
        if s.vram_budget_mib is not None:
            cmd += ["--vram-budget-mib", str(s.vram_budget_mib)]
        if self.device is not None:
            cmd += ["--device", str(self.device)]
        if self.threads:
            cmd += ["--threads", str(self.threads)]
        return cmd

    def single_command(self, rgba: Path, out: Path, fov_rad: float, distance: float, mesh_scale: float,
                       profile: Path) -> list[str]:
        return [str(self.binary), "--backend", self.backend, "--input", str(rgba), "--fov", repr(fov_rad),
                "--distance", repr(distance), "--mesh-scale", repr(mesh_scale)] + self._common(out, profile)

    def multiview_command(self, views: Path, out: Path, profile: Path) -> list[str]:
        return [str(self.binary), "--backend", self.backend, "--views-dir", str(views)] + self._common(out, profile)

    def _finish(self, cmd: list[str], out: Path, work: Path) -> dict:
        ok, missing = self.available()
        if not ok:
            raise ReconstructionError(f"native Pixal3D is not ready: missing {', '.join(missing)}")
        profile = work / "profile.json"
        started = time.perf_counter()
        text = _run(cmd, work / "pixal3d_native.log", self.settings.timeout)
        stats = {}
        for line in reversed(text.splitlines()):
            try:
                candidate = json.loads(line)
            except ValueError:
                continue
            if isinstance(candidate, dict):
                stats = candidate
                break
        return {"runner": "native", "backend": self.backend, "output": str(out),
                "seconds": round(time.perf_counter() - started, 3), "stats": stats,
                "profile": json.loads(profile.read_text()) if profile.is_file() else {},
                "mesh": glb_summary(out), "log": str(work / "pixal3d_native.log")}

    def single(self, rgba, out, work, *, fov_rad: float, mesh_scale: float = 1.0) -> dict:
        out, work = Path(out).resolve(), Path(work).resolve()
        work.mkdir(parents=True, exist_ok=True)
        distance = camera_distance(fov_rad, mesh_scale)
        result = self._finish(self.single_command(Path(rgba).resolve(), out, fov_rad, distance, mesh_scale,
                                                  work / "profile.json"), out, work)
        result.update(mode="single", fov_rad=fov_rad, distance=distance, mesh_scale=mesh_scale)
        return result

    def multiview(self, views_dir, out, work) -> dict:
        out, work = Path(out).resolve(), Path(work).resolve()
        work.mkdir(parents=True, exist_ok=True)
        result = self._finish(self.multiview_command(Path(views_dir).resolve(), out, work / "profile.json"), out, work)
        result.update(mode="multiview")
        return result


class Pixal3DReference:
    """The pinned upstream PyTorch pipeline, for comparison."""
    name = "reference"

    def __init__(self, backend: str = "cuda", model_dir=MODEL_DIR, settings: ReconSettings | None = None):
        if backend not in ("cuda", "rocm"):
            raise ValueError("the PyTorch reference runs on cuda or rocm")
        self.backend, self.model_dir = backend, Path(model_dir)
        self.settings = (settings or ReconSettings()).validate()

    def available(self) -> tuple[bool, list[str]]:
        env = PIXAL3D / (".venv-reference-cuda310" if self.backend == "cuda" else ".venv-rocm") / "bin/python"
        missing = [str(p) for p in (env, self.model_dir, PIXAL3D / "run_reference_sv.py",
                                    PIXAL3D / "run_reference_mv.py") if not p.exists()]
        return not missing, missing

    def _tail(self) -> list[str]:
        s = self.settings
        cmd = ["--seed", str(s.seed), "--model_path", str(self.model_dir), "--resolution", str(s.resolution)]
        return cmd + (["--low_vram"] if s.low_vram else [])

    def single_command(self, rgba: Path, out: Path, fov_rad: float) -> list[str]:
        return [str(_launcher(self.backend)), self.backend, str(PIXAL3D / "run_reference_sv.py"),
                "--image", str(rgba), "--output", str(out), "--fov", repr(fov_rad)] + self._tail()

    def multiview_command(self, views: Path, out: Path) -> list[str]:
        return [str(_launcher(self.backend)), self.backend, str(PIXAL3D / "run_reference_mv.py"),
                "--views_dir", str(views), "--output", str(out)] + self._tail()

    def _finish(self, cmd: list[str], out: Path, work: Path) -> dict:
        ok, missing = self.available()
        if not ok:
            raise ReconstructionError(f"PyTorch Pixal3D reference is not ready: missing {', '.join(missing)}")
        started = time.perf_counter()
        _run(cmd, work / "pixal3d_reference.log", self.settings.timeout)
        return {"runner": "reference", "backend": self.backend, "output": str(out),
                "seconds": round(time.perf_counter() - started, 3), "mesh": glb_summary(out),
                "log": str(work / "pixal3d_reference.log")}

    def single(self, rgba, out, work, *, fov_rad: float, mesh_scale: float = 1.0) -> dict:
        out, work = Path(out).resolve(), Path(work).resolve()
        work.mkdir(parents=True, exist_ok=True)
        if mesh_scale != 1.0:
            raise ReconstructionError("the PyTorch reference takes mesh_scale 1 in single-view mode")
        result = self._finish(self.single_command(Path(rgba).resolve(), out, fov_rad), out, work)
        result.update(mode="single", fov_rad=fov_rad)
        return result

    def multiview(self, views_dir, out, work) -> dict:
        out, work = Path(out).resolve(), Path(work).resolve()
        work.mkdir(parents=True, exist_ok=True)
        result = self._finish(self.multiview_command(Path(views_dir).resolve(), out), out, work)
        result.update(mode="multiview")
        return result


def compare_meshes(native_glb, reference_glb, work, samples: int = 50000) -> dict:
    """Symmetric Chamfer and normal agreement between two GLBs, through
    ref/pixal3d/compare_outputs.py (as the demo server compares its runs)."""
    text = _run([str(PIXAL3D / "run.sh"), "cpu", str(PIXAL3D / "compare_outputs.py"), str(Path(native_glb).resolve()),
                 str(Path(reference_glb).resolve()), "--samples", str(samples)], Path(work).resolve() / "compare.log",
                1800)
    # It prints one indented JSON document.
    decoder = json.JSONDecoder()
    at = text.find("\n{")
    while at >= 0:
        try:
            measured, _ = decoder.raw_decode(text, at + 1)
        except ValueError:
            measured = None
        if isinstance(measured, dict) and "geometry" in measured:
            return {"samples": measured.get("samples"), **measured["geometry"]}
        at = text.find("\n{", at + 1)
    raise ReconstructionError(f"compare_outputs.py printed no result; see {Path(work) / 'compare.log'}")


def make_reconstructors(which: str, backend: str = "cuda", settings: ReconSettings | None = None, **native_options):
    """[] for none, else the requested runners: native, reference or both."""
    if which == "none":
        return []
    if which not in ("native", "reference", "both"):
        raise ValueError("reconstruct must be none, native, reference or both")
    runners = []
    if which in ("native", "both"):
        runners.append(Pixal3DNative(backend, settings=settings, **native_options))
    if which in ("reference", "both"):
        runners.append(Pixal3DReference(backend, settings=settings))
    return runners


def settings_dict(settings: ReconSettings) -> dict:
    return asdict(settings)
