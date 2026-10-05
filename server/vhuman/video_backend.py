"""Explicit video model adapters; no implicit model or hardware fallback."""
import json
from pathlib import Path

BACKENDS = ("repo", "legacy", "wan", "h3", "h3-fl2va", "hv15-rocm")


class WanBackend:
    frames = tuple(range(5, 82, 4))
    presets = ("quality", "fast12", "fast5")
    hardware = "rocm"
    identity_conditioned = True
    manages_device_lock = True
    default_model = Path("/mnt/disk01/models/wan22")

    def __init__(self):
        from cuda.hunyuan_video15_native import generate
        self.tools = generate
        self.Cancelled = generate.Cancelled
        self.RUNNER = generate.ROOT / "rdna4/wan22/run.sh"

    def load_manifest(self, model, task, preset):
        if task != "i2v" or preset not in self.presets:
            raise ValueError("unsupported Wan task/preset")
        model = Path(model or self.default_model)
        receipt = json.loads((model / "download.json").read_text())
        if not (model / "gguf/Wan2.2-TI2V-5B-Q8_0.gguf").is_file():
            raise ValueError("missing Wan Q8_0 model")
        return receipt

    def generate(self, *, model=None, runner=None, image, prompt, out, frames=81,
                 preset="quality", seed=42, device=0, allow_experimental=False,
                 cancel=None, progress=None, width=480, height=832):
        if not allow_experimental or device != 0:
            raise ValueError("Wan adapter requires experimental opt-in and AMD device 0")
        if frames not in self.frames or preset not in self.presets:
            raise ValueError("unsupported Wan frame count/preset")
        model = Path(model or self.default_model)
        self.load_manifest(model, "i2v", preset)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        steps = {"quality": 50, "fast12": 12, "fast5": 5}[preset]
        command = ["sh", runner or self.RUNNER, "--model", model, "--image", image,
                   "--prompt", prompt, "--out", out, "--width", width, "--height", height,
                   "--frames", frames, "--steps", steps, "--seed", seed, "--backend", "hip",
                   "--hip-gemm", "blaslt"]
        # The child owns the shared AMD lock. Do not lock again in this adapter.
        self.tools.run_process(command, cancel=cancel, progress=progress)
        result = json.loads((out / "manifest.json").read_text())
        if result.get("backend") != "wan22_rocm_hip" or result.get("hip_projection_calls", 0) < 1:
            raise RuntimeError("Wan child did not execute the requested HIP backend")
        (out / "video.mp4").rename(out / "clip.mp4")
        self.tools.run_process(["ffmpeg", "-nostdin", "-v", "error", "-i", out / "clip.mp4",
                                "-frames:v", "1", out / "poster.png"], cancel=cancel)
        result.update(schema="vhuman.wan.video.v1", task="i2v", preset=preset,
                      image_sha256=self.tools.digest(image), identity_conditioned=True,
                      synthetic=True, crop={"mode": "center_crop", "size": [width, height]})
        self.tools.atomic_json(out / "metrics.json", {key: result[key] for key in
                              ("seconds", "denoise_seconds", "peak_allocated_mib", "hip_projection_calls")})
        self.tools.atomic_json(out / "manifest.json", result)
        return result


class Hv15RocmBackend:
    frames = (81,)
    presets = ("quality", "fast12")
    hardware = "rocm"
    identity_conditioned = True
    manages_device_lock = True
    default_model = Path("/mnt/disk01/models/hv15")

    def __init__(self):
        from cuda.hunyuan_video15_native import generate
        self.module = generate
        self.RUNNER = generate.ROCM_RUNNER
        self.Cancelled = generate.Cancelled

    def load_manifest(self, model, task, preset):
        return self.module.model_manifest(model or self.default_model, task, preset, verify=False)[0]

    def generate(self, *, frames=81, model=None, **kwargs):
        if frames not in self.frames:
            raise ValueError("native HV1.5 requires 81 frames")
        kwargs['runner'] = kwargs.get('runner') or self.RUNNER
        result = self.module.generate(model=model or self.default_model, backend="rocm",
                                      gemm="repo", gemm_fallback="error", **kwargs)
        metrics = result.get('metrics', {})
        if (metrics.get('hipblas_gemm_calls') != 0 or metrics.get('fallback_gemm_calls') != 0
                or type(metrics.get('repo_gemm_calls')) is not int or metrics['repo_gemm_calls'] < 1):
            raise RuntimeError('HV1.5 ROCm reported forbidden GEMM fallback')
        result.update(identity_conditioned=True, synthetic=True,
                      crop={"mode": "center_crop", "size": [480, 848]})
        return result


class H3Backend:
    frames = tuple(range(5, 125, 17))
    default_frames = 124
    presets = ("quality", "fast12", "fast5")
    hardware = "rocm"
    identity_conditioned = True
    variant = 'ref2va'
    manages_device_lock = True
    default_model = Path("/mnt/disk01/models/h3/weights")

    def __init__(self):
        from rdna4.minimax_h3 import generate
        self.module = generate
        self.RUNNER = generate.RUNNER
        self.Cancelled = generate.video.Cancelled

    def load_manifest(self, model, task, preset):
        if preset not in self.presets:
            raise ValueError("unsupported H3 preset")
        model = Path(model or self.default_model)
        for name in (f'diffusion_models/minimax_h3_{self.variant}_pruned_int8_convrot.safetensors',
                     *self.module.COMPONENTS[1:]):
            if not (model / name).is_file():
                raise ValueError(f"missing H3 component: {name}")
        return {"task": "r2v" if self.variant == 'ref2va' else "i2v", "identity_conditioned": True}

    def generate(self, *, image, preset="quality", model=None, frames=124, **kwargs):
        if frames not in self.frames or preset not in self.presets:
            raise ValueError("unsupported H3 frame count/preset")
        conditioning = {'reference_images': [image]} if self.variant == 'ref2va' else {'first_frame': image}
        if self.variant == 'ref2va':
            kwargs['prompt'] = 'The person in <Picture 1>. '+kwargs['prompt']
        result = self.module.generate(model=model or self.default_model, backend="rocm", frames=frames,
            steps={"quality": 40, "fast12": 13, "fast5": 6}[preset], width=480, height=832,
            variant=self.variant, **conditioning, **kwargs)
        result.update(task="r2v" if self.variant == 'ref2va' else "i2v", preset=preset,
                      identity_conditioned=True, synthetic=True,
                      source_portrait_sha256=self.module.video.digest(image),
                      source_portrait_used_for_conditioning=True)
        return result


class H3Fl2vaBackend(H3Backend):
    variant = 'fl2va'


class RepositoryBackend:
    frames = (81,)

    def __init__(self):
        from cuda.hunyuan_video15_native import generate
        self.module = generate
        self.RUNNER = generate.RUNNER
        self.Cancelled = generate.Cancelled

    def load_manifest(self, model, task, preset):
        # The generation path validates every component hash before inference.
        return self.module.model_manifest(model, task, preset, verify=False)[0]

    def generate(self, *, frames=81, **kwargs):
        if frames not in self.frames:
            raise ValueError('repository Hunyuan currently supports 81 frames; 121-frame generation requires the explicit legacy backend')
        result = self.module.generate(**kwargs, gemm='repo', gemm_fallback='error')
        metrics = result.get('metrics', {})
        if (metrics.get('cublas_gemm_calls') != 0 or metrics.get('fallback_gemm_calls') != 0 or
                type(metrics.get('repo_gemm_calls')) is not int or metrics['repo_gemm_calls'] < 1):
            raise RuntimeError('repository Hunyuan reported forbidden vendor GEMM fallback')
        fast = kwargs.get('preset', 'quality') == 'fast12'
        result.update(steps=12 if fast else 50, cfg=1 if fast else 6, flow_shift=7 if fast else 5)
        return result


def select(backend='repo'):
    if backend == 'repo':
        return RepositoryBackend()
    if backend == 'legacy':
        from cuda.hunyuan_video15 import native_generate
        return native_generate
    if backend == 'wan':
        return WanBackend()
    if backend == 'hv15-rocm':
        return Hv15RocmBackend()
    if backend == 'h3':
        return H3Backend()
    if backend == 'h3-fl2va':
        return H3Fl2vaBackend()
    raise ValueError(f'video backend must be one of {BACKENDS}')
