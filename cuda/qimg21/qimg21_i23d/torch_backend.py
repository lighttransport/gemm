"""TorchBackend: the PyTorch QwenImage21Pipeline, loaded once in this process.

Used for what the native runner cannot do yet -- more than one reference image
(the pipeline takes up to 10) -- and as the reference implementation of the
same semantics:

- SDEdit: the init image is VAE-encoded at the output size (S), the seed's
  noise drawn as for text-to-image (E), and the pipeline runs the tail of the
  schedule from x = (1 - sigma_K) * S + sigma_K * E. The truncated `sigmas`
  list yields exactly the tail of the full schedule (the scheduler's terminal
  stretch depends only on the last sigma, which truncation keeps).
- Masks: after every step the unmasked latent tokens are reset to
  (1 - sigma) * S + sigma * E, as the fast runner's mask_blend kernel does;
  the mask can also be shown to the model as an extra reference image.

Weights are placed with reference.py's resident plan (BlockRing: as many
transformer blocks on the GPU as fit, the rest streamed exactly), so the numbers
are the one-shot reference's.
"""
from __future__ import annotations

import importlib.util
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import imageops
from .backends import BackendError, GenRequest, GenResult

HERE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/mnt/nvme01/models/qimg-21")


def _reference_module():
    spec = importlib.util.spec_from_file_location("qimg21_reference", HERE / "reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule(scheduler, steps: int, tokens: int):
    """(unshifted sigma inputs, shifted sigmas incl. the final 0, mu) exactly
    as QwenImage21Pipeline builds them for `tokens` target tokens."""
    config = scheduler.config
    m = (config.max_shift - config.base_shift) / (config.max_image_seq_len - config.base_image_seq_len)
    mu = tokens * m + (config.base_shift - m * config.base_image_seq_len)
    inputs = np.linspace(1.0, 1 / steps, steps)
    scheduler.set_timesteps(sigmas=inputs, mu=mu)
    return inputs, [float(s) for s in scheduler.sigmas], mu


class TorchBackend:
    name = "torch"
    max_references = 10
    OPTIONS = ("model", "device", "offload", "reserve_mib", "sdpa_backend", "condition_resolution",
               "park_between_runs")

    def __init__(self, model=DEFAULT_MODEL, device: str = "cuda", offload: str = "resident",
                 reserve_mib: int = 2560, sdpa_backend: str = "efficient", condition_resolution: int = 1024,
                 park_between_runs: bool = False):
        if offload not in ("resident", "group", "sequential"):
            raise BackendError(f"offload must be resident, group or sequential, got {offload!r}")
        self.model = Path(model).resolve()
        self.device_name, self.offload, self.reserve_mib = device, offload, reserve_mib
        self.sdpa_backend = sdpa_backend
        self.condition_resolution = condition_resolution
        self.park_between_runs = park_between_runs
        self.pipe = self.ring = self.torch = self.device = None
        self.load_seconds = None

    # ---- loading ----------------------------------------------------------

    def _load(self):
        if self.pipe is not None:
            return
        started = time.perf_counter()
        import os
        # Requests of different sequence lengths share one process; expandable
        # segments keep the freed activations of one from fragmenting the next.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from diffusers import QwenImage21Pipeline
        reference = _reference_module()
        self.torch = torch
        self.device = reference.resolve_device(torch, self.device_name)
        self.pipe = QwenImage21Pipeline.from_pretrained(str(self.model), dtype=torch.bfloat16, local_files_only=True)
        if self.device.type == "cuda":
            if self.offload == "sequential":
                self.pipe.enable_sequential_cpu_offload(device=self.device)
            else:
                self.ring = reference.place_on_device(torch, self.pipe, self.device, self.offload, self.reserve_mib)
        else:
            self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.load_seconds = time.perf_counter() - started

    def reserve_for(self, request: GenRequest) -> int:
        """Device memory (MiB) a request needs next to the resident blocks.

        The prefix K/V cache is 32 blocks x 2 x 3072 BF16 values per token of
        text and condition image (about 0.4 MB), CFG doubles it, and the
        activations grow with the whole sequence. Condition images are resized
        to condition_resolution^2 pixels, 16 x 16 per token."""
        target = (request.width // 16) * (request.height // 16)
        condition = (self.condition_resolution // 16) ** 2
        references = len(request.references) + (1 if request.mask is not None and request.mask_as_reference else 0)
        prefix = 300 + references * condition
        branches = 2 if request.negative_prompt else 1
        return max(self.reserve_mib, int(2048 + 0.5 * prefix * branches + 0.4 * (prefix + target) * branches))

    def _unpark(self, request: GenRequest):
        if self.ring is None:
            return
        needed = self.reserve_for(request) * 2**20
        if self.ring.slots is None and self.park_between_runs or needed != self.ring.reserve:
            # Re-plan the resident blocks for this request's sequence length;
            # views of one size share a plan, so this runs once per shape.
            self.ring.park()
            self.ring.reserve = needed
            self.ring.unpark()
            self.pipe.vae.to(self.device)

    def _park(self):
        if self.ring is not None and self.park_between_runs:
            self.ring.park()
            self.pipe.vae.to("cpu")
        if self.device is not None and self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    # ---- generation -------------------------------------------------------

    def _images(self, request: GenRequest):
        from PIL import Image
        images = [Image.open(r).convert("RGBA") for r in request.references]
        if request.mask is not None and request.mask_as_reference:
            # The model card's convention: a mask is just another reference
            # image; white marks the region the prompt refers to.
            images.append(Image.open(request.mask).convert("L").convert("RGBA"))
        return images or None

    def _encode(self, rgba: np.ndarray, request: GenRequest, generator):
        torch, pipe = self.torch, self.pipe
        from PIL import Image
        image = Image.fromarray(rgba, "RGBA")
        tensor = pipe.image_processor.preprocess(image, width=request.width, height=request.height)
        tensor = tensor.unsqueeze(2).to(device=self.device, dtype=pipe.vae.dtype)
        latents = pipe._encode_vae_image(tensor, generator)
        h, w = request.height // 16, request.width // 16
        return pipe._pack_latents(latents, 1, latents.shape[1], h, w).float()

    def _call(self, request: GenRequest, **extra):
        torch = self.torch
        from contextlib import nullcontext
        context = nullcontext()
        if self.sdpa_backend == "efficient" and self.device.type == "cuda":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            context = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
        with context, torch.inference_mode():
            return self.pipe(negative_prompt=request.negative_prompt,
                             true_cfg_scale=request.true_cfg_scale if request.negative_prompt else 1.0,
                             height=request.height, width=request.width, output_resolution=self.condition_resolution,
                             **extra).images

    def generate(self, request: GenRequest) -> GenResult:
        request.validate(self.max_references)
        self._load()
        started = time.perf_counter()
        torch, pipe = self.torch, self.pipe
        self._unpark(request)
        try:
            generator = torch.Generator(device=self.device).manual_seed(request.seed)
            images = self._images(request)
            details = {"references": len(request.references),
                       "mask_shown_as_reference": bool(request.mask is not None and request.mask_as_reference)}
            if request.init_image is None or (request.strength >= 1.0 and request.mask is None):
                out = self._call(request, prompt=request.prompt, image=images, num_inference_steps=request.steps,
                                 generator=generator)[0]
            else:
                tokens = (request.height // 16) * (request.width // 16)
                inputs, sigmas, _ = schedule(pipe.scheduler, request.steps, tokens)
                start = request.restart_step()
                noise, _ = pipe.prepare_latents(None, 1, pipe.transformer.config.in_channels, request.height,
                                                request.width, torch.bfloat16, self.device, generator, None)
                noise = noise.float()
                init = imageops.load_rgba(request.init_image)
                if init.shape[:2] != (request.height, request.width):
                    init = imageops.resize_rgba(init, request.width, request.height)
                source = self._encode(init, request, generator)
                sigma0 = sigmas[start]
                latents = ((1.0 - sigma0) * source + sigma0 * noise).to(torch.bfloat16)
                callback = None
                if request.mask is not None:
                    mask = imageops.make_mask((request.width, request.height), image=request.mask, resize=True)
                    weights = torch.from_numpy(imageops.latent_mask(mask, request.height // 16, request.width // 16))
                    weights = weights.to(self.device).view(1, -1, 1)

                    def callback(_pipe, step, _timestep, kwargs):
                        sigma = float(_pipe.scheduler.sigmas[step + 1])
                        keep = (1.0 - sigma) * source + sigma * noise
                        kwargs["latents"] = (weights * kwargs["latents"].float()
                                             + (1.0 - weights) * keep).to(kwargs["latents"].dtype)
                        return kwargs
                out = self._call(request, prompt=request.prompt, image=images, sigmas=list(inputs[start:]),
                                 num_inference_steps=request.steps - start, latents=latents,
                                 callback_on_step_end=callback,
                                 callback_on_step_end_tensor_inputs=["latents"] if callback else None)[0]
                details.update(start_step=start, sigma_start=sigma0)
            Path(request.out).parent.mkdir(parents=True, exist_ok=True)
            out.convert("RGBA").save(request.out)
        finally:
            self._park()
        return GenResult(Path(request.out), time.perf_counter() - started, self.name, details)

    def generate_batch(self, requests: list[GenRequest]) -> list[GenResult]:
        """Several plain requests (same references, size, steps and negative
        prompt; no init image or mask) in one pipeline call, one generator per
        request so each keeps its own noise. Anything else runs one by one."""
        first = requests[0]
        shared = all(r.references == first.references and (r.width, r.height, r.steps) ==
                     (first.width, first.height, first.steps) and r.negative_prompt == first.negative_prompt
                     and r.init_image is None and r.mask is None for r in requests)
        if not shared or len(requests) == 1:
            return [self.generate(r) for r in requests]
        for request in requests:
            request.validate(self.max_references)
        self._load()
        started = time.perf_counter()
        torch = self.torch
        worst = max(requests, key=self.reserve_for)
        self._unpark(replace(worst, negative_prompt=worst.negative_prompt))
        try:
            generators = [torch.Generator(device=self.device).manual_seed(r.seed) for r in requests]
            images = self._call(first, prompt=[r.prompt for r in requests], image=self._images(first),
                                num_inference_steps=first.steps, generator=generators)
        finally:
            self._park()
        each = (time.perf_counter() - started) / len(requests)
        results = []
        for request, image in zip(requests, images):
            Path(request.out).parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGBA").save(request.out)
            results.append(GenResult(Path(request.out), each, self.name, {"batched": len(requests)}))
        return results

    def close(self) -> None:
        if self.ring is not None:
            self.ring.park()
        self.pipe = self.ring = None
        if self.torch is not None and self.device is not None and self.device.type == "cuda":
            self.torch.cuda.empty_cache()
