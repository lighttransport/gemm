#!/usr/bin/env python3
"""Run native Qwen-Image 2.1 denoising with an optional native VAE.

Text-only generation and image-editing conditioning use the native tokenizer,
native vision encoder, and native text encoder. This driver
turns the prompt embedding into an F32 fixture, invokes the native
runtime transformer for the complete FlowMatch schedule, and only loads the
Qwen-Image 2.1 VAE after the native subprocess exits.  That process boundary
lets the transformer release all of its allocations before the decoder claims
VRAM on a 12–16 GB card.
Single-image editing additionally runs native F32 VAE encoding before text
encoding; end-to-end editing parity is still experimental.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Keep this preflight check aligned with QIMG21_EDIT_FUSED_MIN_TOKENS in the HIP runner.
EDIT_FUSED_MIN_TOKENS = 1024

# test_cuda_qimg21_fast presets (defined in the runner, which prints their
# expansion) and the weight format each needs from pack_fast.py.
FAST_PRESET_WEIGHTS = {"low8": "int8", "low8-fp4": "nvfp4", "fast12": "int8", "accurate": None}
DEFAULT_PACKAGES = {
    "int8": "/mnt/nvme01/models/qimg-21-fast/int8-smooth-a0.6",
    "nvfp4": "/mnt/nvme01/models/qimg-21-fast/nvfp4-svd-a0.5-m",
}


def _run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(str(x) for x in command), file=sys.stderr)
    start = time.perf_counter()
    subprocess.run(command, cwd=cwd, check=True)
    print(f"  ({Path(command[0]).name if 'python' not in Path(command[0]).name else Path(command[1]).name}: "
          f"{time.perf_counter() - start:.1f} s)", file=sys.stderr)


def _device_used_mib(backend: str) -> int | None:
    """Device memory in use, from nvidia-smi, or None where it cannot be read."""
    if backend != "cuda":
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                              "--id=0"], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0]) if out.returncode == 0 else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def _wait_device_released(backend: str, baseline: int | None, limit: float = 2.0) -> None:
    """Wait until a finished child's device memory is back, rather than a flat
    two seconds. The next process plans its budget from the free memory it
    sees, so it must not start while the text encoder's context is retiring.
    Without a reading to compare, fall back to the old fixed wait."""
    start = time.perf_counter()
    if baseline is None:
        time.sleep(limit)
        return
    while time.perf_counter() - start < limit:
        used = _device_used_mib(backend)
        if used is None or used <= baseline + 128:
            break
        time.sleep(0.05)
    print(f"timing: device memory released {time.perf_counter() - start:.3f} s", file=sys.stderr)


def _prompt_cache_key(model: Path, text_bin: Path, backend: str, prompt: str) -> str:
    """The embedding depends on the prompt, the checkpoint and the encoder
    build; any of them changing is a different key."""
    import hashlib
    import json
    stat = text_bin.stat()
    blob = json.dumps({"prompt": prompt, "model": str(model), "backend": backend,
                       "encoder": [str(text_bin), stat.st_size, stat.st_mtime_ns]}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _largest_fitting_tile(root, command, limit):
    """Largest --tile-tokens whose plan still fits the budget.

    The runner refuses a plan it cannot afford and says so, so a --plan-only run
    is an exact memory query that loads no weights. The command is the real
    refine invocation, so the probe sees the same editing layout, K/V cache and
    preset the run will use. Memory is monotone in the tile size, so a binary
    search finds the largest fit; the maximum is tried first because on most
    budgets it fits outright and then there is nothing to search.
    """
    def fits(tokens):
        probe = list(command)
        probe[probe.index("--tile-tokens") + 1] = str(tokens)
        probe += ["--plan-only", "--quiet"]
        return subprocess.run(probe, cwd=root, capture_output=True).returncode == 0

    if fits(limit):
        return limit
    lo, hi, best = 1, limit - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(mid):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise SystemExit("no refine tile fits the VRAM budget; raise --vram-budget-mib or lower --preset")
    return best


def _decode_vae(model: Path, latent_path: Path, out_path: Path, height: int, width: int,
                dtype: str, backend: str, tile: bool = False) -> None:
    """Decode normalized [tokens, 64] latents using AutoencoderKLQwenImage21."""
    import torch
    from diffusers import AutoencoderKLQwenImage21
    from PIL import Image

    if not torch.cuda.is_available():
        raise SystemExit(f"{backend.upper()} is unavailable; the Qwen-Image VAE decode requires an accelerator")
    if height % 32 or width % 32:
        raise SystemExit("height and width must be divisible by 32 for Qwen-Image 2.1")

    np_latents = np.load(latent_path)
    if np_latents.ndim == 3 and np_latents.shape[0] == 1:
        np_latents = np_latents[0]
    if np_latents.ndim != 2 or np_latents.shape[1] != 64:
        raise SystemExit(f"expected native latents [tokens,64], got {np_latents.shape}")
    h_tokens, w_tokens = height // 16, width // 16
    if np_latents.shape[0] != h_tokens * w_tokens:
        raise SystemExit(
            f"latent token count {np_latents.shape[0]} does not match {height}x{width} ({h_tokens*w_tokens})"
        )

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    vae = AutoencoderKLQwenImage21.from_pretrained(
        str(model / "vae"), torch_dtype=torch_dtype, local_files_only=True
    ).to(device="cuda")
    vae.eval()
    if tile:
        # The reference decoder tiles spatially for the same reason the native
        # one does: an untiled 2048x2048 decode does not fit an 8 GB card.
        vae.enable_tiling()

    latents = torch.from_numpy(np_latents.astype(np.float32, copy=False)).to(
        device="cuda", dtype=torch_dtype
    )
    # This is the inverse of QwenImage21Pipeline._pack_latents/_unpack_latents.
    latents = latents.reshape(1, h_tokens * w_tokens, 64)
    latents = latents.transpose(1, 2).reshape(1, 64, 1, h_tokens, w_tokens).contiguous()
    mean = torch.tensor(vae.config.latents_mean, device="cuda", dtype=torch_dtype).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda", dtype=torch_dtype).view(1, 64, 1, 1, 1)
    with torch.inference_mode():
        image = vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
        image = (image * 0.5 + 0.5).clamp(0, 1)
        image_u8 = (image[0].float().permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "RGBA" if image_u8.shape[-1] == 4 else "RGB"
    Image.fromarray(image_u8, mode=mode).save(out_path)
    print(f"saved {out_path} ({width}x{height})")
    del vae
    torch.cuda.empty_cache()


def _torch_vae(model: Path, dtype: str, tile: bool):
    import torch
    from diffusers import AutoencoderKLQwenImage21

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    vae = AutoencoderKLQwenImage21.from_pretrained(
        str(model / "vae"), torch_dtype=torch_dtype, local_files_only=True
    ).to(device="cuda")
    vae.eval()
    if tile:
        vae.enable_tiling()
    mean = torch.tensor(vae.config.latents_mean, device="cuda", dtype=torch_dtype).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda", dtype=torch_dtype).view(1, 64, 1, 1, 1)
    return vae, mean, std, torch_dtype


def _decode_rgba_torch(model: Path, latents: np.ndarray, h_tokens: int, w_tokens: int,
                       dtype: str, tile: bool) -> np.ndarray:
    """Normalized [tokens, 64] latents -> RGBA [4, H, W] in [-1, 1]."""
    import torch

    vae, mean, std, torch_dtype = _torch_vae(model, dtype, tile)
    x = torch.from_numpy(latents.astype(np.float32, copy=False)).to(device="cuda", dtype=torch_dtype)
    x = x.reshape(1, h_tokens * w_tokens, 64).transpose(1, 2).reshape(1, 64, 1, h_tokens, w_tokens)
    with torch.inference_mode():
        image = vae.decode(x * std + mean, return_dict=False)[0][:, :, 0]
    out = image[0].float().clamp(-1, 1).cpu().numpy()
    del vae
    torch.cuda.empty_cache()
    return out


def _encode_latents_torch(model: Path, rgba: np.ndarray, dtype: str, tile: bool) -> np.ndarray:
    """RGBA [4, H, W] in [-1, 1] -> normalized [tokens, 64] latents, encoded the
    way QwenImage21Pipeline._encode_vae_image does (argmax, then normalized)."""
    import torch

    vae, mean, std, torch_dtype = _torch_vae(model, dtype, tile)
    x = torch.from_numpy(rgba.astype(np.float32, copy=False)).to(device="cuda", dtype=torch_dtype)
    with torch.inference_mode():
        latents = vae.encode(x[None, :, None]).latent_dist.mode()
        latents = (latents - mean) / std
    out = latents[0, :, 0].float().reshape(64, -1).T.contiguous().cpu().numpy()
    del vae
    torch.cuda.empty_cache()
    return out


def _resize_rgba(rgba: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bicubic resample of an RGBA [4, h, w] image in [-1, 1] to [4, height, width]."""
    import torch

    x = torch.from_numpy(rgba.astype(np.float32, copy=False))[None]
    y = torch.nn.functional.interpolate(x, size=(height, width), mode="bicubic", align_corners=False)
    return y[0].clamp(-1, 1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("cuda", "rocm"), default="cuda")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--negative-prompt")
    ap.add_argument("--true-cfg-scale", type=float, default=1.0)
    ap.add_argument("--image", help="Experimental single-image editing with native F32 VAE encoding")
    ap.add_argument("--condition-resolution", type=int, default=1024,
                    help="Condition image target-area side length, matching the reference output_resolution")
    ap.add_argument("--height", type=int, default=256, help="output height in pixels")
    ap.add_argument("--width", type=int, default=256, help="output width in pixels")
    ap.add_argument("--upscale", type=float, default=1.0,
                    help="generate a base image this much smaller, then refine it up to "
                         "--height/--width (2.0 is the usual high-resolution pass)")
    ap.add_argument("--base-steps", type=int, default=None,
                    help="steps for the base pass (default: --steps)")
    ap.add_argument("--tile-tokens", type=int, default=None,
                    help="refine tile side in latent tokens. The default is the largest tile whose "
                         "plan still fits --vram-budget-mib, which is also the best quality, since "
                         "each tile is composed as its own canvas. Needs --upscale above 1.")
    ap.add_argument("--tile-overlap", type=int, default=8,
                    help="latent tokens neighbouring refine tiles share")
    ap.add_argument("--refine-strength", type=float, default=0.5,
                    help="fraction of the schedule the refine pass re-runs, in (0, 1]")
    ap.add_argument("--refine-seed", type=int, default=0, help="seed for the refine pass's per-tile noise")
    ap.add_argument("--vae-tile", type=int, default=None,
                    help="latent tokens per VAE decode tile (default 48 above 1024 px, else 0 = one pass)")
    ap.add_argument("--vae-tile-overlap", type=int, default=8)
    ap.add_argument("--vae-tile-bleed", type=int, default=2,
                    help="latent tokens discarded at each decode tile edge; must be <= half the overlap")
    ap.add_argument("--profile-steps", action="store_true",
                    help="have the fast denoiser report each step's device time in ms "
                         "(adds a per-step sync, about 1.4%% on low8)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--initial-latents", type=Path,
                    help="reuse the same finite F32 noise tensor across backends")
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--work-dir", default="tmp/qimg21-native-generate")
    ap.add_argument("--out", default="tmp/qimg21-native-generate.png")
    ap.add_argument("--native-bin", default=None)
    ap.add_argument("--native-text-bin", default=None)
    ap.add_argument("--native-vision-bin", default=None)
    ap.add_argument("--native-vae-bin", default=None)
    ap.add_argument("--native-vae-encode-bin", default=None)
    ap.add_argument("--native-attention", choices=(
        "math", "reverse64", "wmma", "wmma-fused", "edit-size-select",
        "mma64", "mma64-flash", "mma64-mixed", "mma64-forward-flash",
        "mma128-efficient", "cutlass-efficient", "flash"), default=None,
        help="oracle attention; with --runner fast it overrides the preset (cutlass-efficient or flash)")
    ap.add_argument("--native-normalization", choices=("default", "vector4"), default=None)
    ap.add_argument("--native-rope", choices=("default", "host-table", "host-table-vector4", "host-table-exact"), default=None)
    ap.add_argument("--quantized-transformer", type=Path,
                    help="Optional experimental row-INT8 transformer package")
    ap.add_argument("--quantize-on-load", choices=("int8-row",))
    ap.add_argument("--int8-tensor-core", action="store_true",
                    help="dynamic W8A8 custom tensor-core execution (requires package)")
    ap.add_argument("--int8-bf16-tail-blocks", type=int, default=0,
                    help="reconstruct this many final transformer blocks to BF16")
    ap.add_argument("--native-vae", action="store_true", help="Decode with the native F32 VAE (experimental)")
    ap.add_argument("--native-vae-conv", choices=("direct", "cudnn"), default=None,
                    help="native VAE convolutions: direct F32 kernel (parity) or cuDNN F32 "
                         "(default: cudnn with --runner fast, direct otherwise)")
    ap.add_argument("--runner", choices=("oracle", "fast"), default="oracle",
                    help="denoiser: the parity harness or test_cuda_qimg21_fast (CUDA only)")
    ap.add_argument("--preset", choices=tuple(FAST_PRESET_WEIGHTS), default="accurate",
                    help="fast-runner memory/precision preset (low8/low8-fp4: <8 GB, fast12: ~12 GB)")
    ap.add_argument("--restart-from", type=Path,
                    help="refine an earlier result: its final [tokens, 64] latents, renoised to "
                         "--restart-step of this run's schedule with the --initial-latents noise")
    ap.add_argument("--restart-step", type=int, default=0,
                    help="how many of --steps the restart skips; 0 is a fresh run")
    ap.add_argument("--prompt-cache", type=Path,
                    help="directory of cached text embeddings; a repeated prompt skips the text encoder")
    ap.add_argument("--quant-package", type=Path,
                    help="pack_fast.py package for int8/nvfp4 presets (default: the preset's package)")
    args = ap.parse_args()
    fast_attention = args.native_attention
    if args.runner == "fast":
        if args.backend != "cuda":
            ap.error("--runner fast is CUDA only")
        if args.quantized_transformer or args.quantize_on_load or args.int8_tensor_core:
            ap.error("--runner fast takes --preset/--quant-package instead of harness quantization flags")
        if fast_attention not in (None, "cutlass-efficient", "flash"):
            ap.error("--runner fast supports cutlass-efficient or flash attention")
        if args.native_normalization not in (None, "vector4") or args.native_rope not in (None, "host-table-exact"):
            ap.error("--runner fast implements vector4 normalization and host-table-exact RoPE only")
        args.native_attention, args.native_normalization, args.native_rope = (
            "cutlass-efficient", "vector4", "host-table-exact")
    elif args.native_attention == "flash":
        ap.error("flash attention is a --runner fast option")
    if args.native_vae_conv == "cudnn" and args.backend != "cuda":
        ap.error("--native-vae-conv cudnn is CUDA only")
    # The ROCm path has a native decoder and does not require a PyTorch
    # installation. CUDA keeps its existing reference VAE default.
    if args.backend == "rocm":
        args.native_vae = True
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        ap.error("height and width must be positive multiples of 32")
    h_tokens, w_tokens = args.height // 16, args.width // 16
    target_tokens = h_tokens * w_tokens
    # Coarse-to-fine: a base pass at 1/upscale the output size, then a tiled
    # refine pass that starts from the base picture upscaled in pixel space and
    # re-encoded, and denoises one tile at a time, so device memory and
    # attention cost follow the tile.
    tiled = args.upscale > 1.0
    if tiled and args.runner != "fast":
        ap.error("tiled generation needs --runner fast; the parity harness has no tile path")
    if args.upscale <= 0.0:
        ap.error("--upscale must be positive")
    if args.restart_from is not None:
        if args.runner != "fast" or args.upscale != 1.0 or args.image:
            ap.error("--restart-from needs --runner fast, no --upscale and no --image")
        if args.initial_latents is None:
            ap.error("--restart-from needs --initial-latents: the noise the earlier run started from")
        if not 0 <= args.restart_step < args.steps:
            ap.error("--restart-step must be in [0, --steps)")
    if args.profile_steps and args.runner != "fast":
        ap.error("--profile-steps needs --runner fast; the parity harness has no --profile")
    if args.tile_tokens is not None and not tiled:
        ap.error("--tile-tokens needs --upscale above 1: tiling refines a base grid, "
                 "and without one there is nothing to refine")
    base_h_tokens = base_w_tokens = None
    tile_tokens = args.tile_tokens
    if args.upscale != 1.0:
        base_h = max(32, int(round(args.height / args.upscale)))
        base_w = max(32, int(round(args.width / args.upscale)))
        if base_h % 32 or base_w % 32:
            ap.error(f"--upscale {args.upscale} leaves a {base_h}x{base_w} base size, not a multiple of 32")
        base_h_tokens, base_w_tokens = base_h // 16, base_w // 16
        if base_h_tokens > h_tokens or base_w_tokens > w_tokens:
            ap.error("--upscale must not enlarge the base grid past the output grid")
    if args.tile_tokens is not None and not 1 <= args.tile_tokens <= min(h_tokens, w_tokens):
        ap.error(f"--tile-tokens must be in [1, {min(h_tokens, w_tokens)}] for a "
                 f"{args.height}x{args.width} output")
    if tiled:
        if args.tile_overlap < 0:
            ap.error("--tile-overlap must be >= 0")
        if not 0.0 < args.refine_strength <= 1.0:
            ap.error("--refine-strength must be in (0, 1]")
        if args.base_steps is not None and args.base_steps < 1:
            ap.error("--base-steps must be at least 1")
    if args.vae_tile is None:
        args.vae_tile = 48 if max(h_tokens, w_tokens) > 64 else 0
    if args.vae_tile and not 1 <= args.vae_tile <= min(h_tokens, w_tokens):
        ap.error(f"--vae-tile must be in [1, {min(h_tokens, w_tokens)}] for this output size")
    if args.vae_tile and args.vae_tile_bleed > args.vae_tile_overlap // 2:
        ap.error("--vae-tile-bleed must be at most half of --vae-tile-overlap, or the tiles leave gaps")
    if args.vae_tile:
        print(f"decoding {args.height}x{args.width} in {args.vae_tile}-token VAE tiles with "
              f"{args.vae_tile_overlap} tokens of overlap")
    rocm_bf16_edit = (
        args.backend == "rocm"
        and args.dtype == "bf16"
        and args.image is not None
        and not (args.quantized_transformer or args.quantize_on_load)
    )
    if args.native_attention is None:
        if args.backend == "rocm":
            if rocm_bf16_edit:
                args.native_attention = "edit-size-select"
            else:
                args.native_attention = "math" if args.image else "wmma-fused"
        else:
            args.native_attention = "math"
    if args.native_normalization is None:
        args.native_normalization = "vector4" if rocm_bf16_edit else "default"
    if args.native_rope is None:
        args.native_rope = "host-table-exact" if rocm_bf16_edit else "default"
    if (args.native_attention == "edit-size-select" and
            (args.backend != "rocm" or not args.image)):
        ap.error("edit-size-select attention requires ROCm image editing")
    needs_fused_plugin = (
        args.native_attention == "wmma-fused"
        or (args.native_attention == "edit-size-select"
            and target_tokens >= EDIT_FUSED_MIN_TOKENS)
    )
    if needs_fused_plugin:
        if args.backend != "rocm":
            ap.error("wmma-fused attention supports ROCm only")
        if not (Path(__file__).resolve().parents[2] / "rdna4/qimg21/libq21_hip_attention.so").is_file():
            ap.error("wmma-fused attention plugin missing; run `make -C rdna4/qimg21 fused`")
    if args.quantized_transformer and args.quantize_on_load:
        ap.error("choose a quantized package or quantize-on-load, not both")
    if args.int8_tensor_core and not args.quantized_transformer:
        ap.error("--int8-tensor-core requires --quantized-transformer")
    if args.int8_bf16_tail_blocks < 0 or args.int8_bf16_tail_blocks > 32:
        ap.error("--int8-bf16-tail-blocks must be in [0, 32]")

    root = Path(__file__).resolve().parents[2]
    model = Path(args.model).resolve()
    work = Path(args.work_dir)
    if not work.is_absolute():
        work = root / work
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    defaults = {
        "cuda": {
            "native": "cuda/qimg21/test_cuda_qimg21_native",
            "text": "cuda/qimg21/test_cuda_qimg21_text",
            "vision": "cuda/qimg21/test_cuda_qimg21_vision",
            "vae": "cuda/qimg21/test_cuda_qimg21_vae",
            "vae_encode": "cuda/qimg21/test_cuda_qimg21_vae_encode",
        },
        "rocm": {
            "native": "rdna4/qimg21/test_hip_qimg21_native",
            "text": "rdna4/qimg21/test_hip_qimg21_text",
            "vision": "rdna4/qimg21/test_hip_qimg21_vision",
            "vae": "rdna4/qimg21/test_hip_qimg21_vae",
            "vae_encode": "rdna4/qimg21/test_hip_qimg21_vae_encode",
        },
    }[args.backend]
    def resolve_binary(value, key):
        path = Path(value or defaults[key])
        return path if path.is_absolute() else root / path
    native_bin = resolve_binary(args.native_bin, "native")
    if args.runner == "fast" and not args.native_bin:
        native_bin = root / "cuda/qimg21/test_cuda_qimg21_fast"
    text_bin = resolve_binary(args.native_text_bin, "text")
    vision_bin = resolve_binary(args.native_vision_bin, "vision")
    vae_bin = resolve_binary(args.native_vae_bin, "vae")
    vae_encode_bin = resolve_binary(args.native_vae_encode_bin, "vae_encode")
    if not model.is_dir():
        raise SystemExit(f"model directory does not exist: {model}")
    if not native_bin.exists():
        raise SystemExit(f"native {args.backend} executable not found: {native_bin}")
    if args.native_vae and not vae_bin.exists():
        raise SystemExit(f"native {args.backend} VAE executable missing: {vae_bin}")
    if args.steps < 1 or args.steps > 100:
        raise SystemExit("steps must be between 1 and 100")
    if args.negative_prompt is not None and args.true_cfg_scale <= 1.0:
        raise SystemExit("--true-cfg-scale must be > 1 when --negative-prompt is used")
    condition_hw = None
    condition_dir = work / "condition"
    if args.image:
        encoder = vae_encode_bin
        if not encoder.exists():
            raise SystemExit(f"native {args.backend} encoder missing: {encoder}")
        condition_dir.mkdir(parents=True, exist_ok=False)
        _run([str(encoder), "--model", str(model / "vae"),
              "--input-image", str(Path(args.image).resolve()),
              "--resolution", str(args.condition_resolution),
              "--pipeline-bf16",
              "--preprocessed-out", str(condition_dir / "image.npy"),
              "--resized-out", str(condition_dir / "resized.png"),
              "--out", str(condition_dir / "moments.npy"),
              "--normalized-latents", str(condition_dir / "latents.npy")], cwd=root)
        image_tensor = np.load(condition_dir / "image.npy", mmap_mode="r")
        if image_tensor.ndim != 3 or image_tensor.shape[0] != 4:
            raise SystemExit("native image processor returned an invalid RGBA tensor")
        condition_hw = image_tensor.shape[1] // 16, image_tensor.shape[2] // 16

    prompt_dir = work / "prompt"
    steps_dir = work / "steps"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = prompt_dir / "prompt_embeds.npy"
    if not args.image:
        text_encoder = text_bin
        if not text_encoder.exists():
            raise SystemExit(f"native {args.backend} text encoder missing: {text_encoder}")
        attention = "flash-exact" if args.backend == "cuda" else "custom"
        encoded = False
        baseline = None

        def encode(text: str, output: Path) -> None:
            nonlocal encoded, baseline
            cached = None
            if args.prompt_cache:
                cached = args.prompt_cache / (_prompt_cache_key(model, text_encoder, args.backend, text) + ".npy")
                if cached.is_file():
                    import shutil
                    shutil.copyfile(cached, output)
                    print(f"text: prompt embedding cache hit ({cached.name[:12]})", file=sys.stderr)
                    print("timing: prompt embedding cache hit 0.000 s", file=sys.stderr)
                    return
            if baseline is None:
                baseline = _device_used_mib(args.backend)
            _run([str(text_encoder), "--model", str(model), "--prompt", text,
                  "--attention", attention, "--out", str(output)], cwd=root)
            encoded = True
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                partial = cached.with_suffix(f".{os.getpid()}.tmp")
                import shutil
                shutil.copyfile(output, partial)
                os.replace(partial, cached)

        encode(args.prompt, prompt_path)
        if args.negative_prompt is not None:
            encode(args.negative_prompt, prompt_dir / "negative_prompt_embeds.npy")
    else:
        vision_encoder = vision_bin
        text_encoder = text_bin
        if not vision_encoder.exists() or not text_encoder.exists():
            raise SystemExit(f"native {args.backend} vision/text executables missing: {vision_encoder}, {text_encoder}")
        vision_dir = work / "vision"
        vision_dir.mkdir(parents=True, exist_ok=True)
        _run([
            str(vision_encoder), "--model", str(model), "--image", str(condition_dir / "resized.png"),
            "--max-blocks", "27", "--attention", "flash" if args.backend == "cuda" else "math",
            "--out", str(vision_dir / "blocks.npy"),
            "--merged-out", str(vision_dir / "merged.npy"),
            "--deepstack-dir", str(vision_dir),
        ], cwd=root)
        def encode_multimodal_prompt(text, output, prefix):
            tokens_path = prompt_dir / f"{prefix}tokens.txt"
            _run([
                str(text_encoder), "--model", str(model), "--prompt", text,
                "--vision-merged", str(vision_dir / "merged.npy"),
                "--vision-deepstack-dir", str(vision_dir),
                "--image-grid-height", str(condition_hw[0]),
                "--image-grid-width", str(condition_hw[1]),
                "--attention", "flash-exact" if args.backend == "cuda" else "custom",
                "--out", str(output),
                "--dump-tokens", str(tokens_path),
            ], cwd=root)
            token_ids = np.loadtxt(tokens_path, dtype=np.int64, ndmin=1)
            embeddings = np.load(output, mmap_mode="r", allow_pickle=False)
            token_count = embeddings.shape[-2]
            retained = token_ids[-token_count:]
            if retained.shape != (token_count,):
                raise ValueError("native token and embedding lengths disagree")
            np.save(prompt_dir / f"{prefix}image_pad_mask.npy",
                    (retained == 151655)[None, :].astype(np.int64))
            np.save(prompt_dir / f"{prefix}prompt_mask.npy",
                    np.ones((1, retained.size), dtype=np.int64))

        encode_multimodal_prompt(args.prompt, prompt_path, "")
        if args.negative_prompt is not None:
            encode_multimodal_prompt(args.negative_prompt,
                                     prompt_dir / "negative_prompt_embeds.npy", "negative_")
    if not prompt_path.exists():
        raise SystemExit(f"text runner did not produce {prompt_path}")

    # The text subprocess owns a large accelerator context. Give the driver a
    # moment to retire it before the native process opens cuBLAS/NVRTC;
    # otherwise some 2-step launches can observe stale device allocations even
    # though the child has exited. Nothing to wait for when the embeddings
    # came from the cache.
    if args.image or encoded:
        _wait_device_released(args.backend, None if args.image else baseline)

    latent_path = work / "latents.npy"
    native_latents = work / "native_latents.npy"
    base_latents = work / "base_latents.npy"
    base_grid = work / "base_latents_grid.npy"
    # The base pass denoises its own latent grid; the refine pass reuses the
    # prompt/condition prefix and only needs the resampled base.
    fixture_h, fixture_w = (base_h_tokens, base_w_tokens) if tiled else (h_tokens, w_tokens)
    fixture_command = [
            os.environ.get("QIMG21_PYTHON", sys.executable),
            str(root / "cuda/qimg21/make_native_fixture.py"),
            "--prompt-embeds",
            str(prompt_path),
            "--height-tokens",
            str(fixture_h),
            "--width-tokens",
            str(fixture_w),
            "--seed",
            str(args.seed),
            "--dtype",
            args.dtype,
            "--out-dir",
            str(work),
        ]
    noise_cache = None
    if args.initial_latents:
        fixture_command.extend(("--latents", str(args.initial_latents.resolve())))
    elif args.backend == "cuda":
        # Drawing torch's CUDA noise costs a torch import, about two seconds,
        # for a tensor that depends only on the seed and the grid. Cache it
        # next to the prompt embeddings, keyed by the interpreter that drew it.
        if args.prompt_cache:
            import hashlib
            import json
            python = os.environ.get("QIMG21_PYTHON", sys.executable)
            key = json.dumps({"seed": args.seed, "grid": [fixture_h, fixture_w], "dtype": args.dtype,
                              "rng": "torch-cuda", "python": python}, sort_keys=True)
            noise_cache = args.prompt_cache / ("noise-" + hashlib.sha256(key.encode()).hexdigest() + ".npy")
        if noise_cache is not None and noise_cache.is_file():
            fixture_command.extend(("--latents", str(noise_cache)))
            print(f"text: initial noise cache hit ({noise_cache.name[:18]})", file=sys.stderr)
            noise_cache = None
        else:
            fixture_command.append("--torch-rng")
    _run(fixture_command, cwd=root)
    if noise_cache is not None and latent_path.is_file():
        import shutil
        noise_cache.parent.mkdir(parents=True, exist_ok=True)
        partial = noise_cache.with_suffix(f".{os.getpid()}.tmp")
        shutil.copyfile(latent_path, partial)
        os.replace(partial, noise_cache)
    if args.runner == "fast":
        # The preset sets budget, weights and attention; explicit flags follow it.
        attention_args = ["--preset", args.preset]
        weights = FAST_PRESET_WEIGHTS[args.preset]
        if weights:
            package = args.quant_package or Path(DEFAULT_PACKAGES[weights])
            if not (package / "manifest.json").is_file():
                raise SystemExit(f"preset {args.preset} needs a pack_fast.py {weights} package: {package}")
            attention_args += ["--quant-package", str(package.resolve())]
        if fast_attention:
            attention_args += ["--attention", fast_attention]
    else:
        attention_args = ["--attention", args.native_attention]

    def denoise_command(gh, gw, steps, out, extra, target_hw=None):
        """One denoiser invocation. The editing layout's target block is the grid
        the model actually sees, which is the tile when refining."""
        command = [str(native_bin), *attention_args,
                   "--model", str(model),
                   "--prompt-embeds", str(prompt_path),
                   "--height-tokens", str(gh), "--width-tokens", str(gw),
                   "--steps", str(steps), "--out", str(out), *extra]
        if args.profile_steps:
            command.append("--profile")
        if args.image:
            from editing_inputs import write_layout
            suffix = "" if target_hw is None else f"_{target_hw[0]}x{target_hw[1]}"
            layout = condition_dir / f"positive_layout{suffix}.txt"
            write_layout(prompt_dir, layout, condition_hw, target_hw or (gh, gw))
            command.extend(["--condition-latents", str(condition_dir / "latents.npy"),
                            "--editing-layout", str(layout)])
            if args.negative_prompt is not None:
                negative_layout = condition_dir / f"negative_layout{suffix}.txt"
                write_layout(prompt_dir, negative_layout, condition_hw, target_hw or (gh, gw), negative=True)
                command.extend(["--negative-editing-layout", str(negative_layout)])
        if args.negative_prompt is not None:
            command.extend(["--negative-prompt-embeds",
                            str(prompt_dir / "negative_prompt_embeds.npy"),
                            "--guidance-scale", str(args.true_cfg_scale)])
        if args.quantized_transformer:
            command.extend(["--quantized-transformer", str(args.quantized_transformer.resolve())])
        if args.quantize_on_load:
            command.extend(["--quantize-on-load", args.quantize_on_load])
        if args.int8_tensor_core:
            command.extend(["--int8-tensor-core", "--int8-bf16-tail-blocks",
                            str(args.int8_bf16_tail_blocks)])
        return command

    if tiled:
        base_steps = args.base_steps or args.steps
        print(f"base pass: {base_h_tokens * 16}x{base_w_tokens * 16} px, {base_steps} steps")
        _run(denoise_command(base_h_tokens, base_w_tokens, base_steps, base_latents,
                             ["--latents", str(latent_path), "--normalization", args.native_normalization,
                              "--rope", args.native_rope, "--dump-dir", str(steps_dir)]), cwd=root)
        base = np.load(base_latents, allow_pickle=False)
        if base.shape != (base_h_tokens * base_w_tokens, 64) or not np.isfinite(base).all():
            raise SystemExit("base pass returned an invalid latent grid")
        # Upscale in pixel space, not latent space. A token of this VAE is a
        # 16x16 pixel patch, and interpolating between neighbouring tokens does
        # not interpolate the patches they encode: a bilinear latent upsample
        # decodes to ghosted, gridded copies of every edge, which the refine
        # then only partly repairs. Decode the base, resample the picture, and
        # encode it again, so the refine starts from the base image itself.
        rgba = None
        if args.native_vae:
            base_decoded = work / "base_decoded.npy"
            base_vae_tile = 48 if max(base_h_tokens, base_w_tokens) > 64 else 0
            _run([str(vae_bin), "--model", str(model / "vae"), "--latents", str(base_latents),
                  "--height-tokens", str(base_h_tokens), "--width-tokens", str(base_w_tokens),
                  "--out", str(base_decoded), "--conv", "cudnn",
                  *((["--tile", str(base_vae_tile), "--tile-overlap", "8", "--tile-bleed", "2"])
                    if base_vae_tile else [])], cwd=root)
            rgba = np.load(base_decoded)
        else:
            rgba = _decode_rgba_torch(model, base, base_h_tokens, base_w_tokens, args.dtype,
                                      max(base_h_tokens, base_w_tokens) > 64)
        if rgba.shape != (4, base_h_tokens * 16, base_w_tokens * 16) or not np.isfinite(rgba).all():
            raise SystemExit("base decode returned an invalid RGBA tensor")
        upscaled = _resize_rgba(rgba, args.height, args.width)
        # The native encoder takes up to 1024 px a side; larger refines encode
        # with the reference VAE, tiled.
        if args.native_vae and vae_encode_bin.exists() and max(args.height, args.width) <= 1024:
            upscaled_rgba = work / "upscaled_rgba.npy"
            upscaled_latents = work / "upscaled_latents.npy"
            np.save(upscaled_rgba, np.ascontiguousarray(upscaled, dtype=np.float32))
            _run([str(vae_encode_bin), "--model", str(model / "vae"), "--image", str(upscaled_rgba),
                  "--pipeline-bf16", "--out", str(work / "upscaled_moments.npy"),
                  "--normalized-latents", str(upscaled_latents)], cwd=root)
            fine = np.load(upscaled_latents)
        else:
            fine = _encode_latents_torch(model, upscaled, args.dtype, max(h_tokens, w_tokens) > 64)
        if fine.shape != (h_tokens * w_tokens, 64) or not np.isfinite(fine).all():
            raise SystemExit("upscaled base encode returned an invalid latent grid")
        # The refine pass reads the start as a 3-D grid, so the target is
        # unambiguous and no extra flag can disagree with the data. It is
        # already the output size, so the runner's own resample is the identity.
        np.save(base_grid, np.ascontiguousarray(fine.reshape(h_tokens, w_tokens, 64)))

        # Each tile is composed as its own canvas, so quality falls off with
        # smaller tiles: more tiles means more independently re-drawn detail.
        # The default is therefore the largest tile the budget allows, which on
        # most sizes is the whole grid and means no tiling at all.
        def refine_command(tokens):
            return denoise_command(h_tokens, w_tokens, args.steps, native_latents,
                                   ["--refine-from", str(base_grid), "--tile-tokens", str(tokens),
                                    "--tile-overlap", str(args.tile_overlap),
                                    "--refine-strength", str(args.refine_strength),
                                    "--refine-seed", str(args.refine_seed),
                                    "--normalization", args.native_normalization,
                                    "--rope", args.native_rope,
                                    "--dump-dir", str(steps_dir)],
                                   target_hw=(tokens, tokens))

        if tile_tokens is None:
            limit = min(h_tokens, w_tokens)
            tile_tokens = _largest_fitting_tile(root, refine_command(limit), limit)
            print(f"refine tile: {tile_tokens} of {limit} latent tokens is the largest that fits "
                  f"the {args.preset} budget")
        print(f"refine pass: {args.height}x{args.width} px in {tile_tokens}-token tiles, "
              f"strength {args.refine_strength}")
        _run(refine_command(tile_tokens), cwd=root)
    elif args.restart_from is not None and args.restart_step > 0:
        # Refine an earlier result: renoise it to sigma[K] of this schedule with
        # the noise that run started from, then run the remaining steps.
        restart = np.load(args.restart_from, allow_pickle=False)
        if restart.shape != (h_tokens * w_tokens, 64) or not np.isfinite(restart).all():
            raise SystemExit(f"--restart-from must be finite [{h_tokens * w_tokens}, 64] latents")
        print(f"restart: steps {args.restart_step + 1}-{args.steps} of {args.steps} from "
              f"{args.restart_from.name}", file=sys.stderr)
        _run(denoise_command(h_tokens, w_tokens, args.steps, native_latents,
                             ["--latents", str(args.restart_from.resolve()),
                              "--start-step", str(args.restart_step),
                              "--restart-noise", str(latent_path),
                              "--normalization", args.native_normalization,
                              "--rope", args.native_rope, "--dump-dir", str(steps_dir)]), cwd=root)
    else:
        _run(denoise_command(h_tokens, w_tokens, args.steps, native_latents,
                             ["--latents", str(latent_path), "--normalization", args.native_normalization,
                              "--rope", args.native_rope, "--dump-dir", str(steps_dir)]), cwd=root)
    if args.native_vae:
        from PIL import Image

        vae_conv = args.native_vae_conv or ("cudnn" if args.runner == "fast" else "direct")

        decoded_path = work / "native_decoded.npy"
        _run([
            str(vae_bin),
            "--model", str(model / "vae"), "--latents", str(native_latents),
            "--height-tokens", str(args.height // 16),
            "--width-tokens", str(args.width // 16), "--out", str(decoded_path),
            *(["--conv", vae_conv] if vae_conv != "direct" else []),
            *((["--tile", str(args.vae_tile), "--tile-overlap", str(args.vae_tile_overlap),
                "--tile-bleed", str(args.vae_tile_bleed)]) if args.vae_tile else []),
        ], cwd=root)
        decoded = np.load(decoded_path)
        if decoded.shape != (4, args.height, args.width) or not np.isfinite(decoded).all():
            raise SystemExit("native VAE returned an invalid RGBA tensor")
        pixels = np.rint(np.clip(decoded * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels.transpose(1, 2, 0)).save(out)
    else:
        _decode_vae(model, native_latents, out, args.height, args.width, args.dtype,
                     args.backend, tile=bool(args.vae_tile))
    print(f"native denoise trace: {steps_dir}")
    print(f"fixtures: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
