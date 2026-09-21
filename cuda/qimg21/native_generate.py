#!/usr/bin/env python3
"""Run native Qwen-Image 2.1 denoising with an optional native CUDA VAE.

The text/vision encoder remains the Diffusers boundary for now. This
driver turns its prompt embedding into an F32 fixture, invokes the native
NVRTC/CUDA transformer for the complete FlowMatch schedule, and only loads the
Qwen-Image 2.1 VAE after the native subprocess exits.  That process boundary
lets the transformer release all of its allocations before the decoder claims
VRAM on a 12–16 GB card.
Single-image editing additionally runs native F32 VAE encoding before text
encoding; end-to-end editing parity is still experimental.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(str(x) for x in command), file=sys.stderr)
    subprocess.run(command, cwd=cwd, check=True)


def _decode_vae(model: Path, latent_path: Path, out_path: Path, height: int, width: int, dtype: str) -> None:
    """Decode normalized [tokens, 64] latents using AutoencoderKLQwenImage21."""
    import torch
    from diffusers import AutoencoderKLQwenImage21
    from PIL import Image

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; the Qwen-Image VAE decode requires NVIDIA CUDA")
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--negative-prompt")
    ap.add_argument("--true-cfg-scale", type=float, default=1.0)
    ap.add_argument("--image", help="Experimental single-image editing with native F32 VAE encoding")
    ap.add_argument("--condition-resolution", type=int, default=1024,
                    help="Condition image target-area side length, matching the reference output_resolution")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--work-dir", default="tmp/qimg21-native-generate")
    ap.add_argument("--out", default="tmp/qimg21-native-generate.png")
    ap.add_argument("--native-bin", default="cuda/qimg21/test_cuda_qimg21_native")
    ap.add_argument("--native-attention", choices=("math", "reverse64", "mma64", "mma64-flash", "mma64-mixed", "mma64-forward-flash", "mma128-efficient"), default="math")
    ap.add_argument("--native-normalization", choices=("default", "vector4"), default="default")
    ap.add_argument("--native-rope", choices=("default", "host-table", "host-table-vector4"), default="default")
    ap.add_argument("--quantized-transformer", type=Path,
                    help="Optional experimental row-INT8 transformer package")
    ap.add_argument("--quantize-on-load", choices=("int8-row",))
    ap.add_argument("--native-vae", action="store_true", help="Decode with the native F32 CUDA VAE (experimental)")
    args = ap.parse_args()
    if args.quantized_transformer and args.quantize_on_load:
        ap.error("choose a quantized package or quantize-on-load, not both")

    root = Path(__file__).resolve().parents[2]
    model = Path(args.model).resolve()
    work = Path(args.work_dir)
    if not work.is_absolute():
        work = root / work
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    native_bin = Path(args.native_bin)
    if not native_bin.is_absolute():
        native_bin = root / native_bin
    if not model.is_dir():
        raise SystemExit(f"model directory does not exist: {model}")
    if not native_bin.exists():
        raise SystemExit(f"native executable not found: {native_bin}; run `make -C cuda/qimg21 native` first")
    if args.native_vae and not (root / "cuda/qimg21/test_cuda_qimg21_vae").exists():
        raise SystemExit("native VAE executable missing; run `make -C cuda/qimg21 native`")
    if args.height % 32 or args.width % 32:
        raise SystemExit("height and width must be divisible by 32")
    if args.steps < 1 or args.steps > 100:
        raise SystemExit("steps must be between 1 and 100")
    if args.negative_prompt is not None and args.true_cfg_scale <= 1.0:
        raise SystemExit("--true-cfg-scale must be > 1 when --negative-prompt is used")
    condition_hw = None
    condition_dir = work / "condition"
    if args.image:
        from editing_inputs import prepare_image
        encoder = root / "cuda/qimg21/test_cuda_qimg21_vae_encode"
        if not encoder.exists():
            raise SystemExit("native encoder missing; run `make -C cuda/qimg21 native-vae`")
        condition_hw = prepare_image(Path(args.image).resolve(), condition_dir, args.condition_resolution)
        _run([str(encoder), "--model", str(model / "vae"), "--image", str(condition_dir / "image.npy"),
              "--out", str(condition_dir / "moments.npy"),
              "--normalized-latents", str(condition_dir / "latents.npy")], cwd=root)

    prompt_dir = work / "prompt"
    steps_dir = work / "steps"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)

    # --test-text uses the exact Qwen3-VL tokenizer/text encoder from the
    # reference runner and writes prompt_embeds.npy.  With a negative prompt
    # it also writes a second embedding fixture for native true CFG.
    text_command = [
            sys.executable,
            str(root / "cuda/qimg21/test_cuda_qimg21.py"),
            "--test-text",
            "--model",
            str(model),
            "--prompt",
            args.prompt,
            "--dtype",
            args.dtype,
            "--dump-dir",
            str(prompt_dir),
            "--out",
            str(prompt_dir / "text_smoke.png"),
        ]
    if args.negative_prompt is not None:
        text_command.extend(["--negative-prompt", args.negative_prompt])
    if args.image:
        text_command.extend(["--image", str(condition_dir / "resized.png")])
    _run(text_command, cwd=root)
    prompt_path = prompt_dir / "prompt_embeds.npy"
    if not prompt_path.exists():
        raise SystemExit(f"text runner did not produce {prompt_path}")

    # The reference text subprocess owns a large CUDA context.  Give the
    # driver a moment to retire that context before the native process opens
    # cuBLAS/NVRTC; otherwise some 2-step launches can observe stale device
    # allocations even though the child has exited.
    time.sleep(2.0)

    h_tokens, w_tokens = args.height // 16, args.width // 16
    latent_path = work / "latents.npy"
    _run(
        [
            sys.executable,
            str(root / "cuda/qimg21/make_native_fixture.py"),
            "--prompt-embeds",
            str(prompt_path),
            "--height-tokens",
            str(h_tokens),
            "--width-tokens",
            str(w_tokens),
            "--seed",
            str(args.seed),
            "--dtype",
            args.dtype,
            "--torch-rng",
            "--out-dir",
            str(work),
        ],
        cwd=root,
    )
    native_latents = work / "native_latents.npy"
    native_command = [
            str(native_bin),
            "--attention", args.native_attention,
            "--model",
            str(model),
            "--prompt-embeds",
            str(prompt_path),
            "--latents",
            str(latent_path),
            "--height-tokens",
            str(h_tokens),
            "--width-tokens",
            str(w_tokens),
            "--steps",
            str(args.steps),
            "--dump-dir",
            str(steps_dir),
            "--out",
            str(native_latents),
        ]
    native_command.extend(["--normalization", args.native_normalization, "--rope", args.native_rope])
    if args.image:
        from editing_inputs import write_layout
        layout = condition_dir / "positive_layout.txt"
        write_layout(prompt_dir, layout, condition_hw, (h_tokens, w_tokens))
        native_command.extend(["--condition-latents", str(condition_dir / "latents.npy"),
                               "--editing-layout", str(layout)])
        if args.negative_prompt is not None:
            negative_layout = condition_dir / "negative_layout.txt"
            write_layout(prompt_dir, negative_layout, condition_hw, (h_tokens, w_tokens), negative=True)
            native_command.extend(["--negative-editing-layout", str(negative_layout)])
    if args.negative_prompt is not None:
        native_command.extend([
            "--negative-prompt-embeds",
            str(prompt_dir / "negative_prompt_embeds.npy"),
            "--guidance-scale",
            str(args.true_cfg_scale),
        ])
    if args.quantized_transformer:
        native_command.extend(["--quantized-transformer", str(args.quantized_transformer.resolve())])
    if args.quantize_on_load:
        native_command.extend(["--quantize-on-load", args.quantize_on_load])
    _run(native_command, cwd=root)
    if args.native_vae:
        from PIL import Image

        decoded_path = work / "native_decoded.npy"
        _run([
            str(root / "cuda/qimg21/test_cuda_qimg21_vae"),
            "--model", str(model / "vae"), "--latents", str(native_latents),
            "--height-tokens", str(args.height // 16),
            "--width-tokens", str(args.width // 16), "--out", str(decoded_path),
        ], cwd=root)
        decoded = np.load(decoded_path)
        if decoded.shape != (4, args.height, args.width) or not np.isfinite(decoded).all():
            raise SystemExit("native VAE returned an invalid RGBA tensor")
        pixels = np.rint(np.clip(decoded * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels.transpose(1, 2, 0)).save(out)
    else:
        _decode_vae(model, native_latents, out, args.height, args.width, args.dtype)
    print(f"native denoise trace: {steps_dir}")
    print(f"fixtures: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
