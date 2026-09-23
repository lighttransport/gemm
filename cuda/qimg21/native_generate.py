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


def _run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(str(x) for x in command), file=sys.stderr)
    subprocess.run(command, cwd=cwd, check=True)


def _decode_vae(model: Path, latent_path: Path, out_path: Path, height: int, width: int,
                dtype: str, backend: str) -> None:
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
    ap.add_argument("--backend", choices=("cuda", "rocm"), default="cuda")
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
    ap.add_argument("--native-attention", choices=("math", "reverse64", "wmma", "wmma-fused", "edit-size-select", "mma64", "mma64-flash", "mma64-mixed", "mma64-forward-flash", "mma128-efficient", "cutlass-efficient"), default=None)
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
    args = ap.parse_args()
    # The ROCm path has a native decoder and does not require a PyTorch
    # installation. CUDA keeps its existing reference VAE default.
    if args.backend == "rocm":
        args.native_vae = True
    rocm_bf16_edit = (args.backend == "rocm" and bool(args.image) and
                      not (args.quantized_transformer or args.quantize_on_load))
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
    if args.native_attention == "edit-size-select" and (args.backend != "rocm" or not args.image):
        ap.error("edit-size-select attention requires ROCm image editing")
    if args.native_attention == "wmma-fused" or (args.native_attention == "edit-size-select"
                                                  and args.height * args.width >= 512 * 512):
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
    if args.height % 32 or args.width % 32:
        raise SystemExit("height and width must be divisible by 32")
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
        _run([
            str(text_encoder), "--model", str(model), "--prompt", args.prompt,
            "--attention", "flash-exact" if args.backend == "cuda" else "custom",
            "--out", str(prompt_path),
        ], cwd=root)
        if args.negative_prompt is not None:
            _run([
                str(text_encoder), "--model", str(model), "--prompt", args.negative_prompt,
                "--attention", "flash-exact" if args.backend == "cuda" else "custom", "--out",
                str(prompt_dir / "negative_prompt_embeds.npy"),
            ], cwd=root)
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

    # The reference text subprocess owns a large accelerator context.  Give the
    # driver a moment to retire that context before the native process opens
    # cuBLAS/NVRTC; otherwise some 2-step launches can observe stale device
    # allocations even though the child has exited.
    time.sleep(2.0)

    h_tokens, w_tokens = args.height // 16, args.width // 16
    latent_path = work / "latents.npy"
    fixture_command = [
            os.environ.get("QIMG21_PYTHON", sys.executable),
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
            "--out-dir",
            str(work),
        ]
    if args.initial_latents:
        fixture_command.extend(("--latents", str(args.initial_latents.resolve())))
    elif args.backend == "cuda":
        fixture_command.append("--torch-rng")
    _run(fixture_command, cwd=root)
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
    if args.int8_tensor_core:
        native_command.extend(["--int8-tensor-core", "--int8-bf16-tail-blocks",
                               str(args.int8_bf16_tail_blocks)])
    _run(native_command, cwd=root)
    if args.native_vae:
        from PIL import Image

        decoded_path = work / "native_decoded.npy"
        _run([
            str(vae_bin),
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
        _decode_vae(model, native_latents, out, args.height, args.width, args.dtype, args.backend)
    print(f"native denoise trace: {steps_dir}")
    print(f"fixtures: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
