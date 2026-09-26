#!/usr/bin/env python3
"""Create deterministic PyTorch/Diffusers fixtures for qimg21 comparison."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def resolve_device(torch, name: str):
    """Map a requested device onto a real one, refusing what cannot work and
    saying why. PyTorch exposes ROCm through the `cuda` namespace, so an AMD
    run and an NVIDIA run share the device string and differ only in the build
    that is installed -- which is exactly what the request names.
    """
    if name == "cpu":
        return torch.device("cpu")
    hip = getattr(torch.version, "hip", None)
    if name == "rocm":
        if not hip:
            raise SystemExit("--device rocm needs a ROCm PyTorch build; this interpreter has "
                             f"torch {torch.__version__} with no HIP runtime")
        if not torch.cuda.is_available():
            raise SystemExit("--device rocm needs a visible GPU and none is available")
        return torch.device("cuda")
    if hip:
        raise SystemExit(f"this is a ROCm PyTorch build (HIP {hip}); use --device rocm, "
                         "not --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("--device cuda needs a visible GPU; use --device cpu to run the "
                         "reference on the processor")
    return torch.device("cuda")


def device_label(torch, device) -> str:
    """What actually ran, for the log. `cuda` alone is ambiguous once ROCm is a
    possibility, and a parity report that cannot say which one it used is not
    worth much."""
    if device.type != "cuda":
        return "cpu"
    hip = getattr(torch.version, "hip", None)
    return f"rocm {hip}" if hip else f"cuda {torch.cuda.get_device_name(0)}"


_phase_mark = [time.perf_counter()]


def phase(label: str, seconds: float | None = None) -> None:
    """One "timing:" line per phase, for the demo's breakdown. With no
    `seconds`, the phase is the time since the previous one."""
    now = time.perf_counter()
    print(f"timing: {label} {now - _phase_mark[0] if seconds is None else seconds:.3f} s",
          file=sys.stderr, flush=True)
    _phase_mark[0] = now


def main() -> int:
    run_start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--image", help="optional condition image for editing")
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--true-cfg-scale", type=float, default=1.0)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--device", choices=("cuda", "rocm", "cpu"), default="cuda",
                    help="which PyTorch build and device to run on; rocm and cuda need "
                         "matching PyTorch builds, cpu needs neither")
    ap.add_argument("--sdpa-backend", choices=("default", "efficient"), default="default",
                    help="optionally pin CUDA SDPA for deterministic native parity")
    ap.add_argument(
        "--dump-initial-latents",
        action="store_true",
        help="save the exact packed PyTorch noise tensor used by the denoising loop",
    )
    ap.add_argument(
        "--dump-pred-dir",
        help="save each transformer denoiser prediction as pred_NNN.npy",
    )
    ap.add_argument("--kv-cache", choices=("auto", "on", "off"), default="auto",
                    help="prefix KV cache; auto disables it when dumping predictions")
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--capture-block-dir", help="diagnostic block-0 tensors from the first denoiser call")
    ap.add_argument("--dump-vae-dir", help="save the condition VAE input and posterior moments")
    ap.add_argument("--dump-text-inputs-dir", help="save token IDs and masks passed to the text encoder")
    ap.add_argument("--prompt-fixture-dir", type=Path,
                    help="use captured pre-final-RMSNorm prompt embeddings and image-pad mask")
    args = ap.parse_args()

    _phase_mark[0] = time.perf_counter()
    import torch
    from diffusers import QwenImage21Pipeline
    phase("import torch + diffusers")

    device = resolve_device(torch, args.device)
    on_gpu = device.type == "cuda"
    label = device_label(torch, device)
    if args.sdpa_backend == "efficient" and not on_gpu:
        # The pinned route is the CUDA/HIP flash kernel. Asking for it on the CPU
        # would either raise deep inside SDPA or quietly pick another kernel, and
        # the parity fixtures depend on knowing which one ran.
        print(f"reference: --sdpa-backend efficient has no {label} route; using the default",
              file=sys.stderr)
        args.sdpa_backend = "default"
    print(f"reference: device {label}, dtype {args.dtype}, sdpa {args.sdpa_backend}")
    out = Path(args.dump_dir)
    out.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image) if args.image else None
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    _phase_mark[0] = time.perf_counter()
    pipe = QwenImage21Pipeline.from_pretrained(
        str(Path(args.model).resolve()), dtype=dtype, local_files_only=True
    )
    phase("load pipeline weights (from_pretrained)")
    if on_gpu:
        # The weights are 31 GB and the demo holds a native run in the same
        # device, so the pipeline is streamed module by module rather than
        # resident. On the CPU there is nothing to stream away from.
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        pipe.to(device)
    phase("move to device / CPU offload setup")
    if args.prompt_fixture_dir:
        fixture = args.prompt_fixture_dir
        prompt_array = np.load(fixture / "prompt_embeds.npy", allow_pickle=False)
        prompt_mask_array = np.load(fixture / "prompt_mask.npy", allow_pickle=False)
        image_mask_array = np.load(fixture / "image_pad_mask.npy", allow_pickle=False)
        if (prompt_array.ndim != 3 or prompt_array.shape[0] != 1 or
                prompt_mask_array.shape != prompt_array.shape[:2] or
                image_mask_array.shape != prompt_array.shape[:2]):
            raise ValueError("invalid prompt fixture shapes")
        if not np.isfinite(prompt_array).all() or not np.all(prompt_mask_array):
            raise ValueError("invalid prompt fixture values")

        def fixture_encode_prompt(*_args, device=None, num_images_per_prompt=1, **_kwargs):
            # Falls back to the pipeline's own device, which while offloading is
            # whichever module is resident, and on the CPU is simply the CPU.
            target = device or pipe._execution_device
            embeddings = torch.from_numpy(prompt_array).to(device=target, dtype=dtype)
            prompt_mask = torch.from_numpy(prompt_mask_array).to(device=target, dtype=torch.bool)
            image_mask = torch.from_numpy(image_mask_array).to(device=target, dtype=torch.bool)
            if num_images_per_prompt != 1:
                embeddings = embeddings.repeat_interleave(num_images_per_prompt, dim=0)
                prompt_mask = prompt_mask.repeat_interleave(num_images_per_prompt, dim=0)
                image_mask = image_mask.repeat_interleave(num_images_per_prompt, dim=0)
            return embeddings, None if prompt_mask.all() else prompt_mask, image_mask

        pipe.encode_prompt = fixture_encode_prompt
    text_input_handles = []
    if args.dump_text_inputs_dir:
        text_inputs_dir = Path(args.dump_text_inputs_dir)
        text_inputs_dir.mkdir(parents=True, exist_ok=False)

        def dump_text_inputs(_module, inputs, kwargs):
            for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
                value = kwargs.get(name)
                if value is not None:
                    np.save(text_inputs_dir / f"{name}.npy", value.detach().cpu().numpy())

        text_input_handles.append(
            pipe.text_encoder.register_forward_pre_hook(dump_text_inputs, with_kwargs=True)
        )
    vae_handles = []
    if args.dump_vae_dir:
        vae_dir = Path(args.dump_vae_dir)
        vae_dir.mkdir(parents=True, exist_ok=False)

        def dump_vae_input(_module, inputs):
            np.save(vae_dir / "input.npy", inputs[0].detach().float().cpu().numpy())

        def dump_vae_moments(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            np.save(vae_dir / "moments.npy", value.detach().float().cpu().numpy())

        vae_handles.append(pipe.vae.encoder.register_forward_pre_hook(dump_vae_input))
        vae_handles.append(pipe.vae.encoder.conv_in.register_forward_hook(
            lambda _module, _inputs, output: np.save(
                vae_dir / "encoder_conv_in.npy", output.detach().float().cpu().numpy()
            )
        ))
        for stage, block in enumerate(pipe.vae.encoder.down_blocks):
            vae_handles.append(block.register_forward_hook(
                lambda _module, _inputs, output, stage=stage: np.save(
                    vae_dir / f"encoder_down_{stage}.npy", output.detach().float().cpu().numpy()
                )
            ))
        vae_handles.append(pipe.vae.quant_conv.register_forward_hook(dump_vae_moments))
    if args.capture_block_dir:
        import diffusers.models.transformers.transformer_qwenimage21 as qmod
        capture_dir = Path(args.capture_block_dir)
        capture_dir.mkdir(parents=True, exist_ok=False)
        capture_calls = [0]
        block0 = pipe.transformer.transformer_blocks[0]
        def capture_tensor(name):
            def hook(_module, _inputs, output):
                if capture_calls[0] == 0:
                    value = output[0] if isinstance(output, tuple) else output
                    np.save(capture_dir / f"{name}.npy", value.detach().float().cpu().numpy())
            return hook
        def capture_input(name):
            def hook(_module, inputs):
                if capture_calls[0] == 0:
                    np.save(capture_dir / f"{name}.npy", inputs[0].detach().float().cpu().numpy())
            return hook
        def capture_block_input(_module, inputs, kwargs):
            if capture_calls[0] == 0:
                value = inputs[0] if inputs else kwargs["hidden_states"]
                np.save(capture_dir / "block_00_input.npy", value.detach().float().cpu().numpy())
        def capture_block_output(_module, _inputs, kwargs, output):
            if capture_calls[0] == 0:
                value = output[0] if isinstance(output, tuple) else output
                np.save(capture_dir / "block_00.npy", value.detach().float().cpu().numpy())
            capture_calls[0] += 1
        block0.register_forward_pre_hook(capture_block_input, with_kwargs=True)
        block0.register_forward_hook(capture_block_output, with_kwargs=True)
        block0.img_norm1.register_forward_hook(capture_tensor("norm1"))
        block0.attn.to_q.register_forward_hook(capture_tensor("q_linear"))
        block0.attn.to_k.register_forward_hook(capture_tensor("k_linear"))
        block0.attn.to_v.register_forward_hook(capture_tensor("v_linear"))
        block0.attn.norm_q.register_forward_hook(capture_tensor("q_norm"))
        block0.attn.norm_k.register_forward_hook(capture_tensor("k_norm"))
        block0.attn.to_out[0].register_forward_pre_hook(capture_input("attn_raw"))
        block0.attn.register_forward_hook(capture_tensor("attn_out"))
        block0.img_norm2.register_forward_pre_hook(capture_input("post_attn_hidden"))
        block0.img_norm2.register_forward_hook(capture_tensor("norm2"))
        block0.img_mlp.register_forward_pre_hook(capture_input("mlp_input"))
        block0.img_mlp.register_forward_hook(capture_tensor("mlp_out"))
        original_modulate = block0._modulate
        modulation_calls = [0]
        def capture_modulate(hidden_states, modulation, target_token_mask):
            result = original_modulate(hidden_states, modulation, target_token_mask)
            if capture_calls[0] == 0:
                suffix = modulation_calls[0] + 1
                np.save(capture_dir / f"modulated{suffix}.npy",
                        result[0].detach().float().cpu().numpy())
                np.save(capture_dir / f"gate{suffix}.npy",
                        result[1].detach().float().cpu().numpy())
            modulation_calls[0] += 1
            return result
        block0._modulate = capture_modulate
        original_prepare_qkv = qmod._qwenimage21_prepare_qkv
        prepare_calls = [0]
        def capture_prepare_qkv(*positional, **keywords):
            result = original_prepare_qkv(*positional, **keywords)
            if prepare_calls[0] == 0:
                rotary = positional[2] if len(positional) > 2 else keywords.get("rotary_emb")
                if rotary is not None:
                    np.save(capture_dir / "rope_table.npy",
                            torch.view_as_real(rotary).flatten(-2).detach().float().cpu().numpy())
                for name, value in zip(("rope_q", "rope_k", "v"), result[:3]):
                    np.save(capture_dir / f"{name}.npy",
                            value.flatten(2)[0].detach().float().cpu().numpy())
            prepare_calls[0] += 1
            return result
        qmod._qwenimage21_prepare_qkv = capture_prepare_qkv
    if max(args.height, args.width) > 1024:
        pipe.vae.enable_tiling()
    # The generator has to sit on the same device as the noise it seeds, or
    # prepare_latents has to copy across a bus and the seed stops being exact.
    gen = torch.Generator(device=device).manual_seed(args.seed)
    pred_dir = Path(args.dump_pred_dir) if args.dump_pred_dir else None
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)
    pred_index = [0]
    use_cfg = args.negative_prompt is not None and args.true_cfg_scale > 1.0
    conditional_prediction = [None]
    if pred_dir:
        def dump_timestep(_module, _inputs, kwargs):
            step = pred_index[0] // 2 if use_cfg else pred_index[0]
            is_negative = use_cfg and pred_index[0] % 2 == 1
            embeds = kwargs.get("encoder_hidden_states")
            if step == 0 and embeds is not None:
                name = "negative_prompt_embeds.npy" if is_negative else "prompt_embeds.npy"
                np.save(out / name, embeds.detach().float().cpu().numpy())
                branch = "negative" if is_negative else "positive"
                for key in ("img_mask", "encoder_hidden_states_mask"):
                    value = kwargs.get(key)
                    # An absent key mask means every text key is valid.
                    # Persist that semantic value so native fixtures need
                    # not guess whether a missing file means no padding.
                    if key == "encoder_hidden_states_mask" and value is None:
                        value = torch.ones(embeds.shape[:2], dtype=torch.bool)
                    if value is not None:
                        np.save(pred_dir / f"{branch}_{key}.npy", value.detach().cpu().numpy())
                shapes = kwargs.get("img_shapes")
                if shapes is not None:
                    (pred_dir / f"{branch}_layout.json").write_text(json.dumps({
                        "img_shapes": shapes, "text_slots": int(embeds.shape[1]),
                        "target_tokens": (args.height // 16) * (args.width // 16),
                    }, indent=2) + "\n")
            if is_negative:
                return
            value = kwargs.get("timestep")
            if value is not None:
                np.save(
                    pred_dir / f"timestep_{step:03d}.npy",
                    np.ascontiguousarray(value.detach().float().cpu().numpy()),
                )

        def dump_input(_module, inputs):
            if use_cfg and pred_index[0] % 2:
                return
            if not inputs:
                return
            value = inputs[0]
            np.save(
                pred_dir / f"input_{pred_index[0] // 2 if use_cfg else pred_index[0]:03d}.npy",
                np.ascontiguousarray(value.detach().float().cpu().numpy()),
            )

        def dump_prediction(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            target_tokens = (args.height // 16) * (args.width // 16)
            value = value[:, -target_tokens:]
            step = pred_index[0] // 2 if use_cfg else pred_index[0]
            if use_cfg:
                branch = "negative" if pred_index[0] % 2 else "positive"
                np.save(pred_dir / f"{branch}_{step:03d}.npy", value.detach().float().cpu().numpy())
                if pred_index[0] % 2 == 0:
                    conditional_prediction[0] = value.detach().clone()
                    pred_index[0] += 1
                    return
                # Keep the arithmetic on the same device and dtype as the
                # pipeline, including the BF16 intermediate boundaries.
                value = value + args.true_cfg_scale * (conditional_prediction[0] - value)
                conditional_prediction[0] = None
            np.save(
                pred_dir / f"pred_{step:03d}.npy",
                np.ascontiguousarray(value.detach().float().cpu().numpy()),
            )
            pred_index[0] += 1
        pipe.transformer.register_forward_pre_hook(dump_timestep, with_kwargs=True)
        pipe.transformer.img_in.register_forward_pre_hook(dump_input)
        pipe.transformer.proj_out.register_forward_hook(dump_prediction)
    initial_latents = None
    if args.dump_initial_latents:
        initial_latents, _ = pipe.prepare_latents(
            None,
            1,
            pipe.transformer.config.in_channels,
            args.height,
            args.width,
            dtype,
            device,
            gen,
            None,
        )
        np.save(
            out / "initial_latents.npy",
            np.ascontiguousarray(initial_latents[0].detach().float().cpu().numpy()),
        )

    # Where the pipeline's time goes. The denoise span runs from the first
    # transformer call to the last step callback; prompt encoding and the VAE
    # decode are timed by wrapping them. A device sync bounds each span, so a
    # queued kernel is not counted in the next phase.
    spans = {"encode": 0.0, "decode": 0.0, "first_step": None, "last_step": None}

    def synced():
        if on_gpu:
            torch.cuda.synchronize()
        return time.perf_counter()

    def timed(key, fn):
        def wrapper(*a, **k):
            start = synced()
            try:
                return fn(*a, **k)
            finally:
                spans[key] += synced() - start
        return wrapper

    pipe.encode_prompt = timed("encode", pipe.encode_prompt)
    pipe.vae.decode = timed("decode", pipe.vae.decode)

    def first_transformer_call(_module, _inputs):
        if spans["first_step"] is None:
            spans["first_step"] = synced()
    timing_handle = pipe.transformer.register_forward_pre_hook(first_transformer_call)

    def callback(_pipe, step, _timestep, kwargs):
        spans["last_step"] = synced()
        value = kwargs.get("latents")
        if value is not None:
            np.save(out / f"step_{step:03d}.npy", value.detach().float().cpu().numpy())
        prompt_embeds = kwargs.get("prompt_embeds")
        if prompt_embeds is not None and step == 0:
            np.save(out / "prompt_embeds.npy", prompt_embeds.detach().float().cpu().numpy())
        return kwargs

    use_kv_cache = pred_dir is None if args.kv_cache == "auto" else args.kv_cache == "on"
    t0 = time.perf_counter()
    if args.sdpa_backend == "efficient":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backend_context = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    else:
        backend_context = nullcontext()
    with backend_context:
        result = pipe(
            prompt=args.prompt,
            image=image,
            negative_prompt=args.negative_prompt,
            true_cfg_scale=args.true_cfg_scale,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            generator=None if initial_latents is not None else gen,
            latents=initial_latents,
            # Native denoiser parity replays every step through the same full
            # prefill path.  The cache path is numerically equivalent but uses a
            # different attention execution route, which would measure cache
            # drift instead of kernel parity in the per-step fixtures.
            use_kv_cache=use_kv_cache,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
        )
    timing_handle.remove()
    pipeline_s = time.perf_counter() - t0
    denoise_s = (spans["last_step"] - spans["first_step"]
                 if spans["first_step"] is not None and spans["last_step"] is not None else 0.0)
    phase("prompt encoding (text encoder)", spans["encode"])
    phase(f"denoise {args.steps} steps (image generation)", denoise_s)
    phase("VAE decode", spans["decode"])
    phase("pipeline other (latent prep, scheduler, postprocess)",
          max(0.0, pipeline_s - spans["encode"] - denoise_s - spans["decode"]))
    for handle in vae_handles:
        handle.remove()
    for handle in text_input_handles:
        handle.remove()
    if on_gpu:
        # The elapsed time below is only meaningful once the queued work has
        # actually finished; on the CPU everything already ran in order.
        torch.cuda.synchronize()
    result.images[0].save(out / "reference.png")
    np.save(out / "reference_rgba.npy", np.asarray(result.images[0].convert("RGBA")))
    (out / "run.json").write_text(json.dumps({
        "model": str(Path(args.model).resolve()),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "true_cfg_scale": args.true_cfg_scale,
        "use_true_cfg": use_cfg,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - t0,
        "torch": torch.__version__,
        "device": label,
        "device_requested": args.device,
        "sdpa_backend": args.sdpa_backend,
        "use_kv_cache": use_kv_cache,
        "prompt_fixture_dir": str(args.prompt_fixture_dir.resolve()) if args.prompt_fixture_dir else None,
    }, indent=2) + "\n")
    phase("write image + fixtures")
    print(f"timing: reference total {time.perf_counter() - run_start:.3f} s", file=sys.stderr)
    print(f"saved {out / 'reference.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
