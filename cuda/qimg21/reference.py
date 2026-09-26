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


class BlockRing:
    """--offload resident for the transformer: the first N blocks live on the
    device and the rest stream through two device slots, one block ahead of
    compute, as the native runner's plan does.

    Every block's weights are kept in pinned host memory. unpark() puts the
    first N on the device -- N chosen from the memory free at that moment --
    and park() gives the device back, so a resident reference server can hold
    its weights in host RAM between runs and still leave the GPU to others.

    The blocks share one shape, so two sets of device tensors serve every
    streamed block: before block j runs, its weights are already in its slot
    (copied on a side stream) and its parameters point there; it then starts
    the copy of the next streamed block into the other slot, once the compute
    that last used that slot has finished. Events order both directions, so no
    block ever reads a slot mid-copy. (diffusers' own group offloading with
    some groups pinned resident races exactly there.)"""

    def __init__(self, torch, transformer, device, reserve_mib: int):
        self.torch, self.device, self.reserve = torch, device, reserve_mib * 2**20
        self.blocks = list(transformer.transformer_blocks)
        self.others = [child for name, child in transformer.named_children() if name != "transformer_blocks"]
        shapes = [tuple(p.shape) for p in self.blocks[0].parameters()]
        if any([tuple(p.shape) for p in b.parameters()] != shapes for b in self.blocks):
            raise SystemExit("reference: --offload resident needs identically shaped transformer blocks")
        self.host = [[p.data.pin_memory() for p in b.parameters()] for b in self.blocks]
        for block, tensors in zip(self.blocks, self.host):
            for param, tensor in zip(block.parameters(), tensors):
                param.data = tensor
        self.block_bytes = sum(t.numel() * t.element_size() for t in self.host[0])
        self.resident = len(self.blocks)
        self.slots = None
        for j, block in enumerate(self.blocks):
            block.register_forward_pre_hook(lambda _m, _a, j=j: self.before(j))
            block.register_forward_hook(lambda _m, _a, _o, j=j: self.after(j))
        transformer.register_forward_pre_hook(lambda _m, _a: self.fetch(self.resident))

    def unpark(self) -> int:
        torch = self.torch
        for module in self.others:
            module.to(self.device)
        free, _ = torch.cuda.mem_get_info(self.device)
        room = free - self.reserve - 2 * self.block_bytes  # the two slots
        self.resident = max(0, min(len(self.blocks), int(room // self.block_bytes)))
        for block, tensors in zip(self.blocks[:self.resident], self.host):
            for param, tensor in zip(block.parameters(), tensors):
                param.data = tensor.to(self.device, non_blocking=True)
        if self.resident < len(self.blocks):
            self.slots = [[torch.empty_like(t, device=self.device) for t in self.host[0]] for _ in range(2)]
            self.copy_stream = torch.cuda.Stream(self.device)
            self.copied = [torch.cuda.Event() for _ in range(2)]
            self.released = [torch.cuda.Event() for _ in range(2)]
            for event in self.released:
                event.record()
        torch.cuda.synchronize(self.device)
        return self.resident

    def park(self) -> None:
        for block, tensors in zip(self.blocks, self.host):
            for param, tensor in zip(block.parameters(), tensors):
                param.data = tensor
        for module in self.others:
            module.to("cpu")
        self.slots = None
        self.torch.cuda.synchronize(self.device)
        self.torch.cuda.empty_cache()

    def fetch(self, j: int) -> None:
        if j >= len(self.blocks) or self.slots is None:
            return
        slot = (j - self.resident) % 2
        with self.torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(self.released[slot])
            for dst, src in zip(self.slots[slot], self.host[j]):
                dst.copy_(src, non_blocking=True)
            self.copied[slot].record(self.copy_stream)

    def before(self, j: int) -> None:
        if j < self.resident:
            return
        slot = (j - self.resident) % 2
        self.torch.cuda.current_stream().wait_event(self.copied[slot])
        for param, tensor in zip(self.blocks[j].parameters(), self.slots[slot]):
            param.data = tensor
        self.fetch(j + 1)

    def after(self, j: int) -> None:
        if j >= self.resident:
            self.released[(j - self.resident) % 2].record()


def place_on_device(torch, pipe, device, mode: str, reserve_mib: int):
    """--offload group|resident. Weights stream with diffusers' group
    offloading on a side CUDA stream, so each block's copy overlaps the block
    before it. In resident mode the VAE, the transformer's non-block layers and
    its first N blocks live on the device, N being as many as fit next to
    `reserve_mib`; the text encoder (about 15 GB in BF16) never fits on a 16 GB
    card and is streamed in both modes."""
    from diffusers.hooks import apply_group_offloading

    def stream(module, level="block_level"):
        apply_group_offloading(module, onload_device=device, offload_device=torch.device("cpu"),
                               offload_type=level, num_blocks_per_group=1 if level == "block_level" else None,
                               use_stream=True, record_stream=True)

    # Block level only splits a module's direct ModuleList children, and the
    # text encoder's layers are nested (model.language_model.layers): block
    # level would onload all 15 GB as one group. Leaf level streams each layer.
    stream(pipe.text_encoder, "leaf_level")
    pipe.vae.to(device)
    transformer = pipe.transformer
    blocks = transformer.transformer_blocks
    resident, ring = 0, None
    if mode == "group":
        stream(transformer)
    else:
        ring = BlockRing(torch, transformer, device, reserve_mib)
        resident = ring.unpark()
    print(f"reference: offload {mode}: {resident} of {len(blocks)} transformer blocks resident, "
          f"the rest and the text encoder streamed", file=sys.stderr)
    # Mixed placement leaves the pipeline unable to infer where to run; say so.
    pipe.__class__ = type(pipe.__class__.__name__, (pipe.__class__,),
                          {"_execution_device": property(lambda _self: device)})
    return ring


_phase_mark = [time.perf_counter()]


def phase(label: str, seconds: float | None = None) -> None:
    """One "timing:" line per phase, for the demo's breakdown. With no
    `seconds`, the phase is the time since the previous one."""
    now = time.perf_counter()
    print(f"timing: {label} {now - _phase_mark[0] if seconds is None else seconds:.3f} s",
          file=sys.stderr, flush=True)
    _phase_mark[0] = now


def make_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--offload", choices=("sequential", "group", "resident"), default="sequential",
                    help="GPU weight placement. sequential: every submodule copied at each use "
                         "(least memory, slowest). group: block by block with the next block "
                         "prefetched on a side stream. resident: the VAE and as many transformer "
                         "blocks as fit stay on the device and only the rest stream, as the native "
                         "runner's memory plan does. All three compute the same numbers.")
    ap.add_argument("--resident-reserve-mib", type=int, default=2560,
                    help="device memory --offload resident leaves free for activations and the "
                         "streamed text encoder")
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
    ap.add_argument("--serve", type=Path,
                    help="stay loaded and serve runs on this Unix socket (--offload resident only); "
                         "--dump-dir is then per request")
    return ap


def main() -> int:
    run_start = time.perf_counter()
    ap = make_parser()
    if "--serve" in sys.argv:
        # --dump-dir comes with each request, not with the server.
        for action in ap._actions:
            if action.dest == "dump_dir":
                action.required = False
        return serve(ap, ap.parse_args())
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
    if on_gpu and args.offload == "sequential":
        # The weights are 31 GB and the demo holds a native run in the same
        # device, so the pipeline is streamed module by module rather than
        # resident. On the CPU there is nothing to stream away from.
        pipe.enable_sequential_cpu_offload(device=device)
    elif on_gpu:
        place_on_device(torch, pipe, device, args.offload, args.resident_reserve_mib)
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
        "offload": args.offload if on_gpu else None,
        "use_kv_cache": use_kv_cache,
        "prompt_fixture_dir": str(args.prompt_fixture_dir.resolve()) if args.prompt_fixture_dir else None,
    }, indent=2) + "\n")
    phase("write image + fixtures")
    print(f"timing: reference total {time.perf_counter() - run_start:.3f} s", file=sys.stderr)
    print(f"saved {out / 'reference.png'}")
    return 0

# Options a served run supports; anything else is a one-shot run's job.
SERVE_UNSUPPORTED = ("image", "dump_pred_dir", "capture_block_dir", "dump_vae_dir", "dump_text_inputs_dir",
                     "prompt_fixture_dir")
# Must match what the server was started with.
SERVE_SETUP = ("model", "device", "dtype", "offload")


def serve(ap: argparse.ArgumentParser, setup) -> int:
    """Stay loaded; one run per connection on a Unix socket.

    Same protocol as the native runners: a request is one line of this
    script's usual flags, tab separated; stdout and stderr stream back while it
    runs; the last line is "fast-serve: status N" (0 done, 1 failed, 2 bad
    request, 3 not this server's setup). Weights stay in pinned host memory;
    each run puts the resident blocks and the VAE on the device and takes them
    off again afterwards, so the GPU is free between runs."""
    import io
    import os
    import socket as socketlib
    import traceback
    import torch
    from diffusers import QwenImage21Pipeline

    if setup.offload != "resident" or setup.device == "cpu":
        ap.error("--serve needs --offload resident on a GPU")
    _phase_mark[0] = time.perf_counter()
    device = resolve_device(torch, setup.device)
    dtype = torch.bfloat16 if setup.dtype == "bf16" else torch.float16
    pipe = QwenImage21Pipeline.from_pretrained(str(Path(setup.model).resolve()), dtype=dtype, local_files_only=True)
    ring = place_on_device(torch, pipe, device, "resident", setup.resident_reserve_mib)
    ring.park()
    pipe.vae.to("cpu")
    spans = {"encode": 0.0, "decode": 0.0, "first_step": None, "last_step": None}

    def synced():
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

    def first_call(_module, _inputs):
        if spans["first_step"] is None:
            spans["first_step"] = synced()
    pipe.transformer.register_forward_pre_hook(first_call)

    listener = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_STREAM)
    setup.serve.unlink(missing_ok=True)
    listener.bind(str(setup.serve))
    listener.listen(4)
    listener.settimeout(1.0)
    parent = os.getppid()
    print(f"reference: serving on {setup.serve} ({time.perf_counter() - _phase_mark[0]:.1f} s to load)",
          file=sys.stderr, flush=True)
    while True:
        if os.getppid() != parent:
            return 0
        try:
            conn, _ = listener.accept()
        except socketlib.timeout:
            continue
        conn.settimeout(None)
        stream = conn.makefile("rw", encoding="utf-8", errors="replace", newline="\n", buffering=1)
        saved = sys.stdout, sys.stderr
        status = 0
        try:
            line = stream.readline().rstrip("\n")
            sys.stdout = sys.stderr = stream
            try:
                args = ap.parse_args(line.split("\t"))
            except SystemExit:
                status = 2
            if not status and (any(getattr(args, k) for k in SERVE_UNSUPPORTED) or args.serve or
                               any(getattr(args, k) != getattr(setup, k) for k in SERVE_SETUP) or
                               not args.dump_dir):
                status = 3
            if not status:
                try:
                    run_served(torch, pipe, ring, device, dtype, args, spans)
                except Exception:  # noqa: BLE001 - report it and stay up
                    traceback.print_exc()
                    status = 1
                finally:
                    ring.park()
                    pipe.vae.to("cpu")
                    torch.cuda.empty_cache()
            print(f"fast-serve: status {status}", flush=True)
        except (OSError, ValueError):
            pass
        finally:
            sys.stdout, sys.stderr = saved
            try:
                stream.close()
                conn.close()
            except OSError:
                pass


def run_served(torch, pipe, ring, device, dtype, args, spans) -> None:
    """One text-to-image run on the loaded pipeline: the demo's options only."""
    run_start = time.perf_counter()
    _phase_mark[0] = run_start
    print("timing: reuse loaded PyTorch pipeline 0.000 s", file=sys.stderr)
    resident = ring.unpark()
    pipe.vae.to(device)
    phase(f"onload {resident} resident transformer blocks + VAE")
    for key in spans:
        spans[key] = 0.0 if key in ("encode", "decode") else None
    out = Path(args.dump_dir)
    out.mkdir(parents=True, exist_ok=True)
    if max(args.height, args.width) > 1024:
        pipe.vae.enable_tiling()
    else:
        pipe.vae.disable_tiling()
    gen = torch.Generator(device=device).manual_seed(args.seed)
    initial_latents, _ = pipe.prepare_latents(None, 1, pipe.transformer.config.in_channels, args.height,
                                              args.width, dtype, device, gen, None)
    np.save(out / "initial_latents.npy", np.ascontiguousarray(initial_latents[0].detach().float().cpu().numpy()))

    def callback(_pipe, step, _timestep, kwargs):
        torch.cuda.synchronize()
        spans["last_step"] = time.perf_counter()
        value = kwargs.get("latents")
        if value is not None:
            np.save(out / f"step_{step:03d}.npy", value.detach().float().cpu().numpy())
        prompt_embeds = kwargs.get("prompt_embeds")
        if prompt_embeds is not None and step == 0:
            np.save(out / "prompt_embeds.npy", prompt_embeds.detach().float().cpu().numpy())
        return kwargs

    if args.sdpa_backend == "efficient":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backend_context = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    else:
        backend_context = nullcontext()
    t0 = time.perf_counter()
    use_kv_cache = args.kv_cache != "off"
    with backend_context:
        result = pipe(prompt=args.prompt, negative_prompt=args.negative_prompt,
                      true_cfg_scale=args.true_cfg_scale, height=args.height, width=args.width,
                      num_inference_steps=args.steps, latents=initial_latents, use_kv_cache=use_kv_cache,
                      callback_on_step_end=callback,
                      callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"])
    pipeline_s = time.perf_counter() - t0
    denoise_s = (spans["last_step"] - spans["first_step"]
                 if spans["first_step"] is not None and spans["last_step"] is not None else 0.0)
    phase("prompt encoding (text encoder)", spans["encode"])
    phase(f"denoise {args.steps} steps (image generation)", denoise_s)
    phase("VAE decode", spans["decode"])
    phase("pipeline other (latent prep, scheduler, postprocess)",
          max(0.0, pipeline_s - spans["encode"] - denoise_s - spans["decode"]))
    result.images[0].save(out / "reference.png")
    np.save(out / "reference_rgba.npy", np.asarray(result.images[0].convert("RGBA")))
    (out / "run.json").write_text(json.dumps({
        "model": str(Path(args.model).resolve()), "prompt": args.prompt,
        "negative_prompt": args.negative_prompt, "true_cfg_scale": args.true_cfg_scale,
        "height": args.height, "width": args.width, "steps": args.steps, "seed": args.seed,
        "elapsed_seconds": pipeline_s, "torch": torch.__version__, "device": device_label(torch, device),
        "device_requested": args.device, "sdpa_backend": args.sdpa_backend, "use_kv_cache": use_kv_cache,
        "offload": "resident", "resident_blocks": resident, "served": True,
    }, indent=2) + "\n")
    phase("write image + fixtures")
    print(f"timing: reference total {time.perf_counter() - run_start:.3f} s", file=sys.stderr)
    print(f"saved {out / 'reference.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
