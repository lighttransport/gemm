"""Independent pinned PyTorch references for the repository-owned C++ port.

Encoders never consume native hidden states. Denoising consumes independently
computed conditioning and only the matched native noise. Decode consumes the
reference denoised latent. Run phases sequentially to bound GPU residency.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ref.hunyuan_video15.compare import compare, compare_frames
from cuda.hunyuan_video15_native.generate import UPSTREAM, atomic_json, digest, model_manifest


def convert(directory):
    directory = Path(directory)
    converted = []
    for meta in directory.glob("*.json"):
        raw = meta.with_suffix(".f32")
        if not raw.is_file():
            continue
        info = json.loads(meta.read_text())
        shape = info["shape"]
        if info.get("dtype") != "float32" or not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("invalid native dump metadata")
        if raw.stat().st_size != math.prod(shape) * 4:
            raise ValueError("native dump byte count mismatch")
        value = np.fromfile(raw, dtype="<f4").reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError("nonfinite native tensor")
        np.save(meta.with_suffix(".npy"), value, allow_pickle=False)
        converted.append(meta.stem)
    if not converted:
        raise ValueError("no native captures")
    return converted


def pipeline_names(generation):
    if generation.get("backend") != "hv15n_cuda_experimental":
        raise ValueError("pipeline acceptance requires a completed native generation manifest")
    task, preset = generation.get("task"), generation.get("preset")
    if task not in ("i2v", "t2v") or preset not in ("quality", "fast12") or (task == "t2v" and preset == "fast12"):
        raise ValueError("unsupported pipeline recipe")
    expected = dict(frames=81, fps=24, width=480, height=848,
                    steps=12 if preset == "fast12" else 50,
                    cfg=1 if preset == "fast12" else 6, flow_shift=7 if preset == "fast12" else 5)
    if any(generation.get(k) != v for k, v in expected.items()):
        raise ValueError("generation recipe differs from the accepted profile")
    if generation.get("upstream_reference_revision") != UPSTREAM:
        raise ValueError("generation upstream revision differs from the reference")
    if generation.get("metrics", {}).get("memory_fit") != "pass":
        raise ValueError("pipeline acceptance requires a measured memory-budget pass")
    names = ["noise_input", "qwen_hidden", "byt5_hidden", "dit_first", "latent_final", "vae_decoded"]
    names += [f"latent_step_{i}" for i in range(expected["steps"])]
    if preset == "quality": names.append("qwen_negative_hidden")
    if task == "i2v": names += ["siglip_hidden", "vae_encoded"]
    return names


def validate_pipeline_shapes(directory, names):
    fixed = {"noise_input": (1, 32, 21, 53, 30), "dit_first": (1, 32, 21, 53, 30),
             "latent_final": (1, 32, 21, 53, 30), "vae_encoded": (1, 32, 1, 53, 30),
             "vae_decoded": (1, 3, 81, 848, 480), "siglip_hidden": (1, 729, 1152)}
    for name in names:
        shape = np.load(directory / (name + ".npy"), allow_pickle=False, mmap_mode="r").shape
        expected = (1, 32, 21, 53, 30) if name.startswith("latent_step_") else fixed.get(name)
        if expected is not None:
            valid = shape == expected
        else:
            valid = (len(shape) == 3 and shape[0] == 1 and 1 <= shape[1] <= (256 if name == "byt5_hidden" else 1000)
                     and shape[2] == (1472 if name == "byt5_hidden" else 3584))
        if not valid:
            raise ValueError(f"{name}: bounded/invalid shape cannot pass pipeline acceptance: {shape}")


def validate_configs(model_dir, manifest, receipts, generation):
    names = [manifest["reference_configs"][k] for k in ("qwen", "byt5", "vae", generation["preset"] + "_" + generation["task"])]
    if generation["task"] == "i2v":
        names += ["google_siglip/config.json", "google_siglip/preprocessor_config.json"]
    model_dir = model_dir.resolve()
    for name in names:
        path = (model_dir / name).resolve()
        receipt = manifest.get("sources", {}).get(name, {})
        if Path(name).is_absolute() or not path.is_relative_to(model_dir) or not path.is_file():
            raise ValueError("invalid reference config path")
        if not receipt.get("revision") or receipt.get("bytes") != path.stat().st_size or receipt.get("sha256") != digest(path):
            raise ValueError("reference config receipt is missing or stale")
        receipts[name] = receipt


def reference_provenance(args, generation, receipts):
    # A full generation manifest binds every independently computed reference.
    # Bounded probes can run without one, but cannot pass pipeline acceptance.
    return {"generation_manifest_sha256": digest(args.generation_manifest) if args.generation_manifest else None,
            "weights": {k: v["sha256"] for k, v in receipts.items()}, "upstream_revision": UPSTREAM}


def validate_references(reference, names, provenance):
    records = json.loads((reference / "receipts.json").read_text())
    for name in names:
        receipt = records.get(name, {})
        if receipt.get("provenance") != provenance or receipt.get("sha256") != digest(reference / (name + ".npy")):
            raise ValueError(f"missing or stale independently computed reference: {name}")


def record_references(reference, names, provenance):
    path = reference / "receipts.json"
    records = json.loads(path.read_text()) if path.exists() else {}
    records.update({name: {"sha256": digest(reference / (name + ".npy")), "provenance": provenance} for name in names})
    atomic_json(path, records)


def source(upstream):
    upstream = Path(upstream).resolve()
    if subprocess.check_output(["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip() != UPSTREAM:
        raise ValueError("upstream revision mismatch")
    if subprocess.check_output(["git", "-C", upstream, "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("reference upstream source is modified")
    sys.path.insert(0, str(upstream))
    return upstream


def save(out, name, tensor):
    value = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else tensor
    if not np.isfinite(value).all():
        raise ValueError(f"nonfinite reference {name}")
    np.save(out / (name + ".npy"), value, allow_pickle=False)


def qwen(model_dir, manifest, prompt):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, PreTrainedTokenizerFast
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
    from ref.hunyuan_video15.verify_qwen import SYSTEM
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True).text_config
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen2_5_VLTextModel(config)
    state = {k.removeprefix("model."): v for k, v in load_file(str(model_dir / manifest["components"]["qwen"])).items()
             if k.startswith("model.")}
    model.load_state_dict(state, strict=True, assign=True)
    model.rotary_emb = type(model.rotary_emb)(config, device="cpu")
    del state
    model.eval().to(dtype=torch.float32)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_dir / manifest["components"]["tokenizer"]))
    prefix = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
    crop = len(tokenizer.encode(prefix, add_special_tokens=False))
    ids = tokenizer(prefix + prompt + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False,
                    truncation=True, max_length=crop + 1000, return_tensors="pt").input_ids
    with torch.inference_mode():
        return model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[26][:, crop:].clone()


def byt5(model_dir, manifest, prompt):
    import re
    import torch
    from safetensors.torch import load_file
    from transformers import ByT5Tokenizer, T5Config, T5EncoderModel
    texts = list(dict.fromkeys(a or b for a, b in re.findall(r'"(.*?)"|“(.*?)”', prompt)))
    formatted = "".join(f'Text "{text}". ' for text in texts)
    if not formatted:
        return torch.zeros(1, 1, 1472)
    tokens = ByT5Tokenizer()(formatted, truncation=True, max_length=256, return_tensors="pt")
    config = T5Config.from_json_file(model_dir / manifest["reference_configs"]["byt5"])
    state = load_file(str(model_dir / manifest["components"]["byt5"]))
    config.vocab_size = state["shared.weight"].shape[0]
    state["encoder.embed_tokens.weight"] = state["shared.weight"]
    with torch.device("meta"):
        model = T5EncoderModel(config)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.eval().to(dtype=torch.float32)
    with torch.inference_mode():
        return model(**tokens).last_hidden_state.clone()


def siglip(model_dir, manifest, image, pixels):
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import SiglipVisionConfig, SiglipVisionModel
    from transformers.models.siglip.image_processing_pil_siglip import SiglipImageProcessorPil as SiglipImageProcessor
    config = SiglipVisionConfig.from_dict(json.loads((model_dir / "google_siglip/config.json").read_text())["vision_config"])
    config._attn_implementation = "eager"
    processor = SiglipImageProcessor.from_dict(json.loads((model_dir / "google_siglip/preprocessor_config.json").read_text()))
    with Image.open(image) as im:
        inputs = processor(images=im.convert("RGB"), return_tensors="pt").pixel_values
    actual = np.fromfile(pixels, dtype="<f4").reshape(1, 3, 384, 384)
    if not np.allclose(inputs.numpy(), actual, atol=1.e-7, rtol=0):
        raise ValueError("SigLIP preprocessing mismatch")
    with torch.device("meta"):
        model = SiglipVisionModel(config)
    state = load_file(str(model_dir / manifest["components"]["vision"]))
    expected = model.state_dict()
    state = {(k if k in expected else k.removeprefix("vision_model.")): v for k, v in state.items()
             if k in expected or k.removeprefix("vision_model.") in expected}
    model.load_state_dict(state, strict=True, assign=True)
    del state
    vision = model.vision_model if hasattr(model, "vision_model") else model
    vision.embeddings.position_ids = torch.arange(vision.embeddings.num_patches).expand(1, -1)
    model.eval().to(dtype=torch.float32)
    with torch.inference_mode():
        return model(pixel_values=inputs).last_hidden_state.clone()


def vae_model(model_dir, manifest, upstream):
    import torch
    from safetensors.torch import load_file
    spec = importlib.util.spec_from_file_location("hv15n_reference_vae", upstream / "hyvideo/models/autoencoders/hunyuanvideo_15_vae.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = {k: v for k, v in json.loads((model_dir / manifest["reference_configs"]["vae"]).read_text()).items() if not k.startswith("_")}
    with torch.device("meta"):
        model = module.AutoencoderKLConv3D(**config)
    model.load_state_dict(load_file(str(model_dir / manifest["components"]["vae"])), strict=True, assign=True)
    model.eval().to(device="cuda", dtype=torch.float16)
    model.set_tile_sample_min_size(128, .25)
    model.enable_spatial_tiling()
    return model


def denoise(model_dir, manifest, generation, actual, reference, upstream, bounded=False):
    import torch
    from safetensors.torch import load_file
    from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
    from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
    profile = generation["preset"] + "_" + generation["task"]
    config = {k: v for k, v in json.loads((model_dir / manifest["reference_configs"][profile]).read_text()).items() if not k.startswith("_")}
    config["attn_mode"] = "torch"
    with torch.device("meta"):
        model = HunyuanVideo_1_5_DiffusionTransformer(**config)
    state = load_file(str(model_dir / manifest["checkpoints"][profile]))
    for name in list(state):
        if ".img_attn_qkv." in name or ".txt_attn_qkv." in name:
            value = state.pop(name)
            for suffix, chunk in zip(("q", "k", "v"), value.chunk(3, dim=0)):
                state[name.replace("_qkv.", f"_{suffix}.")] = chunk
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.eval().requires_grad_(False)
    def before(child, unused):
        child.to(device="cuda", dtype=torch.float16)
    def after(child, unused, output):
        child.to(device="cpu", dtype=torch.float16)
    handles = []
    for _, child in model.named_children():
        children = child if isinstance(child, torch.nn.ModuleList) else [child]
        for block in children:
            handles.extend((block.register_forward_pre_hook(before), block.register_forward_hook(after)))
    def tensor(name):
        return torch.from_numpy(np.load(reference / f"{name}.npy", allow_pickle=False)).to("cuda")
    latent = torch.from_numpy(np.load(actual / "noise_input.npy", allow_pickle=False)).to("cuda")
    save(reference, "noise_input", latent)
    condition, mask = torch.zeros_like(latent), torch.zeros_like(latent[:, :1])
    vision = None
    if generation["task"] == "i2v" and not bounded:
        condition[:, :, :1] = tensor("vae_encoded")
        mask[:, :, 0] = 1
        vision = tensor("siglip_hidden")
    glyph = tensor("byt5_hidden")
    glyph_mask = torch.full(glyph.shape[:2], int(bool(torch.any(glyph))), dtype=torch.int64, device="cuda")
    text = tensor("qwen_hidden")
    negative = tensor("qwen_negative_hidden") if generation["preset"] == "quality" and not bounded else None
    steps, shift, cfg = (12, 7, 1) if generation["preset"] == "fast12" else (50, 5, 6)
    if bounded: cfg = 1
    scheduler = FlowMatchDiscreteScheduler(shift=shift, reverse=True, solver="euler")
    scheduler.set_timesteps(steps, device="cuda")
    times = np.float32(1) - np.arange(steps + 1, dtype=np.float32) / np.float32(steps)
    sigmas = np.float32(shift) * times / (np.float32(1) + np.float32(shift - 1) * times)
    if not np.allclose(sigmas, scheduler.sigmas.cpu().numpy(), atol=2.e-7, rtol=0):
        raise ValueError("Euler schedule differs from official recipe")
    scheduler.sigmas = torch.from_numpy(sigmas)
    scheduler.timesteps = (scheduler.sigmas[:-1] * 1000).to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for step in range(1 if bounded else steps):
            t = scheduler.timesteps[step:step + 1]
            next_t = torch.tensor([float(sigmas[step + 1] * np.float32(1000))], device="cuda")
            x = torch.cat((latent, condition, mask), dim=1)
            def predict(encoded, glyphs, glyphs_mask):
                extra = {"byt5_text_states": glyphs, "byt5_text_mask": glyphs_mask}
                return model(x, t, encoded, None, torch.ones(encoded.shape[:2], dtype=torch.int64, device="cuda"),
                             timestep_r=next_t if steps == 12 else None, vision_states=vision,
                             mask_type=generation["task"], extra_kwargs=extra, return_dict=False)[0]
            positive = predict(text, glyph, glyph_mask)
            if step == 0:
                save(reference, "dit_first", positive)
            if cfg != 1:
                uncond = predict(negative, torch.zeros_like(glyph), torch.zeros_like(glyph_mask))
                positive = uncond + cfg * (positive - uncond)
            latent = scheduler.step(positive, t[0], latent, return_dict=False)[0]
            save(reference, f"latent_step_{step}", latent)
            print(f"REFERENCE_STEP {step + 1} {steps}", flush=True)
    if not bounded: save(reference, "latent_final", latent)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "actual", "out"):
        ap.add_argument("--" + key, required=True, type=Path)
    ap.add_argument("--upstream", type=Path, default=ROOT / "tmp/hunyuan-video15-upstream")
    ap.add_argument("--phase", choices=("encoders", "bounded-dit", "bounded-decode", "denoise", "decode", "compare"), required=True)
    ap.add_argument("--components", nargs="+", default=["qwen_hidden", "byt5_hidden", "siglip_hidden", "vae_encoded"])
    ap.add_argument("--generation-manifest", type=Path)
    ap.add_argument("--prompt", default="A person smiles naturally.")
    ap.add_argument("--image", type=Path)
    ap.add_argument("--vision-pixels", type=Path)
    ap.add_argument("--task", choices=("i2v", "t2v"), default="i2v")
    ap.add_argument("--preset", choices=("quality", "fast12"), default="quality")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    scratch = args.out.resolve() / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(scratch / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(scratch / "triton"))
    import torch
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    generation = json.loads(args.generation_manifest.read_text()) if args.generation_manifest else {
        "task": args.task, "preset": args.preset, "prompt": args.prompt, "negative_prompt": ""}
    manifest, receipts = model_manifest(args.model, generation["task"], generation["preset"])
    validate_configs(args.model, manifest, receipts, generation)
    provenance = reference_provenance(args, generation, receipts)
    convert(args.actual)
    reference = args.out / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    upstream = source(args.upstream)
    if args.phase == "encoders":
        for name in args.components:
            if name == "qwen_hidden": value = qwen(args.model, manifest, generation["prompt"])
            elif name == "qwen_negative_hidden": value = qwen(args.model, manifest, generation.get("negative_prompt", ""))
            elif name == "byt5_hidden": value = byt5(args.model, manifest, generation["prompt"])
            elif name == "siglip_hidden": value = siglip(args.model, manifest, args.image, args.vision_pixels)
            elif name == "vae_encoded":
                from PIL import Image
                model = vae_model(args.model, manifest, upstream)
                with Image.open(args.image) as image:
                    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255)
                x = torch.from_numpy(pixels.transpose(2, 0, 1)[None, :, None]).to("cuda")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    value = model.encode(x * 2 - 1).latent_dist.mode() * model.scaling_factor
                del model
            else: raise ValueError("unknown encoder component")
            save(reference, name, value)
            del value
            torch.cuda.empty_cache()
            print("REFERENCE_COMPONENT " + name, flush=True)
        names = args.components
    elif args.phase == "bounded-dit":
        denoise(args.model, manifest, generation, args.actual, reference, upstream, bounded=True)
        names = ["dit_first"]
    elif args.phase == "denoise":
        if not args.generation_manifest:
            raise ValueError("denoising needs exact generation provenance")
        conditioning = ["qwen_hidden", "byt5_hidden"]
        if generation["preset"] == "quality": conditioning.append("qwen_negative_hidden")
        if generation["task"] == "i2v": conditioning += ["siglip_hidden", "vae_encoded"]
        validate_references(reference, conditioning, provenance)
        denoise(args.model, manifest, generation, args.actual, reference, upstream)
        steps = 12 if generation["preset"] == "fast12" else 50
        names = ["noise_input", "dit_first", "latent_final"] + [f"latent_step_{i}" for i in range(steps)]
    elif args.phase in ("decode", "bounded-decode"):
        if args.phase == "decode": validate_references(reference, ["latent_final"], provenance)
        model = vae_model(args.model, manifest, upstream)
        latent = torch.from_numpy(np.load(reference / "latent_final.npy", allow_pickle=False)).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            save(reference, "vae_decoded", model.decode(latent / model.scaling_factor).sample)
        names = ["vae_decoded"]
    else:
        names = pipeline_names(generation)
        validate_pipeline_shapes(args.actual, names)
        validate_pipeline_shapes(reference, names)
        validate_references(reference, names, provenance)
    if args.phase != "compare": record_references(reference, names, provenance)
    result = compare(reference, args.actual, names)
    frames = compare_frames(reference, args.actual) if "vae_decoded" in names else None
    if frames is not None and args.phase != "bounded-decode" and len(frames) != 81:
        raise ValueError("pipeline validation requires all 81 decoded frames")
    passed = all(v["pass"] for v in result.values()) and (frames is None or all(f["pass"] for f in frames))
    report = {"schema": "hv15n.parity.v1", "phase": args.phase, "pass": passed, "results": result,
              "scope": "pipeline" if args.phase == "compare" else "bounded_component_diagnostic" if args.phase.startswith("bounded") else args.phase,
              "decoded_frames": frames, "upstream_revision": UPSTREAM, "torch_version": torch.__version__,
              "reference_weights": receipts, "reference_dtype": "FP32 encoders, FP16 DiT/VAE",
              "generation_manifest_sha256": digest(args.generation_manifest) if args.generation_manifest else None,
              "native_capture_sha256": {n: digest(args.actual / (n + ".f32")) for n in names},
              "reference_capture_sha256": {n: digest(reference / (n + ".npy")) for n in names}}
    atomic_json(args.out / (args.phase + "_parity.json"), report)
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
