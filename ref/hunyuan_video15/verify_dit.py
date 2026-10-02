"""Check fast12 DiT predictions and optional full Euler chain with saved inputs.

This compares the official transformer graph, not end-to-end pipeline parity.
Input dumps must come from the native fast12 I2V first step (12 steps, shift 7).
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from safetensors.torch import load_file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ref.hunyuan_video15.compare import compare
from ref.hunyuan_video15.convert_native import convert
PIN = "60783e704160023913bee78f0b47036d393d4dfa"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--actual", type=Path, help="separate native prediction directory; defaults to --native")
    ap.add_argument("--generation-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference-dtype", choices=("float16", "float32"), default="float16")
    ap.add_argument("--matmul-precision", choices=("highest", "high"), default="highest",
                    help="high permits tensor-core FP32 arithmetic; report separately from highest")
    ap.add_argument("--attention-precision", choices=("ieee", "tf32", "tf32x3"),
                    help="explicit official flex kernel arithmetic override, recorded in the report")
    ap.add_argument("--full-denoise", action="store_true", help="compare all 12 Euler steps and final latent using the official scheduler")
    ap.add_argument("--final-only", action="store_true", help="full denoising with first-prediction/final-latent comparison for pipeline captures")
    ap.add_argument("--reference-inputs", type=Path, help="independently computed official Qwen/SigLIP/VAE conditioning .npy directory")
    ap.add_argument("--first-block", action="store_true", help="diagnostic: one double-stream block plus the unchanged final projection")
    ap.add_argument("--kv-precision", choices=("float16",),
                    help="diagnostic: round attention K/V like the native flash backend")
    args = ap.parse_args()
    if args.full_denoise and args.first_block:
        raise ValueError("full denoising requires all transformer blocks")
    if (args.final_only or args.reference_inputs) and not args.full_denoise:
        raise ValueError('pipeline conditioning and final-only checks require full denoising')
    if args.kv_precision and args.reference_dtype != "float32":
        raise ValueError("KV override requires float32 reference activations")
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.backends.cudnn.allow_tf32 = args.matmul_precision == "high"
    upstream = args.upstream.resolve()
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN:
        raise ValueError("unexpected upstream revision")
    generation = json.loads(args.generation_manifest.read_text())
    bounded = generation.get("validation_stage") == "bounded_dit_from_saved_native_inputs"
    if not bounded and generation.get("frames") not in (81,121):
        raise ValueError("unexpected generation length; bounded crops require an explicit diagnostic manifest")
    if (generation.get("task"), generation.get("preset"), generation.get("steps"),
        generation.get("cfg"), generation.get("flow_shift")) != ("i2v", "fast12", 12, 1, 7):
        raise ValueError("requires an I2V fast12/12-step/CFG1/shift7 native manifest")
    sys.path.insert(0, str(upstream))
    from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
    if args.kv_precision:
        from hyvideo.models.transformers import hunyuanvideo_1_5_transformer as transformer_module
        from hyvideo.models.transformers.modules import token_refiner as token_module
        original_parallel = transformer_module.parallel_attention
        original_attention = token_module.attention
        def rounded(value):
            return value.to(torch.float16).to(value.dtype)
        def parallel_with_native_kv(q, k, v, *a, **kw):
            return original_parallel(q, tuple(rounded(x) for x in k),
                                     tuple(rounded(x) for x in v), *a, **kw)
        def attention_with_native_kv(q, k, v, *a, **kw):
            return original_attention(q, rounded(k), rounded(v), *a, **kw)
        transformer_module.parallel_attention = parallel_with_native_kv
        token_module.attention = attention_with_native_kv
    if args.attention_precision:
        from hyvideo.models.transformers.modules import attention as attention_module
        original_flex = attention_module.flex_attention
        def flex_with_precision(*a, **kw):
            options = dict(kw.get("kernel_options") or {})
            options["FLOAT32_PRECISION"] = repr(args.attention_precision)
            if args.attention_precision == "tf32x3":
                # Consumer Blackwell has 99 KiB shared memory per block.
                options.update(BLOCK_M=64, BLOCK_N=32, num_stages=1)
            kw["kernel_options"] = options
            return original_flex(*a, **kw)
        attention_module.flex_attention = flex_with_precision
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reference = out / "reference"
    reference.mkdir()
    convert(args.native)
    actual = args.actual or args.native
    if actual != args.native:
        convert(actual)
    arrays = {name: np.load(args.native / f"{name}.npy", allow_pickle=False)
              for name in ("noise_input", "vae_encoded", "qwen_hidden", "siglip_hidden")}
    reference_input_hashes = None
    if args.reference_inputs:
        reference_input_hashes = {}
        for name in ('vae_encoded','qwen_hidden','siglip_hidden'):
            path=args.reference_inputs/f'{name}.npy'
            value=np.load(path,allow_pickle=False)
            if value.shape!=arrays[name].shape:
                raise ValueError(f'independent {name} shape disagrees with native capture')
            arrays[name]=value
            reference_input_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("non-finite native inputs")
    noise = arrays["noise_input"]
    if noise.shape != (1,32,(generation["frames"]-1)//4+1,generation["height"]//16,generation["width"]//16):
        raise ValueError("native latent dimensions disagree with generation manifest")
    if arrays["vae_encoded"].shape != (1,32,1,*noise.shape[-2:]):
        raise ValueError("unexpected encoded portrait dimensions")
    config = {k:v for k,v in json.loads(args.config.read_text()).items() if not k.startswith("_")}
    config["attn_mode"] = "torch"  # Official dense flex attention; no sparse/cache path.
    with torch.device("meta"):
        model = HunyuanVideo_1_5_DiffusionTransformer(**config)
    model_manifest = json.loads((args.model / "model.json").read_text())
    checkpoint = args.model / model_manifest["checkpoints"]["fast12_i2v"]
    state = load_file(str(checkpoint))
    # Comfy repack concatenates Q/K/V along the output axis; recover views
    # without another checkpoint copy or changing any tensor values.
    for name in list(state):
        if ".img_attn_qkv." in name or ".txt_attn_qkv." in name:
            value = state.pop(name)
            if value.shape[0] != config["hidden_size"] * 3:
                raise ValueError(f"unexpected fused QKV layout: {name}")
            for suffix, chunk in zip(("q", "k", "v"), value.chunk(3, dim=0)):
                state[name.replace("_qkv.", f"_{suffix}.")] = chunk
    model.load_state_dict(state, strict=True, assign=True)
    del state
    if args.first_block:
        model.double_blocks = torch.nn.ModuleList([model.double_blocks[0]])
    model.eval().requires_grad_(False)
    dtype = getattr(torch, args.reference_dtype)
    # Host parameters stay FP16. Stage/cast one complete block at a time.
    # Replacing Parameters via .to() does not change the frozen reference math.
    staged = []
    for name, child in model.named_children():
        if isinstance(child, torch.nn.ModuleList):
            staged.extend((f"{name}.{i}", block) for i,block in enumerate(child))
        else:
            staged.append((name, child))
    completed = 0
    def before(module, inputs):
        module.to(device="cuda", dtype=dtype)
    def after(module, inputs, output):
        nonlocal completed
        module.to(device="cpu", dtype=torch.float16)
        completed += 1
        # Shared children can execute more than once in a forward pass.
        print(f"reference stage {completed}", flush=True)
    handles = []
    for name, child in staged:
        handles.extend([child.register_forward_pre_hook(before), child.register_forward_hook(after)])
    def tensor(name):
        return torch.from_numpy(arrays[name]).to(device="cuda", dtype=torch.float32 if args.reference_inputs else dtype)
    latents = tensor("noise_input")
    x = latents
    condition = torch.zeros_like(x)
    condition[:,:,0:1] = tensor("vae_encoded")
    mask = torch.zeros_like(x[:,:1]); mask[:,:,0] = 1
    x = torch.cat((x,condition,mask),dim=1)
    qwen = tensor("qwen_hidden")
    vision = tensor("siglip_hidden")
    text_mask = torch.ones(qwen.shape[:2],device="cuda",dtype=torch.int64)
    # Retain the official masked empty ByT5 tokens rather than altering the graph.
    extra = {"byt5_text_states": torch.zeros((1,256,1472),device="cuda",dtype=dtype),
             "byt5_text_mask": torch.zeros((1,256),device="cuda",dtype=torch.int64)}
    t = torch.tensor([1000.0],device="cuda",dtype=torch.float32)
    # Match native float32 flow schedule arithmetic, including the next timestep.
    sigma = np.float32(1)-np.float32(1)/np.float32(12)
    sigma_r = np.float32(7)*sigma/(np.float32(1)+np.float32(6)*sigma)
    r = torch.tensor([float(sigma_r*np.float32(1000))],device="cuda",dtype=torch.float32)
    scheduler = None
    schedule = None
    if args.full_denoise:
        from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
        scheduler = FlowMatchDiscreteScheduler(shift=7.0, reverse=True, solver="euler")
        scheduler.set_timesteps(12, device="cuda")
        schedule = np.asarray(json.loads((actual / "sigmas.json").read_text()), dtype=np.float32)
        if (schedule.shape != (13,) or not np.isfinite(schedule).all() or
            schedule[0] != 1 or schedule[-1] != 0 or not np.all(np.diff(schedule)<0)):
            raise ValueError("invalid native denoising schedule")
        if not np.allclose(schedule,scheduler.sigmas.cpu().numpy(),rtol=0,atol=2e-7):
            raise ValueError("native schedule disagrees with official shift-7 schedule")
        # Use identical float32 schedule values after checking the official recipe.
        scheduler.sigmas = torch.from_numpy(schedule.copy())
        scheduler.timesteps = (scheduler.sigmas[:-1]*1000).to(device="cuda")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    names = ["dit_first"]
    with torch.inference_mode(), torch.autocast("cuda",dtype=dtype,enabled=dtype==torch.float16):
        for step in range(12 if args.full_denoise else 1):
            completed = 0
            if scheduler is not None:
                t = scheduler.timesteps[step:step+1]
                r = torch.tensor([float(schedule[step+1]*np.float32(1000))],device="cuda",dtype=torch.float32)
                x = torch.cat((latents,condition,mask),dim=1)
            prediction = model(x,t,qwen,None,text_mask,timestep_r=r,vision_states=vision,
                               mask_type="i2v",extra_kwargs=extra,return_dict=False)[0]
            if not torch.isfinite(prediction).all():
                raise ValueError(f"non-finite reference prediction at step {step+1}")
            if step==0:
                np.save(reference / "dit_first.npy", prediction.float().cpu().numpy())
            if scheduler is not None:
                prediction_name = f"dit_step_{step+1:02d}"
                latent_name = f"latent_step_{step+1:02d}"
                latents = scheduler.step(prediction,t[0],latents,return_dict=False)[0]
                if not torch.isfinite(latents).all():
                    raise ValueError(f"non-finite reference latent at step {step+1}")
                np.save(reference / f"{prediction_name}.npy",prediction.float().cpu().numpy())
                np.save(reference / f"{latent_name}.npy",latents.float().cpu().numpy())
                if not args.final_only:
                    names.extend((prediction_name,latent_name))
                print(f"DENOISE_STEP {step+1} 12",flush=True)
    torch.cuda.synchronize()
    if scheduler is not None:
        np.save(reference / "latent_final.npy",latents.float().cpu().numpy())
        names.append("latent_final")
    result = compare(reference,actual,names)
    report = {"upstream_revision":PIN, "reference_dtype":args.reference_dtype,
              "torch_version":torch.__version__,
              "matmul_precision":args.matmul_precision,
              "kv_precision_override":args.kv_precision,
              "attention_tiles": [64,32,1] if args.attention_precision == "tf32x3" else None,
              "attention_precision":args.attention_precision or ("tf32" if args.matmul_precision == "high" else "ieee"),
              "transformer_blocks":len(model.double_blocks),
              "scope":"denoising_with_independent_official_conditioning_and_matched_noise" if args.reference_inputs else "full_denoising_with_saved_native_conditioning" if args.full_denoise else "first_block_diagnostic" if args.first_block else "bounded_transformer_with_native_inputs" if bounded else "transformer_only_with_native_conditioning_and_noise", "attention":"official_torch_flex_dense",
              "reference_conditioning_sha256":reference_input_hashes,"final_only":args.final_only,
              "denoising_steps":12 if args.full_denoise else 1, "matched_schedule":schedule.tolist() if schedule is not None else None,
              "timestep":t.tolist(),"timestep_r":r.tolist(),"shape":list(noise.shape),
              "wall_seconds":time.monotonic()-started,
              "peak_torch_allocated_mib":torch.cuda.max_memory_allocated()/1048576,
              "peak_torch_reserved_mib":torch.cuda.max_memory_reserved()/1048576,
              "generation_manifest_sha256":hashlib.sha256(args.generation_manifest.read_bytes()).hexdigest(),
              "native_run_receipt_sha256":hashlib.sha256((actual/"run_receipt.json").read_bytes()).hexdigest()
                                          if (actual/"run_receipt.json").is_file() else None,
              "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "checkpoint_receipt":model_manifest["sources"].get(str(checkpoint.relative_to(args.model))),
              "actual_prediction_sha256":hashlib.sha256((actual/"dit_first.f32").read_bytes()).hexdigest(),
              "actual_final_latent_sha256":hashlib.sha256((actual/"latent_final.f32").read_bytes()).hexdigest() if args.full_denoise else None,
              "actual_output_sha256":{name:hashlib.sha256((actual/f"{name}.f32").read_bytes()).hexdigest()
                                      for name in names},
              "input_sha256":{name:hashlib.sha256((args.native/f"{name}.f32").read_bytes()).hexdigest()
                              for name in arrays if name != "dit_first"},
              "results":result}
    (out / "parity.json").write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if all(v["pass"] for v in result.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
