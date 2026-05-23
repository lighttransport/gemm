#!/usr/bin/env python3
"""PyTorch/ROCm reference benchmark for the Qwen3-VL-30B-A3B vision encoder.

This is the same vision tower (depth=27, hidden=1152, ffn=4304, heads=16,
patch=16, merge=2) used by the qwen36/27b mmproj that rdna4/vlm/test_hip_vision
runs, so the two are directly comparable.

Synthetic patch input at fixed square resolutions; times the .forward() pass.
"""
import sys, json, time, glob, argparse
import torch
from safetensors.torch import load_file
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeVisionModel
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeVisionConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True, help="dir with config.json + the safetensors shard holding model.visual.*")
ap.add_argument("--sizes", default="512,1024,2048")
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"])
args = ap.parse_args()

DT = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[args.dtype]
dev = "cuda"

cfg_full = json.load(open(f"{args.model_dir}/config.json"))
vcfg = Qwen3VLMoeVisionConfig(**cfg_full["vision_config"])
print(f"vision cfg: depth={vcfg.depth} hidden={vcfg.hidden_size} ffn={vcfg.intermediate_size} "
      f"heads={vcfg.num_heads} patch={vcfg.patch_size} merge={vcfg.spatial_merge_size} "
      f"temporal={vcfg.temporal_patch_size} deepstack={vcfg.deepstack_visual_indexes}")

# ---- load model.visual.* weights ----
state = {}
for sf in glob.glob(f"{args.model_dir}/*.safetensors"):
    sd = load_file(sf)
    for k, v in sd.items():
        if k.startswith("model.visual."):
            state[k[len("model.visual."):]] = v
        elif k.startswith("visual."):
            state[k[len("visual."):]] = v
print(f"loaded {len(state)} visual tensors")

model = Qwen3VLMoeVisionModel(vcfg)   # on CPU; ~600M params, fine
missing, unexpected = model.load_state_dict(state, strict=False)
missing = [m for m in missing if not m.endswith("rotary_pos_emb.inv_freq")]
if missing:
    print("WARN missing:", missing[:8], "..." if len(missing) > 8 else "")
if unexpected:
    print("WARN unexpected:", unexpected[:8], "..." if len(unexpected) > 8 else "")
model = model.to(device=dev, dtype=DT).eval()
print(f"loaded on {dev}; params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")

patch_in = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size
torch.manual_seed(0)

print(f"\ndevice={torch.cuda.get_device_name(0)}  dtype={args.dtype}  "
      f"warmup={args.warmup} iters={args.iters}\n")
print(f"{'image':>10} {'patches':>9} {'tokens':>8} {'mean ms':>9} {'min ms':>9} {'b2b ms':>9} {'tok/s':>9}")

for N in [int(s) for s in args.sizes.split(",")]:
    hp = wp = N // vcfg.patch_size           # patches per side
    seq = hp * wp
    tokens = seq // (vcfg.spatial_merge_size ** 2)
    hs = torch.randn(seq, patch_in, device=dev, dtype=DT)
    grid_thw = torch.tensor([[1, hp, wp]], device=dev, dtype=torch.long)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(hs, grid_thw)
        torch.cuda.synchronize()
        # per-iter timing (min = best warm)
        t = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            model(hs, grid_thw)
            torch.cuda.synchronize()
            t.append((time.perf_counter() - t0) * 1e3)
        # back-to-back submission, one sync (steady-state throughput, no idle gaps)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model(hs, grid_thw)
        torch.cuda.synchronize()
        b2b = (time.perf_counter() - t0) * 1e3 / args.iters
    mean = sum(t) / len(t)
    print(f"{N:>8}² {seq:>9} {tokens:>8} {mean:>9.1f} {min(t):>9.1f} {b2b:>9.1f} "
          f"{1000.0 * tokens / b2b:>9.1f}")
