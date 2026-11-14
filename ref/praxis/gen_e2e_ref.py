#!/usr/bin/env python3
"""End-to-end reference for cuda/praxis/praxis_e2e.c, in PyTorch (praxis-1-oss importable for the VAE / tile canvas / policy):

  task string -> ids (Qwen3.5 processor, no chat template) -> Qwen3.5-2B hidden_states[-3] (batch padded to a multiple of 32, padding
  rows removed: praxis TextEncoder) -> 2 views of uint8 frames -> tile canvas (single_arm.tile_clip) -> Wan2.2 VAE encode_eager ->
  policy latent -> anchor + clean chunks -> mid-size policy (text dim 2048, bf16-exact weights, fp32 compute) sample_window_joint
  (ctx_cache=False) with a fixed seed -> last-chunk video latent + action tokens (normalised, no un-normalisation).

Writes under --out: ids.npy ids_u.npy text.npy text_u.npy ext.npy wrist.npy canvas.npy latent.npy proprio.npy
smp_noise.salt0.0.npy smp_noise.salt1.0.npy pred_video.npy pred_action.npy action_rows.npy decoded.npy (last 24 decoded frames, [-1,1]),
and the policy checkpoint under --model-out (optionally sharded). Run: ~/.venv/bin/python -I ref/praxis/gen_e2e_ref.py
"""
import argparse, json, os, sys
import numpy as np, torch
PRAXIS_ROOT = os.environ.get("PRAXIS_ROOT", os.path.expanduser("~/praxis"))

ap = argparse.ArgumentParser()
ap.add_argument("--praxis-src", default=PRAXIS_ROOT + "/praxis-1-oss")
ap.add_argument("--qwen", default=PRAXIS_ROOT + "/qwen3.5-2b")
ap.add_argument("--vae", default=PRAXIS_ROOT + "/wan2.2-ti2v/Wan2.2_VAE.pth")
ap.add_argument("--prompt", default="pick up the red cup and place it on the plate")
ap.add_argument("--out", default="tmp/praxis_ref/e2e")
ap.add_argument("--model-out", default=PRAXIS_ROOT + "/mid2048")
ap.add_argument("--shards", type=int, default=3)
ap.add_argument("--chunks", type=int, default=2, help="K' (>= 2)")
ap.add_argument("--tile-h", type=int, default=64)
ap.add_argument("--tile-w", type=int, default=128)
ap.add_argument("--src-h", type=int, default=90)
ap.add_argument("--src-w", type=int, default=160)
ap.add_argument("--cfg-video", type=float, default=4.0)
ap.add_argument("--cfg-action", type=float, default=1.5)
ap.add_argument("--steps", type=int, default=2)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--qwen-device", default="cuda")
ap.add_argument("--init-std", type=float, default=0.02)
a = ap.parse_args()
sys.path.insert(0, a.praxis_src)
os.makedirs(a.out, exist_ok=True); os.makedirs(a.model_out, exist_ok=True)
from tests.tiny_model import tiny_config
from praxis.model import PraxisPolicy, SamplerSettings
from praxis.sampler import window_time_grid
from praxis.vae import Autoencoder
from praxis.single_arm import tile_clip
from praxis.text_encoder import PAD_TO_MULTIPLE_OF
from safetensors.torch import save_file
from transformers import AutoModel, AutoProcessor
dev = "cuda"
def npy(n, t): np.save(f"{a.out}/{n}.npy", t.detach().float().cpu().numpy() if torch.is_tensor(t) else t)

# ---- text: Qwen3.5-2B, hidden_states[-3] ----
proc = AutoProcessor.from_pretrained(a.qwen, local_files_only=True)
qm = AutoModel.from_pretrained(a.qwen, local_files_only=True, dtype=torch.float32).to(a.qwen_device).eval()
def encode_text(prompt):
    b = proc(text=[prompt if prompt.strip() else " "], padding=True, truncation=False, add_special_tokens=True,
             return_attention_mask=True, pad_to_multiple_of=PAD_TO_MULTIPLE_OF, return_tensors="pt")
    ids_real = proc(text=[prompt if prompt.strip() else " "], padding=False, truncation=False, add_special_tokens=True, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        hs = qm(input_ids=b["input_ids"].to(a.qwen_device), attention_mask=b["attention_mask"].to(a.qwen_device), output_hidden_states=True, use_cache=False).hidden_states[-3]
    n = int(b["attention_mask"].sum())
    return ids_real[0].numpy().astype(np.int64), hs[0, :n].float().cpu(), b["input_ids"].shape[1]
ids, text, padded = encode_text(a.prompt); ids_u, text_u, _ = encode_text(" ")
print("text tokens", len(ids), "padded to", padded, "uncond tokens", len(ids_u))
npy("ids", ids); npy("ids_u", ids_u); npy("text", text); npy("text_u", text_u)
del qm; torch.cuda.empty_cache()

# ---- frames -> tile canvas -> VAE ----
K = a.chunks
T_raw = 21 + 24 * (K - 2)
g = torch.Generator().manual_seed(11)
def make_view(seed):
    gg = torch.Generator().manual_seed(seed)
    base = torch.randn(1, 3, 6, 9, 16, generator=gg)
    v = torch.nn.functional.interpolate(base.flatten(1, 2), size=(a.src_h, a.src_w), mode="bilinear").view(3, 6, a.src_h, a.src_w)
    t = torch.nn.functional.interpolate(v.permute(0, 2, 3, 1).reshape(1, 3 * a.src_h * a.src_w, 6), size=T_raw, mode="linear").view(3, a.src_h, a.src_w, T_raw)
    return (t.permute(3, 1, 2, 0) * 40 + 128).round().clamp(0, 255).to(torch.uint8).numpy()   # [T][H][W][3]
ext, wrist = make_view(1), make_view(2)
npy("ext", ext); npy("wrist", wrist)
clip = tile_clip(ext, wrist, a.tile_h, a.tile_w, device=dev)     # (1,3,T,th,2tw) float in [0,255]
npy("canvas", clip[0].permute(1, 0, 2, 3).to(torch.uint8))        # [T][3][th][2tw]
vae = Autoencoder.from_pretrained(a.vae, device=dev, decoder_impl="reference")
with torch.no_grad():
    lat = vae.encode_eager(clip)                                  # (1,192,Tl,h,w) bf16
lat = lat.float()
npy("latent", lat[0])
F = 6
assert lat.shape[2] == (K - 1) * F, lat.shape
anchor = lat[:, :, :1].contiguous()
cleans = [lat[:, :, i * F:(i + 1) * F].contiguous() for i in range(K - 1)]
print("policy latent", tuple(lat.shape))

# ---- policy (mid size, text dim 2048) ----
cfg = tiny_config(); bb, ex = cfg["backbone"], cfg["action_expert"]
dim, hd, heads, kv, dense = 384, 128, 3, 1, 640
bb["dim"] = dim; bb["timestep_dim"] = 32; bb["dense_ffn_hidden"] = dense
bb["attention"].update(num_heads=heads, num_kv_heads=kv, head_dim=hd); bb["rope"]["effective_dim"] = 32
bb["moe"].update(num_routed_experts=8, top_k_routed=2, expert_hidden=128, dispatch_chunk_size=128)
bb["encoder"]["input_projections"]["TextEncodingTensor"]["c_in"] = 2048
bb["decoder"]["modulation_dim"] = dim
ex.update(dim=128, num_layers=4, num_heads=2, head_dim=64, trunk_dim=dim, num_future_per_chunk=4)
torch.manual_seed(0)
pol = PraxisPolicy.from_config(cfg)
with torch.no_grad():
    for p_ in pol.parameters(): p_.normal_(0.0, a.init_std)
    for n, b_ in pol.named_buffers():
        if n.endswith("expert_bias"): b_.normal_(0.0, 0.01)
        elif n.endswith(("freqs", "phases")): b_.uniform_(-1.0, 1.0)
pol = pol.to(torch.bfloat16)
sd = {k: v.contiguous() for k, v in pol.state_dict().items()}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for f in os.listdir(a.model_out):
    if f == "model.safetensors" or f == "model.safetensors.index.json" or f.startswith("model-0"): os.remove(os.path.join(a.model_out, f))
if a.shards <= 1: save_file(sd, f"{a.model_out}/model.safetensors")
else:
    keys = list(sd.keys()); sizes = {k: sd[k].numel() * sd[k].element_size() for k in keys}
    load = [0] * a.shards; assign = {}
    for k in sorted(keys, key=lambda k: -sizes[k]):
        j = min(range(a.shards), key=lambda j: load[j]); assign[k] = j; load[j] += sizes[k]
    wm = {}
    for j in range(a.shards):
        fn = f"model-{j + 1:05d}-of-{a.shards:05d}.safetensors"; part = {k: sd[k] for k in keys if assign[k] == j}
        if part: save_file(part, f"{a.model_out}/{fn}")
        for k in part: wm[k] = fn
    json.dump({"metadata": {"total_size": sum(sizes.values())}, "weight_map": wm}, open(f"{a.model_out}/model.safetensors.index.json", "w"), indent=1)
json.dump(cfg, open(f"{a.model_out}/config.json", "w"), indent=1)
pol = pol.float().to(dev).eval()
gp = torch.Generator().manual_seed(99)
proprio = torch.randn(1, K, 10, generator=gp).to(dev); npy("proprio", proprio[0])
settings = SamplerSettings(num_steps=a.steps, t_shift=12.5, cfg_scale=a.cfg_action, cfg_video_scale=a.cfg_video)
tr = {}
def trace(name, t):
    n = tr.get(name, 0); tr[name] = n + 1; npy(f"smp_{name}.{n}", t)
with torch.no_grad():
    v_last, a_last = pol.sample_window_joint(text[None].to(dev), text_u[None].to(dev), anchor.to(dev), [c.to(dev) for c in cleans], proprio,
                                             a.seed, settings, trace=trace, ctx_cache=False, overlap_expert=False)
npy("pred_video", v_last); npy("pred_action", a_last)
# action rows: tokens (L, 4*10) -> 24 rows x 10 (depth >= 1), normalised, no clip / un-normalisation (needs the checkpoint NormStats)
from praxis.robot_io import action_rows
npy("action_rows", action_rows(a_last, K - 1, 4, 10))
# decode context + predicted latents (praxis pipeline.decode_video), last 24 frames
with torch.no_grad():
    full = torch.cat([*cleans, v_last.float().cpu().to(dev)], dim=2)
    z = full
    # decode_unit gives the float [-1,1] frames (decode() quantises to uint8)
    frames = vae.decode_unit(z) if hasattr(vae, "decode_unit") else None
if frames is not None: npy("decoded", frames[0][:, -24:])
json.dump(dict(seed=a.seed, chunks=K, cfg_video=a.cfg_video, cfg_action=a.cfg_action, num_steps=a.steps, t_shift=12.5,
               time_grid=window_time_grid(a.steps, 12.5), text_len=len(ids), tile_h=a.tile_h, tile_w=a.tile_w), open(f"{a.out}/meta.json", "w"))
print("done ->", a.out, "model ->", a.model_out)
