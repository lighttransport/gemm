#!/usr/bin/env python3
"""Reference dumps for the praxis-1 CUDA runner, from praxis' TINY random policy.

Builds tests/tiny_model.py::tiny_policy (bf16-initialised, then evaluated in
float32 so the weights are exactly the bf16 values that go into the safetensors
file), saves state_dict (bf16 safetensors) + config.json under --model-out, and
dumps .npy tensors under --out:
  inputs      in_*.npy
  encoder     enc_embed, enc_b{i}_attn/_ffn/_out, enc_cond, (taps = enc_b{tap}_out)
  decoder     dec_embed, dec_mod{i}, dec_b{i}_attn/_ffn/_out, dec_vel
  expert      ae_x0, ae_b{i}_out, ae_out
  sampler     smp_<trace-name>.<n>.npy (all trace tensors of 2-step window_joint),
              smp_out_video, smp_out_action (last chunk), noise_v, noise_a
Run from anywhere:  ~/.venv/bin/python ref/praxis/gen_backbone_ref.py
Needs praxis-1-oss importable (PYTHONPATH or --praxis-src); runs on CUDA (flex attention).
"""
import argparse, json, os, sys
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--praxis-src", default="/mnt/disk1/praxis/praxis-1-oss")
ap.add_argument("--model-out", default="/mnt/disk1/praxis/tiny")
ap.add_argument("--out", default="tmp/praxis_ref/backbone")
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunks", type=int, default=3, help="K' = window chunks (context depth + 1)")
ap.add_argument("--text-len", type=int, default=5)
ap.add_argument("--cfg-video", type=float, default=4.0)
ap.add_argument("--cfg-action", type=float, default=1.5)
ap.add_argument("--shards", type=int, default=0,
                help="write the checkpoint as N safetensors shards + model.safetensors.index.json (real layout) instead of one file")
ap.add_argument("--skip-ref", action="store_true", help="write only the (random) checkpoint, no reference dumps (CPU only; for the RSS test)")
ap.add_argument("--preset", default="tiny", choices=["tiny", "mid", "mid128", "mid2048", "big"],
                help="tiny = praxis tests/tiny_model.py (dim 64); mid = dim 256 / hd 64 / 4+2 heads; mid128 = dim 384 / hd 128 / 3+1 heads "
                     "(dims are multiples of 128 where NVFP4 needs them); mid2048 = mid128 with text dim 2048 (consumes real Qwen3.5-2B output); "
                     "big = dim 768 / 32 experts / hidden 512 / 20 MoE layers, a host-RSS test model (use with --skip-ref)")
ap.add_argument("--init-std", type=float, default=None, help="parameter init std (default 0.05 tiny, 0.02 mid)")
a = ap.parse_args()
sys.path.insert(0, a.praxis_src)
from tests.tiny_model import tiny_policy, tiny_config, TINY_HW
from praxis.model import PraxisPolicy
from praxis.model import SamplerSettings
from praxis.sampler import seeded_normal, window_time_grid
from praxis.layers import MoEFeedForward
from praxis.action_expert import run_expert_on_taps_k
from safetensors.torch import save_file

os.makedirs(a.out, exist_ok=True); os.makedirs(a.model_out, exist_ok=True)
dev = a.device
def npy(name, t):
    np.save(f"{a.out}/{name}.npy", t.detach().float().cpu().numpy())

def mid_config(kind):
    c = tiny_config()
    bb, ex = c["backbone"], c["action_expert"]
    if kind == "mid":
        dim, hd, heads, kv, dense, ts = 256, 64, 4, 2, 384, 32
    elif kind == "big":
        dim, hd, heads, kv, dense, ts = 768, 128, 6, 2, 1280, 32
    else:
        dim, hd, heads, kv, dense, ts = 384, 128, 3, 1, 640, 32
    bb["dim"] = dim; bb["timestep_dim"] = ts; bb["dense_ffn_hidden"] = dense
    bb["attention"].update(num_heads=heads, num_kv_heads=kv, head_dim=hd)
    bb["rope"]["effective_dim"] = 32
    bb["moe"].update(num_routed_experts=8, top_k_routed=2, expert_hidden=128, dispatch_chunk_size=128)
    bb["encoder"]["input_projections"]["TextEncodingTensor"]["c_in"] = 2048 if kind == "mid2048" else 64
    if kind == "big":
        bb["moe"].update(num_routed_experts=32, top_k_routed=4, expert_hidden=512)
        bb["encoder"]["depth"] = 14; bb["decoder"]["depth"] = 8
        bb["encoder"]["tap_blocks"] = [0, 2, 4, 6, 7, 9, 11, 13]
    bb["decoder"]["modulation_dim"] = dim
    ex.update(dim=128, num_layers=4, num_heads=2, head_dim=64, trunk_dim=dim, num_future_per_chunk=4)
    return c

def mid_policy(c, seed, std):
    torch.manual_seed(seed)
    policy = PraxisPolicy.from_config(c)
    with torch.no_grad():
        for p_ in policy.parameters():
            p_.normal_(0.0, std)
        for name, buf in policy.named_buffers():
            if name.endswith("expert_bias"):
                buf.normal_(0.0, 0.01)
            elif name.endswith(("freqs", "phases")):
                buf.uniform_(-1.0, 1.0)
    return policy.to(torch.bfloat16)

if a.preset == "tiny":
    cfg = tiny_config()
    pol_bf16 = tiny_policy(a.seed, torch.bfloat16)
else:
    cfg = mid_config(a.preset)
    if a.preset == "big":
        dev = "cpu"
    pol_bf16 = mid_policy(cfg, a.seed, a.init_std if a.init_std is not None else 0.02)
    TINY_HW = (4, 6)
sd = {k: v.contiguous() for k, v in pol_bf16.state_dict().items()}
def write_ckpt(sd, out, nshards):
    for f in os.listdir(out):   # no stale single file / shards next to a fresh checkpoint
        if f == "model.safetensors" or f == "model.safetensors.index.json" or f.startswith("model-0"):
            os.remove(os.path.join(out, f))
    if nshards <= 1:
        save_file(sd, f"{out}/model.safetensors"); return
    # greedy balanced partition by bytes, keys kept in state_dict order inside a shard
    keys = list(sd.keys()); sizes = {k: sd[k].numel() * sd[k].element_size() for k in keys}
    load = [0] * nshards; assign = {}
    for k in sorted(keys, key=lambda k: -sizes[k]):
        j = min(range(nshards), key=lambda j: load[j]); assign[k] = j; load[j] += sizes[k]
    wm = {}
    for j in range(nshards):
        fn = f"model-{j + 1:05d}-of-{nshards:05d}.safetensors"
        part = {k: sd[k] for k in keys if assign[k] == j}
        if part: save_file(part, f"{out}/{fn}")
        for k in part: wm[k] = fn
    json.dump({"metadata": {"total_size": sum(sizes.values())}, "weight_map": wm}, open(f"{out}/model.safetensors.index.json", "w"), indent=1)
write_ckpt(sd, a.model_out, a.shards)
json.dump(cfg, open(f"{a.model_out}/config.json", "w"), indent=1)
if a.skip_ref:
    print("wrote checkpoint (", max(a.shards, 1), "shard(s) ) ->", a.model_out); sys.exit(0)
pol = pol_bf16.float().to(dev).eval()   # float32 compute, bf16-exact weights
dt = torch.float32
h, w = TINY_HW
TEXT_C = cfg["backbone"]["encoder"]["input_projections"]["TextEncodingTensor"]["c_in"]
lat = cfg["chunking"]["latents_per_chunk"]; K = a.chunks
g = torch.Generator().manual_seed(1234 + a.seed)
def rn(*s): return torch.randn(*s, generator=g).to(dev, dt)
text, text_u = rn(1, a.text_len, TEXT_C), rn(1, a.text_len, TEXT_C)
anchor = rn(1, 192, 1, h, w)
noised = rn(1, 192, K * lat, h, w)
clean = rn(1, 192, (K - 1) * lat, h, w)
t_trunk = torch.tensor([[0.9, 0.9, 0.9][:K] if K <= 3 else [0.9] * K], device=dev, dtype=dt)
t_trunk = torch.linspace(0.95, 0.55, K, device=dev, dtype=dt)[None]
proprio = rn(1, K, 10).float()
noised_a = rn(1, K * lat, cfg["action_expert"]["action_patch_dim"])
for n, t in dict(text=text, text_u=text_u, anchor=anchor, noised=noised, clean=clean,
                 t_trunk=t_trunk, proprio=proprio, noised_a=noised_a).items():
    npy("in_" + n, t)

store = {}
def hook_out(name):
    def f(m, i, o): store[name] = o if torch.is_tensor(o) else o[0]
    return f
def hook_in(name):
    def f(m, i): store[name] = i[0]
    return f
enc, dec, ae = pol.backbone.encoder, pol.backbone.decoder, pol.action_expert
hs = []
for tag, tower in (("enc", enc), ("dec", dec)):
    for i, b in enumerate(tower.core.blocks):
        hs += [b.register_forward_hook(hook_out(f"{tag}_b{i}_out")),
               b.mha.register_forward_hook(hook_out(f"{tag}_b{i}_attn")),
               b.ffn.register_forward_hook(hook_out(f"{tag}_b{i}_ffn"))]
        if isinstance(b.ffn, MoEFeedForward):
            def rh(layer, logits, top_idx, _t=tag, _i=i):
                store[f"{_t}_b{_i}_router_logits"] = logits; store[f"{_t}_b{_i}_top_idx"] = top_idx.float()
            b.ffn.routing_hook = rh
    if tower.core.modulators is not None:
        for i, m in enumerate(tower.core.modulators):
            hs.append(m.register_forward_hook(hook_out(f"{tag}_mod{i}")))
hs.append(enc.core.blocks[0].register_forward_pre_hook(lambda m, i: store.__setitem__("enc_embed", i[0])))
hs.append(dec.core.blocks[0].register_forward_pre_hook(lambda m, i: store.__setitem__("dec_embed", i[0])))
for i, b in enumerate(ae.blocks):
    hs.append(b.register_forward_hook(hook_out(f"ae_b{i}_out")))
    if i == 0: hs.append(b.register_forward_pre_hook(hook_in("ae_x0")))

with torch.no_grad():
    cond, taps = enc.forward_chunk_causal(text, anchor, clean, noised, t_trunk)
    npy("enc_cond", cond)
    vel = dec.forward_chunk_causal(noised, cond, t_trunk)
    npy("dec_vel", vel)
    act_t = t_trunk[:, :K]
    y = run_expert_on_taps_k(ae, taps.taps, taps.embeddings, taps.rope_ids,
        global_spans=taps.global_spans, noised_span=taps.noised_span, clean_span=taps.clean_span,
        n_chunks=K, frames_per_chunk=taps.frames_per_chunk, tokens_per_frame=taps.tokens_per_frame,
        noised_actions=noised_a, action_t_per_chunk=act_t, proprio_per_chunk=proprio)
    npy("ae_out", y)
    npy("enc_rope_ids", taps.rope_ids)
for k, v in store.items(): npy(k, v)
for hh in hs: hh.remove()
for m in list(enc.core.blocks) + list(dec.core.blocks):
    if isinstance(m.ffn, MoEFeedForward): m.ffn.routing_hook = None

# full 2-step sample_window_joint with identical noise
settings = SamplerSettings(num_steps=2, t_shift=12.5, cfg_scale=a.cfg_action, cfg_video_scale=a.cfg_video)
seed = 7
cleans = [clean[:, :, i * lat:(i + 1) * lat].contiguous() for i in range(K - 1)]
tr = {}
def trace(name, t):
    n = tr.get(name, 0); tr[name] = n + 1
    npy(f"smp_{name}.{n}", t)
with torch.no_grad():
    v_last, a_last = pol.sample_window_joint(text, text_u, anchor, cleans, proprio, seed, settings,
                                             trace=trace, ctx_cache=False, overlap_expert=False)
npy("smp_out_video", v_last); npy("smp_out_action", a_last)
json.dump(dict(seed=seed, chunks=K, cfg_video=a.cfg_video, cfg_action=a.cfg_action, num_steps=2, t_shift=12.5,
               time_grid=window_time_grid(2, 12.5), text_len=a.text_len),
          open(f"{a.out}/meta.json", "w"))
print("wrote", len(os.listdir(a.out)), "npy to", a.out, "| model ->", a.model_out)
print("enc cond", tuple(cond.shape), "taps", len(taps.taps), "vel", tuple(vel.shape), "ae", tuple(y.shape),
      "out_video", tuple(v_last.shape), "out_action", tuple(a_last.shape))
