#!/usr/bin/env python3
"""PyTorch oracle for Cosmos3-Edge-Policy-DROID (diffusers Cosmos3OmniPipeline, action mode="policy").

Fixed deterministic synthetic DROID-like observation (exterior + wrist views, concatenated side by side, passed with
view_point="concat_view" as the pipeline expects ONE image), fixed prompt, seeded noise. Exports every tensor the CUDA runner needs
(token ids, noise, VAE-encoded conditioning latent) and the oracle intermediates:

  per layer (step 0, cond+uncond pass): und/gen hidden state after each decoder layer, layer-0 inputs, final norms
  per call: vision/action velocity predictions straight out of the transformer (before masking / CFG)
  per step: CFG-combined velocity, scheduler.step input/output for vision and action latents
  final: action chunk [chunk, raw_dim] and final video latent (decode optional with --decode)

Invocation:  ~/.venv/bin/python -I ref/cosmos3/gen_ref.py [--out tmp/cosmos3_ref] [--steps 4] [--guidance 3.0]
"""
import argparse, copy, json, os, sys, time
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/mnt/disk1/praxis/cosmos3-edge-droid")
ap.add_argument("--out", default="tmp/cosmos3_ref")
ap.add_argument("--prompt", default="pick up the red cup and place it on the plate")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--guidance", type=float, default=3.0)
ap.add_argument("--chunk", type=int, default=32)
ap.add_argument("--fps", type=float, default=15.0, help="conditioning_fps in checkpoint.json")
ap.add_argument("--tier", type=int, default=256)
ap.add_argument("--img-h", type=int, default=180)
ap.add_argument("--img-w", type=int, default=320)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--fp32", action="store_true", help="run the transformer in fp32 (bf16 weights upcast, lm_head dropped): bf16 noise-floor oracle")
ap.add_argument("--decode", action="store_true", help="also VAE-decode the generated video (extra ~1.4GB + time)")
a = ap.parse_args()
import torch as _t; _t.set_grad_enabled(False)
os.makedirs(a.out, exist_ok=True)

from diffusers import Cosmos3OmniPipeline
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition
from diffusers import UniPCMultistepScheduler

def npy(name, t):
    np.save(f"{a.out}/{name}.npy", t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t))

# ---- synthetic observation: two 160x180 views (exterior | wrist), deterministic ----
def synth_view(h, w, seed):
    g = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([128 + 100 * np.sin(xx / 17.0 + seed), 128 + 100 * np.cos(yy / 11.0 - seed), 128 + 60 * np.sin((xx + yy) / 23.0)], -1)
    img += g.randn(h, w, 3) * 6
    # a "red cup" blob and a "plate" ellipse so the scene is not pure texture
    cy, cx = h // 2 + 10 * seed, w // 2
    img[(yy - cy) ** 2 + (xx - cx) ** 2 < 400] = (200, 30, 30)
    img[((yy - h * 0.75) / 20) ** 2 + ((xx - w * 0.3) / 40) ** 2 < 1] = (235, 235, 235)
    return np.clip(img, 0, 255).astype(np.uint8)
frame = np.concatenate([synth_view(a.img_h, a.img_w // 2, 0), synth_view(a.img_h, a.img_w // 2, 1)], axis=1)
Image.fromarray(frame).save(f"{a.out}/obs_frame.png")
npy("obs_frame_u8", frame)

t0 = time.time()
# `accelerate` is not installed, so from_pretrained (which needs low_cpu_mem_usage for the fp32 time_embedder) is bypassed:
# build the components by hand, directly from the checkpoint files (config + safetensors shards), then assemble the pipeline.
from diffusers import Cosmos3OmniTransformer, AutoencoderKLWan
from safetensors.torch import load_file
from transformers import AutoTokenizer
def build(cls, sub, fp32=False):
    cfg = cls.load_config(f"{a.model}/{sub}")
    with torch.device("meta"): m = cls.from_config(cfg)
    sd = {}
    for f in sorted(os.listdir(f"{a.model}/{sub}")):
        if f.startswith("diffusion_pytorch_model") and f.endswith(".safetensors"):
            for k, v in load_file(f"{a.model}/{sub}/{f}", device="cpu" if fp32 else "cuda").items():
                if fp32:
                    if k == "lm_head.weight": continue
                    v = v.float()
                sd[k] = v
    missing = m.load_state_dict(sd, strict=not fp32, assign=True)
    return m.eval()
transformer = build(Cosmos3OmniTransformer, "transformer", fp32=a.fp32)
if a.fp32: transformer.lm_head = torch.nn.Identity()   # unused by the policy (no text decoding)
# non-persistent rotary buffer is lost on the meta device: recompute exactly as Cosmos3VLTextRotaryEmbedding.__init__ does (CPU fp32), then move
transformer.rotary_emb.inv_freq = (1.0 / (transformer.config.rope_theta ** (torch.arange(0, transformer.config.head_dim, 2, dtype=torch.float32) / transformer.config.head_dim))).cuda()
np.save(f"{a.out}/inv_freq.npy", transformer.rotary_emb.inv_freq.cpu().numpy())
vae = build(AutoencoderKLWan, "vae")
tokenizer = AutoTokenizer.from_pretrained(f"{a.model}/text_tokenizer")
sched = UniPCMultistepScheduler.from_pretrained(f"{a.model}/scheduler")
pipe = Cosmos3OmniPipeline(transformer=transformer, text_tokenizer=tokenizer, vae=vae, scheduler=sched, enable_safety_checker=False,
                           default_use_system_prompt=False, use_native_flow_schedule=True)
print(f"load {time.time()-t0:.1f}s; vram after load {torch.cuda.memory_allocated()/2**30:.2f} GiB")
tr = pipe.transformer
torch.cuda.reset_peak_memory_stats()
base_mem = torch.cuda.memory_allocated()

cond = CosmosActionCondition(mode="policy", chunk_size=a.chunk, domain_name="droid_lerobot", resolution_tier=a.tier,
                             image=Image.fromarray(frame), view_point="concat_view")

# ---- noise: build the initial latents with the pipeline's own prepare_latents (cond frame = VAE(x0), rest = seeded noise) ----
gen = torch.Generator(device="cpu").manual_seed(a.seed)
num_frames = a.chunk + 1
probe = pipe.video_processor.preprocess_video([Image.fromarray(frame)])
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _ACTION_RESOLUTION_BINS
H, W = VideoProcessor.classify_height_width_bin(probe.shape[-2], probe.shape[-1], ratios=_ACTION_RESOLUTION_BINS[str(a.tier)])
_enc0 = pipe._encode_video
def _enc_save(x):   # what the VAE sees: frame 0 of the padded canvas video (the other 32 frames repeat the last frame), bf16-rounded, [-1,1]
    if not os.path.exists(f"{a.out}/cond_canvas.npy"): npy("cond_canvas", x[0, :, 0].to(torch.bfloat16))
    return _enc0(x)
pipe._encode_video = _enc_save
if a.fp32:
    _enc = pipe._encode_video; _cache = {}
    def _enc_cached(x):
        if "v" not in _cache: _cache["v"] = _enc(x)
        return _cache["v"]
    pipe._encode_video = _enc_cached
pre = pipe.prepare_latents(image=None, num_frames=num_frames, height=H, width=W, fps=a.fps, generator=gen, device="cuda",
                           dtype=torch.float32, action=cond)
latents0, _, action0, fps_v, _, vmask, _, amask, dom, imsize, rawdim, _ = pre
print("canvas", H, W, "latents", tuple(latents0.shape), "action", tuple(action0.shape), "domain", dom.tolist(), "raw_dim", rawdim, "image_size", imsize.tolist())
if a.fp32:   # the fp32 transformer was built on CPU: now that the VAE encode is done, swap VAE -> CPU and the transformer -> GPU (fits in 16 GB)
    pipe.vae.to("cpu"); torch.cuda.empty_cache(); tr.to("cuda")
# pure noise tensors alone (what a CUDA port draws), plus the VAE-encoded x0 latent for the cond frame
g2 = torch.Generator(device="cpu").manual_seed(a.seed)
from diffusers.utils.torch_utils import randn_tensor
noise_v = randn_tensor(tuple(latents0.shape), generator=g2, device="cuda", dtype=torch.float32)
noise_a = randn_tensor(tuple(action0.shape), generator=g2, device="cuda", dtype=torch.float32)
npy("noise_vision", noise_v); npy("noise_action", noise_a)
npy("latents_init", latents0); npy("action_init", action0); npy("vision_cond_mask", vmask.flatten())
# x0 (normalised VAE posterior mean of the conditioning frame) = latents_init[:, :, 0]
npy("x0_latent_frame0", latents0[:, :, 0])

# ---- instrumentation ----
call = {"n": 0}; layer_dumps = {}
def tag(): return f"s{call['n']//2 if a.guidance != 1.0 else call['n']}_{'cond' if (call['n']%2==0 or a.guidance==1.0) else 'uncond'}"
def pre_hook(mod, args):
    und, gen_s, rot = args
    if call["n"] < 2:
        t = tag()
        npy(f"{t}_layer_in_und", und); npy(f"{t}_layer_in_gen", gen_s)
        if call["n"] == 0:
            for nm, x in zip(("cos_und", "sin_und", "cos_gen", "sin_gen"), rot): npy(nm, x)
for i, lyr in enumerate(tr.layers):
    if i == 0: lyr.register_forward_pre_hook(pre_hook)
    def mk(i):
        def h(mod, args, out):
            if call["n"] < 2:
                t = tag(); npy(f"{t}_layer{i:02d}_und", out[0]); npy(f"{t}_layer{i:02d}_gen", out[1])
        return h
    lyr.register_forward_hook(mk(i))
def norm_hook(name):
    def h(mod, args, out):
        if call["n"] < 2: npy(f"{tag()}_{name}", out)
    return h
tr.norm.register_forward_hook(norm_hook("norm_und")); tr.norm_moe_gen.register_forward_hook(norm_hook("norm_gen"))
tr.embed_tokens.register_forward_hook(lambda m, ar, o: npy(f"{tag()}_embed_text", o) if call["n"] < 2 else None)
tr.proj_in.register_forward_hook(lambda m, ar, o: npy(f"{tag()}_proj_in", o) if call["n"] < 2 else None)
tr.action_proj_in.register_forward_hook(lambda m, ar, o: npy(f"{tag()}_action_proj_in", o) if call["n"] < 2 else None)
orig_fwd = tr.forward
def fwd(*args, **kw):
    t0 = time.time()
    out = orig_fwd(*args, **kw)
    torch.cuda.synchronize()
    t = tag()
    npy(f"{t}_pred_vision", out[0][0]); npy(f"{t}_pred_action", out[2][0])
    if call["n"] == 0:
        npy("position_ids_cond", kw["position_ids"])
        json.dump({k: (v if isinstance(v, (int, float)) else None) for k, v in kw.items() if k in ("und_len", "sequence_length")}, open(f"{a.out}/seq_meta.json", "w"))
        npy("vision_timesteps_s0", kw["vision_timesteps"])
    if call["n"] == 1 or a.guidance == 1.0 and call["n"] == 0:
        pass
    # keep the raw input tokens each call so CUDA can be fed step-exact inputs
    npy(f"{t}_in_vision_tokens", kw["vision_tokens"][0]); npy(f"{t}_in_action_tokens", kw["action_tokens"][0])
    print(f"call {call['n']} ({t}) transformer {time.time()-t0:.3f}s", flush=True)
    call["n"] += 1
    return out
tr.forward = fwd
# uncond ids: capture on first uncond call
orig_tok = pipe.tokenize_prompt
def tok(*args, **kw):
    c, u = orig_tok(*args, **kw)
    np.save(f"{a.out}/input_ids_cond.npy", np.array(c, dtype=np.int64)); np.save(f"{a.out}/input_ids_uncond.npy", np.array(u, dtype=np.int64))
    print("cond ids", len(c), "uncond ids", len(u)); print(pipe.text_tokenizer.decode(c))
    return c, u
pipe.tokenize_prompt = tok
sched_log = []
orig_step = UniPCMultistepScheduler.step
def sstep(self, model_output, timestep, sample, *args, **kw):
    out = orig_step(self, model_output, timestep, sample, *args, **kw)
    k = len(sched_log)
    which = "vision" if sample.numel() == latents0.numel() else "action"
    n = sum(1 for w, *_ in sched_log if w == which)
    npy(f"step{n}_{which}_velocity_cfg", model_output); npy(f"step{n}_{which}_sample_in", sample); npy(f"step{n}_{which}_sample_out", out[0])
    sched_log.append((which, float(timestep)))
    return out
UniPCMultistepScheduler.step = sstep

torch.cuda.synchronize(); t0 = time.time()
res = pipe(prompt=a.prompt, action=cond, num_inference_steps=a.steps, guidance_scale=a.guidance, fps=a.fps,
           latents=latents0, action_latents=action0, output_type="latent" if not a.decode else "np",
           enable_safety_check=False, return_dict=True)
torch.cuda.synchronize(); dt = time.time() - t0
peak = torch.cuda.max_memory_allocated()
act = res.action[0]
npy("final_action", act)
vid = res.video
if not a.decode: npy("final_video_latent", vid)
else: npy("decoded_video", vid)
json.dump({"timesteps": [float(x) for x in pipe.scheduler.timesteps], "sigmas": [float(x) for x in pipe.scheduler.sigmas],
           "steps": a.steps, "guidance": a.guidance, "fps": a.fps, "chunk": a.chunk, "canvas_hw": [H, W], "raw_action_dim": rawdim,
           "domain_id": int(dom.item()), "latent_shape": list(latents0.shape), "seed": a.seed, "prompt": a.prompt,
           "image_size_hwchw": imsize.tolist(), "sched_log": sched_log}, open(f"{a.out}/meta.json", "w"), indent=1)
print(f"pipeline call {dt:.2f}s (steps {a.steps}, cfg {a.guidance}); torch peak allocated {peak/2**30:.2f} GiB (weights {base_mem/2**30:.2f} GiB)")
print("action chunk:", tuple(act.shape)); print(act[:4].numpy().round(4)); print("...", act[-1].numpy().round(4))
