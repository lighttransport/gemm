#!/usr/bin/env python3
"""Reference dumps from the REAL praxis-1-droid checkpoint (54 GB bf16) on a 16 GB GPU.

The model never lives in memory as a whole. It is built on the meta device (like
PraxisPolicy.from_pretrained), the small parts (input/output projections, modulators, registers,
timestep embedders, the 16-block action expert; ~1.2 GB) are loaded to the GPU, and each of the
20 big transformer blocks (14 encoder + 6 decoder, 2-3 GB each) is read from the safetensors shards
just before its forward (forward-pre-hook; the next block is prefetched by a thread) and dropped
(back to meta) right after. The math is exactly the original's: same modules, same bf16 kernels, same
sampler (PraxisPolicy.sample_window_joint with a trace); offload only moves the weights.

Plain-PyTorch reference paths only: every Triton fusion flag of praxis.layers / action_expert is
switched off (--fused keeps the defaults), MoE uses the stock sort+grouped dispatch, attention is
flex-attention (sm_120 -> triton backend).  --fp32 computes in float32 with the bf16-exact weights
upcast per block (tiny/mid-style oracle).

Pipeline (same as praxis.pipeline.Session, minus the H100-only kernels):
  synthetic 2-view DROID-like observation (stack of 22 frames of 234x416, fit rule -> 224x416 tiles hstacked
  to 224x832)  ->  Session state machine (SingleArmServeSession.prepare) with the Wan2.2 VAE encode_eager
  ->  Qwen3.5-2B text (hidden_states[-3], padded to 32, trimmed; blank ' ' for uncond)
  ->  sample_window_joint (2 Euler steps, t_shift 12.5, cfg_video 4.0, cfg 1.0, seed 0 + chunk_index)
call 1 = depth 0 (K'=1; attempted, may hit the reference K'=1 bug), call 2 = depth 1 (K'=2, the main dump).

Outputs (--out, default tmp/praxis_ref/real):
  obs_*.npy/png, canvas_call{1,2}.npy          observation + tile canvas (uint8 [T][3][224][832])
  text.npy (1,S,2048) text_u.npy ids.npy ids_u.npy text_fp32.npy        Qwen conditioning
  call2 root (K'=2):  in_*.npy, enc_embed, enc_b{i}_{attn,ffn,shared,out}, enc_b{i}_router_logits/top_idx/gate (MoE),
        enc_tap{j}, enc_cond, enc_rope_ids, dec_embed, dec_mod{i}, dec_b{i}_*, dec_vel, ae_x0, ae_b{i}_out, ae_out
        (full detail of pass 0 = step 0 cond; other --detail-passes get a s{step}{c|u}_ prefix)
        smp_<trace>.<n>.npy (noise.salt0/1, velocity.in/out, cfg_combine cond/uncond/out, out.*, time_grid),
        pred_video.npy (1,192,6,h,w) pred_action.npy (1,6,40) action_rows_norm.npy (24,10) action_rel.npy (24,10)
        commands_pose.npy (24,12) commands_grip.npy (24,) proprio_all.npy meta.json
  call1/ same for the K'=1 call.
Stages are cached: a finished stage leaves <out>/.done_<stage>; delete it (absolute path!) to redo.
Run (repo root):  ~/.venv/bin/python -I ref/praxis/gen_real_ref.py            (-I needs PYTHONPATH-free: praxis src added by --praxis-src)
"""
import argparse, json, os, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
PRAXIS_ROOT = os.environ.get("PRAXIS_ROOT", os.path.expanduser("~/praxis"))

ap = argparse.ArgumentParser()
ap.add_argument("--praxis-src", default=PRAXIS_ROOT + "/praxis-1-oss")
ap.add_argument("--ckpt", default=PRAXIS_ROOT + "/praxis-1-droid")
ap.add_argument("--qwen", default=PRAXIS_ROOT + "/qwen3.5-2b")
ap.add_argument("--vae", default=PRAXIS_ROOT + "/wan2.2-ti2v/Wan2.2_VAE.pth")
ap.add_argument("--out", default="tmp/praxis_ref/real")
ap.add_argument("--prompt", default="pick up the red cup and place it on the plate")
ap.add_argument("--device", default="cuda")
ap.add_argument("--fp32", action="store_true", help="compute in float32 (weights bf16-exact, upcast per block)")
ap.add_argument("--fused", action="store_true", help="keep praxis Triton fusion defaults (same-bits claims) instead of plain paths")
ap.add_argument("--calls", default="2,1", help="which calls to sample, in order (2 = K'=2 main, 1 = K'=1 attempt)")
ap.add_argument("--detail-passes", default="0", help="passes (step*2+branch) with full per-block dumps (call 2 root, others prefixed)")
ap.add_argument("--no-prefetch", action="store_true")
ap.add_argument("--selftest", default=None, help="checkpoint dir of a small model (tiny/mid): compare offloaded run to a resident run, bit-exact")
ap.add_argument("--tag", default="", help="suffix appended to --out (e.g. _fp32)")
ap.add_argument("--obs-npz", default=None, help="observation stack (22 rows) of a sim call (exterior, wrist, ee_pose, grip) instead of the synthetic one")
ap.add_argument("--descriptor", default=None, help="observation descriptor (e.g. praxis.sim)")
ap.add_argument("--stages", default="obs,text,inputs,sample")
a = ap.parse_args()
sys.path.insert(0, a.praxis_src)
a.out = a.out + a.tag
os.makedirs(a.out, exist_ok=True)
dev = torch.device(a.device)
CDT = torch.float32 if a.fp32 else torch.bfloat16
T0 = time.time()
def log(*m):
    print(f"[{time.time() - T0:7.1f}s | gpu peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB]", *m, flush=True)
def npy(path, t):
    if torch.is_tensor(t): t = t.detach().float().cpu().numpy()
    np.save(path, t)
def done(stage): return os.path.exists(f"{a.out}/.done_{stage}")
def mark(stage): open(f"{a.out}/.done_{stage}", "w").write(time.strftime("%F %T"))

from praxis import layers as L, action_expert as AE, model as M
from praxis.model import PraxisPolicy, SamplerSettings
from praxis.checkpoint import load_config, weight_map
from praxis.sampler import seeded_normal, window_time_grid
from praxis import single_arm as SA
from praxis.robot_io import NormStats, text_encoder_prompt
from safetensors import safe_open

if not a.fused:   # plain eager reference paths: no Triton fusions, no fused MoE
    L.FUSED_NORM = False; L.FUSED_ROPE = False; L.FUSED_ROPE_QK_NORM = False; L.FUSED_GATE = False
    L.ROUTER_FUSED = False; L.FUSED_MOD_NORM = False; L.SHARED_SILU_MUL = False
    L.FUSED_MOE = False; L.FUSED_MOE_DEC = False; L.FUSED_MOE_SOLO = False; L.CUTE_EXPERTS = False
    L.CUTE_DENSE_FFN = False; L.MOE_TAIL_NORM = False; L.PAIR_DEC = False
    AE.FUSED_AE_NORM = False; AE.FUSED_KV_SOURCE = False

# ---------------------------------------------------------------------------------------------------
# Offload: lazy per-block weights
# ---------------------------------------------------------------------------------------------------
class Offload:
    """Meta-built policy; resident small parts on GPU; encoder/decoder blocks streamed per forward."""
    def __init__(self, ckpt, device, cdt, prefetch=True):
        self.ckpt, self.dev, self.cdt = ckpt, device, cdt
        self.config = load_config(ckpt)
        with torch.device("meta"):
            self.model = PraxisPolicy.from_config(self.config)
        self.wm = weight_map(ckpt)
        self.blocks = []   # (prefix, module) in cyclic call order
        for tower in ("encoder", "decoder"):
            for i, b in enumerate(getattr(self.model.backbone, tower).core.blocks):
                self.blocks.append((f"backbone.{tower}.core.blocks.{i}.", b))
        self.block_ids = {id(b): k for k, (_, b) in enumerate(self.blocks)}
        self.pool = ThreadPoolExecutor(1) if prefetch else None
        self.pending = {}
        self.load_s = 0.0; self.load_bytes = 0; self.n_loads = 0; self.n_miss = 0
        self._load_resident()
        for k, (prefix, b) in enumerate(self.blocks):
            b.register_forward_pre_hook(lambda m, i, k=k: self._acquire(k))   # registered first: runs before dump hooks
        self._release_hooks_installed = False

    def _tensors(self, prefix, names):
        by_file = {}
        for n in names: by_file.setdefault(self.wm[prefix + n], []).append(n)
        out = {}
        for f, ns in by_file.items():
            with safe_open(os.path.join(self.ckpt, f), framework="pt", device="cpu") as s:
                for n in ns: out[n] = s.get_tensor(prefix + n)
        return out

    def _slots(self, module):
        """[(submodule, kind, attr, local name)] of all params/buffers of a module tree."""
        r = []
        for mn, sub in module.named_modules():
            for pn in list(sub._parameters):
                if sub._parameters[pn] is not None: r.append((sub, "p", pn, (mn + "." if mn else "") + pn))
            for bn in list(sub._buffers):
                if sub._buffers[bn] is not None: r.append((sub, "b", bn, (mn + "." if mn else "") + bn))
        return r

    def _cvt(self, t):
        return t.to(self.dev, self.cdt) if (self.cdt != torch.bfloat16 and t.dtype == torch.bfloat16) else t.to(self.dev)

    def _install(self, slots, tensors):
        for sub, kind, attr, name in slots:
            t = tensors[name]
            if kind == "p": sub._parameters[attr] = torch.nn.Parameter(t, requires_grad=False)
            else: sub._buffers[attr] = t

    def _load_resident(self):
        blk_prefixes = tuple(p for p, _ in self.blocks)
        slots = [s for s in self._slots(self.model)]
        # full names relative to the model root
        names = {}
        for mn, sub in self.model.named_modules():
            for pn in list(sub._parameters) + list(sub._buffers):
                names[(id(sub), pn)] = (mn + "." if mn else "") + pn
        n = 0
        for sub, kind, attr, local in slots:
            full = names[(id(sub), attr)]
            if full.startswith(blk_prefixes): continue
            if full not in self.wm:
                log(f"WARNING: no checkpoint tensor for {full} (non-persistent buffer?) shape={tuple((sub._parameters.get(attr) if kind == 'p' else sub._buffers[attr]).shape)}"); continue
            with safe_open(os.path.join(self.ckpt, self.wm[full]), framework="pt", device="cpu") as s:
                t = self._cvt(s.get_tensor(full))
            if kind == "p": sub._parameters[attr] = torch.nn.Parameter(t, requires_grad=False)
            else: sub._buffers[attr] = t
            n += t.numel() * t.element_size()
        log(f"resident parts on GPU: {n / 2**30:.2f} GiB")

    def _read_block(self, k):
        prefix, b = self.blocks[k]
        slots = self._slots(b)
        t0 = time.time()
        cpu = self._tensors(prefix, [s[3] for s in slots])
        return slots, cpu, time.time() - t0

    def _acquire(self, k):
        fut = self.pending.pop(k, None)
        t0 = time.time()
        if fut is not None: slots, cpu, _ = fut.result()
        else:
            slots, cpu, _ = self._read_block(k); self.n_miss += 1
        if self.pool is not None:
            nxt = (k + 1) % len(self.blocks)
            if nxt not in self.pending: self.pending[nxt] = self.pool.submit(self._read_block, nxt)
        tens = {name: self._cvt(cpu[name]) for _, _, _, name in slots}
        self._install(slots, tens)
        torch.cuda.synchronize()
        self.load_s += time.time() - t0; self.n_loads += 1
        self.load_bytes += sum(t.numel() * t.element_size() for t in tens.values())

    def release(self, k):
        _, b = self.blocks[k]
        for sub, kind, attr, _ in self._slots(b):
            t = sub._parameters[attr] if kind == "p" else sub._buffers[attr]
            m = torch.empty(t.shape, dtype=t.dtype, device="meta")
            if kind == "p": sub._parameters[attr] = torch.nn.Parameter(m, requires_grad=False)
            else: sub._buffers[attr] = m

    def install_release_hooks(self):
        for k, (_, b) in enumerate(self.blocks):
            b.register_forward_hook(lambda m, i, o, k=k: self.release(k))   # registered last: after dump hooks


# ---------------------------------------------------------------------------------------------------
# Dump hooks
# ---------------------------------------------------------------------------------------------------
class Dumper:
    def __init__(self, pol: PraxisPolicy, root):
        self.pol, self.root, self.on, self.prefix = pol, root, False, ""
        self.detail_passes = {int(x) for x in a.detail_passes.split(",") if x != ""}
        self.first_detail = min(self.detail_passes) if self.detail_passes else -1
        self.pass_idx = -1
        self.tr = {}
        bb = pol.backbone
        enc, dec, ae = bb.encoder, bb.decoder, pol.action_expert
        for tag, tower in (("enc", enc), ("dec", dec)):
            for i, b in enumerate(tower.core.blocks):
                b.register_forward_hook(self._out(f"{tag}_b{i}_out"))
                b.mha.register_forward_hook(self._out(f"{tag}_b{i}_attn"))
                b.ffn.register_forward_hook(self._out(f"{tag}_b{i}_ffn"))
                if isinstance(b.ffn, L.MoEFeedForward):
                    b.ffn.shared.register_forward_hook(self._out(f"{tag}_b{i}_shared"))
                    b.ffn.routing_hook = self._router(f"{tag}_b{i}")   # always on: forces the stock routing path in every pass
                if i == 0:
                    b.register_forward_pre_hook(lambda m, inp, t=tag: self._save(f"{t}_embed", inp[0]))
            if tower.core.modulators is not None:
                for i, m in enumerate(tower.core.modulators):
                    m.register_forward_hook(self._out(f"{tag}_mod{i}"))
        for i, b in enumerate(ae.blocks):
            b.register_forward_hook(self._out(f"ae_b{i}_out"))
            if i == 0: b.register_forward_pre_hook(lambda m, inp: self._save("ae_x0", inp[0]))
        # towers / expert entry points
        orig_enc = enc.forward_chunk_causal
        def enc_fwd(*args, **kw):
            cond, taps = orig_enc(*args, **kw)
            if self.on:
                self._save("enc_cond", cond); self._save("enc_rope_ids", taps.rope_ids)
                for j, t in enumerate(taps.taps): self._save(f"enc_tap{j}", t)
            return cond, taps
        enc.forward_chunk_causal = enc_fwd
        orig_dec = dec.forward_chunk_causal
        def dec_fwd(*args, **kw):
            v = orig_dec(*args, **kw)
            if self.on and v is not None: self._save("dec_vel", v)
            return v
        dec.forward_chunk_causal = dec_fwd
        orig_exp = M.run_expert_on_taps_k
        def exp_fwd(*args, **kw):
            y = orig_exp(*args, **kw)
            if self.on: self._save("ae_out", y)
            return y
        M.run_expert_on_taps_k = exp_fwd
        self._restore = lambda: setattr(M, "run_expert_on_taps_k", orig_exp)

    def _save(self, name, t):
        if self.on:
            npy(f"{self.root}/{self.prefix}{name}.npy", t if torch.is_tensor(t) else t[0])
    def _out(self, name):
        def f(m, i, o): self._save(name, o if torch.is_tensor(o) else o[0])
        return f
    def _router(self, name):
        def f(layer, logits, top_idx):
            if not self.on: return
            sc = torch.sigmoid(logits)
            picked = sc.gather(-1, top_idx); gate = picked / picked.sum(-1, keepdim=True)
            self._save(f"{name}_router_logits", logits); self._save(f"{name}_top_idx", top_idx.float()); self._save(f"{name}_gate", gate)
            self._save(f"{name}_expert_bias", layer.expert_bias)
        return f

    def trace(self, name, t):
        n = self.tr.get(name, 0); self.tr[name] = n + 1
        npy(f"{self.root}/smp_{name}.{n}.npy", t)
        if name == "pass.marker":
            step, br = int(t[0]), int(t[1]); self.pass_idx = step * 2 + br
            self.on = self.pass_idx in self.detail_passes
            self.prefix = "" if self.pass_idx == self.first_detail else f"s{step}{'cu'[br]}_"
            log(f"pass {self.pass_idx} (step {step}, {'uncond' if br else 'cond'}) detail={self.on}")


# ---------------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------------
def make_observation(T=22, H=234, W=416, seed=2024):
    """Smooth seeded DROID-like stacks: exterior + wrist (T,H,W,3) uint8, pose (T,7) wxyz, grip (T,)."""
    def view(s):
        rs = np.random.RandomState(s)
        key = torch.from_numpy(rs.rand(1, 3, 4, 6, 10).astype(np.float32))                # 4 key times, coarse grid
        base = torch.nn.functional.interpolate(key, size=(T, H, W), mode="trilinear", align_corners=True)[0]   # (3,T,H,W)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
        for t in range(T):                                                                  # moving blob + gradient
            cx, cy = 0.3 + 0.4 * t / (T - 1), 0.5 + 0.1 * np.sin(t / 3)
            blob = torch.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / 0.01))
            base[:, t] = 0.6 * base[:, t] + 0.25 * (xx + yy)[None] / 2 + 0.35 * blob[None] * torch.tensor([1.0, 0.2, 0.1])[:, None, None]
        return (base.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 3, 0).numpy()
    rs = np.random.RandomState(seed + 7)
    t = np.linspace(0, 1, T)[:, None]
    xyz = np.array([0.52, 0.0, 0.32])[None] + 0.05 * np.stack([np.sin(2 * t[:, 0]), np.cos(3 * t[:, 0]) - 1, 0.5 * t[:, 0]], 1)
    ang = 0.15 * t[:, 0]
    q = np.stack([np.cos(ang / 2), np.zeros(T), np.zeros(T), np.sin(ang / 2)], 1)           # wxyz, yaw rotation
    q0 = np.array([0.0, 1.0, 0.0, 0.0])                                                      # flange pointing down (180 deg about x)
    def qmul(p, r):
        pw, px, py, pz = p.T; rw, rx, ry, rz = r.T
        return np.stack([pw*rw-px*rx-py*ry-pz*rz, pw*rx+px*rw+py*rz-pz*ry, pw*ry-px*rz+py*rw+pz*rx, pw*rz+px*ry-py*rx+pz*rw], 1)
    qq = qmul(q, np.broadcast_to(q0, q.shape)); qq /= np.linalg.norm(qq, axis=1, keepdims=True)
    pose = np.concatenate([xyz, qq], 1).astype(np.float32)
    grip = np.clip(0.4 * np.linspace(0, 1, T) ** 2, 0, 1).astype(np.float32)
    return view(seed + 1), view(seed + 2), pose, grip

def stage_obs():
    if a.obs_npz:
        z = np.load(a.obs_npz); ext, wr, pose, grip = z["exterior"], z["wrist"], z["ee_pose"], z["grip"]
    else:
        ext, wr, pose, grip = make_observation()
    np.save(f"{a.out}/obs_exterior.npy", ext); np.save(f"{a.out}/obs_wrist.npy", wr)
    np.save(f"{a.out}/obs_ee_pose.npy", pose); np.save(f"{a.out}/obs_grip.npy", grip)
    from PIL import Image
    for t in (0, ext.shape[0] - 1):
        Image.fromarray(np.concatenate([ext[t], wr[t]], 1)).save(f"{a.out}/obs_frame{t}.png")
    log("obs stage done", ext.shape)

def load_obs():
    return dict(exterior=np.load(f"{a.out}/obs_exterior.npy"), wrist=np.load(f"{a.out}/obs_wrist.npy"),
                pose=np.load(f"{a.out}/obs_ee_pose.npy"), grip=np.load(f"{a.out}/obs_grip.npy"))

def stage_text():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from praxis.text_encoder import PAD_TO_MULTIPLE_OF
    proc = AutoProcessor.from_pretrained(a.qwen, local_files_only=True)
    full = AutoModelForImageTextToText.from_pretrained(a.qwen, local_files_only=True, dtype=torch.bfloat16)  # NOT AutoModel (re-inits weights)
    qm = full.model.language_model.to(dev).eval(); del full
    def enc(prompt, dtype):
        qm.to(dtype)
        b = proc(text=[text_encoder_prompt(prompt)], padding=True, truncation=False, add_special_tokens=True, return_attention_mask=True,
                 pad_to_multiple_of=PAD_TO_MULTIPLE_OF, return_tensors="pt")
        with torch.no_grad():
            hs = qm(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev), output_hidden_states=True, use_cache=False).hidden_states[-3]
        n = int(b["attention_mask"].sum())
        return hs[:, :n].cpu(), b["input_ids"][0, :n].numpy().astype(np.int64), b["input_ids"].shape[1]
    t, ids, pad = enc(a.prompt, torch.bfloat16); tu, idsu, _ = enc(" ", torch.bfloat16)
    log(f"text tokens {len(ids)} (padded to {pad}), uncond {len(idsu)}")
    torch.save(dict(text=t, text_u=tu), f"{a.out}/text.pt")
    npy(f"{a.out}/text.npy", t); npy(f"{a.out}/text_u.npy", tu); np.save(f"{a.out}/ids.npy", ids); np.save(f"{a.out}/ids_u.npy", idsu)
    t32, _, _ = enc(a.prompt, torch.float32); tu32, _, _ = enc(" ", torch.float32)
    npy(f"{a.out}/text_fp32.npy", t32); npy(f"{a.out}/text_u_fp32.npy", tu32)
    log(f"text bf16 vs fp32 rel-L2: {((t.float() - t32).norm() / t32.norm()).item():.4e}  uncond {((tu.float() - tu32).norm() / tu32.norm()).item():.4e}")
    del qm; torch.cuda.empty_cache()

def stage_inputs():
    from praxis.vae import Autoencoder
    cfg = load_config(a.ckpt)
    v = cfg["video"]
    vae = Autoencoder.from_pretrained(a.vae, device=dev, decoder_impl="reference")
    norm = cfg["normalization"]
    stats = NormStats(*(np.asarray(norm[k], dtype=np.float32) for k in ("action_q01", "action_q99", "proprio_mean", "proprio_std")))
    st = SA.SingleArmServeSession(encode=vae.encode_eager, stats=stats, device=dev, prompt=a.prompt, max_context_chunks=1, fps=15.0,
                                  dims=SA.q99_dims(cfg), control_mode="droid_impedance", exec_horizon=24, n_chunks=int(cfg["chunking"]["n_chunks"]),
                                  tile_size=(v["tile_height"], v["tile_width"]), aspect_tol=v["aspect_tol"], antialias=v["antialias"])
    o = load_obs()
    def obs(rows):   # a stack whose last row is the current frame
        return {"ee_pose": o["pose"][:rows], "grip": o["grip"][:rows], "exterior_image": o["exterior"][:rows], "wrist_image": o["wrist"][:rows],
                "task": a.prompt, **({"descriptor": a.descriptor} if a.descriptor else {})}
    for call, rows in ((1, 1), (2, 22)):
        inp = st.prepare(obs(rows))
        d = f"{a.out}/inputs_call{call}"; os.makedirs(d, exist_ok=True)
        npy(f"{d}/anchor.npy", inp.anchor)
        for i, c in enumerate(inp.cleans): npy(f"{d}/clean{i}.npy", c)
        npy(f"{d}/proprio_all.npy", inp.proprio_all); np.save(f"{d}/boundary_poses.npy", inp.boundary_poses); np.save(f"{d}/pose8_now.npy", inp.pose8_now)
        torch.save(dict(anchor=inp.anchor.cpu(), cleans=[c.cpu() for c in inp.cleans], proprio_all=inp.proprio_all.cpu(), depth=inp.depth,
                        pose8_now=inp.pose8_now, chunk_index=inp.chunk_index, prompt=inp.prompt, boundary_poses=inp.boundary_poses), f"{d}/inputs.pt")
        # canvas of the encoded window, for the runner's VAE parity: tiles of the frames this call encoded
        if st.stream._clip is not None:
            np.save(f"{a.out}/canvas_call{call}.npy", st.stream._clip[0].permute(1, 0, 2, 3).to(torch.uint8).cpu().numpy())
        cur = SA.current_clip(obs(rows), dev, st.tile_size, st.aspect_tol, st.antialias)
        np.save(f"{a.out}/canvas_cur_call{call}.npy", cur[0].permute(1, 0, 2, 3).to(torch.uint8).cpu().numpy())
        log(f"call {call}: depth {inp.depth}, anchor {tuple(inp.anchor.shape)}, cleans {[tuple(c.shape) for c in inp.cleans]}, proprio_all {tuple(inp.proprio_all.shape)}")
        if call == 1:   # the session needs the previous call's actions only for chunk_index; advance it like finish()
            st.chunk_index += 1
    from PIL import Image
    cv = np.load(f"{a.out}/canvas_cur_call2.npy")[0].transpose(1, 2, 0)
    Image.fromarray(cv).save(f"{a.out}/canvas_cur_call2.png")
    # VAE latent shapes for the runner
    del vae; torch.cuda.empty_cache()

DRV = None
def sample_call(call, off: Offload, drv_hooks_installed=None):
    root = a.out if call == 2 else f"{a.out}/call{call}"
    os.makedirs(root, exist_ok=True)
    tin = torch.load(f"{a.out}/inputs_call{call}/inputs.pt", weights_only=False)
    tx = torch.load(f"{a.out}/text.pt", weights_only=False)
    text, text_u = tx["text"].to(dev, CDT), tx["text_u"].to(dev, CDT)
    cfg = off.config
    s = load_yaml_sampler()
    seed = 0 + tin["chunk_index"]
    cleans = [c.to(dev) for c in tin["cleans"]]; anchor = tin["anchor"].to(dev)
    proprio_all = tin["proprio_all"].to(dev)
    kk = len(cleans) + 1; lat = int(cfg["chunking"]["latents_per_chunk"])
    c, h, w = anchor.shape[1], anchor.shape[3], anchor.shape[4]
    # explicit inputs, same names as the tiny dumps
    nz = seeded_normal((1, c, kk * lat, h, w), seed, 0); na = seeded_normal((kk * lat, int(cfg["action_expert"]["action_patch_dim"])), seed, 1)
    npy(f"{root}/in_text.npy", text); npy(f"{root}/in_text_u.npy", text_u); npy(f"{root}/in_anchor.npy", anchor)
    if cleans: npy(f"{root}/in_clean.npy", torch.cat(cleans, 2))
    npy(f"{root}/in_noised.npy", nz); npy(f"{root}/in_noised_a.npy", na[None]); npy(f"{root}/in_proprio.npy", proprio_all)
    npy(f"{root}/in_t_trunk.npy", torch.full((1, kk), window_time_grid(s.num_steps, s.t_shift)[0]))
    global DRV
    if DRV is None:
        DRV = Dumper(off.model, root); off.install_release_hooks()
    DRV.root, DRV.tr, DRV.pass_idx, DRV.on, DRV.prefix = root, {}, -1, False, ""
    drv = DRV
    t0 = time.time(); off.load_s = 0; off.load_bytes = 0; off.n_loads = 0
    log(f"== call {call} (K'={kk}, seed {seed}) sampling ...")
    try:
        with torch.no_grad():
            v_last, a_last = off.model.sample_window_joint(text, text_u, anchor, cleans, proprio_all.float(), seed, s, trace=drv.trace,
                                                           ctx_cache=False, overlap_expert=False, skip_last_video=False)
    finally:
        drv.on = False
    wall = time.time() - t0
    npy(f"{root}/pred_video.npy", v_last); npy(f"{root}/pred_action.npy", a_last)
    npy(f"{root}/smp_out_video.npy", v_last); npy(f"{root}/smp_out_action.npy", a_last)   # tiny-reference names (last chunk)
    stats = NormStats(*(np.asarray(cfg["normalization"][k], dtype=np.float32) for k in ("action_q01", "action_q99", "proprio_mean", "proprio_std")))
    rows, rel = SA.decode_actions(a_last, stats, tin["depth"], SA.q99_dims(cfg))
    np.save(f"{root}/action_rows_norm.npy", rows); np.save(f"{root}/action_rel.npy", rel)
    wire = SA.rel10_to_wire(rel, tin["pose8_now"], 1.0 / 15.0, SA.control_mode_payload("droid_impedance"), SA.FR3_HAND)
    poses, grips = SA.wire_to_arrays(wire)
    np.save(f"{root}/commands_pose.npy", poses); np.save(f"{root}/commands_grip.npy", grips)
    json.dump(dict(call=call, kk=kk, seed=seed, depth=tin["depth"], cfg_video=s.cfg_video_scale, cfg_action=s.cfg_scale, num_steps=s.num_steps,
                   t_shift=s.t_shift, time_grid=window_time_grid(s.num_steps, s.t_shift), compute_dtype=str(CDT), plain_paths=not a.fused,
                   wall_s=wall, weight_load_s=off.load_s, weight_loaded_GiB=off.load_bytes / 2**30, block_loads=off.n_loads, prefetch_misses=off.n_miss,
                   gpu_peak_GiB=torch.cuda.max_memory_allocated() / 2**30, detail_passes=sorted(drv.detail_passes), prompt=a.prompt,
                   pred_video_shape=list(v_last.shape), pred_action_shape=list(a_last.shape)),
              open(f"{root}/meta.json", "w"), indent=1)
    log(f"call {call} done in {wall:.0f}s; weight streaming {off.load_s:.0f}s / {off.load_bytes / 2**30:.0f} GiB; video {tuple(v_last.shape)} action {tuple(a_last.shape)}")

def load_yaml_sampler():
    return SamplerSettings(num_steps=2, t_shift=12.5, cfg_scale=1.0, cfg_video_scale=4.0)   # configs/inference_droid.yaml

# ---------------------------------------------------------------------------------------------------
def selftest():
    """Offloaded run == resident run (bit-exact) on a small checkpoint."""
    ck = a.selftest
    cfg = load_config(ck)
    cdt = torch.bfloat16
    res = PraxisPolicy.from_pretrained(ck, device=str(dev)).to(cdt)
    kk, lat = 2, int(cfg["chunking"]["latents_per_chunk"])
    g = torch.Generator().manual_seed(5)
    c_in = cfg["backbone"]["encoder"]["input_projections"]["TextEncodingTensor"]["c_in"]
    h, w = 4, 6
    text = torch.randn(1, 5, c_in, generator=g).to(dev, cdt); text_u = torch.randn(1, 3, c_in, generator=g).to(dev, cdt)
    anchor = torch.randn(1, 192, 1, h, w, generator=g).to(dev, cdt); clean = torch.randn(1, 192, lat, h, w, generator=g).to(dev, cdt)
    pro = torch.randn(1, kk, 10, generator=g).to(dev)
    s = load_yaml_sampler()
    kwargs = dict(trace=None, ctx_cache=False, overlap_expert=False)
    with torch.no_grad():
        v0, a0 = res.sample_window_joint(text, text_u, anchor, [clean], pro, 3, s, **kwargs)
    del res; torch.cuda.empty_cache()
    off = Offload(ck, dev, cdt, prefetch=not a.no_prefetch); off.install_release_hooks()
    with torch.no_grad():
        v1, a1 = off.model.sample_window_joint(text, text_u, anchor, [clean], pro, 3, s, **kwargs)
    log(f"selftest: video max|diff| {(v0 - v1).abs().max().item():.3e}  action max|diff| {(a0 - a1).abs().max().item():.3e}  "
        f"(video absmax {v0.abs().max().item():.3f})  loads {off.n_loads} misses {off.n_miss}")
    # leftover meta tensors?
    bad = [n for n, p in list(off.model.named_parameters()) + list(off.model.named_buffers()) if p.is_meta and ".core.blocks." not in n]
    log("meta tensors outside blocks:", bad[:5])

if a.selftest:
    selftest(); sys.exit(0)

stages = a.stages.split(",")
if "obs" in stages and not done("obs"): stage_obs(); mark("obs")
if "text" in stages and not done("text"): stage_text(); mark("text")
if "inputs" in stages and not done("inputs"): stage_inputs(); mark("inputs")
if "sample" in stages:
    off = Offload(a.ckpt, dev, CDT, prefetch=not a.no_prefetch)
    log("model built (meta) + resident parts loaded")
    hooks = False
    for call in [int(x) for x in a.calls.split(",")]:
        if done(f"sample_call{call}"): log(f"call {call} cached"); continue
        try:
            sample_call(call, off, hooks); hooks = True; mark(f"sample_call{call}")
        except Exception:
            hooks = True
            err = traceback.format_exc(); log(f"call {call} FAILED:\n{err}")
            os.makedirs(f"{a.out}/call{call}" if call != 2 else a.out, exist_ok=True)
            open(f"{a.out}/call{call}_error.txt", "w").write(err)
log("all done")
