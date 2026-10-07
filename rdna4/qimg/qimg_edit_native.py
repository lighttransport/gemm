"""Qwen-Image-Edit-2511 with the native RDNA4 INT4 W4A16 DiT (libhip_qimg.so) under diffusers.

diffusers' QwenImageEditPlusPipeline keeps prompt/vision encoding (Qwen2.5-VL), VAE and the
scheduler on the CPU; only the 60-block DiT runs natively on the GPU, fully resident in INT4.
The diffusers transformer is an empty meta-device shell whose forward() calls
hip_qimg_set_edit_layout + hip_qimg_dit_step (see EDIT_PORT_PLAN.md).

    pipe = load_pipeline('/mnt/disk01/models/qwen-image-edit-2511/base', 'edit2511-int4.safetensors')
    image = pipe(image=[portrait, target], prompt=..., height=1024, width=1024, ...).images[0]
"""
import ctypes
import os
from pathlib import Path
import time

import numpy as np
import torch

LIB = Path(__file__).resolve().parent / 'libhip_qimg.so'

# No shipped MIOpen find-db for gfx1201 here ("fdb.txt unreadable"): every new process re-searched the VAE's
# conv3d kernels. Persist MIOpen's user db / kernel cache and use the fast find mode.
_MIOPEN_CACHE = Path(os.environ.get('QIMG_MIOPEN_CACHE', '/mnt/disk01/tmp/miopen-cache'))
os.environ.setdefault('MIOPEN_USER_DB_PATH', str(_MIOPEN_CACHE / 'db'))
os.environ.setdefault('MIOPEN_CUSTOM_CACHE_DIR', str(_MIOPEN_CACHE / 'kernels'))
os.environ.setdefault('MIOPEN_FIND_MODE', 'FAST')
for _d in ('db', 'kernels'):
    (_MIOPEN_CACHE / _d).mkdir(parents=True, exist_ok=True)


class NativeDiT:
    def __init__(self, int4_path, device=0, verbose=1):
        self.lib = ctypes.CDLL(str(LIB))
        L = self.lib
        L.hip_qimg_init.restype = ctypes.c_void_p
        L.hip_qimg_init.argtypes = [ctypes.c_int, ctypes.c_int]
        L.hip_qimg_load_dit_int4.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.hip_qimg_set_edit_layout.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        fp = ctypes.POINTER(ctypes.c_float)
        L.hip_qimg_dit_step.argtypes = [ctypes.c_void_p, fp, ctypes.c_int, fp, ctypes.c_int, ctypes.c_float, fp]
        L.hip_qimg_free.argtypes = [ctypes.c_void_p]
        L.hip_qimg_set_mod_vectors.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_float)] * 3
        self.r = L.hip_qimg_init(device, verbose)
        if not self.r:
            raise RuntimeError('hip_qimg_init failed')
        if L.hip_qimg_load_dit_int4(self.r, str(int4_path).encode()) != 0:
            raise RuntimeError('hip_qimg_load_dit_int4 failed: ' + str(int4_path))
        self.layout = None
        self.seconds = 0.
        self.calls = 0

    def step(self, tokens, txt, t1000, img_shapes, zero_cond_t):
        """tokens [N,64] f32 (noisy + refs), txt [T,3584] f32 -> velocity [N,64] (ref rows meaningless)."""
        fhw = tuple(int(v) for s in img_shapes for v in s)
        if fhw != self.layout:
            arr = (ctypes.c_int * len(fhw))(*fhw)
            self.lib.hip_qimg_set_edit_layout(self.r, len(img_shapes), arr, int(zero_cond_t))
            self.layout = fhw
        tokens = np.ascontiguousarray(tokens, np.float32)
        txt = np.ascontiguousarray(txt, np.float32)
        out = np.zeros_like(tokens)
        fp = ctypes.POINTER(ctypes.c_float)
        started = time.time()
        rc = self.lib.hip_qimg_dit_step(self.r, tokens.ctypes.data_as(fp), len(tokens), txt.ctypes.data_as(fp),
                                        len(txt), ctypes.c_float(t1000), out.ctypes.data_as(fp))
        self.seconds += time.time() - started
        if rc != 0:
            raise RuntimeError('hip_qimg_dit_step failed')
        return out

    def set_mod_vectors(self, img, txt, img0):
        fp = ctypes.POINTER(ctypes.c_float)
        ptr = lambda a: None if a is None else np.ascontiguousarray(a, np.float32).ctypes.data_as(fp)
        keep = [np.ascontiguousarray(a, np.float32) if a is not None else None for a in (img, txt, img0)]
        if self.lib.hip_qimg_set_mod_vectors(self.r, *[None if a is None else a.ctypes.data_as(fp) for a in keep]) != 0:
            raise RuntimeError('hip_qimg_set_mod_vectors failed')

    def close(self):
        if self.r:
            self.lib.hip_qimg_free(self.r)
            self.r = None


def tkey(t):
    """Cache key for a timestep: the forward sees float32 t/1000, the scheduler list is double."""
    return round(float(np.float32(float(t))), 5)


class HostModulation:
    """Exact BF16 modulation vectors (img_mod / txt_mod per block) computed off the INT4 DiT.

    RTN-INT4 modulation weights dominated the native edit error (late-step rel_l2 21% -> 2% with BF16 mods), but
    6.8 B parameters do not fit beside the DiT. Mods are a mat-vec on the timestep embedding only, so the BF16
    weights stay in host RAM, are streamed per block to the GPU, and the vectors are cached per timestep.
    """

    def __init__(self, base):
        import json
        from safetensors import safe_open
        from diffusers.models.embeddings import Timesteps, TimestepEmbedding
        tdir = Path(base) / 'transformer'
        where = json.load(open(tdir / 'diffusion_pytorch_model.safetensors.index.json'))['weight_map']
        files = {}
        def get(name):
            f = where[name]
            if f not in files:
                files[f] = safe_open(str(tdir / f), 'pt', 'cpu')
            return files[f].get_tensor(name)
        self.n = 1 + max(int(k.split('.')[1]) for k in where if k.startswith('transformer_blocks.'))
        self.proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.emb = TimestepEmbedding(in_channels=256, time_embed_dim=3072)
        self.emb.load_state_dict({k.split('timestep_embedder.', 1)[1]: get(k) for k in where
                                  if k.startswith('time_text_embed.timestep_embedder.')})
        self.emb = self.emb.to('cuda', torch.bfloat16)
        self.w = {kind: [(get(f'transformer_blocks.{b}.{kind}.1.weight'), get(f'transformer_blocks.{b}.{kind}.1.bias'))
                         for b in range(self.n)] for kind in ('img_mod', 'txt_mod')}
        self.cache = {}

    @torch.no_grad()
    def prefetch(self, ts):
        """Compute and cache all timesteps in one pass over the streamed weights (13 x 6.8 B MACs, one transfer)."""
        ts = sorted({tkey(t) for t in ts} | {0.0})
        todo = [t for t in ts if t not in self.cache]
        if not todo:
            return
        x = torch.nn.functional.silu(self.emb(self.proj(torch.tensor(todo, device='cuda')).to(torch.bfloat16)))
        res = {}
        for kind in ('img_mod', 'txt_mod'):
            res[kind] = torch.stack([torch.nn.functional.linear(x, w.to('cuda', non_blocking=True), b.to('cuda')).float()
                                     for w, b in self.w[kind]], 1).cpu().numpy()          # [len(todo), n_blocks, 6*dim]
        for i, t in enumerate(todo):
            self.cache[t] = (res['img_mod'][i], res['txt_mod'][i])

    def __call__(self, t):
        key = tkey(t)
        if key not in self.cache or 0.0 not in self.cache:
            self.prefetch([key])
        return self.cache[key][0], self.cache[key][1], self.cache[0.0][0]


def load_pipeline(base, int4_path, *, zero_cond_t=None):
    """diffusers Edit-Plus pipeline on CPU with the transformer forward replaced by NativeDiT."""
    from accelerate import init_empty_weights
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    config = QwenImageTransformer2DModel.load_config(str(Path(base) / 'transformer'))
    with init_empty_weights():
        shell = QwenImageTransformer2DModel.from_config(config).to(torch.bfloat16)
    zc = bool(config.get('zero_cond_t', False)) if zero_cond_t is None else zero_cond_t
    host_mod = os.environ.setdefault('QIMG_HOST_MOD', '1') == '1'   # read by hip_qimg_load_dit_int4
    native = NativeDiT(int4_path)
    mods = HostModulation(base) if host_mod else None
    holder = {}

    def forward(hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None, timestep=None,
                img_shapes=None, txt_seq_lens=None, guidance=None, attention_kwargs=None, return_dict=True, **_):
        outs = []
        native.calls += 1
        for b in range(hidden_states.shape[0]):
            txt = encoder_hidden_states[b]
            if encoder_hidden_states_mask is not None:   # drop padding: native attention has no mask
                txt = txt[encoder_hidden_states_mask[b].bool()]
            if mods is not None:
                sched = getattr(holder.get('pipe'), 'scheduler', None)
                ts = getattr(sched, 'timesteps', None)
                if ts is not None and tkey(timestep[b]) not in mods.cache:
                    mods.prefetch([float(v) / 1000.0 for v in ts])   # whole schedule in one weight pass
                native.set_mod_vectors(*mods(float(timestep[b])))
            v = native.step(hidden_states[b].float().cpu().numpy(), txt.float().cpu().numpy(),
                            float(timestep[b]) * 1000.0, img_shapes[b], zc)
            outs.append(torch.from_numpy(v))
            dump = os.environ.get('QIMG_EDIT_DUMP')
            want = int(os.environ.get('QIMG_EDIT_DUMP_CALL', '1'))   # 1-based DiT call to dump (CFG: 2 per step)
            if dump and native.calls == want and not Path(dump).exists():   # parity oracle inputs/outputs
                np.savez(dump, tokens=hidden_states[b].float().cpu().numpy(), txt=txt.float().cpu().numpy(),
                         timestep=float(timestep[b]), img_shapes=np.array(img_shapes[b]), out=v)
        out = torch.stack(outs).to(hidden_states.device, hidden_states.dtype)
        return Transformer2DModelOutput(sample=out) if return_dict else (out,)

    shell.forward = forward
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(base), transformer=shell, torch_dtype=torch.bfloat16)
    if os.environ.get('QIMG_ENCODER_DTYPE', 'fp32') == 'fp32':
        # No native BF16 GEMM on this CPU: the Qwen2.5-VL encoder runs several times faster in FP32 (~30 GB RAM).
        pipe.text_encoder.to(torch.float32)
    if os.environ.get('QIMG_PIPE_FLOW', 'cuda') == 'cpu':
        # Legacy CPU-device flow with VAE/vision/encode wrappers. Kept for A/B: it lost the subject's identity on
        # side views (the same native DiT kept it under the cuda flow), so it is not the default.
        type(pipe)._execution_device = property(lambda self: torch.device('cpu'))
        # bf16 conv3d on the CPU is single-threaded and takes >1 h at 1024^2: run the VAE on the GPU next to the
        # resident INT4 DiT (tiled), moving tensors across so the rest of the pipeline stays on the CPU.
        pipe.vae.to('cuda'); pipe.vae.enable_tiling()
        enc, dec = pipe.vae.encode, pipe.vae.decode

        def to_cpu(x):
            if torch.is_tensor(x):
                return x.cpu()
            if hasattr(x, 'latent_dist'):
                d = x.latent_dist
                for k in ('parameters', 'mean', 'logvar', 'std', 'var'):
                    setattr(d, k, getattr(d, k).cpu())
                return x
            if hasattr(x, 'sample') and torch.is_tensor(x.sample):
                x.sample = x.sample.cpu()
                return x
            if isinstance(x, tuple):
                return tuple(to_cpu(v) for v in x)
            return x

        phase = {}

        def timed(name, fn):
            def run(*a, **k):
                t = time.time()
                try:
                    return fn(*a, **k)
                finally:
                    phase[name] = phase.get(name, 0.) + time.time() - t
            return run

        pipe.phase_seconds = phase

        def released(fn):
            # torch's caching allocator would otherwise keep VAE activations the native DiT needs.
            def run(x, *a, **k):
                try:
                    return to_cpu(fn(x.to('cuda', pipe.vae.dtype), *a, **k))   # FP32 encoder makes inputs float32
                finally:
                    torch.cuda.empty_cache()
            return run

        enc_cache = {}

        def cached_encode(x, *a, **k):
            # Reference latents use the distribution mode (deterministic), and the portrait recurs every view.
            import hashlib, copy
            key = (hashlib.sha1(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                   tuple(x.shape), str(x.dtype))
            if key not in enc_cache:
                if len(enc_cache) >= 8:
                    enc_cache.pop(next(iter(enc_cache)))
                enc_cache[key] = released(enc)(x, *a, **k)
            else:
                pipe.vae_cache_hits += 1
            return copy.copy(enc_cache[key])

        pipe.vae_cache_hits = 0
        pipe.vae.encode = timed('vae_encode', cached_encode if not os.environ.get('QIMG_NO_VAE_CACHE') else released(enc))
        pipe.vae.decode = timed('vae_decode', released(dec))
        pipe.encode_prompt = timed('prompt_encode', pipe.encode_prompt)

        # The negative-prompt encode re-runs the Qwen2.5-VL vision tower on the same reference images
        # (~16k patches per view on the CPU). Its output depends only on pixels and grid, so memoize it.
        visual = pipe.text_encoder.model.visual
        vis_forward, vis_cache = visual.forward, {}

        def cached_visual(pixel_values, grid_thw=None, **kwargs):
            import copy, hashlib
            key = (hashlib.sha1(pixel_values.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                   None if grid_thw is None else tuple(grid_thw.flatten().tolist()), tuple(sorted(kwargs)))
            if key not in vis_cache:
                if len(vis_cache) >= 4:
                    vis_cache.pop(next(iter(vis_cache)))
                vis_cache[key] = copy.copy(vis_forward(pixel_values, grid_thw=grid_thw, **kwargs))
            else:
                pipe.vision_cache_hits += 1
            # get_image_features mutates the output in place (pooler_output -> tuple of splits): hand out copies.
            import copy
            return copy.copy(vis_cache[key])

        if not os.environ.get('QIMG_NO_VISION_CACHE'):
            visual.forward = cached_visual
        pipe.vision_cache_hits = 0
    else:
        # Same flow as the GGUF editor (identity verified): pipeline on cuda, tiled VAE on cuda, prompts
        # encoded on the CPU by the caller (NativeEditor) and passed as embeddings.
        pipe.text_encoder.to(torch.bfloat16)
        pipe.vae.to('cuda'); pipe.vae.enable_tiling()
        type(pipe)._execution_device = property(lambda self: torch.device('cuda'))
        pipe.phase_seconds = {}; pipe.vision_cache_hits = 0
    pipe.native = native
    holder['pipe'] = pipe
    return pipe
