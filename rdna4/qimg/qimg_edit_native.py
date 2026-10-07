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
        self.r = L.hip_qimg_init(device, verbose)
        if not self.r:
            raise RuntimeError('hip_qimg_init failed')
        if L.hip_qimg_load_dit_int4(self.r, str(int4_path).encode()) != 0:
            raise RuntimeError('hip_qimg_load_dit_int4 failed: ' + str(int4_path))
        self.layout = None
        self.seconds = 0.

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

    def close(self):
        if self.r:
            self.lib.hip_qimg_free(self.r)
            self.r = None


def load_pipeline(base, int4_path, *, zero_cond_t=None):
    """diffusers Edit-Plus pipeline on CPU with the transformer forward replaced by NativeDiT."""
    from accelerate import init_empty_weights
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    config = QwenImageTransformer2DModel.load_config(str(Path(base) / 'transformer'))
    with init_empty_weights():
        shell = QwenImageTransformer2DModel.from_config(config).to(torch.bfloat16)
    zc = bool(config.get('zero_cond_t', False)) if zero_cond_t is None else zero_cond_t
    native = NativeDiT(int4_path)

    def forward(hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None, timestep=None,
                img_shapes=None, txt_seq_lens=None, guidance=None, attention_kwargs=None, return_dict=True, **_):
        outs = []
        for b in range(hidden_states.shape[0]):
            txt = encoder_hidden_states[b]
            if encoder_hidden_states_mask is not None:   # drop padding: native attention has no mask
                txt = txt[encoder_hidden_states_mask[b].bool()]
            v = native.step(hidden_states[b].float().cpu().numpy(), txt.float().cpu().numpy(),
                            float(timestep[b]) * 1000.0, img_shapes[b], zc)
            outs.append(torch.from_numpy(v))
            dump = os.environ.get('QIMG_EDIT_DUMP')
            if dump and not Path(dump).exists():   # first call only: parity oracle inputs/outputs
                np.savez(dump, tokens=hidden_states[b].float().cpu().numpy(), txt=txt.float().cpu().numpy(),
                         timestep=float(timestep[b]), img_shapes=np.array(img_shapes[b]), out=v)
        out = torch.stack(outs).to(hidden_states.dtype)
        return Transformer2DModelOutput(sample=out) if return_dict else (out,)

    shell.forward = forward
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(base), transformer=shell, torch_dtype=torch.bfloat16)
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

    def released(fn):
        # torch's caching allocator would otherwise keep VAE activations the native DiT needs.
        def run(x, *a, **k):
            try:
                return to_cpu(fn(x.to('cuda'), *a, **k))
            finally:
                torch.cuda.empty_cache()
        return run

    pipe.vae.encode, pipe.vae.decode = released(enc), released(dec)
    pipe.native = native
    return pipe
