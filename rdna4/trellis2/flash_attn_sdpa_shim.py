"""Shim module that exposes flash_attn-varlen entrypoints backed by torch SDPA.

Install via `install_as_flash_attn()` *before* any trellis2 module imports flash_attn.
Only the three varlen calls used by trellis2 sparse attention are provided.
"""

import torch
import torch.nn.functional as F


def _per_seq_sdpa(q, k, v, cu_q, cu_kv):
    """q:[Tq,H,C] k,v:[Tkv,H,C]. Apply SDPA per-sequence slice then concat."""
    outs = []
    cu_q = cu_q.cpu().tolist()
    cu_kv = cu_kv.cpu().tolist()
    for i in range(len(cu_q) - 1):
        q_s, q_e = cu_q[i], cu_q[i + 1]
        k_s, k_e = cu_kv[i], cu_kv[i + 1]
        # SDPA wants [*, H, L, C]
        qi = q[q_s:q_e].transpose(0, 1).unsqueeze(0)   # [1,H,Lq,C]
        ki = k[k_s:k_e].transpose(0, 1).unsqueeze(0)
        vi = v[k_s:k_e].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi)   # [1,H,Lq,C]
        outs.append(oi.squeeze(0).transpose(0, 1))         # [Lq,H,C]
    return torch.cat(outs, dim=0)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, *args, **kwargs):
    q, k, v = qkv.unbind(dim=1)
    return _per_seq_sdpa(q, k, v, cu_seqlens, cu_seqlens)


def flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_kv, max_q, max_kv, *args, **kwargs):
    k, v = kv.unbind(dim=1)
    return _per_seq_sdpa(q, k, v, cu_q, cu_kv)


def flash_attn_varlen_func(q, k, v, cu_q, cu_kv, max_q, max_kv, *args, **kwargs):
    return _per_seq_sdpa(q, k, v, cu_q, cu_kv)


def install_as_flash_attn():
    import sys
    import types
    import importlib.machinery
    mod = types.ModuleType('flash_attn')
    mod.__spec__ = importlib.machinery.ModuleSpec('flash_attn', loader=None)
    mod.__file__ = '<flash_attn_sdpa_shim>'
    mod.__version__ = '2.0.0-sdpa-shim'
    mod.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    mod.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules['flash_attn'] = mod
