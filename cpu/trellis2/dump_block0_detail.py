#!/usr/bin/env python3
"""Dump detailed block 0 intermediates for CUDA comparison."""
import sys, os
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trellis2_repo'))

import torch, numpy as np, torch.nn.functional as F
from safetensors import safe_open
from trellis2.modules.attention.rope import RotaryPositionEmbedder

path = '/mnt/disk01/models/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors'
def get(name):
    with safe_open(path, framework='pt', device='cuda') as f:
        return f.get_tensor(name).float()

def stats(name, t):
    v = t.flatten()
    print(f"  {name}: std={v.std():.6f} mean={v.mean():.6f} [:4]={v[:4].cpu().numpy()}")

noise = torch.from_numpy(np.load('pytorch_noise_seed42.npy')).float().cuda()
features = torch.from_numpy(np.load('official_teapot_features.npy')).float().unsqueeze(0).cuda()

# Input embedding
x = noise.view(1, 8, -1).permute(0, 2, 1).contiguous()  # [1, 4096, 8]
h = F.linear(x, get('input_layer.weight'), get('input_layer.bias'))  # [1, 4096, 1536]
stats("input_emb", h)

# Timestep embedding
freqs = torch.exp(-np.log(10000) * torch.arange(128, dtype=torch.float32, device='cuda') / 128)
t_freq = torch.cat([torch.cos(1000 * freqs), torch.sin(1000 * freqs)])
te = F.silu(F.linear(t_freq, get('t_embedder.mlp.0.weight'), get('t_embedder.mlp.0.bias')))
te = F.linear(te, get('t_embedder.mlp.2.weight'), get('t_embedder.mlp.2.bias'))

# Modulation
mod = F.linear(F.silu(te), get('adaLN_modulation.1.weight'), get('adaLN_modulation.1.bias'))
mod = mod + get('blocks.0.modulation')
shift, scale, gate, shift_m, scale_m, gate_m = mod.chunk(6)

# === Self-attention ===
print("\n=== Block 0: Self-attention ===")

# 1. adaLN
normed = F.layer_norm(h, [1536], eps=1e-6)
normed = normed * (1 + scale) + shift
stats("adaln_out", normed)

# 2. QKV projection
qkv = F.linear(normed, get('blocks.0.self_attn.to_qkv.weight'), get('blocks.0.self_attn.to_qkv.bias'))
q, k, v = qkv.chunk(3, dim=-1)
stats("Q_raw", q)
stats("K_raw", k)
stats("V_raw", v)

# 3. RMSNorm on Q and K
q_gamma = get('blocks.0.self_attn.q_rms_norm.gamma')  # [12, 128]
k_gamma = get('blocks.0.self_attn.k_rms_norm.gamma')

q3 = q.view(1, 4096, 12, 128)
rms = torch.rsqrt(q3.pow(2).mean(-1, keepdim=True) + 1e-6)
q3 = q3 * rms * q_gamma
q_normed = q3.view(1, 4096, 1536)
stats("Q_rmsnorm", q_normed)

k3 = k.view(1, 4096, 12, 128)
rms = torch.rsqrt(k3.pow(2).mean(-1, keepdim=True) + 1e-6)
k3 = k3 * rms * k_gamma
k_normed = k3.view(1, 4096, 1536)
stats("K_rmsnorm", k_normed)

# 4. RoPE
rope = RotaryPositionEmbedder(128, 3)
coords = torch.meshgrid(*[torch.arange(16, device='cuda') for _ in range(3)], indexing='ij')
coords = torch.stack(coords, dim=-1).reshape(-1, 3).float()
phases = rope(coords)  # [4096, 64] complex

q_h = q_normed.view(1, 4096, 12, 128)
k_h = k_normed.view(1, 4096, 12, 128)

q_rot = RotaryPositionEmbedder.apply_rotary_embedding(q_h, phases)
k_rot = RotaryPositionEmbedder.apply_rotary_embedding(k_h, phases)
stats("Q_rope", q_rot.view(1, 4096, 1536))
stats("K_rope", k_rot.view(1, 4096, 1536))

# 5. Attention
q_a = q_rot.squeeze(0).transpose(0, 1)  # [12, 4096, 128]
k_a = k_rot.squeeze(0).transpose(0, 1)
v_a = v.view(4096, 12, 128).transpose(0, 1)
attn_out = F.scaled_dot_product_attention(q_a, k_a, v_a)
attn_out = attn_out.transpose(0, 1).contiguous().view(4096, 1536)
stats("attn_out", attn_out)

# 6. Output projection + gated residual
sa_proj = F.linear(attn_out, get('blocks.0.self_attn.to_out.weight'), get('blocks.0.self_attn.to_out.bias'))
stats("sa_proj", sa_proj)
h_after_sa = h.squeeze(0) + gate * sa_proj
stats("h_after_sa", h_after_sa)

# === Cross-attention ===
print("\n=== Block 0: Cross-attention ===")
normed2 = F.layer_norm(h_after_sa.unsqueeze(0), [1536],
                        weight=get('blocks.0.norm2.weight'), bias=get('blocks.0.norm2.bias'), eps=1e-6)
stats("norm2_out", normed2)

cross_q = F.linear(normed2, get('blocks.0.cross_attn.to_q.weight'), get('blocks.0.cross_attn.to_q.bias'))
cross_kv = F.linear(features, get('blocks.0.cross_attn.to_kv.weight'), get('blocks.0.cross_attn.to_kv.bias'))
cross_k, cross_v = cross_kv.chunk(2, dim=-1)

# RMSNorm on cross Q and K
cq_gamma = get('blocks.0.cross_attn.q_rms_norm.gamma')
ck_gamma = get('blocks.0.cross_attn.k_rms_norm.gamma')
cq3 = cross_q.view(1, 4096, 12, 128)
rms = torch.rsqrt(cq3.pow(2).mean(-1, keepdim=True) + 1e-6)
cross_q_normed = (cq3 * rms * cq_gamma).view(1, 4096, 1536)

ck3 = cross_k.view(1, 1029, 12, 128)
rms = torch.rsqrt(ck3.pow(2).mean(-1, keepdim=True) + 1e-6)
cross_k_normed = (ck3 * rms * ck_gamma).view(1, 1029, 1536)

stats("cross_Q_norm", cross_q_normed)
stats("cross_K_norm", cross_k_normed)

# Cross attention (no RoPE on cross-attn)
cq = cross_q_normed.view(1, 4096, 12, 128).squeeze(0).transpose(0, 1)
ck = cross_k_normed.view(1, 1029, 12, 128).squeeze(0).transpose(0, 1)
cv = cross_v.view(1, 1029, 12, 128).squeeze(0).transpose(0, 1)
ca_out = F.scaled_dot_product_attention(cq, ck, cv)
ca_out = ca_out.transpose(0, 1).contiguous().view(4096, 1536)
stats("ca_out", ca_out)

ca_proj = F.linear(ca_out, get('blocks.0.cross_attn.to_out.weight'), get('blocks.0.cross_attn.to_out.bias'))
stats("ca_proj", ca_proj)
h_after_ca = h_after_sa + ca_proj
stats("h_after_ca", h_after_ca)

# === MLP ===
print("\n=== Block 0: MLP ===")
normed3 = F.layer_norm(h_after_ca.unsqueeze(0), [1536], eps=1e-6)
normed3 = normed3 * (1 + scale_m) + shift_m
ff = F.linear(normed3, get('blocks.0.mlp.mlp.0.weight'), get('blocks.0.mlp.mlp.0.bias'))
ff = F.gelu(ff)
ff = F.linear(ff, get('blocks.0.mlp.mlp.2.weight'), get('blocks.0.mlp.mlp.2.bias'))
h_final = h_after_ca + gate_m * ff.squeeze(0)
stats("h_block0_final", h_final)
