#!/usr/bin/env python3
"""
Qwen-Image full pipeline PyTorch reference — generates a ground-truth image.

Loads GGUF MMDiT weights and safetensors VAE, runs the complete denoising
pipeline on GPU (block-by-block to fit in 16GB VRAM), then VAE decode.

Usage:
    uv run python run_full_pipeline.py \
        --dit-path /mnt/disk01/models/qwen-image/diffusion-models/qwen-image-Q4_0.gguf \
        --vae-path /mnt/disk01/models/qwen-image/vae/qwen_image_vae.safetensors \
        --text-hidden ../../cpu/qwen_image/text_hidden_states.npy \
        --output output/ground_truth.png \
        --height 256 --width 256 --steps 30 --seed 42
"""

import argparse, math, os, struct, time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


# ---- GGUF reader (reuse from run_dit_block_reference.py) ----

def read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")

def skip_value(f, vtype):
    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if vtype == 8: read_string(f)
    elif vtype == 9:
        at = struct.unpack("<I", f.read(4))[0]
        al = struct.unpack("<Q", f.read(8))[0]
        if at == 8:
            for _ in range(al): read_string(f)
        else: f.read(al * sizes.get(at, 4))
    else: f.read(sizes.get(vtype, 4))

def dequant_q4_0(data, n_elements):
    """Vectorized Q4_0 dequantization. GGML: low nibbles→0-15, high→16-31."""
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 18)
    scales = raw[:, :2].view(np.float16).astype(np.float32).flatten()  # [n_blocks]
    quants = raw[:, 2:]  # [n_blocks, 16]
    lo = (quants & 0x0F).astype(np.int8) - 8  # [n_blocks, 16]
    hi = (quants >> 4).astype(np.int8) - 8     # [n_blocks, 16]
    # GGML layout: low nibbles at positions 0-15, high at 16-31
    result = np.empty((len(scales), 32), dtype=np.float32)
    result[:, :16] = lo.astype(np.float32) * scales[:, None]
    result[:, 16:] = hi.astype(np.float32) * scales[:, None]
    return result.flatten()[:n_elements]

def dequant_q4_1(data, n_elements):
    """Vectorized Q4_1 dequantization."""
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 20)
    scales = raw[:, :2].view(np.float16).astype(np.float32).flatten()
    mins = raw[:, 2:4].view(np.float16).astype(np.float32).flatten()
    quants = raw[:, 4:]
    lo = (quants & 0x0F).astype(np.float32)
    hi = (quants >> 4).astype(np.float32)
    result = np.empty((len(scales), 32), dtype=np.float32)
    result[:, :16] = lo * scales[:, None] + mins[:, None]
    result[:, 16:] = hi * scales[:, None] + mins[:, None]
    return result.flatten()[:n_elements]


class GGUFFile:
    def __init__(self, path):
        self.path = path
        self.tensors = {}
        with open(path, "rb") as f:
            assert f.read(4) == b'GGUF'
            struct.unpack("<I", f.read(4))
            nt = struct.unpack("<Q", f.read(8))[0]
            nk = struct.unpack("<Q", f.read(8))[0]
            for _ in range(nk):
                read_string(f)
                skip_value(f, struct.unpack("<I", f.read(4))[0])
            for _ in range(nt):
                name = read_string(f)
                nd = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
                tt = struct.unpack("<I", f.read(4))[0]
                off = struct.unpack("<Q", f.read(8))[0]
                self.tensors[name] = (dims, tt, off)
            self.data_start = ((f.tell() + 31) // 32) * 32

    def load(self, name, device='cpu'):
        dims, tt, off = self.tensors[name]
        ne = 1
        for d in dims: ne *= d
        with open(self.path, "rb") as f:
            f.seek(self.data_start + off)
            if tt == 0:
                data = np.frombuffer(f.read(ne*4), dtype=np.float32).copy()
            elif tt == 30:  # BF16
                raw = np.frombuffer(f.read(ne*2), dtype=np.uint16).copy()
                data = (raw.astype(np.uint32) << 16).view(np.float32)
            elif tt == 2:
                data = dequant_q4_0(f.read((ne//32)*18), ne)
            elif tt == 3:
                data = dequant_q4_1(f.read((ne//32)*20), ne)
            elif tt in (31,32,33):
                data = dequant_q4_0(f.read((ne//32)*18), ne)
            else:
                raise ValueError(f"Unsupported type {tt} for {name}")
        if len(dims) == 2:
            data = data.reshape(dims[1], dims[0])
        t = torch.tensor(data, dtype=torch.float32)
        if device != 'cpu': t = t.to(device)
        return t


# ---- Scheduler ----

def make_schedule(n_steps, img_seq_len, base_shift=0.5, max_shift=0.9, shift_terminal=0.02):
    log_s = math.log(img_seq_len)
    mu = (log_s - math.log(256)) / (math.log(8192) - math.log(256))
    mu = max(0, min(1, mu))
    shift = base_shift + (max_shift - base_shift) * mu
    sigmas = []
    for i in range(n_steps + 1):
        t = i / n_steps
        s = 1.0 - t * (1.0 - shift_terminal)
        es = math.exp(shift)
        s = es * s / (1 + (es - 1) * s)
        sigmas.append(s)
    timesteps = [s * 1000.0 for s in sigmas[:n_steps]]
    dt = [sigmas[i+1] - sigmas[i] for i in range(n_steps)]
    return timesteps, dt, sigmas


# ---- RMSNorm ----

def rmsnorm(x, w, eps=1e-6):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return x / rms * w

def rmsnorm_per_head(x, w, nh, hd, eps=1e-6):
    N, D = x.shape
    x = x.reshape(N, nh, hd)
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return (x / rms * w.view(1, 1, hd)).reshape(N, D)


# ---- RoPE ----

def apply_rope_2d(q, k, n_img, nh, hd, axes_dims, theta=10000.0):
    t_dim, h_dim, w_dim = axes_dims
    hp = int(n_img**0.5); wp = n_img // hp
    for tok in range(n_img):
        ph, pw = tok // wp, tok % wp
        for h in range(nh):
            off = h * hd
            for i in range(h_dim // 2):
                freq = 1.0 / (theta ** (2.0*i/h_dim))
                cs, sn = math.cos(ph*freq), math.sin(ph*freq)
                idx = off + t_dim + 2*i
                for a in [q, k]:
                    v0, v1 = a[tok,idx].item(), a[tok,idx+1].item()
                    a[tok,idx] = v0*cs - v1*sn
                    a[tok,idx+1] = v0*sn + v1*cs
            for i in range(w_dim // 2):
                freq = 1.0 / (theta ** (2.0*i/w_dim))
                cs, sn = math.cos(pw*freq), math.sin(pw*freq)
                idx = off + t_dim + h_dim + 2*i
                for a in [q, k]:
                    v0, v1 = a[tok,idx].item(), a[tok,idx+1].item()
                    a[tok,idx] = v0*cs - v1*sn
                    a[tok,idx+1] = v0*sn + v1*cs

def apply_rope_1d(q, k, n_txt, nh, hd, theta=10000.0):
    for tok in range(n_txt):
        for h in range(nh):
            off = h * hd
            for i in range(hd // 2):
                freq = 1.0 / (theta ** (2.0*i/hd))
                cs, sn = math.cos(tok*freq), math.sin(tok*freq)
                for a in [q, k]:
                    v0, v1 = a[tok,off+2*i].item(), a[tok,off+2*i+1].item()
                    a[tok,off+2*i] = v0*cs - v1*sn
                    a[tok,off+2*i+1] = v0*sn + v1*cs


# ---- DiT block forward (on GPU) ----

@torch.no_grad()
def dit_block_forward(img, txt, t_emb, gguf, block_idx, cfg, dev):
    dim = cfg['dim']; nh = cfg['nh']; hd = cfg['hd']
    mlp_h = cfg['mlp_h']; eps = 1e-6
    n_img, n_txt = img.shape[0], txt.shape[0]
    pre = f"transformer_blocks.{block_idx}"

    def load(suffix):
        return gguf.load(f"{pre}.{suffix}", dev)

    # Modulation
    t_silu = F.silu(t_emb)
    img_mod = t_silu @ load("img_mod.1.weight").T + load("img_mod.1.bias")
    txt_mod = t_silu @ load("txt_mod.1.weight").T + load("txt_mod.1.bias")
    ish1,isc1,ig1,ish2,isc2,ig2 = img_mod.reshape(6, dim)
    tsh1,tsc1,tg1,tsh2,tsc2,tg2 = txt_mod.reshape(6, dim)

    # adaLN
    def adaln(x, shift, scale):
        m = x.mean(-1, keepdim=True)
        v = x.var(-1, keepdim=True, unbiased=False)
        return (x - m) / torch.sqrt(v + eps) * (1 + scale) + shift

    img_n = adaln(img, ish1, isc1)
    txt_n = adaln(txt, tsh1, tsc1)

    # QKV
    img_q = img_n @ load("attn.to_q.weight").T + load("attn.to_q.bias")
    img_k = img_n @ load("attn.to_k.weight").T + load("attn.to_k.bias")
    img_v = img_n @ load("attn.to_v.weight").T + load("attn.to_v.bias")
    txt_q = txt_n @ load("attn.add_q_proj.weight").T + load("attn.add_q_proj.bias")
    txt_k = txt_n @ load("attn.add_k_proj.weight").T + load("attn.add_k_proj.bias")
    txt_v = txt_n @ load("attn.add_v_proj.weight").T + load("attn.add_v_proj.bias")

    # QK norm
    img_q = rmsnorm_per_head(img_q, load("attn.norm_q.weight"), nh, hd, eps)
    img_k = rmsnorm_per_head(img_k, load("attn.norm_k.weight"), nh, hd, eps)
    txt_q = rmsnorm_per_head(txt_q, load("attn.norm_added_q.weight"), nh, hd, eps)
    txt_k = rmsnorm_per_head(txt_k, load("attn.norm_added_k.weight"), nh, hd, eps)

    # RoPE (on CPU for simplicity)
    img_q_c, img_k_c = img_q.cpu(), img_k.cpu()
    txt_q_c, txt_k_c = txt_q.cpu(), txt_k.cpu()
    apply_rope_2d(img_q_c, img_k_c, n_img, nh, hd, cfg['axes'], cfg['theta'])
    apply_rope_1d(txt_q_c, txt_k_c, n_txt, nh, hd, cfg['theta'])
    img_q, img_k = img_q_c.to(dev), img_k_c.to(dev)
    txt_q, txt_k = txt_q_c.to(dev), txt_k_c.to(dev)

    # Joint attention
    N = n_txt + n_img
    q = torch.cat([txt_q, img_q]).reshape(N, nh, hd).permute(1,0,2)
    k = torch.cat([txt_k, img_k]).reshape(N, nh, hd).permute(1,0,2)
    v = torch.cat([txt_v, img_v]).reshape(N, nh, hd).permute(1,0,2)
    attn = torch.bmm(q, k.transpose(1,2)) * (hd**-0.5)
    attn = torch.softmax(attn, dim=-1)
    out = torch.bmm(attn, v).permute(1,0,2).reshape(N, dim)

    txt_a, img_a = out[:n_txt], out[n_txt:]
    img_a = img_a @ load("attn.to_out.0.weight").T + load("attn.to_out.0.bias")
    txt_a = txt_a @ load("attn.to_add_out.weight").T + load("attn.to_add_out.bias")
    img = img + ig1 * img_a
    txt = txt + tg1 * txt_a

    # MLP
    img_n2 = adaln(img, ish2, isc2)
    img_f = F.gelu(img_n2 @ load("img_mlp.net.0.proj.weight").T + load("img_mlp.net.0.proj.bias"), approximate='tanh')
    img_f = img_f @ load("img_mlp.net.2.weight").T + load("img_mlp.net.2.bias")
    img = img + ig2 * img_f

    txt_n2 = adaln(txt, tsh2, tsc2)
    txt_f = F.gelu(txt_n2 @ load("txt_mlp.net.0.proj.weight").T + load("txt_mlp.net.0.proj.bias"), approximate='tanh')
    txt_f = txt_f @ load("txt_mlp.net.2.weight").T + load("txt_mlp.net.2.bias")
    txt = txt + tg2 * txt_f

    return img, txt


# ---- VAE decode ----

def causal_pad(x, kd, kh, kw):
    return F.pad(x, ((kw-1)//2, (kw-1)//2, (kh-1)//2, (kh-1)//2, kd-1, 0), mode='replicate')

@torch.no_grad()
def vae_decode(latent, vae_path, device):
    """latent: [1, 16, H, W] tensor."""
    st = safe_open(vae_path, framework="pt", device="cpu")

    def get(name):
        return st.get_tensor(name).float().to(device)

    # Add temporal dim
    x = latent.unsqueeze(2)  # [1,16,1,H,W]

    # post_quant_conv
    x = F.conv3d(x, get("conv2.weight"), get("conv2.bias"))

    # decoder.conv1
    x = F.conv3d(causal_pad(x,3,3,3), get("decoder.conv1.weight"), get("decoder.conv1.bias"))

    # ResBlock helper
    def resblock(x, prefix):
        h = F.group_norm(x, 32, weight=get(f"{prefix}.residual.0.gamma").view(-1))
        h = F.silu(h)
        h = F.conv3d(causal_pad(h,3,3,3), get(f"{prefix}.residual.2.weight"), get(f"{prefix}.residual.2.bias"))
        h = F.group_norm(h, 32, weight=get(f"{prefix}.residual.3.gamma").view(-1))
        h = F.silu(h)
        h = F.conv3d(causal_pad(h,3,3,3), get(f"{prefix}.residual.6.weight"), get(f"{prefix}.residual.6.bias"))
        try:
            x = F.conv3d(x, get(f"{prefix}.shortcut.weight"), get(f"{prefix}.shortcut.bias"))
        except: pass
        return x + h

    # Middle
    x = resblock(x, "decoder.middle.0")
    # Mid attention
    B,C,T,H,W = x.shape; S = T*H*W
    xn = F.group_norm(x, 32, weight=get("decoder.middle.1.norm.gamma").view(-1))
    xf = xn.reshape(B,C,S)
    qkv = torch.einsum('oi,bis->bos', get("decoder.middle.1.to_qkv.weight").view(3*C,C), xf) + get("decoder.middle.1.to_qkv.bias").view(1,3*C,1)
    q,k,v = qkv.chunk(3,dim=1)
    a = torch.softmax(torch.bmm(q.permute(0,2,1), k) * (C**-0.5), dim=-1)
    o = torch.bmm(v, a.permute(0,2,1))
    o = torch.einsum('oi,bis->bos', get("decoder.middle.1.proj.weight").view(C,C), o) + get("decoder.middle.1.proj.bias").view(1,C,1)
    x = x + o.reshape(B,C,T,H,W)
    x = resblock(x, "decoder.middle.2")

    # Upsample blocks
    for i in range(15):
        pfx = f"decoder.upsamples.{i}"
        try:
            st.get_tensor(f"{pfx}.residual.2.weight")
            x = resblock(x, pfx)
        except: pass
        try:
            rs_w = get(f"{pfx}.resample.1.weight")
            rs_b = get(f"{pfx}.resample.1.bias")
            x = x.repeat_interleave(2,dim=3).repeat_interleave(2,dim=4)
            x2 = F.conv2d(x[:,:,0], rs_w, rs_b, padding=1)
            x = x2.unsqueeze(2)
        except: pass

    # Head
    x = F.group_norm(x, 32, weight=get("decoder.head.0.gamma").view(-1))
    x = F.silu(x)
    x = F.conv3d(causal_pad(x,3,3,3), get("decoder.head.2.weight"), get("decoder.head.2.bias"))
    return x[:,:,0]  # [1,3,H,W]


# ---- Timestep embedding ----

def sinusoidal_embed(t, dim=256):
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, dtype=torch.float32) / half * math.log(10000))
    angles = t * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)])


# ---- Main pipeline ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dit-path', required=True)
    parser.add_argument('--vae-path', required=True)
    parser.add_argument('--text-hidden', required=True, help='.npy from C text encoder')
    parser.add_argument('--output', default='output/ground_truth.png')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Config
    dim = 3072; nh = 24; hd = 128; txt_dim = 3584; mlp_h = 12288; ps = 2
    n_blocks = 60
    cfg = {'dim':dim, 'nh':nh, 'hd':hd, 'mlp_h':mlp_h,
           'axes':[16,56,56], 'theta':10000.0}

    # Load text hidden states
    txt_raw = torch.tensor(np.load(args.text_hidden), dtype=torch.float32)
    n_txt = txt_raw.shape[0]
    print(f"Text hidden states: {txt_raw.shape}")

    # Load GGUF
    print(f"Loading DiT: {args.dit_path}")
    gguf = GGUFFile(args.dit_path)

    # Latent dims
    lat_h = args.height // 8; lat_w = args.width // 8
    n_img = (lat_h // ps) * (lat_w // ps)
    print(f"Image: {args.height}x{args.width}, latent: {lat_h}x{lat_w}, patches: {n_img}, text: {n_txt}")

    # Initialize noise
    torch.manual_seed(args.seed)
    latent = torch.randn(1, 16, lat_h, lat_w)
    np.save(args.output.replace('.png','_noise.npy'), latent.numpy())

    # Patchify
    def patchify(z):
        """[1, C, H, W] -> [H/ps*W/ps, C*ps*ps]"""
        B,C,H,W = z.shape
        return z.reshape(B, C, H//ps, ps, W//ps, ps).permute(0,2,4,1,3,5).reshape(-1, C*ps*ps)

    def unpatchify(tokens, C, H, W):
        """[seq, C*ps*ps] -> [1, C, H, W]"""
        hp, wp = H//ps, W//ps
        return tokens.reshape(1, hp, wp, C, ps, ps).permute(0,3,1,4,2,5).reshape(1, C, H, W)

    # Schedule
    timesteps, dt, sigmas = make_schedule(args.steps, n_img)

    # Text input projection (load to GPU, apply once)
    txt_norm_w = gguf.load("txt_norm.weight", device)
    txt_in_w = gguf.load("txt_in.weight", device)
    txt_in_b = gguf.load("txt_in.bias", device)
    txt_normed = rmsnorm(txt_raw.to(device), txt_norm_w)
    txt_proj = txt_normed @ txt_in_w.T + txt_in_b
    del txt_norm_w, txt_in_w, txt_in_b, txt_normed
    torch.cuda.empty_cache() if device == 'cuda' else None

    # Timestep embedder weights (load once)
    t_fc1_w = gguf.load("time_text_embed.timestep_embedder.linear_1.weight", device)
    t_fc1_b = gguf.load("time_text_embed.timestep_embedder.linear_1.bias", device)
    t_fc2_w = gguf.load("time_text_embed.timestep_embedder.linear_2.weight", device)
    t_fc2_b = gguf.load("time_text_embed.timestep_embedder.linear_2.bias", device)

    # Image input projection
    img_in_w = gguf.load("img_in.weight", device)
    img_in_b = gguf.load("img_in.bias", device)

    # Output projection
    norm_out_w = gguf.load("norm_out.linear.weight", device)
    norm_out_b = gguf.load("norm_out.linear.bias", device)
    proj_out_w = gguf.load("proj_out.weight", device)
    proj_out_b = gguf.load("proj_out.bias", device)

    # Denoising loop
    print(f"\nDenoising ({args.steps} steps)...")
    total_t0 = time.time()

    for step in range(args.steps):
        t_val = timesteps[step]
        step_t0 = time.time()

        # Patchify
        img_tokens = patchify(latent).to(device)

        # Timestep embedding
        t_sin = sinusoidal_embed(t_val, 256).to(device)
        t_emb = F.silu(t_sin @ t_fc1_w.T + t_fc1_b) @ t_fc2_w.T + t_fc2_b

        # Image input projection
        img = img_tokens @ img_in_w.T + img_in_b
        txt = txt_proj.clone()

        # Run all 60 blocks
        for L in range(n_blocks):
            img, txt = dit_block_forward(img, txt, t_emb, gguf, L, cfg, device)
            if device == 'cuda':
                torch.cuda.empty_cache()

        # Final output
        t_silu = F.silu(t_emb)
        final_mod = t_silu @ norm_out_w.T + norm_out_b
        shift, scale = final_mod.reshape(2, dim)
        m = img.mean(-1, keepdim=True)
        v = img.var(-1, keepdim=True, unbiased=False)
        img = (img - m) / torch.sqrt(v + 1e-6) * (1 + scale) + shift
        vel_tokens = img @ proj_out_w.T + proj_out_b

        # Unpatchify velocity
        vel_latent = unpatchify(vel_tokens.cpu(), 16, lat_h, lat_w)

        # Euler step
        latent = latent + dt[step] * vel_latent

        elapsed = time.time() - step_t0
        print(f"  step {step+1}/{args.steps}  t={t_val:.1f}  "
              f"latent: [{latent.min():.3f}, {latent.max():.3f}]  {elapsed:.1f}s")

    total_elapsed = time.time() - total_t0
    print(f"\nDenoising done: {total_elapsed:.1f}s total")

    # Save final latent
    np.save(args.output.replace('.png','_latent.npy'), latent.numpy())

    # VAE decode
    print("\nVAE decode...")
    vae_t0 = time.time()
    rgb = vae_decode(latent.to(device), args.vae_path, device)
    print(f"VAE decode: {time.time()-vae_t0:.1f}s, output: {rgb.shape}")

    # Save image
    img_np = rgb[0].cpu().permute(1,2,0).numpy()
    img_np = np.clip(img_np * 0.5 + 0.5, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    from PIL import Image
    Image.fromarray(img_np).save(args.output)
    print(f"Saved {args.output}")

    # Also save .npy for comparison
    np.save(args.output.replace('.png','.npy'), rgb[0].cpu().numpy())

    # Save C-compatible outputs for comparison
    np.save(args.output.replace('.png','_latent_final.npy'), latent.numpy())


if __name__ == "__main__":
    main()
