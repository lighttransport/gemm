#!/usr/bin/env python3
"""Trace block 0 intermediates in ComfyUI and save for comparison with C."""
import os, sys, math
import numpy as np
import torch

def our_prng_noise(seed, n):
    state = seed
    vals = np.zeros(n, dtype=np.float32)
    for i in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1<<64)-1)
        u1 = (state >> 11) / (1 << 53)
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1<<64)-1)
        u2 = (state >> 11) / (1 << 53)
        if u1 < 1e-10: u1 = 1e-10
        vals[i] = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return vals

sys.path.insert(0, '/mnt/disk01/ComfyUI')
os.chdir('/mnt/disk01/ComfyUI')
import comfy.sd, comfy.model_management

model_dir = '/mnt/disk01/models/qwen-image-st'
dit_path = os.path.join(model_dir, 'diffusion_models', 'qwen_image_fp8_e4m3fn.safetensors')
print("Loading DiT...")
model_patcher = comfy.sd.load_diffusion_model(dit_path)
comfy.model_management.load_models_gpu([model_patcher])
device = comfy.model_management.get_torch_device()
dm = model_patcher.model.diffusion_model

# Text
clip_path = os.path.join(model_dir, 'text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
print("Loading CLIP...")
clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
tokens = clip.tokenize("a red apple on a white table")
positive = clip.encode_from_tokens_scheduled(tokens)
txt_hs = positive[0][0].to(device).bfloat16()

# Noise
noise = our_prng_noise(42, 16*32*32).reshape(1, 16, 1, 32, 32)
x = torch.from_numpy(noise).to(device).bfloat16()

with torch.no_grad():
    hidden, img_ids, orig = dm.process_img(x)
    img_proj = dm.img_in(hidden)
    txt_proj = dm.txt_in(dm.txt_norm(txt_hs))

    ms = model_patcher.model.model_sampling
    sigma = torch.tensor([1.0], device=device)
    timestep = ms.timestep(sigma)
    temb = dm.time_text_embed(timestep, img_proj)

    # Save img_proj for comparison
    ip = img_proj.float().detach().cpu().numpy()[0]
    np.save('output/cf_img_projected.npy', ip)
    print(f"img_projected: {ip.shape} std={ip.std():.4f}")

    # Block 0 internals with hooks
    blk = dm.transformer_blocks[0]

    # Modulation
    mod = blk.img_mod(temb)
    mp = mod.float().detach().cpu().numpy()[0]
    np.save('output/cf_block0_mod.npy', mp)
    print(f"block0 mod: shift1={np.abs(mp[:3072]).max():.2f} scale1={np.abs(mp[3072:6144]).max():.2f}")

    # adaLN
    img_mod1, img_mod2 = mod.chunk(2, dim=-1)
    txt_mod1, txt_mod2 = blk.txt_mod(temb).chunk(2, dim=-1)

    img_normed = blk.img_norm1(img_proj)
    img_modulated, img_gate1 = blk._modulate(img_normed, img_mod1)
    np.save('output/cf_block0_img_adaln.npy', img_modulated.float().detach().cpu().numpy()[0])
    print(f"img_adaln: std={img_modulated.float().std():.4f}")

    txt_normed = blk.txt_norm1(txt_proj)
    txt_modulated, txt_gate1 = blk._modulate(txt_normed, txt_mod1)

    # Attention QKV
    attn = blk.attn
    img_q = attn.to_q(img_modulated)
    img_k = attn.to_k(img_modulated)
    img_v = attn.to_v(img_modulated)
    txt_q = attn.add_q_proj(txt_modulated)
    txt_k = attn.add_k_proj(txt_modulated)

    np.save('output/cf_block0_img_q.npy', img_q.float().detach().cpu().numpy()[0])
    print(f"img_q: std={img_q.float().std():.4f}")

    # Run full block
    enc_out, img_out = blk(
        hidden_states=img_proj,
        encoder_hidden_states=txt_proj,
        encoder_hidden_states_mask=None,
        temb=temb,
        image_rotary_emb=dm.pe_embedder(
            torch.cat((
                torch.arange(8, 20, device=device).reshape(1,-1,1).repeat(1,1,3).bfloat16(),
                img_ids
            ), dim=1)
        ).to(torch.bfloat16).contiguous(),
    )
    np.save('output/cf_block0_img_out.npy', img_out.float().detach().cpu().numpy()[0])
    print(f"block0 img_out: std={img_out.float().std():.4f}")

    # Run all blocks, save after block 0 and final
    h = img_proj.clone()
    e = txt_proj.clone()
    ids = torch.cat((
        torch.arange(8, 20, device=device).reshape(1,-1,1).repeat(1,1,3).bfloat16(),
        img_ids
    ), dim=1)
    pe = dm.pe_embedder(ids).to(torch.bfloat16).contiguous()

    for i, b in enumerate(dm.transformer_blocks):
        e, h = b(hidden_states=h, encoder_hidden_states=e,
                 encoder_hidden_states_mask=None, temb=temb, image_rotary_emb=pe)
        if i == 0:
            np.save('output/cf_after_block0_img.npy', h.float().detach().cpu().numpy()[0])
            print(f"after block 0: img std={h.float().std():.4f}")

    h = dm.norm_out(h, temb)
    h = dm.proj_out(h)
    fo = h.float().detach().cpu().numpy()[0, :256]
    np.save('output/cf_final_output.npy', fo)
    print(f"final output: std={fo.std():.4f}")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Done. Saved to output/cf_*.npy")
