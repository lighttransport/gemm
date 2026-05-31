#!/usr/bin/env python3
"""Pack the bf16 Qwen-Image DiT to a simple RTN int4-g16 checkpoint that the HIP
runner loads resident (~half-byte/param) and runs via gemm_int4w_g16.

For the 12 per-block linears + img_mod.1/txt_mod.1, each weight [out,in] (in%16==0)
becomes:
  <key>.qint4  uint8  [out, in/2]   packed signed nibbles (-7..7), kp even=low, odd=high
  <key>.wscale bf16   [out, in/16]  per-(out,group16) scale = max(|w|)/7
  <key>.bias   bf16                 copied
Everything else (norms, img_in/txt_in, time embed, norm_out/proj_out, biases) is
copied bf16 unchanged. Matches the keys hip_qimg_load_dit_int4 reads; group_size=16
is auto-detected from wscale and dispatched to the simple kernel.
"""
import sys, torch
from safetensors import safe_open
from safetensors.torch import save_file

LIN = {"attn.to_q","attn.to_k","attn.to_v","attn.to_out.0",
       "attn.add_q_proj","attn.add_k_proj","attn.add_v_proj","attn.to_add_out",
       "img_mlp.net.0.proj","img_mlp.net.2","txt_mlp.net.0.proj","txt_mlp.net.2",
       "img_mod.1","txt_mod.1"}

def is_lin_weight(name):
    if not name.endswith(".weight"): return None
    body = name[:-len(".weight")]
    # body = transformer_blocks.<N>.<suffix>
    for suf in LIN:
        if body.endswith("." + suf) and body.startswith("transformer_blocks."):
            return body
    return None

def pack(w):  # w: [out, in] float32, in%16==0
    out, k = w.shape
    g = w.reshape(out, k//16, 16)
    scale = (g.abs().amax(dim=2, keepdim=True) / 7.0).clamp(min=1e-12).to(torch.bfloat16)
    scq = scale.float()
    q = torch.round((g / scq).clamp(-7, 7)).to(torch.int8)        # [out, k/16, 16]
    q = q.reshape(out, k)
    lo = (q[:, 0::2] & 0xF).to(torch.uint8)
    hi = (q[:, 1::2] & 0xF).to(torch.uint8)
    qint4 = (lo | (hi << 4))                                       # [out, k/2]
    return qint4.contiguous(), scale.reshape(out, k//16).contiguous()

def main():
    src, dst = sys.argv[1], sys.argv[2]
    out = {}
    packed = {}  # body -> out_dim
    with safe_open(src, "pt") as f:
        keys = list(f.keys())
        for name in keys:
            body = is_lin_weight(name)
            if body is not None:
                w = f.get_tensor(name).float()
                qint4, wscale = pack(w)
                out[body + ".qint4"] = qint4
                out[body + ".wscale"] = wscale
                packed[body] = w.shape[0]
            else:
                out[name] = f.get_tensor(name).to(torch.bfloat16)  # bf16 passthrough (incl .bias)
    # The loader requires a .bias per int4 linear; synthesize zeros where absent.
    for body, odim in packed.items():
        if body + ".bias" not in out:
            out[body + ".bias"] = torch.zeros(odim, dtype=torch.bfloat16)
    print(f"packed {len(packed)} linears; {len(out)} tensors total -> {dst}")
    save_file(out, dst)
    print("done")

if __name__ == "__main__":
    main()
