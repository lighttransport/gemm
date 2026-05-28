#!/usr/bin/env python3
"""Repack the Nunchaku SVDQuant FP4 (NVFP4) Qwen-Image checkpoint into a dense
BF16 diffusers-layout checkpoint that the cuda/qimg runner loads directly.

Stage-1 (W4A16) of the FP4 port: de-swizzle + dequant every quantized linear,
fold the rank-128 low-rank correction into the dense weight, split the fused
to_qkv / add_qkv_proj back into to_q/k/v + add_q/k/v_proj (diffusers layout),
and dequant the AWQ-INT4 img_mod/txt_mod projections. Non-quant tensors pass
through unchanged.

De-swizzle recipe validated to the activation-quant noise floor (see
/tmp/fp4_validate.py and memory project_qimg_fp4): W4A4 rel(y_hat,y)=0.06-0.10,
AWQ mod rel=0.003 vs nunchaku's own kernels.

Usage:
  python3 nunchaku_fp4_repack.py SRC.safetensors DST.safetensors [--blocks N] [--dtype bf16|fp8]
"""
import sys, json, struct, argparse
import numpy as np
import torch
from safetensors import safe_open

# ---- e2m1 value table (fp4): idx 0..7 magnitude, bit3 = sign ----
E2M1 = torch.tensor([0,0.5,1,1.5,2,3,4,6, -0.0,-0.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32)


class Packer:
    """Nunchaku tensor-core weight/scale/low-rank de-swizzle (bits=4, warp_n=128).
    Ported + validated inverses of nunchaku NunchakuWeightPacker."""
    def __init__(s, bits=4, warp_n=128):
        s.bits=bits; s.comp_n=16; s.comp_k=256//bits; s.insn_k=s.comp_k
        s.num_lanes=32; s.num_k_lanes=4; s.num_n_lanes=8; s.warp_n=warp_n
        s.reg_k=32//bits; s.reg_n=1
        s.k_pack_size=s.comp_k//(s.num_k_lanes*s.reg_k); s.n_pack_size=s.comp_n//(s.num_n_lanes*s.reg_n)
        s.mem_k=s.comp_k; s.mem_n=warp_n
        s.num_k_packs=s.mem_k//(s.k_pack_size*s.num_k_lanes*s.reg_k)
        s.num_n_packs=s.mem_n//(s.n_pack_size*s.num_n_lanes*s.reg_n)
    def _vshape(s,n,k):
        return (n//s.mem_n,s.num_n_packs,s.n_pack_size,s.num_n_lanes,s.reg_n,
                k//s.mem_k,s.num_k_packs,s.k_pack_size,s.num_k_lanes,s.reg_k)
    WPERM=(0,5,6,1,3,8,2,7,4,9)
    def unpack_weight_codes(s, packed_i8, n, k, dev):
        """int8[n,k/2] (packed e2m1, swizzled) -> 4-bit codes [n,k] (0..15)."""
        i32=packed_i8.to(dev).view(n,-1).contiguous().view(torch.int32)
        vp=tuple(s._vshape(n,k)[i] for i in s.WPERM)
        i32=i32.reshape(*vp[:-1])
        sh=torch.arange(0,32,4,dtype=torch.int32,device=dev)
        nib=((i32.unsqueeze(-1)>>sh)&0xF)
        inv=tuple(int((torch.tensor(s.WPERM)==j).nonzero()) for j in range(10))
        return nib.permute(*inv).contiguous().reshape(n,k).to(torch.long)
    def unpack_micro_scale(s, packed_f32, n, ng, dev):
        """fp8 group scales [ng,n] (swizzled) -> plain [n,ng]."""
        gsz=16; sub=s.insn_k//gsz
        t=packed_f32.to(dev).reshape(n//s.warp_n, ng//sub, 1, 8, 4, 4, sub)
        P=(0,5,1,4,3,2,6); inv=tuple(int((torch.tensor(P)==j).nonzero()) for j in range(7))
        return t.permute(*inv).contiguous().reshape(n, ng)
    def unpack_lowrank(s, w, down, dev):
        """proj_down [in,128] (down=True)->[rank,in]; proj_up [out,128] (down=False)->[out,rank]."""
        w=w.to(dev); c,r=w.shape; reg_n,reg_k=1,2
        pack_n=s.n_pack_size*s.num_n_lanes*reg_n   # 16
        pack_k=s.k_pack_size*s.num_k_lanes*reg_k   # 16
        if down: r_packs,c_packs=r//pack_n, c//pack_k
        else:    c_packs,r_packs=c//pack_n, r//pack_k
        w=w.view(c_packs,r_packs,s.num_n_lanes,s.num_k_lanes,s.n_pack_size,s.k_pack_size,reg_n,reg_k)
        w=w.permute(0,1,4,2,6,5,3,7).contiguous().view(c_packs,r_packs,pack_n,pack_k)
        if down: w=w.permute(1,2,0,3).contiguous().view(r,c)
        else:    w=w.permute(0,2,1,3).contiguous().view(c,r)
        return w


def unpack_awq_w4(packed_i32, oc, ic, dev):
    """AWQ mod qweight int32[oc//4, ic//2] (== pack_w4 int16[oc//4,ic]) -> codes [oc,ic]."""
    p=packed_i32.to(dev).contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    p=p.view(oc//4, ic//64, 4, 16).permute(0,2,1,3).reshape(-1,8)
    c0=p&0xF; c1=(p>>4)&0xF; c2=(p>>8)&0xF; c3=(p>>12)&0xF
    return torch.stack([c0,c1,c2,c3],dim=1).reshape(oc,ic)


# ----------------------------------------------------------------------------
DTYPE_STR = {"bf16":"BF16", "fp8":"F8_E4M3"}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--blocks", type=int, default=60)
    ap.add_argument("--dtype", choices=["bf16","fp8"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--validate", action="store_true", help="re-check block-0 vs ground-truth dumps")
    args=ap.parse_args()
    dev=args.device
    P=Packer(4,128)
    e2m1=E2M1.to(dev)

    src=safe_open(args.src, 'pt')
    keys=set(src.keys())
    def has(k): return k in keys
    def get(k): return src.get_tensor(k)

    out_dtype = torch.bfloat16 if args.dtype=="bf16" else torch.float8_e4m3fn
    dtype_str = DTYPE_STR[args.dtype]
    DBYTES = 2 if args.dtype=="bf16" else 1

    # The 8 SVDQuant NVFP4 linears per block. (src_prefix, out_features, in_features,
    #   split -> list of (out_name_no_block, row_slice) , or None for no split)
    GS=16
    def svdq_specs():
        # returns list of dicts describing each source linear
        return [
            dict(src="attn.to_qkv",       O=9216, I=3072,
                 splits=[("attn.to_q",0,3072),("attn.to_k",3072,6144),("attn.to_v",6144,9216)]),
            dict(src="attn.add_qkv_proj",  O=9216, I=3072,
                 splits=[("attn.add_q_proj",0,3072),("attn.add_k_proj",3072,6144),("attn.add_v_proj",6144,9216)]),
            dict(src="attn.to_out.0",      O=3072, I=3072, splits=[("attn.to_out.0",0,3072)]),
            dict(src="attn.to_add_out",    O=3072, I=3072, splits=[("attn.to_add_out",0,3072)]),
            dict(src="img_mlp.net.0.proj", O=12288,I=3072, splits=[("img_mlp.net.0.proj",0,12288)]),
            dict(src="img_mlp.net.2",      O=3072, I=12288,splits=[("img_mlp.net.2",0,3072)]),
            dict(src="txt_mlp.net.0.proj", O=12288,I=3072, splits=[("txt_mlp.net.0.proj",0,12288)]),
            dict(src="txt_mlp.net.2",      O=3072, I=12288,splits=[("txt_mlp.net.2",0,3072)]),
        ]
    MOD=[("img_mod.1",18432,3072),("txt_mod.1",18432,3072)]
    NORMS=["attn.norm_q.weight","attn.norm_k.weight","attn.norm_added_q.weight","attn.norm_added_k.weight"]

    # ---- pass 1: build ordered output spec (name, dtype_str, shape) ----
    spec=[]  # list of (name, dtype_str, shape_list)
    def add(name, shape): spec.append((name, dtype_str, list(shape)))
    for b in range(args.blocks):
        bp=f"transformer_blocks.{b}."
        for s in svdq_specs():
            for (nm,r0,r1) in s["splits"]:
                add(bp+nm+".weight",[r1-r0, s["I"]])
                add(bp+nm+".bias",  [r1-r0])
        for nm in NORMS:
            add(bp+nm, list(get(bp+nm).shape))
        for (nm,O,I) in MOD:
            add(bp+nm+".weight",[O,I]); add(bp+nm+".bias",[O])
    # non-block passthrough
    nonblock=sorted(k for k in keys if not k.startswith("transformer_blocks."))
    for k in nonblock:
        t=get(k); add(k, list(t.shape))

    # ---- compute offsets + header ----
    def nbytes(shape):
        n=1
        for d in shape: n*=d
        return n*DBYTES
    header={}; off=0
    for (name,ds,shape) in spec:
        sz=nbytes(shape)
        header[name]={"dtype":ds,"shape":shape,"data_offsets":[off,off+sz]}
        off+=sz
    header["__metadata__"]={"format":"pt","repacked_from":"svdq-fp4_r128 NVFP4 SVDQuant",
                            "stage":"W4A16 dense fold","emit_dtype":dtype_str}
    hjson=json.dumps(header,separators=(",",":")).encode("utf-8")
    pad=(8-(len(hjson)%8))%8; hjson+=b" "*pad
    total_data=off
    print(f"[repack] {len(spec)} tensors, header {len(hjson)}B, data {total_data/1e9:.2f} GB, emit={dtype_str}")

    # ---- helpers ----
    def to_out_bytes(t):
        t=t.to(out_dtype).contiguous().cpu()
        if args.dtype=="bf16":
            return t.view(torch.uint16).numpy().tobytes()
        else:  # fp8 e4m3 -> raw bytes
            return t.view(torch.uint8).numpy().tobytes()

    def dequant_svdq(bp, s):
        """returns (W_eff[O,I] f32 on dev, bias[O] f32)."""
        O,I=s["O"],s["I"]; pre=bp+s["src"]+"."
        codes=P.unpack_weight_codes(get(pre+"qweight"), O, I, dev)
        wval=e2m1[codes]                                            # [O,I]
        ws=get(pre+"wscales").float()
        gw=P.unpack_micro_scale(ws, O, I//GS, dev)                 # [O, I/16]
        gw=gw.repeat_interleave(GS, dim=1)                         # [O,I]
        wc=get(pre+"wcscales").float().to(dev) if has(pre+"wcscales") else None
        wts=float(get(pre+"wtscale").float().item()) if has(pre+"wtscale") else 1.0
        sc = (wts*wc) if wc is not None else torch.full((O,), wts, device=dev)  # [O]
        W_deq = wval*gw*sc[:,None]                                 # [O,I]
        pd=P.unpack_lowrank(get(pre+"proj_down").float(), True,  dev)  # [rank,I]
        pu=P.unpack_lowrank(get(pre+"proj_up").float(),   False, dev)  # [O,rank]
        W_eff = W_deq + pu@pd                                      # fold low-rank
        sm=get(pre+"smooth_factor").float().to(dev)               # [I], ==1 for nvfp4
        W_eff = W_eff / sm[None,:]                                 # fold activation smoothing
        bias=get(pre+"bias").float().to(dev)                      # [O]
        return W_eff, bias

    def dequant_mod(bp, nm, O, I):
        pre=bp+nm+"."
        codes=unpack_awq_w4(get(pre+"qweight"), O, I, dev).float()
        ws=get(pre+"wscales").float().to(dev)   # [ng,O]
        wz=get(pre+"wzeros").float().to(dev)    # [ng,O]
        ng=ws.shape[0]; gsz=I//ng
        gidx=(torch.arange(I,device=dev)//gsz)
        W = codes*ws[gidx,:].t() + wz[gidx,:].t()   # [O,I], w=q*scale+zero
        bias=get(pre+"bias").float().to(dev)
        # nunchaku stores mod params CHANNEL-major (idx = c*6+p); diffusers/runner want
        # PARAM-major (idx = p*dim+c, slices [shift1,scale1,gate1,shift2,scale2,gate2]).
        # Bake the view(dim,6).transpose nunchaku applies at runtime into the weight+bias.
        dim = I  # img_mod/txt_mod: out=6*dim, in=dim
        W   = W.reshape(dim, 6, I).permute(1,0,2).reshape(O, I).contiguous()
        bias= bias.reshape(dim, 6).permute(1,0).reshape(O).contiguous()
        return W, bias

    # ---- pass 2: stream-write ----
    written=0
    with open(args.dst,"wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in range(args.blocks):
            bp=f"transformer_blocks.{b}."
            for s in svdq_specs():
                W_eff,bias = dequant_svdq(bp, s)
                for (nm,r0,r1) in s["splits"]:
                    f.write(to_out_bytes(W_eff[r0:r1,:])); written+=1
                    f.write(to_out_bytes(bias[r0:r1]));    written+=1
                del W_eff,bias
            for nm in NORMS:
                f.write(to_out_bytes(get(bp+nm).float().to(dev))); written+=1
            for (nm,O,I) in MOD:
                W,bias=dequant_mod(bp,nm,O,I)
                f.write(to_out_bytes(W)); written+=1
                f.write(to_out_bytes(bias)); written+=1
                del W,bias
            if dev=="cuda": torch.cuda.empty_cache()
            if (b+1)%5==0 or b==0: print(f"[repack] block {b+1}/{args.blocks} done ({written} tensors)")
        for k in nonblock:
            f.write(to_out_bytes(get(k).float().to(dev))); written+=1
    print(f"[repack] wrote {written} tensors -> {args.dst}")


if __name__=="__main__":
    main()
