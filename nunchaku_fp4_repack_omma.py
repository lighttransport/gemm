#!/usr/bin/env python3
"""Repack the Nunchaku SVDQuant FP4 checkpoint into an OMMA-layout checkpoint for
the cuda/qimg runner's native W4A4 path (Stage 2).

Per W4A4 linear (fused qkv split into q/k/v + add_q/k/v): emit
  .qweight_fp4  I32  [out, in/8]   8 e2m1 codes / uint32, plain de-swizzled, row-major
  .wscales_fp4  U8   [out, in/16]  de-swizzled e4m3 group scales (raw bytes)
  .wcwt         F32  [out]         wcscales * wtscale (per-output-channel)
  .proj_down_fp8 F8_E4M3 [128, in] low-rank down (for op_gemm lora_act = x@pd^T)
  .proj_up_fp8   F8_E4M3 [out,128] low-rank up
  .bias         F32  [out]
img_mod.1/txt_mod.1 -> dense FP8 .weight (channel->param permute, Stage-1 fix) + bias.
Non-block + norms pass through (FP8 for *.weight, F32/BF16 otherwise).

Usage: python3 nunchaku_fp4_repack_omma.py SRC DST [--blocks N]
"""
import sys, argparse, numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor([0,0.5,1,1.5,2,3,4,6,-0.,-0.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float32)

class Packer:
    def __init__(s,bits=4,warp_n=128):
        s.bits=bits;s.comp_n=16;s.comp_k=256//bits;s.insn_k=s.comp_k
        s.num_lanes=32;s.num_k_lanes=4;s.num_n_lanes=8;s.warp_n=warp_n
        s.reg_k=32//bits;s.reg_n=1
        s.k_pack_size=s.comp_k//(s.num_k_lanes*s.reg_k);s.n_pack_size=s.comp_n//(s.num_n_lanes*s.reg_n)
        s.mem_k=s.comp_k;s.mem_n=warp_n
        s.num_k_packs=s.mem_k//(s.k_pack_size*s.num_k_lanes*s.reg_k)
        s.num_n_packs=s.mem_n//(s.n_pack_size*s.num_n_lanes*s.reg_n)
    def _vshape(s,n,k):
        return (n//s.mem_n,s.num_n_packs,s.n_pack_size,s.num_n_lanes,s.reg_n,
                k//s.mem_k,s.num_k_packs,s.k_pack_size,s.num_k_lanes,s.reg_k)
    WPERM=(0,5,6,1,3,8,2,7,4,9)
    def unpack_weight_codes(s,packed,n,k,dev):
        i32=packed.to(dev).view(n,-1).contiguous().view(torch.int32)
        vp=tuple(s._vshape(n,k)[i] for i in s.WPERM);i32=i32.reshape(*vp[:-1])
        sh=torch.arange(0,32,4,dtype=torch.int32,device=dev);nib=((i32.unsqueeze(-1)>>sh)&0xF)
        inv=tuple(int((torch.tensor(s.WPERM)==j).nonzero()) for j in range(10))
        return nib.permute(*inv).contiguous().reshape(n,k).to(torch.int32)   # [n,k] 0..15
    def unpack_micro_bytes(s,packed_u8,n,ng,dev):
        gsz=16;sub=s.insn_k//gsz
        t=torch.from_numpy(packed_u8.astype(np.int16)).to(dev).reshape(n//s.warp_n,ng//sub,1,8,4,4,sub)
        P=(0,5,1,4,3,2,6);inv=tuple(int((torch.tensor(P)==j).nonzero()) for j in range(7))
        return t.permute(*inv).contiguous().reshape(n,ng).to(torch.uint8)     # [n,ng] e4m3 bytes
    def unpack_lowrank(s,w,down,dev):
        w=w.to(dev);c,r=w.shape;reg_n,reg_k=1,2
        pk_n=s.n_pack_size*s.num_n_lanes*reg_n;pk_k=s.k_pack_size*s.num_k_lanes*reg_k
        if down: r_p,c_p=r//pk_n,c//pk_k
        else:    c_p,r_p=c//pk_n,r//pk_k
        w=w.view(c_p,r_p,s.num_n_lanes,s.num_k_lanes,s.n_pack_size,s.k_pack_size,reg_n,reg_k)
        w=w.permute(0,1,4,2,6,5,3,7).contiguous().view(c_p,r_p,pk_n,pk_k)
        if down: w=w.permute(1,2,0,3).contiguous().view(r,c)
        else:    w=w.permute(0,2,1,3).contiguous().view(c,r)
        return w
def unpack_awq_w4(packed_i32,oc,ic,dev):
    p=packed_i32.to(dev).contiguous().view(torch.int16).to(torch.int32)&0xFFFF
    p=p.view(oc//4,ic//64,4,16).permute(0,2,1,3).reshape(-1,8)
    c0=p&0xF;c1=(p>>4)&0xF;c2=(p>>8)&0xF;c3=(p>>12)&0xF
    return torch.stack([c0,c1,c2,c3],dim=1).reshape(oc,ic)

def pack_codes_u32(codes):  # [O,I] int32 0..15 -> [O,I/8] int32 (8 codes/uint, low nibble low k)
    O,I=codes.shape; c=codes.reshape(O,I//8,8)
    sh=torch.arange(0,32,4,dtype=torch.int32,device=codes.device)
    return (c.to(torch.int32)<<sh).sum(-1).to(torch.int32)   # [O,I/8]

# E2M1 magnitudes (positive codes 0..7) and round-to-nearest midpoint thresholds.
_E2M1_MAG = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.])
_E2M1_THR = torch.tensor([0.25,0.75,1.25,1.75,2.5,3.5,5.0])

def quant_nvfp4(W, dev, GS=16):
    """Quantize a weight matrix W[O,I] to the OMMA NVFP4 layout that the runtime
    decodes as  W[o,k] ≈ E2M1[code] * e4m3(wscales[o,k//GS]) * wcwt[o].
    Two-level scale: per-row wcwt = max group amax / 6; per-16-group e4m3 scale
    is the group amax relative to that (in (0,1], well inside e4m3 range). Codes
    are round-to-nearest e2m1. Returns (qweight_fp4 [O,I/8] i32, wscales_fp4
    [O,I/GS] u8, wcwt [O] f32) on CPU. Memory-frugal (no [.,16] broadcast)."""
    O, I = W.shape
    W = W.to(dev).float()
    Wg = W.reshape(O, I // GS, GS)
    amax_g = Wg.abs().amax(dim=2)                       # [O, I/GS]
    wcwt = (amax_g.amax(dim=1) / 6.0).clamp_min(1e-12)  # [O]
    gscale_rel = (amax_g / 6.0) / wcwt.unsqueeze(1)     # [O, I/GS] in (0,1]
    ws_e4m3 = gscale_rel.to(torch.float8_e4m3fn)        # quantize group scale to e4m3
    eff = (ws_e4m3.float() * wcwt.unsqueeze(1)).clamp_min(1e-12)  # [O, I/GS] effective scale
    Wn = (Wg / eff.unsqueeze(2))                        # normalized to ~[-6,6]
    mag = Wn.abs()
    thr = _E2M1_THR.to(dev)
    midx = torch.bucketize(mag, thr).to(torch.int32)    # 0..7 -> {0,.5,1,1.5,2,3,4,6}
    codes = (midx + (Wn < 0).to(torch.int32) * 8).reshape(O, I)  # negatives -> 8..15
    qw = pack_codes_u32(codes)                          # [O, I/8] i32
    return qw.contiguous().cpu(), ws_e4m3.view(torch.uint8).contiguous().cpu(), wcwt.float().cpu()

def _decode_nvfp4(qw, ws_u8, wcwt, O, I, dev, GS=16):
    """Inverse of quant_nvfp4 (for the self-test): reconstruct W[O,I]."""
    e2 = E2M1.to(dev)
    sh = torch.arange(0,32,4,dtype=torch.int32,device=dev)
    codes = ((qw.to(dev).view(O, I//8, 1) >> sh) & 0xF).reshape(O, I)   # [O,I]
    vals = e2[codes.long()]                                            # [O,I]
    ws = ws_u8.to(dev).view(torch.float8_e4m3fn).float().view(O, I//GS)
    gidx = torch.arange(I, device=dev) // GS
    return vals * ws[:, gidx] * wcwt.to(dev).unsqueeze(1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("src",nargs="?");ap.add_argument("dst",nargs="?")
    ap.add_argument("--blocks",type=int,default=60); ap.add_argument("--device",default="cuda")
    ap.add_argument("--mod-fp4",action="store_true",dest="mod_fp4",
                    help="quantize img_mod.1/txt_mod.1 to NVFP4 (W4A16) instead of dense FP8 "
                         "so all 60 blocks fit resident (lossy adaLN; see cuda/qimg runner QIMG_MOD_FP4)")
    ap.add_argument("--selftest",action="store_true",help="validate quant_nvfp4 round-trip and exit")
    a=ap.parse_args(); dev=a.device; P=Packer(); e2m1=E2M1.to(dev); GS=16
    if a.selftest:
        torch.manual_seed(0)
        for (O,I) in [(18432,3072),(256,512)]:
            W=(torch.randn(O,I,device=dev)*0.05)
            qw,ws,wcwt=quant_nvfp4(W,dev)
            R=_decode_nvfp4(qw,ws,wcwt,O,I,dev)
            rel=(R-W.to(dev)).norm()/W.to(dev).norm()
            print(f"[selftest] W[{O},{I}] qweight_fp4{tuple(qw.shape)} wscales_fp4{tuple(ws.shape)} "
                  f"wcwt{tuple(wcwt.shape)}  rel_L2={rel.item():.4f}")
        return
    src=safe_open(a.src,'pt'); keys=set(src.keys())
    def has(k):return k in keys
    def G(k):return src.get_tensor(k)
    out={}
    SV=[("attn.to_qkv",9216,3072,[("attn.to_q",0,3072),("attn.to_k",3072,6144),("attn.to_v",6144,9216)]),
        ("attn.add_qkv_proj",9216,3072,[("attn.add_q_proj",0,3072),("attn.add_k_proj",3072,6144),("attn.add_v_proj",6144,9216)]),
        ("attn.to_out.0",3072,3072,[("attn.to_out.0",0,3072)]),
        ("attn.to_add_out",3072,3072,[("attn.to_add_out",0,3072)]),
        ("img_mlp.net.0.proj",12288,3072,[("img_mlp.net.0.proj",0,12288)]),
        ("img_mlp.net.2",3072,12288,[("img_mlp.net.2",0,3072)]),
        ("txt_mlp.net.0.proj",12288,3072,[("txt_mlp.net.0.proj",0,12288)]),
        ("txt_mlp.net.2",3072,12288,[("txt_mlp.net.2",0,3072)])]
    NORMS=["attn.norm_q.weight","attn.norm_k.weight","attn.norm_added_q.weight","attn.norm_added_k.weight"]
    def fp8(t): return t.to(torch.float8_e4m3fn)
    for b in range(a.blocks):
        bp=f"transformer_blocks.{b}."
        for (sname,O,I,splits) in SV:
            pre=bp+sname+"."
            codes=P.unpack_weight_codes(G(pre+"qweight"),O,I,dev)          # [O,I]
            wsb=P.unpack_micro_bytes(G(pre+"wscales").view(torch.uint8).numpy(),O,I//GS,dev) # [O,I/16]
            wc=G(pre+"wcscales").float().to(dev) if has(pre+"wcscales") else torch.ones(O,device=dev)
            wts=float(G(pre+"wtscale").float().item()) if has(pre+"wtscale") else 1.0
            wcwt=(wc*wts).float()
            pd=P.unpack_lowrank(G(pre+"proj_down").float(),True,dev)       # [128,I]
            pu=P.unpack_lowrank(G(pre+"proj_up").float(),False,dev)        # [O,128]
            bias=G(pre+"bias").float().to(dev)
            qw_u32=pack_codes_u32(codes)                                  # [O,I/8]
            for (nm,r0,r1) in splits:
                op=bp+nm+"."
                out[op+"qweight_fp4"]=qw_u32[r0:r1,:].contiguous().cpu()
                out[op+"wscales_fp4"]=wsb[r0:r1,:].contiguous().cpu()
                out[op+"wcwt"]=wcwt[r0:r1].contiguous().cpu()
                out[op+"proj_down_fp8"]=fp8(pd).contiguous().cpu()        # shared [128,I]
                out[op+"proj_up_fp8"]=fp8(pu[r0:r1,:]).contiguous().cpu() # [r1-r0,128]
                out[op+"bias"]=bias[r0:r1].contiguous().cpu()
        for nm in NORMS: out[bp+nm]=G(bp+nm).float().cpu()
        # mod layers -> dense FP8 with channel->param permute (Stage-1 fix)
        for nm in ["img_mod.1","txt_mod.1"]:
            pre=bp+nm+"."; O,I=18432,3072
            codes=unpack_awq_w4(G(pre+"qweight"),O,I,dev).float()
            ws=G(pre+"wscales").float().to(dev); wz=G(pre+"wzeros").float().to(dev)
            ng=ws.shape[0]; gsz=I//ng; gidx=torch.arange(I,device=dev)//gsz
            W=codes*ws[gidx,:].t()+wz[gidx,:].t()
            bias=G(pre+"bias").float().to(dev)
            W=W.reshape(I,6,I).permute(1,0,2).reshape(O,I).contiguous()   # dim=I=3072
            bias=bias.reshape(I,6).permute(1,0).reshape(O).contiguous()
            if a.mod_fp4:
                qw,ws_u8,wcwt=quant_nvfp4(W,dev)     # NVFP4 (W4A16): half the FP8 footprint
                out[bp+nm+".qweight_fp4"]=qw; out[bp+nm+".wscales_fp4"]=ws_u8
                out[bp+nm+".wcwt"]=wcwt;       out[bp+nm+".bias"]=bias.float().cpu()
            else:
                out[bp+nm+".weight"]=fp8(W).cpu(); out[bp+nm+".bias"]=bias.float().cpu()
        if (b+1)%10==0 or b==0: print(f"[omma] block {b+1}/{a.blocks}")
        if dev=="cuda": torch.cuda.empty_cache()
    for k in sorted(x for x in keys if not x.startswith("transformer_blocks.")):
        t=G(k)
        out[k]=fp8(t.float().to(dev)).cpu() if k.endswith(".weight") and t.dim()==2 else t.cpu()
    print(f"[omma] saving {len(out)} tensors -> {a.dst}")
    save_file(out,a.dst,metadata={"format":"pt","stage":"W4A4 OMMA"})
    print("[omma] done")

if __name__=="__main__": main()
