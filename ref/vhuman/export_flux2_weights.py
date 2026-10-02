"""Lossless Diffusers -> native FLUX.2 Klein DiT tensor layout conversion.

Streams BF16 tensor bytes without importing Torch or an inference runtime.
Names, QKV concatenation and final scale/shift permutation invert Diffusers'
documented Flux2 checkpoint conversion. No quantization is performed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

GLOBALS={'x_embedder':'img_in','context_embedder':'txt_in',
         'time_guidance_embed.timestep_embedder.linear_1':'time_in.in_layer',
         'time_guidance_embed.timestep_embedder.linear_2':'time_in.out_layer',
         'double_stream_modulation_img.linear':'double_stream_modulation_img.lin',
         'double_stream_modulation_txt.linear':'double_stream_modulation_txt.lin',
         'single_stream_modulation.linear':'single_stream_modulation.lin','proj_out':'final_layer.linear'}
DOUBLE={'attn.norm_q':'img_attn.norm.query_norm','attn.norm_k':'img_attn.norm.key_norm',
        'attn.to_out.0':'img_attn.proj','ff.linear_in':'img_mlp.0','ff.linear_out':'img_mlp.2',
        'attn.norm_added_q':'txt_attn.norm.query_norm','attn.norm_added_k':'txt_attn.norm.key_norm',
        'attn.to_add_out':'txt_attn.proj','ff_context.linear_in':'txt_mlp.0','ff_context.linear_out':'txt_mlp.2'}
SINGLE={'attn.to_qkv_mlp_proj':'linear1','attn.to_out':'linear2','attn.norm_q':'norm.query_norm','attn.norm_k':'norm.key_norm'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()


def export(source,output):
    source,output=Path(source),Path(output)
    with source.open('rb') as f:
        header_size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(header_size))
    tensors={k:v for k,v in header.items() if k!='__metadata__'}
    data_bytes=source.stat().st_size-8-header_size
    for spec in tensors.values():
        lo,hi=spec['data_offsets'];shape=spec['shape']
        if (spec['dtype']!='BF16' or not shape or any(type(n) is not int or n<1 for n in shape)
                or not 0<=lo<hi<=data_bytes or hi-lo!=2*math.prod(shape)):
            raise ValueError('expected valid BF16 tensor ranges and shapes')
    plan={};consumed=set()
    for key,spec in tensors.items():
        if key in consumed:continue
        prefix,kind=key.rsplit('.',1);pieces=[spec['data_offsets']];shape=spec['shape'].copy()
        if prefix in GLOBALS:new=GLOBALS[prefix]+'.'+kind
        elif prefix=='norm_out.linear':
            new='final_layer.adaLN_modulation.1.'+kind
            lo,hi=pieces[0];mid=(lo+hi)//2;pieces=[(mid,hi),(lo,mid)]
        elif prefix.startswith('single_transformer_blocks.'):
            _,block,rest=prefix.split('.',2)
            new='single_blocks.'+block+'.'+SINGLE[rest]+('.scale' if rest in ('attn.norm_q','attn.norm_k') else '.'+kind)
        elif prefix.startswith('transformer_blocks.'):
            _,block,rest=prefix.split('.',2)
            group=next((keys for keys in (('attn.to_q','attn.to_k','attn.to_v'),('attn.add_q_proj','attn.add_k_proj','attn.add_v_proj')) if rest in keys),None)
            if group:
                keys=['transformer_blocks.'+block+'.'+v+'.'+kind for v in group]
                parts=[tensors[k] for k in keys]
                if any(p['dtype']!=spec['dtype'] or p['shape']!=shape for p in parts):raise ValueError('incompatible QKV')
                pieces=[p['data_offsets'] for p in parts];shape[0]*=3;consumed.update(keys)
                new='double_blocks.'+block+'.'+('img_attn' if rest in ('attn.to_q','attn.to_k','attn.to_v') else 'txt_attn')+'.qkv.'+kind
            else:
                mapped=DOUBLE[rest];new='double_blocks.'+block+'.'+mapped+('.scale' if '.norm.' in mapped else '.'+kind)
        else:raise ValueError('unsupported tensor '+key)
        consumed.add(key)
        plan[new]=(spec['dtype'],shape,pieces)
    offset=0;out_header={}
    for name,(dtype,shape,pieces) in plan.items():
        size=sum(hi-lo for lo,hi in pieces)
        out_header[name]=dict(dtype=dtype,shape=shape,data_offsets=[offset,offset+size]);offset+=size
    raw=json.dumps(out_header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix(output.suffix+'.partial')
    if output.exists():raise ValueError('output already exists')
    try:
        with source.open('rb') as src,partial.open('wb') as dst:
            dst.write(struct.pack('<Q',len(raw)));dst.write(raw)
            for _,_,pieces in plan.values():
                for lo,hi in pieces:
                    src.seek(8+header_size+lo);remaining=hi-lo
                    while remaining:
                        data=src.read(min(8<<20,remaining))
                        if not data:raise ValueError('truncated source weights')
                        dst.write(data);remaining-=len(data)
        partial.replace(output)
    finally:partial.unlink(missing_ok=True)
    receipt=dict(format='vhuman.flux2_native_weights.v1',conversion='lossless_names_qkv_scale_shift',
                 source_sha256=digest(source),sha256=digest(output),tensors=len(plan))
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    print(json.dumps(export(**vars(p.parse_args())),indent=2))
