#!/usr/bin/env python3
"""Independent single-token NumPy reference derived from inference/model.py.

Reads original safetensors, not the C loader or staged binary layout. Intended
for bounded correctness runs, not serving/performance. BLAS thread count should
be explicitly limited when run on the frontend. No PyTorch/GPU parity claim.
"""
import argparse
import json
import math
import os
import struct
from pathlib import Path
import numpy as np

FP4 = np.array([0,.5,1,1.5,2,3,4,6],np.float32)
codes=np.arange(128,dtype=np.int32)
FP8=np.where(codes<8,codes/512,(1+(codes%8)/8)*np.exp2(codes//8-7)).astype(np.float32)
FP8[-1]=np.nan


def bf(x):
    x=np.asarray(x,dtype=np.float32)
    bits=x.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def nearest(x,table):
    a=np.abs(x)
    hi=np.minimum(np.searchsorted(table,a),len(table)-1)
    lo=np.maximum(hi-1,0)
    dl=a-table[lo];dh=table[hi]-a
    idx=np.where((dl<dh)|((dl==dh)&((lo%2)==0)),lo,hi)
    return np.copysign(table[idx],x)


def aq(x):
    x=bf(x);shape=x.shape;v=x.reshape(-1,32)
    maximum=np.maximum(np.max(np.abs(v),axis=-1,keepdims=True),np.float32(1e-4))
    scale=np.exp2(np.ceil(np.log2(maximum/np.float32(448)))).astype(np.float32)
    return (nearest(v/scale,FP8[:-1])*scale).reshape(shape)


def q4(x,group=32,e4=False):
    x=bf(x);shape=x.shape;v=x.reshape(-1,group)
    minimum=6/512 if e4 else 6*2**-126
    a=np.maximum(np.max(np.abs(v),axis=-1,keepdims=True),np.float32(minimum))
    scale=nearest(a/6,FP8[:-1]) if e4 else np.exp2(np.ceil(np.log2(a/6))).astype(np.float32)
    return bf(nearest(v/scale,FP4)*scale).reshape(shape)


def sigmoid(x):
    e=np.exp(-np.abs(x));return np.where(x>=0,1/(1+e),e/(1+e))


def swiglu(g,u):
    g=np.minimum(g,10);u=np.clip(u,-10,10)
    return g*sigmoid(g)*u


def hc_post(x,h,post,comb):
    # Match model.py's explicit FP32 multiply/reduce/add boundaries. A BLAS
    # matmul may fuse the products, unlike the upstream eager torch.sum.
    return bf(post[:,None]*x+np.sum(comb[:,:,None]*h[:,None,:],axis=0,dtype=np.float32))


class Reference:
    def __init__(self,model,metadata=None):
        self.model=Path(model);self.tensors={}
        for path in sorted(self.model.glob('*.safetensors')):
            with path.open('rb') as f:
                length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length))
            for name,d in header.items():
                if name!='__metadata__':self.tensors[name]=(path,8+length+d['data_offsets'][0],d)
        self.windows={};self.latents={};self.keys={};self.pool={};self.selected=np.array([],np.int32)
        self.candidates=None;self.history=[]
        if metadata:
            with open(metadata,'rb') as f:
                if f.read(9)!=b'DS41FENG1':raise ValueError('metadata magic')
                raw,compressed,heads=struct.unpack('<III',f.read(12))
                if (raw,compressed,heads)!=(129280,99092,8):raise ValueError('metadata contract')
                self.token_map=np.frombuffer(f.read(raw*4),dtype='<u4')
                self.multipliers=np.frombuffer(f.read(2*4*8),dtype='<u8').reshape(2,4)
                self.primes=np.frombuffer(f.read(2*3*8*8),dtype='<u8').reshape(2,3,8)

    def raw(self,name):
        path,offset,d=self.tensors[name]
        dtype={'BF16':'<u2','F32':'<f4','F8_E4M3':'u1','F8_E8M0':'u1','I8':'u1'}[d['dtype']]
        return np.memmap(path,mode='r',dtype=dtype,offset=offset,shape=tuple(d['shape']))

    def tensor(self,name):
        dtype=self.tensors[name][2]['dtype'];raw=self.raw(name)
        if dtype=='BF16':return (raw.astype(np.uint32)<<16).view(np.float32)
        if dtype=='F32':return np.asarray(raw,dtype=np.float32)
        raise ValueError('not a scalar tensor '+name)

    def byte_rows(self,name,ids):
        path,offset,desc=self.tensors[name];columns=desc['shape'][1]
        out=np.empty((len(ids),columns),np.uint8)
        with path.open('rb',buffering=0) as f:
            for i,row in enumerate(ids):
                block=os.pread(f.fileno(),columns,offset+int(row)*columns)
                if len(block)!=columns:raise IOError('short row read')
                out[i]=np.frombuffer(block,np.uint8)
        return out

    def weight(self,base):
        name=base+'.weight';raw=self.raw(name);dtype=self.tensors[name][2]['dtype']
        if dtype in ('BF16','F32'):return self.tensor(name)
        scale=np.exp2(self.raw(base+'.scale').astype(np.float32)-127)
        if dtype=='F8_E4M3':
            w=np.copysign(FP8[raw&127],np.where(raw&128,-1,1)).astype(np.float32)
            rows,cols=w.shape
            w.reshape(rows//32,32,cols//32,32)[:]*=scale[:,None,:,None]
            return w
        if dtype=='I8':
            rows,cols=raw.shape;w=np.empty((rows,cols*2),np.float32)
            for shift in (0,4):
                code=(raw>>shift)&15
                w[:,shift//4::2]=FP4[code&7]*np.where(code&8,-1,1)
            w.reshape(rows,-1,32)[:]*=scale[:,:,None]
            return w
        raise ValueError(dtype)

    def linear(self,base,x,raw=False):
        dtype=self.tensors[base+'.weight'][2]['dtype']
        y=self.weight(base)@(aq(x) if not raw and dtype in ('F8_E4M3','I8') else x)
        return y if raw else bf(y)

    def norm(self,base,x):
        return bf(x/np.sqrt(np.mean(x*x,dtype=np.float32)+np.float32(1e-20))*self.tensor(base+'.weight'))

    def rope(self,x,layer,pos,inverse=False):
        x=x.copy();rd=64;theta=10000 if layer<2 else 160000
        j=np.arange(rd//2,dtype=np.float64);freq=theta**(-2*j/rd)
        if layer>=2:
            low=max(math.floor(rd*math.log(65536/(32*2*math.pi))/(2*math.log(theta))),0)
            high=min(math.ceil(rd*math.log(65536/(2*math.pi))/(2*math.log(theta))),rd-1)
            ramp=np.clip((j-low)/max(high-low,1e-3),0,1)
            freq*=1-ramp+ramp/16
        angle=pos*freq*(-1 if inverse else 1);c=np.cos(angle);s=np.sin(angle)
        tail=x[..., -64:].reshape(-1,32,2);a=tail[:,:,0].copy();b=tail[:,:,1].copy()
        tail[:,:,0]=a*c-b*s;tail[:,:,1]=a*s+b*c
        return bf(x)

    def attention(self,layer,pos,x):
        base=f'layers.{layer}.attn';src=2 if layer<8 else 8 if layer<14 else 14 if layer<20 else 20
        ratio=2 if layer<20 else 1
        qr=self.norm(base+'.q_norm',self.linear(base+'.wq_a',x))
        q=self.rope(self.linear(base+'.wq_b',qr).reshape(64,512),layer,pos)
        kv=self.norm(base+'.kv_norm',self.linear(base+'.wkv',x))
        kv=aq(self.rope(kv,layer,pos));window=self.windows.setdefault(layer,[])
        window.append(kv);window[:]=window[-128:]
        source=layer in (2,8,14,20);index=source or layer in (24,28,32,36)
        if source:
            cbase=base+'.compressor';value=self.linear(cbase+'.wkv',x,raw=ratio==2)
            latent=None
            if ratio==2:
                score=self.linear(cbase+'.wgate',x,raw=True)
                if pos%2==0:self.pool[layer]=(value,score)
                else:
                    old,old_score=self.pool[layer];alpha=sigmoid(old_score-score)
                    latent=bf(old*alpha+value*(1-alpha))
            else:latent=value
            if latent is not None:
                latent=self.norm(cbase+'.norm',latent)
                key=self.norm(base+'.indexer.k_norm',self.linear(base+'.indexer.wk',latent))
                key=q4(self.rope(key,layer,pos+1-ratio))
                self.keys.setdefault(layer,[]).append(key)
                latent=q4(self.rope(latent,layer,pos+1-ratio),16,True)
                self.latents.setdefault(layer,[]).append(latent)
        count=(pos+1)//ratio
        if index:
            self.selected=np.array([],np.int32)
            if count:
                iq=q4(self.rope(self.linear(base+'.indexer.wq_b',qr).reshape(32,128),layer,pos))
                iw=bf(self.linear(base+'.indexer.weights_proj',x)/64)
                score=bf(np.sum(bf(np.maximum(bf(iq@np.stack(self.keys[src]).T),0)*iw[:,None]),axis=0,dtype=np.float32))
                if layer==20:
                    blocks=np.array([np.max(score[i:i+8]) for i in range(0,count,8)],np.float32);blocks[-1]=np.inf
                    order=np.lexsort((np.arange(len(blocks)),-blocks))[:2048]
                    self.candidates=np.zeros(len(blocks),bool);self.candidates[order]=True
                elif layer>20:score=np.where(np.repeat(self.candidates,8)[:count],score,-np.inf)
                chosen=np.lexsort((np.arange(count),-score))[:min(512,count)]
                self.selected=np.sort(chosen[np.isfinite(score[chosen])])
        rows=list(window)
        if layer>=2:rows.extend(self.latents[src][i] for i in self.selected)
        rows=np.stack(rows);score=(q@rows.T)/np.float32(math.sqrt(512));sink=self.tensor(base+'.attn_sink')
        mx=np.maximum(score.max(axis=-1),sink);p=np.exp(score-mx[:,None]);p/=p.sum(axis=-1,keepdims=True)+np.exp(sink-mx)[:,None]
        out=self.rope(bf(p@rows),layer,pos,True).reshape(8,4096)
        wa=bf(self.weight(base+'.wo_a')).reshape(8,1024,4096)
        projected=bf(np.einsum('grd,gd->gr',wa,out,optimize=True)).reshape(-1)
        return self.linear(base+'.wo_b',projected)

    def hashes(self,token):
        current=int(self.token_map[token]);past=[current]+list(reversed(self.history[-3:]))
        past+= [int(self.token_map[2])]*(4-len(past));out=[]
        for layer in range(2):
            ids=[];offset=0
            for n in range(3):
                value=0
                for j in range(n+2):value^=past[j]*int(self.multipliers[layer,j])
                for prime in self.primes[layer,n]:
                    ids.append(offset+value%int(prime));offset+=int(prime)
            out.append(ids)
        self.history.append(current);return out

    def engram(self,layer,h,ids):
        base=f'layers.{layer}.engram';raw=self.byte_rows(base+'.embed.weight',ids);scale=self.byte_rows(base+'.embed.scale',ids)
        rows=(FP8[raw&127]*np.where(raw&128,-1,1)).astype(np.float32)
        rows.reshape(24,8,32)[:]*=np.exp2(scale.astype(np.float32)-127)[:,:,None]
        kv=self.linear(base+'.wkv',bf(rows).reshape(-1));key=kv[:20480].reshape(4,5120);value=kv[20480:]
        weight=self.tensor(base+'.q_weight')*self.tensor(base+'.k_weight')
        dot=np.sum(h*key*weight,axis=-1)/np.sqrt((np.mean(h*h,axis=-1)+1e-20)*(np.mean(key*key,axis=-1)+1e-20)*5120)
        gate=sigmoid(np.copysign(np.sqrt(np.maximum(np.abs(dot),1e-6)),dot))
        return bf(h+gate[:,None]*value)

    def mixes(self,layer,kind,h):
        base=f'layers.{layer}.hc_{kind}';v=self.tensor(base+'_fn')@h.reshape(-1)
        v/=np.sqrt(np.mean(h*h,dtype=np.float32)+1e-20)
        b=self.tensor(base+'_base');s=self.tensor(base+'_scale')
        pre=sigmoid(v[:4]*s[0]+b[:4])+1e-6;post=2*sigmoid(v[4:8]*s[1]+b[4:8])
        comb=(v[8:]*s[2]+b[8:]).reshape(4,4);comb=np.exp(comb-comb.max(axis=1,keepdims=True));comb/=comb.sum(axis=1,keepdims=True);comb+=1e-6
        for i in range(20):
            if i:comb/=comb.sum(axis=1,keepdims=True)+1e-6
            comb/=comb.sum(axis=0,keepdims=True)+1e-6
        return pre,post,comb

    def expert(self,base,x,weight=1):
        gate=self.linear(base+'.w1',x);up=self.linear(base+'.w3',x)
        return self.linear(base+'.w2',bf(swiglu(gate,up)*weight))

    def forward(self,token,pos):
        raw=self.raw('embed.weight')[token];row=(raw.astype(np.uint32)<<16).view(np.float32)
        h=np.tile(row,(4,1));pre=np.array([1,0,0,0],np.float32);hashes=self.hashes(token)
        for l in range(40):
            base=f'layers.{l}'
            if l in (1,14):h=self.engram(l,h,hashes[0 if l==1 else 1])
            ap,post,comb=self.mixes(l,'attn',h)
            x=self.norm(base+'.attn_norm',bf(np.sum(pre[:,None]*h,axis=0)))
            x=self.attention(l,pos,x);h=hc_post(x,h,post,comb)
            pre,post,comb=self.mixes(l,'ffn',h)
            x=self.norm(base+'.ffn_norm',bf(np.sum(ap[:,None]*h,axis=0)))
            score=self.linear(base+'.ffn.gate',x,raw=True);score=np.sqrt(np.logaddexp(np.float32(0),score))
            biased=score+self.tensor(base+'.ffn.gate.bias');ids=np.lexsort((np.arange(384),-biased))[:6]
            prob=score[ids]/(score[ids].sum()+1e-20)*1.5
            y=np.zeros(5120,np.float32)
            for e,p in zip(ids,prob):y+=self.expert(base+f'.ffn.experts.{e}',x,p)
            y=bf(y+self.expert(base+'.ffn.shared_experts',x));h=hc_post(y,h,post,comb)
            print('REF_LAYER pos={} layer={} norm={} experts={}'.format(pos,l,np.linalg.norm(h),','.join(map(str,ids))),flush=True)
        x=self.norm('norm',bf(np.sum(pre[:,None]*h,axis=0)))
        return self.linear('head',x,raw=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--metadata');ap.add_argument('--output',required=True)
    ap.add_argument('--attention-only',action='store_true');ap.add_argument('--token',type=int,default=0)
    ap.add_argument('--attention-layer',type=int,choices=(0,2,8,14,20),default=0)
    ap.add_argument('--positions',type=int,default=3)
    ap.add_argument('--prompt-ids');ap.add_argument('--generate',type=int,default=1)
    args=ap.parse_args();ref=Reference(args.model,args.metadata)
    if args.attention_only:
        values=[]
        if not 1<=args.positions<=2048:ap.error('positions must be in [1, 2048]')
        for pos in range(args.positions):
            x=np.sin(np.arange(5120,dtype=np.float32)*np.float32(.013)+np.float32(pos)*np.float32(.07)).astype(np.float32)
            out=ref.attention(args.attention_layer,pos,bf(x));values.append(out);print('REF_ATTENTION',args.attention_layer,pos,np.linalg.norm(out),flush=True)
        np.stack(values).tofile(args.output)
    elif args.prompt_ids:
        ids=[int(x) for x in Path(args.prompt_ids).read_text().split()]
        next_token=0
        for pos in range(len(ids)+args.generate-1):
            token=ids[pos] if pos<len(ids) else next_token
            logits=ref.forward(token,pos);logits.tofile(args.output+'.pos'+str(pos)+'.bin')
            next_token=int(np.argmax(logits));print('REF_TOKEN',pos,token,next_token,flush=True)
    else:
        logits=ref.forward(args.token,0);logits.tofile(args.output);print('REF_ARGMAX',int(np.argmax(logits)),float(np.max(logits)),flush=True)


if __name__=='__main__':main()
