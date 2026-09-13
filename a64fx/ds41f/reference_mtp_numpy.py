#!/usr/bin/env python3
"""Independent DSpark replay from original tensors and captured backbone taps.

Uses checkpoint model.py semantics; no C staging, packed kernels or draft
outputs are reused. NumPy reductions are a numerical reference, not GPU parity.
"""
import argparse
import json
import math
import re
from pathlib import Path
import numpy as np
from reference_numpy import Reference, bf, aq, sigmoid, hc_post, swiglu


class MTPReference(Reference):
    def linear_batch(self, base, x, raw=False):
        dtype = self.tensors[base + '.weight'][2]['dtype']
        y = (aq(x) if not raw and dtype in ('F8_E4M3', 'I8') else x) @ self.weight(base).T
        return y if raw else bf(y)

    def norm_batch(self, base, x):
        return bf(x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True, dtype=np.float32)
                              + np.float32(1e-20)) * self.tensor(base+'.weight'))

    def commit(self, taps, pos):
        x = self.norm_batch('mtp.0.main_norm', self.linear_batch('mtp.0.main_proj', taps))
        for stage in range(3):
            base = f'mtp.{stage}.attn'
            kv = self.norm_batch(base+'.kv_norm', self.linear_batch(base+'.wkv', x))
            kv = aq(self.rope(kv, 0, pos))
            self.windows.setdefault(stage, np.zeros((128,512), np.float32))[pos % 128] = kv

    def mix(self, stage, kind, h):
        base = f'mtp.{stage}.hc_{kind}'
        value = self.tensor(base+'_fn') @ h.reshape(-1)
        value /= np.sqrt(np.mean(h*h, dtype=np.float32)+np.float32(1e-20))
        bias, scale = self.tensor(base+'_base'), self.tensor(base+'_scale')
        pre = sigmoid(value[:4]*scale[0]+bias[:4])+np.float32(1e-6)
        post = 2*sigmoid(value[4:8]*scale[1]+bias[4:8])
        comb = (value[8:]*scale[2]+bias[8:]).reshape(4,4)
        comb = np.exp(comb-comb.max(axis=1,keepdims=True))
        comb /= comb.sum(axis=1,keepdims=True)
        comb += np.float32(1e-6)
        for iteration in range(20):
            if iteration:
                comb /= comb.sum(axis=1,keepdims=True)+np.float32(1e-6)
            comb /= comb.sum(axis=0,keepdims=True)+np.float32(1e-6)
        return pre, post, comb

    def attend(self, stage, pos, x):
        base = f'mtp.{stage}.attn'
        qr = self.norm_batch(base+'.q_norm', self.linear_batch(base+'.wq_a',x))
        q = self.linear_batch(base+'.wq_b',qr).reshape(5,64,512)
        kv = self.norm_batch(base+'.kv_norm',self.linear_batch(base+'.wkv',x))
        for i in range(5):
            q[i] = self.rope(q[i],0,pos+1+i)
            kv[i] = aq(self.rope(kv[i],0,pos+1+i))
        rows = np.concatenate((self.windows[stage][:min(128,pos+1)],kv))
        score = (q @ rows.T)/np.float32(math.sqrt(512))
        sink = self.tensor(base+'.attn_sink')[None,:]
        maximum = np.maximum(score.max(axis=-1),sink)
        p = np.exp(score-maximum[...,None])
        p /= p.sum(axis=-1,keepdims=True)+np.exp(sink-maximum)[...,None]
        out = bf(p @ rows)
        for i in range(5):
            out[i] = self.rope(out[i],0,pos+1+i,True)
        wa = bf(self.weight(base+'.wo_a')).reshape(8,1024,4096)
        projected = bf(np.einsum('grd,sgd->sgr',wa,out.reshape(5,8,4096),optimize=True)).reshape(5,8192)
        return self.linear_batch(base+'.wo_b',projected)

    def head_batch(self, base, x):
        raw = self.raw(base+'.weight')
        if self.tensors[base+'.weight'][2]['dtype'] != 'BF16':
            raise ValueError('expected BF16 vocabulary rows')
        out = np.empty((len(x),len(raw)),np.float32)
        # At most 80 MiB of dequantized vocabulary weights at a time.
        for first in range(0,len(raw),4096):
            w = (raw[first:first+4096].astype(np.uint32)<<16).view(np.float32)
            out[:,first:first+len(w)] = x @ w.T
        return out

    def draft(self, seed, pos):
        raw = self.raw('embed.weight')[[seed,128799,128799,128799,128799]]
        x = (raw.astype(np.uint32)<<16).view(np.float32)
        h = np.repeat(x[:,None,:],4,axis=1)
        pre = np.tile(np.array([1,0,0,0],np.float32),(5,1))
        for stage in range(3):
            base = f'mtp.{stage}'
            mixes = [self.mix(stage,'attn',v) for v in h]
            x = bf(np.sum(pre[...,None]*h,axis=1,dtype=np.float32))
            y = self.attend(stage,pos,self.norm_batch(base+'.attn_norm',x))
            for i in range(5):
                ap,post,comb = mixes[i]
                h[i] = hc_post(y[i],h[i],post,comb)
                pre[i],post,comb = self.mix(stage,'ffn',h[i])
                x = self.norm_batch(base+'.ffn_norm',bf(np.sum(ap[:,None]*h[i],axis=0,dtype=np.float32)))
                score = self.linear(base+'.ffn.gate',x,raw=True)
                score = np.sqrt(np.logaddexp(np.float32(0),score))
                ids = np.lexsort((np.arange(128),-(score+self.tensor(base+'.ffn.gate.bias'))))[:3]
                probability = score[ids]/(score[ids].sum()+np.float32(1e-20))*np.float32(1.5)
                y_ffn = np.zeros(5120,np.float32)
                for expert,weight in zip(ids,probability):
                    y_ffn += self.expert(base+f'.ffn.experts.{expert}',x,weight)
                y_ffn = bf(y_ffn+self.expert(base+'.ffn.shared_experts',x))
                h[i] = hc_post(y_ffn,h[i],post,comb)
                print('MTP_REF_LAYER',stage,i,'experts',','.join(map(str,ids)),flush=True)
        collapsed = bf(np.sum(pre[...,None]*h,axis=1,dtype=np.float32))
        logits = self.head_batch('head',self.norm_batch('mtp.2.norm',collapsed))
        outputs = [seed]
        confidence = []
        for i in range(5):
            raw = self.raw('mtp.2.markov_head.embed.weight')[outputs[-1]]
            markov_x = (raw.astype(np.uint32)<<16).view(np.float32)
            logits[i] += self.head_batch('mtp.2.markov_head.head',markov_x[None])[0]
            outputs.append(int(logits[i].argmax()))
            confidence.append(float(self.linear('mtp.2.confidence_head.proj',
                np.concatenate((collapsed[i],markov_x)),raw=True)[0]))
        return outputs,logits,confidence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--position',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    if not 1 <= args.position <= 2048:
        ap.error('bounded reference position must be 1..2048')
    args.output.mkdir()
    log = (args.run/'inference.rank00.log').read_text()
    match = re.search(r'MTP_DRAFT pos='+str(args.position)+r' seed=(\d+) ',log)
    if not match:
        ap.error('missing draft seed at requested position')
    ref = MTPReference(args.model)
    for pos in range(args.position+1):
        taps = np.fromfile(args.run/f'mtp-taps.pos{pos}.bin',dtype='<f4')
        if taps.size != 15360 or not np.isfinite(taps).all():
            raise ValueError('invalid tap record')
        ref.commit(taps,pos)
    outputs,logits,confidence = ref.draft(int(match[1]),args.position)
    for i,row in enumerate(logits):
        row.tofile(args.output/f'logits.step{i}.bin')
    result = dict(position=args.position,outputs=outputs,confidence=confidence)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
