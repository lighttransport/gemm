"""Resident plugin kernel checks against PyTorch; runs in the per-backend uv env."""
import argparse
import ctypes as C
from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--backend', choices=['cuda', 'rocm'], required=True)
p.add_argument('--kernels', choices=['auto', 'blas', 'mma'], default='mma')
p.add_argument('--benchmark',action='store_true')
a = p.parse_args()
assert torch.cuda.is_available() and bool(torch.version.hip) == (a.backend == 'rocm')
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(17)
root = Path(__file__).resolve().parents[2]
lib = C.CDLL(str(root / ('cuda' if a.backend == 'cuda' else 'rdna4') / 'pixal3d' / f'libpixal3d_{a.backend}.so'))
class Op(C.Structure):
    _fields_ = [(x, C.c_int) for x in ['op','precision','n','c','k','heads','offset','extra']] + [('epsilon',C.c_float)] + [(x,C.c_void_p) for x in ['out','x','w','b','v']]
lib.px_gpu_create.argtypes=[C.c_int,C.c_size_t];lib.px_gpu_create.restype=C.c_void_p
lib.px_gpu_allocate.argtypes=[C.c_void_p,C.c_size_t];lib.px_gpu_allocate.restype=C.c_void_p
lib.px_gpu_release.argtypes=[C.c_void_p]
lib.px_gpu_destroy.argtypes=[C.c_void_p]
lib.px_gpu_error.argtypes=[C.c_void_p];lib.px_gpu_error.restype=C.c_char_p
lib.px_gpu_copy.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
lib.px_gpu_execute.argtypes=[C.c_void_p,C.POINTER(Op)]
lib.px_gpu_configure.argtypes=[C.c_void_p,C.c_int,C.c_int]
lib.px_gpu_trim.argtypes=[C.c_void_p]
class Metrics(C.Structure):
    _fields_=[(name,C.c_uint64) for name in ['uploads','downloads','allocations','gemms','mma_gemms','attentions','mma_attentions']]+[('kernel_ms',C.c_double)]+[(name,C.c_uint64) for name in ['effective_budget_bytes','active_bytes','pooled_bytes','peak_active_bytes','largest_allocation_bytes']]
lib.px_gpu_metrics.argtypes=[C.c_void_p,C.POINTER(Metrics)]
assert lib.px_gpu_device_version()==2
d=lib.px_gpu_create(0,1024**3)
assert d
assert lib.px_gpu_configure(d,['auto','blas','mma'].index(a.kernels),0)==0
allocated=[]
def alloc(n):
    b=lib.px_gpu_allocate(d,n);assert b,lib.px_gpu_error(d);allocated.append(b);return b
def upload(x):
    v=x.cpu().contiguous()
    b=alloc(v.numel()*v.element_size())
    assert lib.px_gpu_copy(d,b,v.data_ptr(),v.numel()*v.element_size(),0)==0
    return b
def run(op,shape):
    out=np.empty(shape,np.float32);op.out=alloc(out.nbytes)
    assert lib.px_gpu_execute(d,C.byref(op))==0,lib.px_gpu_error(d)
    assert lib.px_gpu_copy(d,op.out,out.ctypes.data,out.nbytes,1)==0,lib.px_gpu_error(d)
    return torch.from_numpy(out)
def check(label,actual,expected,precision):
    x=actual.flatten().double();y=expected.float().cpu().flatten().double()
    err=float((x-y).norm()/y.norm().clamp_min(1e-30));cos=float(F.cosine_similarity(x,y,dim=0))
    print(json.dumps(dict(test=label,backend=a.backend,kernels=a.kernels,nrmse=err,cosine=cos)),flush=True)
    assert torch.isfinite(x).all() and err <= (.02 if precision else .0001) and cos>=.999
    if precision==0:torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)
def free():
    for b in allocated:lib.px_gpu_release(b)
    allocated.clear()
try:
    for prec,dt in [(0,torch.float32),(1,torch.bfloat16),(2,torch.float16)]:
        for n,ci,co in [(1,16,16),(17,31,47),(137,160,96),(65,1536,128)]:
            x=torch.randn(n,ci)*.2;w=torch.randn(co,ci)*.2;b=torch.randn(co)*.2
            expected=(x.to('cuda',dt).float()@w.to('cuda',dt).float().T+b.to('cuda',dt).float()).to(dt)
            op=Op(op=0,precision=prec,n=n,c=co,k=ci,x=upload(x),w=upload(w.to(dt)),b=upload(b))
            check(f'gemm_{prec}_{n}_{ci}_{co}',run(op,(n,co)),expected,prec);free()
    for prec,dt in [(0,torch.float32),(1,torch.bfloat16),(2,torch.float16)]:
        for n,m in [(1,5),(17,33),(137,137)]:
            q,k,v=[torch.randn(s,2,128).to(dt).float() for s in [n,m,m]]
            ts=[z.to('cuda',dt).transpose(0,1)[None] for z in [q,k,v]]
            expected=F.scaled_dot_product_attention(*ts)[0].transpose(0,1)
            op=Op(op=11,precision=prec,n=n,c=128,k=m,heads=2,x=upload(q),w=upload(k),v=upload(v))
            check(f'attention_{prec}_{n}_{m}',run(op,(n,2,128)),expected,prec);free()
    # Cached RoPE phases must preserve the original per-head calculation exactly.
    for prec,dt in [(0,torch.float32),(1,torch.bfloat16),(2,torch.float16)]:
        n,heads,hd=67,2,128
        coords=torch.randint(0,64,(n,4),dtype=torch.int32);coords[:,0]=0
        positions=upload(coords);phases=alloc(n*hd*4)
        op=Op(op=22,n=n,c=hd,w=positions,out=phases)
        assert lib.px_gpu_execute(d,C.byref(op))==0,lib.px_gpu_error(d)
        q=upload(torch.randn(n,heads,hd).to(dt).float())
        old=run(Op(op=3,precision=prec,n=n*heads,c=hd,k=heads,x=q,w=positions),(n,heads,hd))
        cached=run(Op(op=3,precision=prec,n=n*heads,c=hd,k=heads,x=q,w=phases,extra=1),(n,heads,hd))
        torch.testing.assert_close(cached,old,rtol=0,atol=0)
        free()
    print('Cached RoPE equivalence PASS',flush=True)
    if a.benchmark:
        for n in [1024,4096]:
            q,k,v=[torch.randn(n,12,128).bfloat16().float() for _ in range(3)]
            op=Op(op=11,precision=1,n=n,c=128,k=n,heads=12,x=upload(q),w=upload(k),v=upload(v),out=alloc(n*12*128*4))
            for mode in [1,2]:
                assert lib.px_gpu_configure(d,mode,0)==0
                seconds=[]
                for repeat in range(4):
                    torch.cuda.synchronize();start=time.monotonic()
                    assert lib.px_gpu_execute(d,C.byref(op))==0,lib.px_gpu_error(d)
                    torch.cuda.synchronize();seconds.append(time.monotonic()-start)
                print(json.dumps(dict(benchmark='attention',backend=a.backend,tokens=n,kernels=['auto','blas','mma'][mode],seconds=seconds[1:])),flush=True)
            free()
    # The allocator must reject an oversized request without invalidating the context.
    assert not lib.px_gpu_allocate(d,2*1024**3)
    assert not lib.px_gpu_allocate(d,0)
    assert lib.px_gpu_configure(d,-1,0)==-1
    free();assert lib.px_gpu_trim(d)==0
    before=Metrics();after=Metrics();assert lib.px_gpu_metrics(d,C.byref(before))==0
    for _ in range(10):
        value=torch.tensor([7.0]);b=upload(value)
        op=Op(op=5,n=1,c=1,x=b,out=b)
        assert lib.px_gpu_execute(d,C.byref(op))==0,lib.px_gpu_error(d)
        result=np.empty(1,np.float32)
        assert lib.px_gpu_copy(d,b,result.ctypes.data,4,1)==0 and result[0]==7
        op.n=2
        assert lib.px_gpu_execute(d,C.byref(op))==-1, 'Extent overrun must be rejected'
        free()
    assert lib.px_gpu_metrics(d,C.byref(after))==0
    assert after.allocations-before.allocations==1, 'Repeated workspace should be reused'
    assert after.effective_budget_bytes==1024**3 and after.active_bytes==0
    assert after.pooled_bytes>0 and after.peak_active_bytes>=4 and after.largest_allocation_bytes>=4
    assert lib.px_gpu_trim(d)==0
    trimmed=Metrics();assert lib.px_gpu_metrics(d,C.byref(trimmed))==0
    assert trimmed.active_bytes==0 and trimmed.pooled_bytes==0
    print('Resident kernel validation PASS',flush=True)
finally:
    free();lib.px_gpu_destroy(d)
