"""Independent gfx12 GEMM/attention checks; run from the repository root with ROCm PyTorch."""
import ctypes as C, torch
assert torch.version.hip and torch.cuda.is_available()
torch.manual_seed(4)
def ptr(t):return C.c_void_p(t.data_ptr())
def lib(name):return C.CDLL('rdna4/qimg21/libq21_hip_fast_'+name+'.so')
s=C.c_void_p(torch.cuda.current_stream().cuda_stream)
g=lib('gemm');g.q21f_bf16_gemm.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_void_p]+[C.c_int]*4+[C.c_void_p]
for m,n,k in [(1,17,32),(33,129,64),(128,128,128)]:
 x=torch.randn(m,k+16,device='cuda',dtype=torch.bfloat16);w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16);y=torch.empty(m,n+16,device='cuda',dtype=torch.bfloat16)
 assert g.q21f_bf16_gemm(ptr(y),n+16,ptr(w),ptr(x),k+16,m,n,k,s)==0
 torch.cuda.synchronize();r=(x[:,:k].float()@w.float().T).bfloat16();err=(y[:,:n].float()-r.float()).abs().max().item();print('bf16',m,n,k,err,flush=True);assert err<.13
g.q21f_i8_gemm.argtypes=[C.c_void_p,C.c_int]+[C.c_void_p]*4+[C.c_int]*4+[C.c_void_p]
for m,n,k in [(1,17,31),(33,129,64),(128,128,128)]:
 x=torch.randint(-127,128,(m,k),device='cuda',dtype=torch.int8);w=torch.randint(-127,128,(n,k),device='cuda',dtype=torch.int8)
 xs=torch.rand(m,device='cuda')*.01;ws=torch.rand(n,device='cuda')*.01;y=torch.empty(m,n+16,device='cuda',dtype=torch.bfloat16)
 assert g.q21f_i8_gemm(ptr(y),n+16,ptr(x),ptr(xs),ptr(w),ptr(ws),m,n,k,-1,s)==0
 torch.cuda.synchronize();acc=x.cpu().int()@w.cpu().int().T;ref=(acc.float().to('cuda')*xs[:,None]*ws[None,:]).bfloat16()
 torch.testing.assert_close(y[:,:n],ref,rtol=0,atol=0)
 print('int8 exact',m,n,k,flush=True)
for name in ['attention','sage']:
 l=lib(name);fn=l.q21f_sage_attention if name=='sage' else l.q21f_attention;fn.argtypes=[C.c_void_p]*4+[C.c_int]*(8 if name=='sage' else 7)+[C.c_void_p]
 for nq,nk in [(31,65),(129,131),(128,128)]:
  h=2;stride=h*128+128
  q=torch.randn(nq,stride,device='cuda',dtype=torch.bfloat16);k=torch.randn(nk,stride,device='cuda',dtype=torch.bfloat16);v=torch.randn(nk,stride,device='cuda',dtype=torch.bfloat16);o=torch.empty(nq,stride,device='cuda',dtype=torch.bfloat16)
  for mask in ([0] if name=='sage' else [0,1,2]):
   rc=fn(ptr(o),ptr(q),ptr(k),ptr(v),nq,nk,h,stride,stride,stride,mask,*([1] if name=='sage' else []),s);assert rc==0,rc
   torch.cuda.synchronize();qh=q[:,:h*128].reshape(nq,h,128).transpose(0,1).float();kh=k[:,:h*128].reshape(nk,h,128).transpose(0,1).float();vh=v[:,:h*128].reshape(nk,h,128).transpose(0,1).float()
   scores=qh@kh.transpose(-1,-2)/128**.5
   if mask:scores.masked_fill_(torch.arange(nk,device='cuda')[None,:]>torch.arange(nq,device='cuda')[:,None]+(nk-nq if mask==2 else 0),-float('inf'))
   ref=(scores.softmax(-1)@vh).transpose(0,1).reshape(nq,h*128);err=(o[:,:h*128].float()-ref).abs();print(name,nq,nk,mask,'max',err.max().item(),'rms',err.square().mean().sqrt().item(),flush=True);assert torch.isfinite(o[:,:h*128]).all();assert err.max()<(.14 if name=='sage' else .04)
print('PASS',flush=True)
