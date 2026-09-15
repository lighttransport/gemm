"""DINOv3 and NAF conditioning against PyTorch, using real checkpoint weights."""
import argparse
import ctypes as C
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--backend",choices=["cpu","cuda","rocm"],required=True)
parser.add_argument("--natten",action="store_true")
parser.add_argument("--dino",default="/mnt/disk2/models/dinov3-vitl16/model.safetensors")
parser.add_argument("--gpu-execution",choices=["legacy","resident"],default="legacy")
parser.add_argument("--gpu-kernels",choices=["auto","blas","mma"],default="auto")
args=parser.parse_args()
backend=["cpu","cuda","rocm"].index(args.backend)
device="cuda" if backend else "cpu"
torch.set_num_threads(16)
if backend:
    assert torch.cuda.is_available() and bool(torch.version.hip)==(args.backend=="rocm")
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
lib=C.CDLL(str(ROOT.parent.parent/"cpu/pixal3d/libpixal3d_validation.so"))
lib.px_test_error.restype=C.c_char_p
lib.px_test_set_gpu.argtypes=[C.c_int,C.c_int]
assert lib.px_test_set_gpu(int(args.gpu_execution=="resident"),["auto","blas","mma"].index(args.gpu_kernels))==0
fp=np.ctypeslib.ndpointer(dtype=np.float32,flags="C_CONTIGUOUS")
lib.px_test_dino.argtypes=[C.c_int,C.c_char_p,fp,fp,C.c_int,C.c_int]
lib.px_test_naf.argtypes=[C.c_int,C.c_char_p,fp,fp,fp,C.c_int,fp,C.c_int,C.c_int,fp,C.c_int]
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def check(name,actual,expected):
    x=actual.astype(np.float64).ravel();y=expected.detach().cpu().numpy().astype(np.float64).ravel()
    error=np.linalg.norm(x-y)/np.linalg.norm(y)
    cosine=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    print(json.dumps(dict(backend=args.backend,test=name,max_abs=float(abs(x-y).max()),nrmse=error,cosine=cosine)),flush=True)
    assert np.isfinite(x).all() and error<=.02 and cosine>=.999
torch.manual_seed(134)
reference=load_module("trellis_dino_reference",ROOT.parent/"trellis2/gen_stage1_ref.py")
model=reference.load_dinov3_from_timm(args.dino,device)
image=torch.rand(1,3,64,64,device=device)
with torch.inference_mode():
    h=model.embeddings(image,bool_masked_pos=None)
    rope=model.rope_embeddings(image)
    for layer in model.layer:h=layer(h,position_embeddings=rope)
    expected=F.layer_norm(h,h.shape[-1:])
actual=np.empty((21,1024),np.float32)
rc=lib.px_test_dino(backend,args.dino.encode(),actual,image.cpu().numpy(),64,24)
assert rc==0,lib.px_test_error().decode()
check("dinov3_24_blocks",actual,expected)
del model
if backend:torch.cuda.empty_cache()

# The original NAF convolution and RoPE modules have no CUDA dependency.
conv=load_module("naf_convolutions",ROOT/"naf-upstream/src/layers/convolutions.py")
rope=load_module("naf_rope",ROOT/"naf-upstream/src/layers/rope.py")
class Guide(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=conv.encoder(3,128,kernel_size=1,ks_res=1,num_layers=2)
        self.sem_encoder=conv.encoder(3,128,kernel_size=3,ks_res=3,num_layers=2)
        self.rope=rope.RoPE(embed_dim=256,num_heads=4,base=100.,rescale_coords=2.)
    def forward(self,image,target):
        return self.rope(F.adaptive_avg_pool2d(torch.cat([self.encoder(image),self.sem_encoder(image)],1),target))
model=Guide().eval()
weights=ROOT/"weights/naf_release.safetensors"
model.load_state_dict({k.removeprefix("image_encoder."):v for k,v in load_file(weights).items()})
model.to(device)
target,grid,size=40,10,48
image=torch.rand(1,3,size,size,device=device)
patches=torch.randn(grid,grid,1024,device=device)
xy=torch.rand(73,2,device=device)*3-1.5
with torch.inference_mode():
    guide=model(image,target)
    q=guide[0].permute(1,2,0).reshape(target,target,4,64)
    keys=F.adaptive_avg_pool2d(guide,grid)[0].permute(1,2,0).reshape(grid,grid,4,64)
    # Exact dilated NATTEN window on a grid divisible by its dilation.
    # Boundary windows shift inward; they do not zero-pad or repeat keys.
    offsets=torch.arange(9,device=device)
    rows=(torch.arange(target,device=device)//(target//grid)-4).clamp(0,grid-9)[:,None]+offsets
    k=keys[rows[:,None,:,None],rows[None,:,None,:]]
    v=patches.reshape(grid,grid,4,256)[rows[:,None,:,None],rows[None,:,None,:]]
    scores=torch.einsum("xyhd,xyijhd->xyhij",q,k)*.125
    scores=scores.flatten(-2).softmax(-1).reshape(target,target,4,9,9)
    up=torch.einsum("xyhij,xyijhd->xyhd",scores,v).reshape(target,target,1024)
    expected=F.grid_sample(up.permute(2,0,1)[None],xy[None,None],align_corners=False,padding_mode="border")[0,:,0].T
if args.natten:
    import sys
    assert args.backend=="cuda", "Official NATTEN validation uses CUDA"
    sys.path.insert(0,str(ROOT/"naf-upstream"))
    import src.layers.attentions as attention_module
    from natten import na2d
    # The released wheel has no SM120 CUTLASS image. Use NATTEN's own
    # portable PyTorch backend, preserving the original neighborhood mask.
    def portable_natten(q,k,v,**kw):
        # Flex requires matching Q/K/V head widths. Independent value-column
        # tiles preserve the same attention probabilities and neighborhood.
        return torch.cat([na2d(q,k,part.contiguous(),**{**kw,"backend":"flex-fna"})
                          for part in v.split(q.shape[-1],dim=-1)],dim=-1)
    attention_module.na2d=portable_natten
    CrossAttention=attention_module.CrossAttention
    with torch.inference_mode():
        full=CrossAttention(256,4).to(device)(guide,F.adaptive_avg_pool2d(guide,grid),patches.permute(2,0,1)[None])
        check("naf_original_natten",up.cpu().numpy(),full[0].permute(1,2,0))
        expected=F.grid_sample(full,xy[None,None],align_corners=False,padding_mode="border")[0,:,0].T
actual=np.empty((len(xy),1024),np.float32)
actual_guide=np.empty((target,target,256),np.float32)
rc=lib.px_test_naf(backend,str(weights).encode(),actual,actual_guide,
    image[0].permute(1,2,0).contiguous().cpu().numpy(),size,patches.cpu().numpy(),grid,target,xy.cpu().numpy(),len(xy))
assert rc==0,lib.px_test_error().decode()
check("naf_guide",actual_guide,guide[0].permute(1,2,0))
check("naf_projected",actual,expected)
print(f"{args.backend}: conditioning PASS",flush=True)
