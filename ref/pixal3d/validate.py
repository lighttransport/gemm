"""Compare native Pixal3D operations with PyTorch and pinned upstream code."""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from upstream_import import ROOT, prepare

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["cpu", "cuda", "rocm"], required=True)
p.add_argument("--flow", action="store_true")
p.add_argument("--model-dir", type=Path, default=Path("/mnt/disk2/models/Pixal3D"))
p.add_argument("--gpu-execution",choices=["legacy","resident"],default="legacy")
p.add_argument("--gpu-kernels",choices=["auto","blas","mma"],default="auto")
a = p.parse_args()
backend = ["cpu", "cuda", "rocm"].index(a.backend)
device = "cpu" if not backend else "cuda"
torch.set_num_threads(16)
torch.manual_seed(923)
if backend:
    assert torch.cuda.is_available()
    assert bool(torch.version.hip) == (a.backend == "rocm")
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
lib = C.CDLL(str(ROOT.parent.parent / "cpu/pixal3d/libpixal3d_validation.so"))
lib.px_test_error.restype = C.c_char_p
lib.px_test_set_gpu.argtypes=[C.c_int,C.c_int]
assert lib.px_test_set_gpu(int(a.gpu_execution=="resident"),["auto","blas","mma"].index(a.gpu_kernels))==0
fp = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
lib.px_test_gemm.argtypes = [C.c_int, fp, fp, fp, fp] + [C.c_int]*4
lib.px_test_attention.argtypes = [C.c_int, fp, fp, fp, fp] + [C.c_int]*4
lib.px_test_flow.argtypes = [C.c_int, C.c_char_p, fp, fp, ip, C.c_int, C.c_int,
                           fp, C.c_int, fp, C.c_int, C.c_float, C.c_int, C.c_int]

def call(fn, *args):
    rc = fn(*args)
    assert rc == 0, lib.px_test_error().decode()

def check(name, actual, expected, stage=False):
    x = actual.astype(np.float64).ravel()
    y = expected.detach().float().cpu().numpy().astype(np.float64).ravel()
    assert np.isfinite(x).all() and np.isfinite(y).all()
    error = np.linalg.norm(x-y) / max(np.linalg.norm(y), 1e-30)
    cosine = np.dot(x,y) / max(np.linalg.norm(x)*np.linalg.norm(y), 1e-30)
    print(json.dumps(dict(backend=a.backend, test=name, max_abs=float(abs(x-y).max()),
                          nrmse=float(error), cosine=float(cosine))), flush=True)
    if stage:
        assert cosine >= .999 and error <= .02
    else:
        np.testing.assert_allclose(x, y, atol=1e-5, rtol=1e-4)

rng = np.random.default_rng(173)
rand = lambda shape: rng.standard_normal(shape).astype(np.float32) * .2
for bf in [0, 1]:
    x,w,b = rand((139, 160)),rand((96,160)),rand((96,))
    dtype = torch.bfloat16 if bf else torch.float32
    expected = F.linear(torch.tensor(x,device=device,dtype=dtype),
                        torch.tensor(w,device=device,dtype=dtype),
                        torch.tensor(b,device=device,dtype=dtype))
    actual = np.empty((139,96),np.float32)
    call(lib.px_test_gemm,backend,actual,x,w,b,139,96,160,bf)
    check(f"gemm_{dtype}",actual,expected,stage=bool(bf))
for n,m in [(137,137),(137,5),(1037,1139)]:
    q,k,v = rand((n,4,32)),rand((m,4,32)),rand((m,4,32))
    tensors = [torch.tensor(x,device=device).transpose(0,1)[None] for x in (q,k,v)]
    expected = F.scaled_dot_product_attention(*tensors)[0].transpose(0,1)
    actual = np.empty_like(q)
    call(lib.px_test_attention,backend,actual,q,k,v,n,m,4,32)
    check(f"attention_{n}_{m}",actual,expected)

lib.pixal3d_sample_features.argtypes = [fp,C.c_int,C.c_int,C.c_int,fp,C.c_size_t,fp]
features,xy = rand((7,9,13)),rand((39,2))*10
actual=np.empty((39,13),np.float32)
call(lib.pixal3d_sample_features,features,7,9,13,xy,39,actual)
expected=F.grid_sample(torch.tensor(features).permute(2,0,1)[None],
                       torch.tensor(xy)[None,None],align_corners=False,padding_mode="border")
check("grid_sample_border",actual,expected[0,:,0].T)
lib.pixal3d_cascade_coords.argtypes = [ip,C.c_size_t,C.c_int,C.c_int,ip]
lib.pixal3d_cascade_coords.restype = C.c_int64
coords = np.column_stack([np.zeros(8192,np.int32),rng.integers(0,512,(8192,3),dtype=np.int32)])
actual=np.empty_like(coords)
n=lib.pixal3d_cascade_coords(coords,len(coords),512,1024,actual)
expected=torch.unique(torch.cat([torch.zeros(len(coords),1,dtype=torch.int32),
                        ((torch.tensor(coords[:,1:]).float()+.5)/512*63).round().int()],1),dim=0)
np.testing.assert_array_equal(actual[:n],expected.numpy())
print(f"{a.backend}: exact cascade coordinates PASS",flush=True)

if a.flow:
    prepare()
    from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel
    stem=a.model_dir/"ckpts/ss_flow_img_dit_1_3B_64_bf16"
    config=json.loads(stem.with_suffix(".json").read_text())["args"]
    config.update(num_blocks=1,resolution=4)
    xyz=np.indices((4,4,4)).reshape(3,-1).T.astype(np.int32)
    coords=np.ascontiguousarray(np.column_stack([np.zeros(len(xyz),np.int32),xyz]))
    x,global_cond,proj=rand((64,8)),rand((5,1024)),rand((64,1024))
    for bf in [0,1]:
        config["dtype"]="bfloat16" if bf else "float32"
        model=SparseStructureFlowModel(**config).eval()
        with safe_open(str(stem.with_suffix(".safetensors")),framework="pt") as weights:
            state={k:weights.get_tensor(k) for k in model.state_dict() if k!="rope_phases"}
        missing,unexpected=model.load_state_dict(state,strict=False)
        assert missing==["rope_phases"] and not unexpected,(missing,unexpected)
        del state
        model.to(device)
        with torch.inference_mode():
            expected=model(torch.tensor(x.T.reshape(1,8,4,4,4),device=device),
                           torch.tensor([730.],device=device),
                           {"global":torch.tensor(global_cond[None],device=device),
                            "proj":torch.tensor(proj[None],device=device)})
        expected=expected.reshape(8,64).T
        actual=np.empty_like(x)
        call(lib.px_test_flow,backend,str(stem.with_suffix(".safetensors")).encode(),
             actual,x,coords,64,8,global_cond,1024,proj,1024,.73,1,bf)
        check(f"upstream_ss_flow_block_{config['dtype']}",actual,expected,stage=True)
        del model
        if backend: torch.cuda.empty_cache()
print(f"{a.backend}: PASS",flush=True)
