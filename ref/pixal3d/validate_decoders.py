"""Checkpoint-backed upstream/native decoder comparison."""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sparse_reference import install
from upstream_import import ROOT

parser=argparse.ArgumentParser()
parser.add_argument("--backend",choices=["cpu","cuda","rocm"],required=True)
parser.add_argument("--precision",choices=["float32","float16"],default="float16")
parser.add_argument("--guided",type=Path,help="Validate with upstream subdivisions and report threshold sensitivity")
parser.add_argument("--noise-device",choices=["cpu","backend"],default="backend")
parser.add_argument("--reference-device",choices=["cpu","cuda"],help="Override the PyTorch oracle device, independently of the native backend")
parser.add_argument("--stage",choices=["structure","shape","texture"],required=True)
parser.add_argument("--model-dir",type=Path,default=Path("/mnt/disk2/models/Pixal3D"))
parser.add_argument("--gpu-execution",choices=["legacy","resident"],default="legacy")
parser.add_argument("--gpu-kernels",choices=["auto","blas","mma"],default="auto")
args=parser.parse_args()
if args.stage=="texture" and not args.guided:parser.error("texture requires --guided DIR for shape subdivisions")
backend=["cpu","cuda","rocm"].index(args.backend)
precision=2 if args.precision=="float16" else 0
device=args.reference_device or ("cuda" if backend else "cpu")
torch.set_num_threads(16)
torch.manual_seed(55)
if device=="cuda":
    assert torch.cuda.is_available()
    if backend:assert bool(torch.version.hip)==(args.backend=="rocm")
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
lib=C.CDLL(str(ROOT.parent.parent/"cpu/pixal3d/libpixal3d_validation.so"))
lib.px_test_error.restype=C.c_char_p
lib.px_test_set_gpu.argtypes=[C.c_int,C.c_int]
assert lib.px_test_set_gpu(int(args.gpu_execution=="resident"),["auto","blas","mma"].index(args.gpu_kernels))==0
fp=np.ctypeslib.ndpointer(dtype=np.float32,flags="C_CONTIGUOUS")
ip=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
install()
stem=args.model_dir/"ckpts"/("ss_dec_conv3d_16l8_fp16" if args.stage=="structure" else "shape_dec_next_dc_f16c32_fp16")
config=json.loads(stem.with_suffix(".json").read_text())["args"]
config["use_fp16"]=bool(precision)
path=stem.with_suffix(".safetensors")
if args.stage=="structure":
    from pixal3d.models.sparse_structure_vae import SparseStructureDecoder
    model=SparseStructureDecoder(**config)
else:
    from pixal3d.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    from pixal3d.modules.sparse import SparseTensor
    config.pop("resolution")
    model=SparseUnetVaeDecoder(out_channels=7,**config)
model.load_state_dict(load_file(path))
model.eval().to(device)
if args.stage=="structure":
    x=(torch.randn(1,8,16,16,16,device="cpu" if args.noise_device=="cpu" else device)*.3).to(device)
    with torch.inference_mode():expected=model(x).float().cpu().numpy().ravel()
    actual=np.empty(64**3,np.float32)
    lib.px_test_structure.argtypes=[C.c_int,C.c_char_p,fp,fp,C.c_int]
    rc=lib.px_test_structure(backend,str(path).encode(),actual,x[0].permute(1,2,3,0).contiguous().cpu().numpy(),precision)
else:
    coords=torch.tensor([[0,16,16,16],[0,16,16,17],[0,16,17,16],[0,17,16,16]],dtype=torch.int32,device=device)
    x=(torch.randn(4,32,device="cpu" if args.noise_device=="cpu" else device)*3).to(device)
    with torch.inference_mode():ref,subs=model(SparseTensor(feats=x,coords=coords),return_subs=True)
    if args.guided:
        args.guided.mkdir(parents=True,exist_ok=True)
        for i,sub in enumerate(subs):
            save_file({"coords":sub.coords.cpu().contiguous(),"feats":sub.feats.float().cpu().contiguous()},str(args.guided/f"reference_sub_{i}.safetensors"))
    if args.stage=="texture":
        del model,ref
        if backend:torch.cuda.empty_cache()
        stem=args.model_dir/"ckpts/tex_dec_next_dc_f16c32_fp16"
        config=json.loads(stem.with_suffix(".json").read_text())["args"]
        config.pop("resolution",None);config["use_fp16"]=bool(precision)
        model=SparseUnetVaeDecoder(**config)
        path=stem.with_suffix(".safetensors")
        model.load_state_dict(load_file(path));model.eval().to(device)
        x=torch.randn_like(x)
        with torch.inference_mode():ref=model(SparseTensor(feats=x,coords=coords),guide_subs=subs)
    expected=ref.feats.float().cpu().numpy()
    values=C.POINTER(C.c_float)();out_coords=C.POINTER(C.c_int32)();rows=C.c_int()
    lib.px_test_decode.argtypes=[C.c_int,C.c_char_p,C.POINTER(C.POINTER(C.c_float)),
        C.POINTER(C.POINTER(C.c_int32)),C.POINTER(C.c_int),fp,ip,C.c_int,C.c_int,C.c_char_p]
    rc=lib.px_test_decode(backend,str(path).encode(),C.byref(values),C.byref(out_coords),C.byref(rows),
                        x.cpu().numpy(),coords.cpu().numpy(),4,precision,str(args.guided).encode() if args.guided else None)
    assert rc==0,lib.px_test_error().decode()
    actual=np.ctypeslib.as_array(values,shape=(rows.value,6 if args.stage=="texture" else 7)).copy()
    native_coords=np.ctypeslib.as_array(out_coords,shape=(rows.value,4)).copy()
    lib.px_test_free.argtypes=[C.c_void_p]
    lib.px_test_free(values);lib.px_test_free(out_coords)
    np.testing.assert_array_equal(native_coords,ref.coords.cpu().numpy())
    print(f"Exact {args.stage} coordinates: {rows.value}",flush=True)
    if args.guided and args.stage=="shape":
        for i,sub in enumerate(subs):
            native=load_file(args.guided/f"native_sub_{i}.safetensors")
            np.testing.assert_array_equal(native["coords"].numpy(),sub.coords.cpu().numpy())
            actual_logits=native["feats"].numpy();expected_logits=sub.feats.float().cpu().numpy()
            nrmse=np.linalg.norm(actual_logits-expected_logits)/np.linalg.norm(expected_logits)
            changed=(actual_logits>0)!=(expected_logits>0)
            margin=float(np.abs(expected_logits[changed]).max()) if changed.any() else 0.
            print(json.dumps(dict(stage=i,logit_nrmse=float(nrmse),changed_subdivision_decisions=int(changed.sum()),max_changed_margin=margin)),flush=True)
            assert nrmse<=.02 and margin<=.02, "Subdivision differences exceed rounding tolerance"
assert rc==0,lib.px_test_error().decode()
x=actual.astype(np.float64).ravel();y=expected.astype(np.float64).ravel()
error=np.linalg.norm(x-y)/np.linalg.norm(y);cosine=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
print(json.dumps(dict(backend=args.backend,reference_device=device,stage=args.stage,precision=args.precision,
    max_abs=float(abs(x-y).max()),nrmse=error,cosine=cosine)),flush=True)
assert np.isfinite(x).all() and error<=.02 and cosine>=.999
print("PASS",flush=True)
