"""Compare dumped complete native flow steps with the original upstream sampler."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from upstream_import import prepare

p=argparse.ArgumentParser()
p.add_argument("--backend",choices=["cpu","cuda","rocm"],required=True)
p.add_argument("--dump-dir",type=Path,required=True)
p.add_argument("--stage",choices=["structure","shape512","shape1024","texture"],required=True)
p.add_argument("--step",type=int,default=1)
p.add_argument("--trajectory",action="store_true",help="Diagnose accumulated error over all 12 steps; the isolated-step threshold is reported separately")
p.add_argument("--attention",choices=["auto","math"],default="auto",help="Select the PyTorch SDPA reference kernel")
p.add_argument("--save-reference",type=Path,help="Save reference output before checking tolerance")
p.add_argument("--model-dir",type=Path,default=Path("/mnt/disk2/models/Pixal3D"))
p.add_argument("--reference-device",choices=["cpu","cuda"],default=None,
               help="PyTorch oracle device; native dump/backend remains --backend")
a=p.parse_args()
torch.set_num_threads(16)
device=a.reference_device or ("cpu" if a.backend=="cpu" else "cuda")
if device=="cuda":
    assert torch.cuda.is_available() and bool(torch.version.hip)==(a.backend=="rocm")
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=False
prepare()
from pixal3d.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from pixal3d.modules.sparse import SparseTensor
from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel
from pixal3d.models.structured_latent_flow import SLatFlowModel

pipeline=json.loads((a.model_dir/"pipeline.json").read_text())["args"]
keys={"structure":("sparse_structure_flow_model","sparse_structure_sampler"),
      "shape512":("shape_slat_flow_model_512","shape_slat_sampler"),
      "shape1024":("shape_slat_flow_model_1024","shape_slat_sampler"),
      "texture":("tex_slat_flow_model_1024","tex_slat_sampler")}
model_key,sampler_key=keys[a.stage]
stem=a.model_dir/pipeline["models"][model_key]
config=json.loads(stem.with_suffix(".json").read_text())["args"]
model=(SparseStructureFlowModel if a.stage=="structure" else SLatFlowModel)(**config)
model.load_state_dict(load_file(stem.with_suffix(".safetensors")))
model.eval().to(device)
def read(name):return {k:v.to(device) for k,v in load_file(a.dump_dir/f"{name}.safetensors").items()}
data=read(a.stage+"_noise" if a.trajectory or a.step==1 else a.stage+f"_step_{a.step-1}")
global_cond=read(a.stage+"_global")["feats"][None]
projected=read(a.stage+"_projected")["feats"]
if a.stage=="structure":
    sample=data["feats"].T.reshape(1,8,16,16,16).contiguous()
    projected=projected[None]
else:
    sample=SparseTensor(feats=data["feats"],coords=data["coords"])
    projected=SparseTensor(feats=projected,coords=data["coords"])
negative=torch.zeros_like(projected) if a.stage=="structure" else projected.replace(torch.zeros_like(projected.feats))
cond={"global":global_cond,"proj":projected}
neg_cond={"global":torch.zeros_like(global_cond),"proj":negative}
spec=pipeline[sampler_key];params=dict(spec["params"]);steps=params.pop("steps");rescale_t=params.pop("rescale_t")
times=np.linspace(1,0,steps+1);times=rescale_t*times/(1+(rescale_t-1)*times)
extra={}
if a.stage=="texture":
    shape=read("shape1024_step_12")
    extra["concat_cond"]=SparseTensor(feats=shape["feats"],coords=shape["coords"])
attention_context=torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if a.attention=="math" else nullcontext()
with torch.inference_mode(),attention_context:
    sampler=FlowEulerGuidanceIntervalSampler(**spec["args"])
    if a.trajectory:
        expected=sampler.sample(model,sample,cond=cond,neg_cond=neg_cond,steps=steps,rescale_t=rescale_t,**params,**extra).samples
        a.step=steps
    else:
        expected=sampler.sample_once(model,sample,float(times[a.step-1]),float(times[a.step]),
            cond=cond,neg_cond=neg_cond,**params,**extra).pred_x_prev
expected=expected[0].flatten(1).T if a.stage=="structure" else expected.feats
if a.save_reference:
    a.save_reference.parent.mkdir(parents=True,exist_ok=True)
    save_file({"feats":expected.float().cpu().contiguous()},str(a.save_reference))
x=read(a.stage+f"_step_{a.step}")["feats"].double().flatten()
y=expected.double().flatten()
error=((x-y).norm()/y.norm()).item();cosine=torch.nn.functional.cosine_similarity(x,y,dim=0).item()
within_step_tolerance=error<=.02 and cosine>=.999
print(json.dumps(dict(backend=a.backend,stage=a.stage,step=a.step,trajectory=a.trajectory,attention=a.attention,nrmse=error,cosine=cosine,max_abs=(x-y).abs().max().item(),within_step_tolerance=within_step_tolerance)),flush=True)
assert torch.isfinite(x).all() and torch.isfinite(y).all()
if a.trajectory:
    print("Full BF16 trajectory diagnostic complete; compare with PyTorch backend/kernel baselines",flush=True)
else:
    assert within_step_tolerance
    print("Complete upstream flow-step PASS",flush=True)
