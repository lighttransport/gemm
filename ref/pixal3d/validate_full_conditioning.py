"""Full image conditioning against upstream DINOv3, NAF and camera math.

NAF evaluates only the projected bilinear neighbors, using the neighborhood
layout independently checked against original NATTEN by validate_conditioning.py.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
from typing import *
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
from upstream_import import ROOT
p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['cpu','cuda','rocm'],required=True)
p.add_argument('--dump-dir',type=Path,required=True)
p.add_argument('--stage',choices=['structure','shape512','shape1024','texture'],required=True)
p.add_argument('--fov',type=float,required=True)
p.add_argument('--distance',type=float,default=0)
p.add_argument('--mesh-scale',type=float,default=1)
a=p.parse_args();torch.set_num_threads(8)
device='cpu' if a.backend=='cpu' else 'cuda'
if device=='cuda':
    assert torch.cuda.is_available() and bool(torch.version.hip)==(a.backend=='rocm')
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
size=512 if a.stage in ['structure','shape512'] else 1024
grid={'structure':16,'shape512':32,'shape1024':64,'texture':64}[a.stage]
target={'structure':0,'shape512':512,'shape1024':512,'texture':1024}[a.stage]
image=load_file(a.dump_dir/f'image{size}.safetensors')['feats'].reshape(size,size,3).to(device)
saved=load_file(a.dump_dir/f'{a.stage}_projected.safetensors')
coords=saved['coords'].to(device)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def check(name,actual,expected):
    x=actual.to('cpu',dtype=torch.float64).flatten();y=expected.to('cpu',dtype=torch.float64).flatten()
    error=((x-y).norm()/y.norm()).item();cos=F.cosine_similarity(x,y,dim=0).item()
    print(json.dumps(dict(backend=a.backend,stage=a.stage,test=name,nrmse=error,cosine=cos,max_abs=(x-y).abs().max().item())),flush=True)
    assert torch.isfinite(x).all() and error<=.02 and cos>=.999
reference=module('trellis_dino_reference',ROOT.parent/'trellis2/gen_stage1_ref.py')
model=reference.load_dinov3_from_timm('/mnt/disk2/models/dinov3-vitl16/model.safetensors',device)
normalized=((image-torch.tensor([.485,.456,.406],device=device))/torch.tensor([.229,.224,.225],device=device)).permute(2,0,1)[None]
with torch.inference_mode():
    h=model.embeddings(normalized,bool_masked_pos=None);rope=model.rope_embeddings(normalized)
    for layer in model.layer:h=layer(h,position_embeddings=rope)
    features=F.layer_norm(h,h.shape[-1:])[0]
check('global',load_file(a.dump_dir/f'{a.stage}_global.safetensors')['feats'],features[:5])
patches=features[5:].reshape(size//16,size//16,1024)
del model,h,features,rope,normalized
if device=='cuda':torch.cuda.empty_cache()
source=ast.parse((ROOT/'upstream/pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py').read_text())
nodes=[n for n in source.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['project_points_to_image_batch','ProjGrid']]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'upstream_projection','exec'),globals())
projection=ProjGrid(grid,size).to(device)
index=(coords[:,1]*grid+coords[:,2])*grid+coords[:,3]
points=projection.grid_points[index][None]/a.mesh_scale/2
transform=projection.front_view_transform_matrix[None].clone()
transform[:,1,3]=-(a.distance or .5/(np.tan(a.fov/2)*a.mesh_scale))
xy,_,_=project_points_to_image_batch(points,transform,torch.tensor([a.fov],device=device),size)
xy=(xy[0]+.5)/size*2-1
low=F.grid_sample(patches.permute(2,0,1)[None],xy[None,None],padding_mode='border',align_corners=False)[0,:,0].T
if not target:
    check('projected',saved['feats'],low)
else:
    conv=module('naf_convolutions',ROOT/'naf-upstream/src/layers/convolutions.py')
    rope=module('naf_rope',ROOT/'naf-upstream/src/layers/rope.py')
    encoder,RoPE=conv.encoder,rope.RoPE
    source=ast.parse((ROOT/'naf-upstream/src/model/naf.py').read_text())
    nodes=[n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='ImageEncoder']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'upstream_naf_image_encoder','exec'),globals())
    model=ImageEncoder(heads_rope=4,rope_base=100.,rope_rescale=2.).eval()
    model.load_state_dict({k.removeprefix('image_encoder.'):v for k,v in load_file(ROOT/'weights/naf_release.safetensors').items()})
    with torch.inference_mode():guide=model(image.permute(2,0,1)[None].cpu(),(target,target)).to(device)
    keys=F.adaptive_avg_pool2d(guide,size//16)[0].permute(1,2,0).reshape(size//16,size//16,4,64)
    queries=guide[0].permute(1,2,0).reshape(-1,4,64)
    pixel=((xy+1)*target*.5-.5).clamp(0,target-1);base=pixel.floor().int();frac=pixel-base
    neighbors=torch.stack([base,base+torch.tensor([1,0],device=device),base+torch.tensor([0,1],device=device),base+1],1).clamp(max=target-1)
    ids=neighbors[:,:,1]*target+neighbors[:,:,0]
    unique,inverse=ids.unique(return_inverse=True)
    high=torch.empty(len(unique),1024,device=device)
    offset=torch.arange(9,device=device);patch_grid=size//16;dilation=target//patch_grid
    values=patches.reshape(patch_grid,patch_grid,4,256)
    with torch.inference_mode():
        for start in range(0,len(unique),64):
            chosen=unique[start:start+64]
            rows=(chosen//target//dilation-4).clamp(0,patch_grid-9)[:,None]+offset
            cols=(chosen%target//dilation-4).clamp(0,patch_grid-9)[:,None]+offset
            k=keys[rows[:,:,None],cols[:,None,:]];v=values[rows[:,:,None],cols[:,None,:]]
            scores=torch.einsum('rhd,rijhd->rhij',queries[chosen],k)*.125
            scores=scores.flatten(-2).softmax(-1).reshape(-1,4,9,9)
            high[start:start+64]=torch.einsum('rhij,rijhd->rhd',scores,v).flatten(1)
    corners=high[inverse.reshape(-1)].reshape(-1,4,1024);x,y=frac[:,0:1],frac[:,1:2]
    high=((1-x)*corners[:,0]+x*corners[:,1])*(1-y)+((1-x)*corners[:,2]+x*corners[:,3])*y
    check('DINO projected',saved['feats'][:,:1024],low)
    check('NAF projected',saved['feats'][:,1024:],high)
print('Full conditioning PASS',flush=True)
