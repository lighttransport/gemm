#!/usr/bin/env python3
"""Offline MoGe2 camera-path export and oracle. Runtime needs no Torch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--image',type=Path)
    ap.add_argument('--side',type=int,default=512)
    ap.add_argument('--tokens',type=int,default=3600)
    a=ap.parse_args()
    if a.side<1 or a.side>1024 or a.tokens<1 or a.tokens>4096:ap.error('invalid bounded dimensions/tokens')
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    torch.set_num_threads(4)
    data=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    cfg=data['model_config']
    if cfg['encoder']!={'backbone':'dinov2_vitl14','intermediate_layers':[5,11,17,23],'dim_out':1024} or cfg['remap_output']!='exp':
        raise ValueError('unsupported MoGe encoder/remap')
    for key,num in [('neck',2),('points_head',1),('mask_head',1)]:
        expected=dict(dim_in=[1026,2,2,2,2] if key=='neck' else [1024,256,128,64,32],
            dim_out=None if key=='neck' else [None,None,None,None,3 if key=='points_head' else 1],
            dim_res_blocks=[1024,256,128,64,32],num_res_blocks=[0,num,num,num,0],
            res_block_in_norm='none',res_block_hidden_norm='none',
            resamplers=['conv_transpose','conv_transpose','conv_transpose','bilinear'])
        if cfg[key]!=expected:raise ValueError('unsupported MoGe stack '+key)
    a.out.mkdir(parents=True,exist_ok=True)
    state=data['model'];prefix='encoder.backbone.'
    save_file({k[len(prefix):]:v.float().contiguous() for k,v in state.items() if k.startswith(prefix)},str(a.out/'dinov2.safetensors'))
    save_file({k:v.float().contiguous() for k,v in state.items() if not k.startswith(prefix)},str(a.out/'heads.safetensors'))
    manifest={'format':'vhuman.moge2_camera.v1','model_config':cfg,'source_sha256':digest(a.checkpoint),
              'files':{p:digest(a.out/p) for p in ('dinov2.safetensors','heads.safetensors')}}
    (a.out/'native.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if a.image:
        sys.path.insert(0,str(ROOT/'ref/pixal3d/moge-upstream'))
        from moge.model.v2 import MoGeModel
        from moge.utils.geometry_torch import recover_focal_shift
        model=MoGeModel(**cfg)
        model.load_state_dict(state,strict=True,assign=True);model.eval()
        im=Image.open(a.image).convert('RGB').resize((a.side,a.side))
        x=torch.from_numpy(np.array(im).transpose(2,0,1).copy()).float()[None]/255
        x.numpy().astype('<f4').tofile(a.out/'input.f32')
        with torch.inference_mode():
            result=model(x,num_tokens=a.tokens)
            focal,shift=recover_focal_shift(result['points'],result['mask']>.5)
        output=torch.cat((result['points'].permute(0,3,1,2),result['mask'][:,None]),1)
        output.numpy().astype('<f4').tofile(a.out/'reference.f32')
        (a.out/'fixture.json').write_text(json.dumps({'task':'moge','side':a.side,'tokens':a.tokens,
            'grid':round(a.tokens**.5),'focal':float(focal[0]),'shift':float(shift[0]),'torch':torch.__version__},indent=2)+'\n')
    print('MoGe export/reference saved',a.out,flush=True)


if __name__=='__main__':main()
