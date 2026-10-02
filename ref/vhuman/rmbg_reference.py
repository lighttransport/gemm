#!/usr/bin/env python3
"""Offline full RMBG2 oracle, using the pinned local implementation unchanged."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--image',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--side',type=int,default=1024)
    a=ap.parse_args()
    if a.side<32 or a.side>1024 or a.side%32:ap.error('side must be a multiple of 32 in [32,1024]')
    source=a.model/'birefnet.py'
    if hashlib.sha256(source.read_bytes()).hexdigest()!='e499d75224b8819e985e68fb78b7a8e8c99316840474e74e16b5529f03ca2860':
        raise ValueError('unrecognized upstream source')
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from torchvision import transforms
    torch.set_num_threads(4)
    pkg=types.ModuleType('rmbg_reference_package');pkg.__path__=[str(a.model.resolve())]
    sys.modules[pkg.__name__]=pkg
    mod=importlib.import_module(pkg.__name__+'.birefnet')
    with torch.device('meta'):
        m=mod.BiRefNet(config=mod.BiRefNetConfig(bb_pretrained=False))
    m.load_state_dict(load_file(str(a.model/'model.safetensors')),strict=True,assign=True)
    m.eval()
    image=Image.open(a.image).convert('RGB')
    x=transforms.Compose([transforms.Resize((a.side,a.side)),transforms.ToTensor(),
                          transforms.Normalize([.485,.456,.406],[.229,.224,.225])])(image)[None]
    a.out.mkdir(parents=True,exist_ok=True)
    x.numpy().astype('<f4').tofile(a.out/'input.f32')
    with torch.inference_mode():y=m(x)[-1]
    y.numpy().astype('<f4').tofile(a.out/'reference.f32')
    probability=y.sigmoid()[0,0].numpy()
    Image.fromarray((probability*255).astype('uint8')).resize(image.size).save(a.out/'reference-mask.png')
    (a.out/'fixture.json').write_text(json.dumps({'task':'rmbg','side':a.side,'image':str(a.image.resolve()),
        'model':str((a.model/'model.safetensors').resolve()),'torch':torch.__version__},indent=2)+'\n')
    print('reference saved',a.out,flush=True)


if __name__=='__main__':main()
