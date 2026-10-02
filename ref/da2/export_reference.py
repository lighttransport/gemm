#!/usr/bin/env python3
"""Offline export/oracle for the pinned vhuman Depth Anything V2 Small model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import cv2
import numpy as np
import torch
from safetensors.torch import save_file


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


class Compose:
    def __init__(self, transforms):
        self.transforms=transforms

    def __call__(self, x):
        for transform in self.transforms:
            x=transform(x)
        return x


def load_reference(installation):
    manifest=json.loads((installation/"installation.json").read_text())
    if manifest["model"]!="Depth-Anything-V2-Small":
        raise ValueError("only Small is supported")
    weights=installation/"depth_anything_v2_vits.pth"
    if sha256(weights)!=manifest["weights_sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    source=installation/"source"
    revision=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(source),"status","--porcelain","--untracked-files=no"],text=True)
    if revision!=manifest["code_revision"] or dirty:
        raise ValueError("reference checkout must match the pinned clean source")
    sys.path.insert(0,str(source))
    text=(source/"depth_anything_v2/dpt.py").read_text()
    old="from torchvision.transforms import Compose"
    if text.count(old)!=1:
        raise ValueError("unsupported reference source")
    module=types.ModuleType("depth_anything_v2.dpt")
    module.__package__="depth_anything_v2"
    module.__dict__["Compose"]=Compose
    exec(compile(text.replace(old,""),str(source/"depth_anything_v2/dpt.py"),"exec"),module.__dict__)
    model=module.DepthAnythingV2(encoder="vits",features=64,out_channels=[48,96,192,384])
    state=torch.load(weights,map_location="cpu",weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.cpu().eval(), state, manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--installation",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--image",type=Path)
    ap.add_argument("--input-size",type=int,default=518)
    args=ap.parse_args()
    torch.set_num_threads(4)
    model,state,manifest=load_reference(args.installation)
    args.out.mkdir(parents=True,exist_ok=True)
    for prefix,name in (("pretrained.","dinov2.safetensors"),("depth_head.","depth_head.safetensors")):
        tensors={k[len(prefix):]:v.float().contiguous() for k,v in state.items() if k.startswith(prefix)}
        save_file(tensors,str(args.out/name))
    converted={"version":1,"model":"Depth-Anything-V2-Small","source":manifest,
               "dtype":"F32","files":{name:sha256(args.out/name) for name in ("dinov2.safetensors","depth_head.safetensors")}}
    (args.out/"native.json").write_text(json.dumps(converted,indent=2))
    if args.image:
        raw=cv2.imread(str(args.image))
        if raw is None:
            raise ValueError("reference image unreadable")
        image,(h,w)=model.image2tensor(raw,args.input_size)
        image=image.cpu()
        image.numpy().tofile(args.out/"input.f32")
        ih,iw=image.shape[-2:]
        def capture(_module,_args,value):
            value.detach().cpu().numpy().tofile(args.out/"ref_fused.f32")
        model.depth_head.scratch.refinenet1.register_forward_hook(capture)
        with torch.inference_mode():
            features=model.pretrained.get_intermediate_layers(image,[2,5,8,11],return_class_token=True)
            for i,(patch,_cls) in enumerate(features):
                patch.numpy().tofile(args.out/f"ref_feature_{i}.f32")
            depth=model.depth_head(features,ih//14,iw//14).relu()
            depth.numpy().tofile(args.out/"ref_depth_model.f32")
            output=torch.nn.functional.interpolate(depth,(h,w),mode="bilinear",align_corners=True)[0,0]
            output.numpy().tofile(args.out/"reference.f32")
        (args.out/"fixture.json").write_text(json.dumps({"width":iw,"height":ih,"output_width":w,
                    "output_height":h,"input_size":args.input_size,"image":str(args.image.resolve())},indent=2))
        print(f"reference {iw}x{ih} -> {w}x{h}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
