"""CPU oracle for native edge/corner blending using official spatial_tiled_decode."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
PIN='60783e704160023913bee78f0b47036d393d4dfa'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--upstream',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    upstream=args.upstream.resolve()
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN:
        raise ValueError('unexpected upstream revision')
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(upstream))
    spec=importlib.util.spec_from_file_location('hv15_tiling_oracle',upstream/'hyvideo/models/autoencoders/hunyuanvideo_15_vae.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    class TileDecoder(torch.nn.Module):
        def forward(self,x):
            value=x.repeat_interleave(16,-2).repeat_interleave(16,-1)
            y=torch.arange(value.shape[-2],dtype=torch.float32)[:,None]
            x=torch.arange(value.shape[-1],dtype=torch.float32)[None,:]
            return value+(x+2*y)/4096
    # Construct only the tiling shell. Neural layers are replaced by a known
    # tile-local transform, so different blends cannot hide behind uniform data.
    model=module.AutoencoderKLConv3D.__new__(module.AutoencoderKLConv3D)
    torch.nn.Module.__init__(model)
    model.tile_sample_min_size=64
    model.tile_latent_min_size=4
    model.tile_overlap_factor=0.25
    model._tile_parallelism_enabled=False
    model.decoder=TileDecoder()
    value=torch.arange(7*9*2*2,dtype=torch.float32).reshape(1,2,2,9,7)/256
    with torch.inference_mode():
        reference=model.spatial_tiled_decode(value).numpy()
    subprocess.run([str(ROOT/'cuda/hunyuan_video15/test_cuda_hunyuan_video15_vae_tiling'),str(out/'native.f32')],check=True)
    actual=np.fromfile(out/'native.f32',dtype='<f4').reshape(reference.shape)
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('non-finite tiling output')
    error=float(np.abs(reference-actual).max())
    report={'upstream_revision':PIN,'scope':'CPU spatial tiling edge/corner ordering',
            'shape':list(reference.shape),'max_absolute_error':error,'pass':error<=1e-6}
    (out/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['pass'] else 1

if __name__=='__main__':
    raise SystemExit(main())
