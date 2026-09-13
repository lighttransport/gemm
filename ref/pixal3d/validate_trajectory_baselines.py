"""Measure full BF16 trajectory sensitivity across original PyTorch backends.

This is a diagnostic, not a replacement for strict isolated-step comparisons.
Each original sampler starts from the same native noise and conditioning.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from safetensors.numpy import load_file
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument('--dump-dir',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--stage',choices=['structure','shape512','shape1024','texture'],required=True)
p.add_argument('--include-math',action='store_true',help='Also run the dense PyTorch mathematical SDPA kernel; this needs more VRAM')
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
variants=[('cuda','auto'),('rocm','auto')]
if a.include_math:variants.append(('rocm','math'))
outputs={'native':load_file(a.dump_dir/f'{a.stage}_step_12.safetensors')['feats'].astype(np.float64).ravel()}
commands=[]
for backend,attention in variants:
    stem=a.output_dir/f'{a.stage}-{backend}-{attention}'
    command=['bash',str(ROOT/'ref/pixal3d/run.sh'),backend,str(ROOT/'ref/pixal3d/validate_flow_stage.py'),
        '--backend',backend,'--dump-dir',str(a.dump_dir.resolve()),'--stage',a.stage,'--trajectory',
        '--attention',attention,'--save-reference',str(stem.with_suffix('.safetensors').resolve())]
    commands.append(command)
    with stem.with_suffix('.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=log,check=True)
    outputs[f'pytorch-{backend}-{attention}']=load_file(stem.with_suffix('.safetensors'))['feats'].astype(np.float64).ravel()
results=[]
for i,(name,x) in enumerate(outputs.items()):
    assert np.isfinite(x).all()
    for other,y in list(outputs.items())[i+1:]:
        assert x.shape==y.shape and np.isfinite(y).all()
        result=dict(a=name,b=other,nrmse=float(np.linalg.norm(x-y)/np.linalg.norm(y)),
                    cosine=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))))
        results.append(result);print(json.dumps(result),flush=True)
(a.output_dir/f'{a.stage}-comparison.json').write_text(json.dumps(dict(stage=a.stage,commands=commands,comparisons=results),indent=2)+'\n')
print('Trajectory sensitivity measurements complete; no equivalence threshold applied',flush=True)
