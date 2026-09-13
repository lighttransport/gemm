"""Fetch pinned reference sources and convert official NAF weights locally."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import torch
from safetensors.torch import save_file

root=Path(__file__).resolve().parent
sources=json.loads((root/'sources.json').read_text())
for name,directory in [('pixal3d','upstream'),('naf','naf-upstream'),('cumesh','cumesh-upstream'),('flexgemm','flexgemm-upstream')]:
    source=sources[name];dest=root/directory
    if not dest.exists():
        dest.mkdir()
        subprocess.run(['git','init',str(dest)],check=True)
        subprocess.run(['git','-C',str(dest),'remote','add','origin',source['url']],check=True)
        subprocess.run(['git','-C',str(dest),'fetch','--depth','1','origin',source['revision']],check=True)
        subprocess.run(['git','-C',str(dest),'checkout','--detach',source['revision']],check=True)
    actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
    if actual!=source['revision']:
        raise RuntimeError(f"{dest}: expected {source['revision']}, found {actual}; preserve local changes and select the pinned revision")
path=root/'weights/naf_release.pth'
path.parent.mkdir(exist_ok=True)
if not path.exists():
    temporary=path.with_suffix('.partial')
    urllib.request.urlretrieve(sources['naf_weights'],temporary)
    temporary.replace(path)
digest=hashlib.sha256(path.read_bytes()).hexdigest()
if digest!=sources['naf_checkpoint_sha256']:raise RuntimeError('Unexpected NAF checkpoint SHA256: '+digest)
weights=torch.load(path,map_location='cpu',weights_only=True)
save_file({k:v.contiguous() for k,v in weights.items()},str(path.with_suffix('.safetensors')))
print('NAF checkpoint sha256:',hashlib.sha256(path.read_bytes()).hexdigest())
revision=sources['trellis2']['revision']
for name,source in [('o_voxel_postprocess.py','o-voxel/o_voxel/postprocess.py'),
                    ('flexible_dual_grid.py','o-voxel/o_voxel/convert/flexible_dual_grid.py')]:
    path=root/'deps'/name;path.parent.mkdir(exist_ok=True)
    url=f'https://raw.githubusercontent.com/microsoft/TRELLIS.2/{revision}/{source}'
    urllib.request.urlretrieve(url,path)
    print(name,hashlib.sha256(path.read_bytes()).hexdigest())
