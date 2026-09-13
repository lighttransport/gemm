"""Compare a saved full-resolution native remesh with the original CuMesh path."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors.numpy import load_file,save_file
from upstream_import import ROOT
p=argparse.ArgumentParser()
p.add_argument('--dump-dir',type=Path,required=True)
p.add_argument('--resolution',type=int,default=1024)
a=p.parse_args()
torch.set_num_threads(8)
assert torch.cuda.is_available() and not torch.version.hip
torch.cuda.set_per_process_memory_fraction(.45)
sys.path.insert(0,str(ROOT/'cumesh-upstream'))
import cumesh
original=load_file(a.dump_dir/'mesh_fdg.safetensors')
native=load_file(a.dump_dir/'mesh_remeshed.safetensors')
cache=a.dump_dir/f'mesh_reference_{a.resolution}.safetensors'
digest=hashlib.sha256()
with (a.dump_dir/'mesh_fdg.safetensors').open('rb') as source:
    while chunk:=source.read(4*1024*1024):digest.update(chunk)
provenance=dict(input_sha256=digest.hexdigest(),resolution=a.resolution,
                cumesh=json.loads((ROOT/'sources.json').read_text())['cumesh']['revision'])
metadata=cache.with_suffix('.json')
if cache.exists() and metadata.exists() and json.loads(metadata.read_text())==provenance:reference=load_file(cache)
else:
    with torch.inference_mode():
        vertices,faces=cumesh.remeshing.remesh_narrow_band_dc(
            torch.tensor(original['vertices'],device='cuda'),torch.tensor(original['faces'],device='cuda'),
            torch.zeros(3,device='cuda'),(a.resolution+3)/a.resolution,a.resolution,verbose=True)
    reference={'vertices':vertices.cpu().numpy(),'faces':faces.cpu().numpy()}
    save_file(reference,str(cache));metadata.write_text(json.dumps(provenance,indent=2)+'\n');del vertices,faces
    torch.cuda.empty_cache()

def compare(name,actual,expected):
    stats=[]
    for source,target in [(actual,expected),(expected,actual)]:
        bvh=cumesh.cuBVH(torch.tensor(target['vertices'],device='cuda'),torch.tensor(target['faces'],device='cuda'))
        # Every vertex and face centroid, in bounded batches. No nearest-vertex proxy.
        matched=count=0;maximum=0.
        for mode in ['vertices','centroids']:
            total=len(source['vertices'] if mode=='vertices' else source['faces'])
            for start in range(0,total,131072):
                points=source['vertices'][start:start+131072] if mode=='vertices' else source['vertices'][source['faces'][start:start+131072]].mean(1)
                distances=bvh.unsigned_distance(torch.tensor(points,device='cuda'))[0].cpu().numpy()
                matched+=int((distances<=2/a.resolution).sum());count+=len(distances)
                maximum=max(maximum,float(distances.max()))
        stats.append((matched/count,maximum));del bvh
        torch.cuda.empty_cache()
    precision,recall=[s[0] for s in stats]
    result=dict(stage=name,resolution=a.resolution,native_vertices=len(actual['vertices']),native_faces=len(actual['faces']),
                reference_vertices=len(expected['vertices']),reference_faces=len(expected['faces']),
                precision=precision,recall=recall,fscore=2*precision*recall/max(precision+recall,1e-30),
                max_surface_distance=max(s[1] for s in stats))
    print(json.dumps(result),flush=True)
    assert result['fscore']>=.99
    return result
results=[compare('remesh',native,reference)]
simplified_path=a.dump_dir/'mesh_simplified.safetensors'
if simplified_path.exists():
    mesh=cumesh.CuMesh();mesh.init(torch.tensor(native['vertices'],device='cuda'),torch.tensor(native['faces'],device='cuda'))
    mesh.simplify(1000000);mesh.remove_degenerate_faces()
    v,f=mesh.read()
    results.append(compare('simplify',load_file(simplified_path),{'vertices':v.cpu().numpy(),'faces':f.cpu().numpy()}))
(a.dump_dir/'mesh-reference-results.json').write_text(json.dumps(results,indent=2)+'\n')
print('Full mesh-stage reference: PASS',flush=True)
