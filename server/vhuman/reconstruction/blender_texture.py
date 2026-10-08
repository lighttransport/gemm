"""Prepare a checksummed skin surface for Blender visibility, USD and shading review."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from .generated_skin import atlas_surface
from .observations import sha256
from .provenance import validate_candidate
from .reference import Camera


def prepare(candidate, out, *, source=None):
    candidate,out=Path(candidate).resolve(),Path(out).resolve()
    source=Path(source or candidate).resolve()
    manifest=validate_candidate(candidate);original=validate_candidate(source)
    if manifest['geometry_sha256']!=original['geometry_sha256']:
        raise ValueError('review and source geometry must match')
    if manifest['portrait_sha256']!=original['portrait_sha256']:
        raise ValueError('review and source portrait must match')
    if len(original['geometry']['fitted_cameras'])!=1:
        raise ValueError('Blender visibility audit currently requires one source camera')
    if out.exists() and any(out.iterdir()):raise ValueError('Blender review output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    with np.load(candidate/'geometry.npz',allow_pickle=False) as data:geometry=dict(data)
    resolution=Image.open(candidate/'skin_basecolor.png').width
    valid,points,normals=atlas_surface(geometry,resolution)
    camera=Camera.from_dict(original['geometry']['fitted_cameras'][0])
    pixels,depth=camera.project(points)
    np.savez_compressed(out/'surface.npz',valid=valid,points=points,normals=normals,
        pixels=pixels,depth=depth,camera_origin=camera.origin,
        observed=np.asarray(Image.open(source/'skin_coverage.png'))[valid]>0,
        confidence=np.asarray(Image.open(source/'skin_confidence.png'))[valid]/255.)
    record=dict(schema='vhuman.blender_texture.v1',candidate=str(candidate),source=str(source),out=str(out),
        geometry_sha256=manifest['geometry_sha256'],basecolor_sha256=sha256(candidate/'skin_basecolor.png'),
        source_basecolor_sha256=sha256(source/'skin_basecolor.png'),surface_sha256=sha256(out/'surface.npz'),
        coordinates='GNM metres Y-up; Blender metres Z-up via (x,-z,y)',
        device='OPTIX',samples=32,resolution=512,audit=True,render=True)
    (out/'request.json').write_text(json.dumps(record,indent=2))
    return record


def run(candidate,out, *, source=None,blender,device='OPTIX',samples=32,resolution=512,audit=True,render=True):
    if device not in ('OPTIX','CUDA','CPU'):raise ValueError('unsupported Cycles device')
    if samples<1 or resolution<32:raise ValueError('invalid render size/samples')
    request=prepare(candidate,out,source=source);out=Path(request['out'])
    request.update(device=device,samples=samples,resolution=resolution,audit=audit,render=render)
    (out/'request.json').write_text(json.dumps(request,indent=2))
    cache=out/'cache';cache.mkdir()
    env=dict(os.environ,TMPDIR=str(cache),TEMP=str(cache),TMP=str(cache))
    with (out/'blender.log').open('w') as log:
        subprocess.run([str(Path(blender).expanduser()),'-b','--factory-startup','--python-exit-code','1',
            '--python',str(Path(__file__).with_name('blender_texture_worker.py')),'--',str(out/'request.json')],
            env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((out/'result.json').read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True);p.add_argument('--source');p.add_argument('--out',required=True)
    p.add_argument('--blender',required=True);p.add_argument('--device',choices=('OPTIX','CUDA','CPU'),default='OPTIX')
    p.add_argument('--samples',type=int,default=32);p.add_argument('--resolution',type=int,default=512)
    p.add_argument('--no-audit',dest='audit',action='store_false');p.add_argument('--no-render',dest='render',action='store_false')
    print(json.dumps(run(**vars(p.parse_args())),indent=2))


if __name__=='__main__':main()
