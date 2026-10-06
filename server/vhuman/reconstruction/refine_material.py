"""Rebake a candidate into a new directory while preserving fitted identity."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .provenance import validate_candidate
from .reference import Camera
from .materials import bake_portrait


def refine(candidate,out):
    candidate,out=Path(candidate).resolve(),Path(out).resolve()
    manifest=validate_candidate(candidate)
    if out.exists() and any(out.iterdir()):raise ValueError('output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    for file in candidate.iterdir():
        if file.is_file():shutil.copyfile(file,out/file.name)
    observations=json.loads((out/'observations.json').read_text())['views']
    for view in observations:
        view['image_path']=str(out/view['image'])
        if view.get('exclusion_mask'):view['exclusion_mask_path']=str(out/view['exclusion_mask'])
    cameras=[Camera.from_dict(c) for c in manifest['geometry']['fitted_cameras']]
    with np.load(out/'geometry.npz',allow_pickle=False) as geometry:
        material=bake_portrait(geometry['captured'],geometry['triangles'],geometry['triangle_uvs'],
            observations,cameras,out,res=manifest['config']['texture_res'],
            roughness=manifest['material']['roughness']['value'],f0=manifest['material']['f0']['value'],
            spatial_materials=manifest['config'].get('spatial_materials',False))
    if manifest['material'].get('authored_detail'):
        shutil.copyfile(candidate/'skin_normal.png',out/'skin_normal.png')
        material['authored_detail']=manifest['material']['authored_detail']
    manifest['material']=material
    manifest['material_refinement']=dict(source=str(candidate),geometry_unchanged=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    validate_candidate(out)
    return material


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate');p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(refine(a.candidate,a.out),indent=2))


if __name__=='__main__':main()
