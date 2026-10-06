"""Rebake a candidate into a new directory while preserving fitted identity."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .provenance import validate_candidate
from .reference import Camera
from .materials import bake_portrait


def refine(candidate,out, *, exclude_mouth=False, exclude_non_skin=False):
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
    if exclude_mouth or exclude_non_skin:
        from PIL import Image
        from ..face_parsing import FaceParser
        from .occlusion import mouth_mask, skin_bake_mask
        from ..face_assets import PARSING_SHA256
        from .observations import sha256
        parser=FaceParser();reports=[]
        for i,view in enumerate(observations):
            image=np.asarray(Image.open(view['image_path']).convert('RGB'))
            labels,confidence=parser.predict(image)
            excluded=skin_bake_mask(labels,confidence) if exclude_non_skin else mouth_mask(labels,confidence)
            automatic=int(excluded.sum())
            if view.get('exclusion_mask_path'):
                excluded|=np.asarray(Image.open(view['exclusion_mask_path']).convert('L'))>0
            prefix='skin' if exclude_non_skin else 'mouth'
            path=out/f'{prefix}_exclusion_{i}.png'
            Image.fromarray(excluded.astype(np.uint8)*255).save(path)
            view['exclusion_mask_path']=str(path);view['exclusion_mask']=path.name
            view['exclusion_mask_sha256']=sha256(path)
            label_path=out/f'{prefix}_labels_{i}.png';Image.fromarray(labels).save(label_path)
            reports.append(dict(automatic_pixels=automatic,total_excluded_pixels=int(excluded.sum()),
                mouth_pixels=int(mouth_mask(labels,confidence).sum()),
                source='face parsing; manual masks unioned; lips protected',
                parser_sha256=PARSING_SHA256,labels=label_path.name,labels_sha256=sha256(label_path),
                independent_ground_truth=False))
        manifest['skin_texture_exclusions' if exclude_non_skin else 'mouth_texture_exclusions']=reports
        document=json.loads((out/'observations.json').read_text())
        document['views']=[{k:v for k,v in view.items() if not k.endswith('_path')} for view in observations]
        (out/'observations.json').write_text(json.dumps(document,indent=2))
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
    p.add_argument('--exclude-mouth',action='store_true',help='exclude parsed cavity pixels, retaining lips and manual masks')
    p.add_argument('--exclude-non-skin',action='store_true',help='exclude parsed background, clothes, accessories, hair, eyeballs and mouth from the skin bake')
    a=p.parse_args();print(json.dumps(refine(a.candidate,a.out,exclude_mouth=a.exclude_mouth,exclude_non_skin=a.exclude_non_skin),indent=2))


if __name__=='__main__':main()
