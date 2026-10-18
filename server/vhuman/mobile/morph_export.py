"""A morph-target runtime package from a built facial rig.

The package is the rig's glTF (skinned, sparse morph targets), rig.json
(controls, correctives, joint matrix, eye frames), the ML deformer when
present, and the analytic eye material inputs (flat eye_* files), with the
same checksummed avatar.json as the native package (`runtime: "morph"`).
The browser player evaluates the rig on the GPU through morph targets and
skinning (web/vhuman_rig_eval.js) and shades the carried runtime eyeballs
with the analytic mesh eye material; no WASM evaluator is needed.
"""
import argparse
import json
import shutil
from pathlib import Path

from .export import validate_package, write_eye_files, sha256
from .profile import IPHONE12

RIG_FILES=('rig.glb','rig.json','deformer.lrm','rig_lod1.glb','rig_lod2.glb')


def export(rig,out,*,head=None,eye_params=None,profile='iphone12'):
    rig,out=Path(rig),Path(out)
    if profile!='iphone12':raise ValueError('unsupported mobile profile')
    if not (rig/'rig.glb').is_file() or not (rig/'rig.json').is_file():raise ValueError('rig.glb and rig.json are required')
    if out.exists() and any(out.iterdir()):raise ValueError('mobile output must be empty')
    definition=json.loads((rig/'rig.json').read_text())
    if not definition.get('controls') or not definition.get('blendshapes'):raise ValueError('rig.json has no controls or shapes')
    report=json.loads((rig/'rig_report.json').read_text()) if (rig/'rig_report.json').is_file() else {}
    meshes=(report.get('gltf') or {}).get('meshes') or []
    triangles=int(sum(m['triangles'] for m in meshes))
    if triangles>IPHONE12['triangle_budgets'][0]:raise ValueError(f'rig exceeds the mobile triangle budget: {triangles}')
    out.mkdir(parents=True,exist_ok=True)
    copied=[]
    for name in RIG_FILES:
        if (rig/name).is_file():shutil.copyfile(rig/name,out/name);copied.append(name)
    if eye_params is None and head is not None:
        fit=json.loads((Path(head)/'fit.json').read_text())
        eye_params=fit.get('eye_params')
    eye_files=write_eye_files(eye_params,out) if eye_params else []
    result=dict(schema='vhuman.mobile_avatar.v1',runtime='morph',profile=IPHONE12,rig=dict(files=copied,
        controls=len(definition['controls']),blendshapes=len(definition['blendshapes']),joints=[j['name'] for j in definition['joints']],
        face_model=definition.get('face_model'),ml_deformer=bool(definition.get('ml_deformer')),eyes=definition.get('eyes'),
        lods=[n for n in copied if n.startswith('rig_lod')]),
        parts=[dict(name=m['name'],triangles=m['triangles'],vertices=m.get('vertices')) for m in meshes],triangles=triangles,
        eyes=dict(material='analytic mesh eye material from the eye_* files' if eye_files else 'standard glTF materials',files=eye_files),
        lid_clearance=report.get('lid_clearance'),
        validation=dict(device_measured=False,production_ready=False),
        limitations=['morph-target reference runtime; live speech streaming needs the semantic control stream',
            'no mouth occlusion or contact deformer in the morph runtime yet; the viewer projects no contacts',
            'source image and generated-reference redistribution terms require separate receipts'])
    if definition.get('portrait_provenance'):result['portrait_provenance']=definition['portrait_provenance']
    if definition.get('input_provenance'):result['input_provenance']=definition['input_provenance']
    result['files']={p.name:dict(sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.is_file()}
    (out/'avatar.json').write_text(json.dumps(result,indent=2));validate_package(out)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rig',required=True,help='a built rig folder (rig.glb, rig.json, ...)')
    p.add_argument('--out',required=True);p.add_argument('--head',help='the head folder whose fit.json supplies the eye parameters')
    p.add_argument('--profile',default='iphone12',choices=['iphone12'])
    a=p.parse_args();r=export(a.rig,a.out,head=a.head,profile=a.profile)
    print(json.dumps(dict(out=a.out,triangles=r['triangles'],rig=r['rig']['files'],eyes=r['eyes']['files']),indent=2))


if __name__=='__main__':main()
