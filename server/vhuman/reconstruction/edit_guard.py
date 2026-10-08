"""Reject garment-bearing edited views before projection; retain failed images for diagnosis."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from ..face_parsing import FaceParser,LABELS
from .observations import sha256


def clothing_fraction(labels,confidence,valid, *, threshold=.5):
    labels,confidence,valid=map(np.asarray,(labels,confidence,valid))
    if labels.shape!=valid.shape or confidence.shape!=valid.shape or not valid.any():
        raise ValueError('nonempty, aligned view masks required')
    garment=np.isin(labels,[LABELS.index(n) for n in ('clothes','necklace','hat')])&(confidence>=threshold)&valid
    return float(garment.sum()/valid.sum())


def guard(work,backend, *, parsing_model,max_clothing_fraction=.005):
    from .mv_texture import check,conditions
    if not 0<=max_clothing_fraction<=1:raise ValueError('invalid clothing threshold')
    if Path(backend).name!=backend or backend in ('.','..'):raise ValueError('invalid backend directory')
    work=Path(work);record=check(work);folder=work/backend
    info=json.loads((folder/'generation.json').read_text());names=list(info.get('view_runs',{}))
    if not names:raise ValueError('edit guard requires recorded independent raw edits')
    _,views=conditions(record['candidate'],record['resolution'],only=names)
    parser=FaceParser(model=parsing_model);results={}
    for v in views:
        path=folder/v['name']/'edited.png';digest=sha256(path)
        if digest!=info['view_runs'][v['name']]['edited_sha256']:raise ValueError('edited view changed')
        image=Image.open(path).convert('RGB').resize(v['valid'].shape[::-1],Image.Resampling.LANCZOS)
        labels,confidence=parser.predict(np.asarray(image));fraction=clothing_fraction(labels,confidence,v['valid'])
        results[v['name']]=dict(edited_sha256=digest,clothing_fraction=fraction,approved=fraction<=max_clothing_fraction)
    report=dict(schema='vhuman.edit_guard.v1',generation_sha256=sha256(folder/'generation.json'),
        source_geometry_sha256=record['geometry_sha256'],source_basecolor_sha256=record['basecolor_sha256'],
        parsing_model_sha256=sha256(parsing_model),max_clothing_fraction=max_clothing_fraction,views=results,
        approved=all(v['approved'] for v in results.values()),
        limitations=['semantic garment rejection is not a visual identity or anatomy guarantee'])
    (folder/'quality.json').write_text(json.dumps(report,indent=2));return report


def require_approved(folder,view,digest,record):
    folder=Path(folder)
    try:report=json.loads((folder/'quality.json').read_text())
    except (OSError,ValueError) as exc:raise ValueError('raw edit requires a quality.json guard receipt') from exc
    if (report.get('schema')!='vhuman.edit_guard.v1' or report.get('generation_sha256')!=sha256(folder/'generation.json')
        or report.get('source_geometry_sha256')!=record['geometry_sha256']
        or report.get('source_basecolor_sha256')!=record['basecolor_sha256']):
        raise ValueError('edit guard provenance mismatch')
    status=report.get('views',{}).get(view,{})
    if not status.get('approved') or status.get('edited_sha256')!=digest:raise ValueError('raw edited view rejected by clothing guard: '+view)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',required=True);p.add_argument('--backend',required=True);p.add_argument('--parsing-model',required=True)
    p.add_argument('--max-clothing-fraction',type=float,default=.005)
    report=guard(**vars(p.parse_args()));print(json.dumps(report,indent=2))
    if not report['approved']:raise SystemExit(2)


if __name__=='__main__':main()
