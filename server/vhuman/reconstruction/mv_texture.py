"""Geometry-conditioned multiview texture completion with a shared evaluation.

Stages:
  prepare   render GNM conditions (six MV-Adapter orthographic cameras)
  generate  run one backend into work/<backend>/view_<name>.png
  bake      project views onto the atlas, fuse, harmonise and fill unseen texels
  eval      score every baked backend and write work/eval.html

Unlike generated_skin (bounded detail residual on a flat fill) this replaces
the colour of unseen texels. Photographed texels remain byte-identical.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from . import generated_skin as skin
from . import mv_conditioning as cond
from .observations import sha256
from .provenance import validate_candidate
from .reference import srgb_to_linear, linear_to_srgb

BACKENDS=('mvadapter','qwen_seq','qwen_grid')
RES=768


def load(candidate):
    candidate=Path(candidate)
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:geometry=dict(z)
    base=np.asarray(Image.open(candidate/'skin_basecolor.png').convert('RGB'))
    observed=np.asarray(Image.open(candidate/'skin_coverage.png'))>0
    return geometry,base,observed


def conditions(candidate, resolution=RES):
    geometry,base,observed=load(candidate)
    return cond.render_conditions(geometry,srgb_to_linear(base/255),observed.astype(float),resolution)


def prepare(candidate, work):
    candidate,work=Path(candidate).resolve(),Path(work).resolve()
    manifest=validate_candidate(candidate)
    folder=work/'conditions';folder.mkdir(parents=True,exist_ok=True)
    _,views=conditions(candidate)
    for v in views:
        Image.fromarray(v['rgb']).save(folder/f"{v['name']}_rgb.png")
        Image.fromarray(np.uint8(v['position']*255+.5)).save(folder/f"{v['name']}_position.png")
        Image.fromarray(np.uint8(v['normal']*255+.5)).save(folder/f"{v['name']}_normal.png")
        Image.fromarray(np.uint8((v['known']>.5)*255)).save(folder/f"{v['name']}_known.png")
    record=dict(schema='vhuman.mv_texture.v1',candidate=str(candidate),geometry_sha256=manifest['geometry_sha256'],
        basecolor_sha256=sha256(candidate/'skin_basecolor.png'),resolution=RES,
        views=[dict(name=v['name'],elevation=v['elevation'],azimuth=v['azimuth']) for v in views],
        conditions={p.name:sha256(p) for p in sorted(folder.iterdir())})
    skin.write_json(work/'mv_texture.json',record)
    return record


def check(work):
    work=Path(work);record=json.loads((work/'mv_texture.json').read_text())
    if sha256(Path(record['candidate'])/'skin_basecolor.png')!=record['basecolor_sha256']:
        raise ValueError('candidate material changed since prepare')
    for name,digest in record['conditions'].items():
        if sha256(work/'conditions'/name)!=digest:raise ValueError('condition checksum mismatch: '+name)
    return record


def generate(work, backend, **options):
    record=check(work);work=Path(work);candidate=Path(record['candidate'])
    out=work/backend;out.mkdir(exist_ok=True)
    frame,views=conditions(candidate,record['resolution'])
    if backend=='mvadapter':
        from . import mvadapter_backend as mva
        ref=mva.reference_image(candidate/'portrait.png',candidate/'parsing_0_silhouette.png',record['resolution'])
        ref.save(out/'reference.png')
        images,info=mva.generate(views,ref,**options)
    elif backend in ('qwen_seq','qwen_grid'):
        from . import mv_qwen
        images,info=(mv_qwen.sequential if backend=='qwen_seq' else mv_qwen.grid)(candidate,frame,views,out,**options)
    else:raise ValueError('unknown backend '+backend)
    for v,im in zip(views,images):Image.fromarray(im).save(out/f"view_{v['name']}.png")
    info.update(synthetic=True,geometry_evidence=False,
        views={f"view_{v['name']}.png":sha256(out/f"view_{v['name']}.png") for v in views})
    skin.write_json(out/'generation.json',info)
    return info


def fuse(colors, weights):
    """Facing-weighted mean plus per-texel cross-view disagreement (linear RGB)."""
    w=weights.sum(0);mean=(colors*weights[...,None]).sum(0)/np.maximum(w[:,None],1e-12)
    spread=np.sqrt((weights*((colors-mean)**2).sum(-1)).sum(0)/np.maximum(w,1e-12))
    return mean,w,spread,(weights>.01).sum(0)


def bake(work, backend, out):
    record=check(work);work,out=Path(work),Path(out);candidate=Path(record['candidate'])
    info=json.loads((work/backend/'generation.json').read_text())
    for name,digest in info['views'].items():
        if sha256(work/backend/name)!=digest:raise ValueError('generated view checksum mismatch: '+name)
    if out.exists() and any(out.iterdir()):raise ValueError('bake output must be empty')
    geometry,base,observed=load(candidate)
    frame,views=conditions(candidate,record['resolution'])
    valid,points,normals=skin.atlas_surface(geometry,len(base))
    images=[np.asarray(Image.open(work/backend/f"view_{v['name']}.png").convert('RGB')) for v in views]
    colors,weights=cond.project_views(frame,views,images,points,normals)
    gen,support,spread,count=fuse(colors,weights)
    linear=srgb_to_linear(base[valid]/255);seen=observed[valid]
    # Generators relight; match their per-channel level to photographed skin
    # over texels both observed and generated (robust median ratio).
    both=seen&(support>.05)
    gain=np.median(linear[both],0)/np.maximum(np.median(gen[both],0),1e-6) if both.sum()>500 else np.ones(3)
    gen=np.clip(gen*gain,0,1)
    blend=skin.unseen_weight(points,seen,distance=.01)*np.clip(support/.05,0,1)
    colour=linear*(1-blend[:,None])+gen*blend[:,None]
    result=base.copy()
    result[valid]=np.uint8(np.clip(linear_to_srgb(colour)*255+.5,0,255))
    result[observed]=base[observed]
    out.mkdir(parents=True,exist_ok=True)
    for path in candidate.iterdir():
        if path.is_file():shutil.copyfile(path,out/path.name)
    distance,near=distance_transform_edt(~valid,return_indices=True);gutter=(~valid)&(distance<=4)&~observed
    result[gutter]=result[near[0][gutter],near[1][gutter]]
    Image.fromarray(result).save(out/'skin_basecolor.png')
    sup=np.zeros(valid.shape);sup[valid]=blend
    Image.fromarray(np.uint8(np.clip(sup*.25,0,1)*255+.5)).save(out/'skin_generated_support.png')
    multi=count>=2
    report=dict(schema='vhuman.synthetic_skin_completion.v1',method='mv_texture',backend=backend,
        source_geometry_sha256=record['geometry_sha256'],work=str(work.resolve()),
        generator=info['generator'],license=info['license'],synthetic=True,generation=info,
        basecolor_sha256=sha256(out/'skin_basecolor.png'),gain=gain.tolist(),
        photographed_texels_changed=int(np.any(result[observed]!=base[observed],axis=-1).sum()),
        generated_texels=int((blend>.01).sum()),
        unseen_texels=int((~seen).sum()),unseen_covered=float((support[~seen]>.05).mean()),
        multiview_texels=int(multi.sum()),
        multiview_spread_linear=float(np.median(spread[multi&~seen])) if (multi&~seen).any() else None,
        limitations=['generated colour is a synthetic appearance prior, not recovered anatomy',
                     'cross-view agreement measures generator self-consistency, not accuracy'])
    manifest=validate_candidate(candidate)
    manifest['material']['synthetic_completion']=report
    manifest['material_refinement']=dict(source=str(candidate.resolve()),geometry_unchanged=True)
    skin.write_json(out/'manifest.json',manifest);skin.write_json(out/'skin_material.json',manifest['material'])
    skin.write_json(out/'generated_skin.json',report)
    validate_candidate(out)
    return report


def seam_energy(points, valid_result, seen, radius=.004):
    """Mean linear-RGB jump between observed texels and nearby generated texels."""
    tree=cKDTree(points[~seen]);pairs=tree.query(points[seen],distance_upper_bound=radius)
    hit=np.isfinite(pairs[0])
    a=valid_result[seen][hit];b=valid_result[~seen][pairs[1][hit]]
    return float(np.abs(a-b).mean()) if hit.any() else None


def evaluate(work, outs):
    record=check(work);work=Path(work);candidate=Path(record['candidate'])
    geometry,base,observed=load(candidate)
    valid,points,_=skin.atlas_surface(geometry,len(base));seen=observed[valid]
    frame,views=conditions(candidate,384)
    rows=[];scores={}
    for label,out in [('input',candidate)]+[(Path(o).name,Path(o)) for o in outs]:
        atlas=np.asarray(Image.open(out/'skin_basecolor.png').convert('RGB'))
        lin=srgb_to_linear(atlas/255)
        _,rendered=cond.render_conditions(geometry,lin,observed.astype(float),384)
        strip=np.concatenate([v['rgb'] for v in rendered],1)
        Image.fromarray(strip).save(work/f'eval_{label}.jpg',quality=92)
        s=dict(seam_energy=seam_energy(points,lin[valid],seen),
               photographed_texels_changed=int(np.any(atlas[observed]!=base[observed],axis=-1).sum()))
        gen=out/'generated_skin.json'
        if gen.exists() and label!='input':
            r=json.loads(gen.read_text())
            s.update({k:r.get(k) for k in ('backend','generator','license','unseen_covered','multiview_spread_linear')})
            s['seconds']=r.get('generation',{}).get('seconds')
        scores[label]=s
        rows.append(f'<h2>{label}</h2><pre>{json.dumps(s,indent=1)}</pre><img src="eval_{label}.jpg">')
    skin.write_json(work/'eval.json',scores)
    (work/'eval.html').write_text('<!doctype html><meta charset="utf-8"><title>Multiview texture eval</title>'
        '<style>body{background:#191b1e;color:#eee;font:15px system-ui;margin:24px}img{width:100%}</style>'
        '<h1>Multiview texture completion</h1><p>Views: front, right, back, left, top, bottom. '
        'Photographed texels must be unchanged (0). Seam energy = mean linear-RGB jump across the observed boundary.</p>'+''.join(rows))
    return scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','generate','bake','eval'))
    p.add_argument('--candidate');p.add_argument('--work',required=True)
    p.add_argument('--backend',choices=BACKENDS,default='mvadapter');p.add_argument('--out',nargs='*')
    p.add_argument('--steps',type=int);p.add_argument('--seed',type=int,default=317)
    a=p.parse_args()
    if a.stage=='prepare':print(json.dumps(prepare(a.candidate,a.work),indent=1)[:400])
    if a.stage=='generate':
        opts=dict(seed=a.seed);opts.update(steps=a.steps) if a.steps else None
        print(json.dumps(generate(a.work,a.backend,**opts),indent=1))
    if a.stage=='bake':print(json.dumps(bake(a.work,a.backend,a.out[0]),indent=1))
    if a.stage=='eval':print(json.dumps(evaluate(a.work,a.out or []),indent=1))


if __name__=='__main__':main()
