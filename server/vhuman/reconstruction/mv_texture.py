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
import warnings

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from . import generated_skin as skin
from . import mv_conditioning as cond
from .observations import sha256
from .provenance import validate_candidate
from .reference import srgb_to_linear, linear_to_srgb

POLAR_VIEWS=('top','bottom')
FACELESS_VIEWS=('back','top','bottom')
DEFAULT_HYBRID='front=qwen_edit_seq:raw,right=qwen_edit_seq:raw,left=qwen_edit_seq:raw,back=mvadapter:view,top=mvadapter:view,bottom=mvadapter:view'
BACKENDS=('mvadapter','qwen_seq','qwen_grid','qwen_edit_seq')
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
    elif backend=='qwen_edit_seq':
        from . import mv_qwen
        from .qwen_edit_backend import make_editor
        images,info=mv_qwen.sequential(candidate,frame,views,out,editor=make_editor(),**{'steps':12,**options})
    else:raise ValueError('unknown backend '+backend)
    for v,im in zip(views,images):Image.fromarray(im).save(out/f"view_{v['name']}.png")
    info.update(synthetic=True,geometry_evidence=False,
        views={f"view_{v['name']}.png":sha256(out/f"view_{v['name']}.png") for v in views})
    skin.write_json(out/'generation.json',info)
    return info


def compose(work, spec, name='hybrid'):
    """Assemble a per-view hybrid generation (e.g. Edit-2511 for face-seeing views, MV-Adapter elsewhere).

    spec: 'view=<backend dir>:<raw|view>,...'. 'raw' takes <dir>/<view>/edited.png (full editor output),
    'view' takes <dir>/view_<view>.png. Writes <work>/<name>/view_*.png and a generation.json whose licence
    is the union of the sources (MV-Adapter's SDXL base makes the result evaluation-only).
    """
    record=check(work);work=Path(work);out=work/name;out.mkdir(exist_ok=True)
    res=record['resolution'];sources={};licences=set();generators={}
    for item in spec.split(','):
        view,src=item.split('=');backend,kind=src.split(':')
        path=work/backend/view/'edited.png' if kind=='raw' else work/backend/f'view_{view}.png'
        im=Image.open(path).convert('RGB')
        if im.size!=(res,res):im=im.resize((res,res),Image.Resampling.LANCZOS)
        im.save(out/f'view_{view}.png')
        info=json.loads((work/backend/'generation.json').read_text())
        licences.add(info.get('license','unknown'));generators[view]=f"{info.get('generator',backend)} [{kind}]"
        sources[view]=dict(backend=backend,kind=kind,sha256=sha256(path))
    names=[v['name'] for v in record['views']]
    if set(sources)!=set(names):raise ValueError('hybrid spec must cover every view: '+','.join(names))
    info=dict(generator='hybrid: '+'; '.join(f'{v}={generators[v]}' for v in names),
              license=' + '.join(sorted(licences)),sources=sources,synthetic=True,geometry_evidence=False,
              views={f'view_{v}.png':sha256(out/f'view_{v}.png') for v in names})
    skin.write_json(out/'generation.json',info)
    return info


def fuse(colors, weights):
    """Facing-weighted mean plus per-texel cross-view disagreement (linear RGB)."""
    w=weights.sum(0);mean=(colors*weights[...,None]).sum(0)/np.maximum(w[:,None],1e-12)
    spread=np.sqrt((weights*((colors-mean)**2).sum(-1)).sum(0)/np.maximum(w,1e-12))
    return mean,w,spread,(weights>.01).sum(0)


def bake(work, backend, out, *, delight=True, source='auto', polar='auto', two_band=True, band_sigma=4.,
         exclude_clothing=True):
    record=check(work);work,out=Path(work),Path(out);candidate=Path(record['candidate'])
    info=json.loads((work/backend/'generation.json').read_text())
    for name,digest in info['views'].items():
        if sha256(work/backend/name)!=digest:raise ValueError('generated view checksum mismatch: '+name)
    if out.exists() and any(out.iterdir()):raise ValueError('bake output must be empty')
    geometry,base,observed=load(candidate)
    frame,views=conditions(candidate,record['resolution'])
    valid,points,normals=skin.atlas_surface(geometry,len(base))
    # source='raw': fuse each view's full editor output (<name>/edited.png) instead of the sequential
    # composite view_<name>.png, whose earlier-view regions are re-renders (hard commit seams). 'auto'
    # uses raw outputs when every view has one (sequential backends).
    def load_view(v):
        raw=work/backend/v['name']/'edited.png'
        # Top/bottom raw edits hallucinate faces (portrait-conditioned editor on a top-down head): keep
        # their sequential composite, which only adds the hole no earlier view covered.
        path=raw if use_raw and raw.exists() and (v['name'] not in POLAR_VIEWS or polar=='raw') else work/backend/f"view_{v['name']}.png"
        im=Image.open(path).convert('RGB')
        return np.asarray(im.resize((record['resolution'],)*2,Image.Resampling.LANCZOS) if im.size[0]!=record['resolution'] else im)
    raw_all=all((work/backend/v['name']/'edited.png').exists() for v in views)
    polar_ok=(work/backend/'polar_regen.json').exists()   # top/bottom re-edited without the portrait
    use_raw=source=='raw' or (source=='auto' and raw_all)
    images=[load_view(v) for v in views]
    exclusion_report=None
    if exclude_clothing:
        # Editors paint the subject's 'usual' clothing (suit/shirt/tie) on the untextured neck/shoulders even when
        # told not to; parse every view and treat clothing/accessory pixels as not visible, so those texels come
        # from skin seen elsewhere or nearest-fill.
        from scipy.ndimage import binary_dilation
        from ..face_parsing import FaceParser, LABELS
        parser=FaceParser();drop=[LABELS.index(n) for n in ('glasses','earring','necklace','clothes','hat')]
        # Views that cannot see the face: portrait-conditioned edits paint one anyway (a face on the back of the head).
        face_parts=[LABELS.index(n) for n in ('left_brow','right_brow','left_eye','right_eye','nose','mouth','upper_lip','lower_lip')]
        exclusion_report={};views=[dict(v) for v in views]
        for v,im in zip(views,images):
            labels,conf=parser.predict(im)
            bad=np.isin(labels,drop)&(conf>=.5)
            if v['name'] in FACELESS_VIEWS:
                feats=np.isin(labels,face_parts)&(conf>=.4)
                if feats.sum()>.002*v['valid'].sum():      # a hallucinated face: drop its whole skin region too
                    bad|=binary_dilation(feats,iterations=40)
            bad=binary_dilation(bad,iterations=4)
            exclusion_report[v['name']]=float((bad&v['valid']).sum()/max(v['valid'].sum(),1))
            v['valid']=v['valid']&~bad
    delight_report=None
    if delight:
        from .mv_delight import delight as remove_light
        delight_report={}
        for i,v in enumerate(views):
            # structure views of a hybrid (MV-Adapter) also get hue-blotch flattening
            structure=bool(info.get('sources')) and info['sources'].get(v['name'],{}).get('kind')!='raw'
            images[i],delight_report[v['name']]=remove_light(images[i],v['normal'],v['valid'],chroma=structure)
    # polar='skip': top/bottom generations are unreliable (portrait-conditioned edits paint faces on the crown;
    # portrait-free ones ignore the top-down camera). Use only horizontal views, accepting grazing angles
    # (facing >= 0.05, still weighted by facing^2) for texels no view sees head-on; fill_unsupported covers the rest.
    polar_mode=('skip' if use_raw else 'composite') if polar=='auto' else polar
    unseen=grazing_facing=None
    is_polar=np.array([v['name'] in POLAR_VIEWS for v in views])
    def project(imgs):
        """Projection with the polar policy applied; returns colors, weights, unseen, grazing facing."""
        colors,weights=cond.project_views(frame,views,imgs,points,normals)
        unseen=grazing=None
        if polar_mode=='skip':
            side=[i for i,p in enumerate(is_polar) if not p]
            gc,gw=cond.project_views(frame,[views[i] for i in side],[imgs[i] for i in side],points,normals,min_facing=.05)
            weights[is_polar]=0
            unseen=weights[~is_polar].sum(0)<=.01        # no horizontal view sees these head-on
            grazing=np.sqrt(gw.max(0))                    # weights are facing^2 (0 where not visible)
            for k,i in enumerate(side):
                colors[i]=np.where(unseen[:,None],gc[k],colors[i]);weights[i]=np.where(unseen,gw[k],weights[i])
        elif use_raw and not polar_ok:   # polar composites only fill texels the horizontal (raw) views do not see
            side_seen=weights[~is_polar].sum(0)>.01
            weights[is_polar]*=np.where(side_seen,.05,1.)[None]
        return colors,weights,unseen,grazing
    colors,weights,unseen,grazing_facing=project(images)
    # Two-band fusion: overlapping generations have misaligned fine detail (stubble), so averaging ghosts it
    # (the crown "star"). Tone (low band) is fused with facing^2 weights; detail (high band) with facing^8,
    # i.e. nearly winner-take-all from the best-facing view.
    if two_band:
        from scipy.ndimage import gaussian_filter
        low_images=[np.uint8(np.clip(gaussian_filter(im.astype(np.float32),(band_sigma,band_sigma,0))+.5,0,255)) for im in images]
        colors_low,_,_,_=project(low_images)
    linear=srgb_to_linear(base[valid]/255);seen=observed[valid]
    weights*=neck_cut_weight(geometry,points)[None]
    # Reject per-view outliers (cast shadows, collars) against the cross-view
    # luminance median before fusing.
    luma=colors@np.array([.2126,.7152,.0722]);has=weights>.01
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)  # texels no view sees
        ref=np.nanmedian(np.where(has,luma,np.nan),0)
    weights*=~(has&((luma<.6*ref)|(luma>1.6*ref)))
    hybrid=info.get('sources')
    edit_v=np.array([bool(hybrid) and hybrid.get(v['name'],{}).get('kind')=='raw' for v in views])
    if hybrid and edit_v.any() and (~edit_v).any():
        # Hybrid: pull the structure views' (MV-Adapter) tone toward the edited views, measured where they overlap.
        ge,we,_,_=fuse(colors[edit_v],weights[edit_v]);gm,wm,_,_=fuse(colors[~edit_v],weights[~edit_v])
        both_g=(we>.05)&(wm>.05)
        if both_g.sum()>500:
            # One global per-channel gain: a spatial field interpolated from the left and right overlaps switched
            # sides at the back midline and drew a vertical seam.
            lr=np.clip(np.log(np.maximum(ge[both_g],1e-4))-np.log(np.maximum(gm[both_g],1e-4)),-1,1)
            field=np.broadcast_to(np.median(lr,0),(len(points),3))
            colors[~edit_v]=np.clip(colors[~edit_v]*np.exp(field)[None],0,1)
            if two_band:   # tone lives in the low band, which is fused from the separate blurred projection
                colors_low[~edit_v]=np.clip(colors_low[~edit_v]*np.exp(field)[None],0,1)
    gen,support,spread,count=fuse(colors,weights)
    if two_band:
        low,_,_,_=fuse(colors_low,weights);high,_,_,_=fuse(colors-colors_low,weights**4)
        # Grazing projections smear one pixel over many texels (radial streaks -> crown "star"): fade the
        # projected detail out with the best facing cosine and substitute world-space triplanar detail.
        best=np.sqrt(np.maximum(weights.max(0),0))
        synth=synthetic_detail(points,normals,work/backend)
        if synth is not None:
            a=np.clip((best-.25)/.35,0,1)[:,None]
            high=a*high+(1-a)*synth
        if hybrid and edit_v.any() and (~edit_v).any():
            # Structure views (MV-Adapter) carry little fine detail: where they dominate, take the high band from
            # the edited side view's scalp/neck instead (triplanar, so no UV seams).
            side=synthetic_detail(points,normals,work/backend,src=work/backend/'view_right.png',
                                  scalp_box=(.08,.32,.45,.72),neck_box=(.55,.72,.45,.62))
            if side is not None:
                share=np.clip(weights[edit_v].sum(0)/np.maximum(weights.sum(0),1e-12)/.5,0,1)[:,None]
                high=share*high+(1-share)*side
        gen=np.clip(low+high,0,1)
    gen,support=fill_unsupported(points,gen,support,seen)
    if unseen is not None and unseen.any():
        detail_src=work/backend/'back'/'edited.png'
        gen=fill_pole(points,normals,gen,support,seen,unseen,grazing_facing,
                      np.asarray(Image.open(detail_src).convert('RGB')) if detail_src.exists() else None)
    # Generators relight; match their per-channel level to photographed skin
    # over texels both observed and generated (robust median ratio).
    both=seen&(support>.05)
    gain=np.median(linear[both],0)/np.maximum(np.median(gen[both],0),1e-6) if both.sum()>500 else np.ones(3)
    gen=np.clip(gen*gain*local_gain(points,linear,gen*gain,both),0,1)
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
        basecolor_sha256=sha256(out/'skin_basecolor.png'),gain=gain.tolist(),delight=delight_report is not None,
        view_source=('raw edits' if use_raw else 'composites')+f', polar={polar_mode}'+(f', two-band sigma {band_sigma}' if two_band else ''),
        clothing_excluded_fraction=exclusion_report,
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


def neck_cut_weight(geometry, points, width=.015):
    """Fade generated colour near the lowest open boundary (the neck cut)."""
    from ..rig.common import boundary_edges
    p=geometry['captured'][0].astype(float);edges=boundary_edges(geometry['triangles'])
    ids=np.unique(edges);low=ids[p[ids,1]<p[:,1].min()+.25*np.ptp(p[:,1])]
    if not len(low):return np.ones(len(points))
    d,_=cKDTree(p[low]).query(points)
    return np.clip(d/width-1,0,1)


def local_gain(points, photo, gen, both, k=48, falloff=.03):
    """Per-channel ratio field matching photographed shading near the boundary.

    Log-ratios measured on overlap texels are averaged over k neighbours and
    fade to 1 with distance, so far unseen texels keep the delit albedo.
    """
    if both.sum()<k:return np.ones_like(gen)
    log=np.clip(np.log(np.maximum(photo[both],1e-4))-np.log(np.maximum(gen[both],1e-4)),-1,1)
    d,j=cKDTree(points[both]).query(points,k=k)
    w=1/(d+.002);field=(log[j]*w[...,None]).sum(1)/w.sum(1)[:,None]
    return np.exp(field*np.exp(-d[:,0]/falloff)[:,None])


def synthetic_detail(points, normals, gen_dir, *, period=.03, limit=.06, src=None,
                     scalp_box=(.10,.34,.38,.62), neck_box=(.62,.80,.38,.62)):
    """Triplanar high-band detail from an edited view (default: the back-view edit): scalp patch for upward
    normals, neck patch below. Boxes are (y0, y1, x0, x1) image fractions."""
    src=Path(src) if src else Path(gen_dir)/'back'/'edited.png'
    if not src.exists():return None
    from scipy.ndimage import gaussian_filter
    from ..mobile.baking import sample
    im=srgb_to_linear(np.asarray(Image.open(src).convert('RGB'))/255);h=im.shape[0]
    def residual(y0,y1,x0,x1):
        patch=im[int(y0*h):int(y1*h),int(x0*h):int(x1*h)]
        return np.clip(patch-gaussian_filter(patch,(4,4,0)),-limit,limit)
    scalp,neck=residual(*scalp_box),residual(*neck_box)
    tw=np.abs(normals)**4;tw/=np.maximum(tw.sum(1,keepdims=True),1e-12)
    out=np.zeros_like(points)
    for patch,mask in ((scalp,normals[:,1]>=0),(neck,normals[:,1]<0)):
        for axis,pair in enumerate(((1,2),(0,2),(0,1))):
            uv=points[mask][:,pair]/period;uv=1-np.abs(np.mod(uv,2)-1)
            out[mask]+=sample(patch,uv)*tw[mask][:,axis,None]
    return out


def fill_pole(points, normals, gen, support, seen, unseen, facing, detail_image, *, sigma=.01, full=.3):
    """Replace the grazing-angle star at the crown/chin poles with a smooth fill plus world-space detail.

    Low frequency: Gaussian (sigma metres) average of nearby well-seen generated texels. High frequency:
    stubble/skin residual from a scalp patch of the back-view edit, mapped triplanar (no UV seams, no
    directional pinch). Grazing projections keep weight where facing >= full; the fill takes over below.
    """
    good=(~unseen)&(~seen)&(support>.05)
    if good.sum()<64:return gen
    tree=cKDTree(points[good]);d,j=tree.query(points[unseen],k=32)
    w=np.exp(-.5*(d/sigma)**2)+1e-12;low=(gen[good][j]*w[...,None]).sum(1)/w.sum(1)[:,None]
    fill=low
    if detail_image is not None:
        from scipy.ndimage import gaussian_filter
        h=detail_image.shape[0];patch=srgb_to_linear(detail_image[int(.12*h):int(.38*h),int(.35*h):int(.65*h)]/255)
        residual=np.clip(patch-gaussian_filter(patch,(6,6,0)),-.06,.06)
        tw=np.abs(normals[unseen])**4;tw/=np.maximum(tw.sum(1,keepdims=True),1e-12)
        from ..mobile.baking import sample
        det=np.zeros_like(low)
        for axis,pair in enumerate(((1,2),(0,2),(0,1))):
            uv=points[unseen][:,pair]/.03;uv=1-np.abs(np.mod(uv,2)-1)
            det+=sample(residual,uv)*tw[:,axis,None]
        fill=np.clip(low+det,0,1)
    a=np.clip(facing[unseen]/full,0,1)[:,None]**2      # 0 at the pole -> synthesized fill
    gen=gen.copy();gen[unseen]=a*gen[unseen]+(1-a)*fill
    return gen


def fill_unsupported(points, gen, support, seen, threshold=.05):
    """Unseen texels no view saw take the nearest supported generated colour."""
    ok=support>threshold;need=(~ok)&(~seen)
    if ok.any() and need.any():
        _,j=cKDTree(points[ok]).query(points[need],k=8)
        gen=gen.copy();gen[need]=gen[ok][j].mean(1)
        support=support.copy();support[need]=threshold
    return gen,support


def seam_energy(points, valid_result, seen, radius=.004, neighbourhood=.008, samples=3000):
    """Low-pass colour jump across the observed boundary (linear RGB).

    Compares 8 mm neighbourhood means on the observed and the generated side
    of boundary points, so real fine texture does not dominate the score.
    """
    tree_un=cKDTree(points[~seen]);tree_seen=cKDTree(points[seen])
    d,_=tree_un.query(points[seen],distance_upper_bound=radius);boundary=np.flatnonzero(np.isfinite(d))
    if not len(boundary):return None
    boundary=boundary[np.linspace(0,len(boundary)-1,min(samples,len(boundary))).astype(int)]
    q=points[seen][boundary];jumps=[]
    for a,b in zip(tree_seen.query_ball_point(q,neighbourhood),tree_un.query_ball_point(q,neighbourhood)):
        if len(a)>=4 and len(b)>=4:
            jumps.append(np.abs(valid_result[seen][a].mean(0)-valid_result[~seen][b].mean(0)).mean())
    return float(np.mean(jumps)) if jumps else None


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
    p.add_argument('stage',choices=('prepare','generate','compose','regen-polar','bake','eval'))
    p.add_argument('--candidate');p.add_argument('--work',required=True)
    p.add_argument('--backend',default='mvadapter',help=f'{BACKENDS} for generate; any work subdir for bake');p.add_argument('--out',nargs='*')
    p.add_argument('--no-delight',action='store_true');p.add_argument('--source',choices=('auto','raw','composite'),default='auto');p.add_argument('--polar',choices=('auto','skip','composite','raw'),default='auto');p.add_argument('--one-band',action='store_true');p.add_argument('--spec',default=DEFAULT_HYBRID);p.add_argument('--steps',type=int);p.add_argument('--seed',type=int,default=317)
    a=p.parse_args()
    if a.stage=='prepare':print(json.dumps(prepare(a.candidate,a.work),indent=1)[:400])
    if a.stage=='generate':
        opts=dict(seed=a.seed);opts.update(steps=a.steps) if a.steps else None
        print(json.dumps(generate(a.work,a.backend,**opts),indent=1))
    if a.stage=='regen-polar':
        from . import mv_qwen
        from .qwen_edit_backend import make_editor
        record=check(a.work);frame,views=conditions(Path(record['candidate']),record['resolution'])
        print(json.dumps(mv_qwen.regen_polar(record['candidate'],frame,views,Path(a.work)/a.backend,make_editor(),
            **({'steps':a.steps} if a.steps else {})),indent=1))
    if a.stage=='compose':print(json.dumps(compose(a.work,a.spec,a.backend if a.backend.startswith('hybrid') else 'hybrid'),indent=1))
    if a.stage=='bake':print(json.dumps(bake(a.work,a.backend,a.out[0],delight=not a.no_delight,source=a.source,polar=a.polar,two_band=not a.one_band),indent=1))
    if a.stage=='eval':print(json.dumps(evaluate(a.work,a.out or []),indent=1))


if __name__=='__main__':main()
