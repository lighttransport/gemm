"""Qwen-Image-2.1 multiview completion modes for mv_texture (research licence).

sequential: a clean-room port of the CAP4D *method* (no CAP4D code or weights,
  which are CC BY-NC / FLAME-derived). Views are generated one at a time; each
  edit is conditioned on GNM geometry (normal map) plus the surface state after
  all previous views have been projected and fused, so later views inherit the
  colours already committed to the shared atlas instead of re-inventing them.
grid: the four horizontal views tiled 2x2 and edited in one 1024px call, so the
  generator sees all of them jointly; top and bottom keep the projected state.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from . import generated_skin as skin
from . import mv_conditioning as cond
from .reference import srgb_to_linear

ORDER=('front','right','left','back','top','bottom')
PROMPT=('Photorealistic texture render of this exact bald human head on a plain gray background. '
    'Fill the flat untextured regions with realistic skin matching the identity portrait: same skin tone, '
    'natural pores, ears with correct anatomy, scalp with very short dark hair stubble where hair grows. '
    'Soft even diffuse light, no shadows, no highlights. Preserve the exact silhouette, pose and existing '
    'facial details. No glasses, hats, jewelry, text or extra features.')

EDIT_PROMPT=('Picture 1 is a photo of a person. The LAST picture is a 3D render of the same bald head, partly '
    'untextured (flat uniform beige). Edit only the last picture: replace the flat beige areas with realistic skin of '
    'the person in Picture 1, matching the skin tone of the already-textured face, natural ear anatomy, and very short '
    'dark hair stubble on the scalp where hair grows. Do not move, zoom, crop or change the camera; keep the silhouette, '
    'all textured regions and the gray background identical. Even diffuse light, no shadows, no highlights.')

def _mask(view):
    return np.uint8((view['valid']&(view['known']<.5))*255)


def _sheet(target, portrait, previous, normal):
    sheet=Image.new('RGB',(1024,1024),(127,127,127))
    for im,pos in zip((target,portrait,previous,normal),((0,0),(512,0),(0,512),(512,512))):
        sheet.paste(Image.fromarray(im).resize((512,512),Image.Resampling.LANCZOS),pos)
    return sheet


def _commit(atlas, known, geometry, frame, view, image, valid, points, normals):
    """Project one generated view onto unknown texels; mark them known."""
    colors,weights=cond.project_views(frame,[view],[image],points,normals)
    take=(weights[0]>.05)&(known[valid]<.5)
    flat=atlas[valid];flat[take]=colors[0][take];atlas[valid]=flat
    k=known[valid];k[take]=1;known[valid]=k


def sequential(candidate, frame, views, out, *, steps=12, seed=317, strength=.9, editor=None):
    """editor: optional qwen_edit_backend.Editor (Apache-2.0 Edit-2511) instead of Qwen-Image-2.1."""
    candidate,out=Path(candidate),Path(out)
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:geometry=dict(z)
    base=np.asarray(Image.open(candidate/'skin_basecolor.png').convert('RGB'))
    atlas=srgb_to_linear(base/255)
    known=(np.asarray(Image.open(candidate/'skin_coverage.png'))>0).astype(float)
    valid,points,normals=skin.atlas_surface(geometry,len(base))
    portrait=np.asarray(Image.open(candidate/'portrait.png').convert('RGB'))
    res=views[0]['depth'].shape[0];by_name={v['name']:v for v in views};results={};seconds=0.;previous=None
    for index,name in enumerate(ORDER):
        _,current=cond.render_conditions(geometry,atlas,known,res)
        view=next(v for v in current if v['name']==name);view['camera']=by_name[name]['camera']
        mask=_mask(view)
        if (mask>0).sum()<.02*view['valid'].sum():
            results[name]=view['rgb'];previous=view['rgb'];continue
        folder=out/name;folder.mkdir(parents=True,exist_ok=True)
        small=lambda a,mode=Image.Resampling.LANCZOS:Image.fromarray(a).resize((512,512),mode)
        small(view['rgb']).save(folder/'input.png');small(mask,Image.Resampling.NEAREST).save(folder/'mask.png')
        _sheet(view['rgb'],portrait,previous if previous is not None else view['rgb'],
               np.uint8(view['normal']*255+.5)).save(folder/'reference.png')
        if editor is None:
            receipt=skin.qwen_generate(folder/'edited.png',PROMPT+' The reference shows: upper left the target view, '
                'upper right the portrait, lower left the previous view, lower right the surface normals.',
                reference=folder/'reference.png',init_image=folder/'input.png',mask=folder/'mask.png',
                steps=steps,seed=seed+index,strength=strength)
            seconds+=receipt['seconds']
        else:
            big=np.asarray(Image.fromarray(view['rgb']).resize((1024,1024),Image.Resampling.LANCZOS))
            inputs=[portrait]+([np.asarray(Image.fromarray(previous).resize((1024,1024)))] if previous is not None else [])+[big]
            edited,took=editor(inputs,EDIT_PROMPT+(' Picture 2 is the previously completed neighbouring view; keep skin '
                'tone and texture consistent with it.' if len(inputs)==3 else ''),steps=steps,seed=seed+index)
            Image.fromarray(edited).save(folder/'edited.png');seconds+=took
        image=np.asarray(Image.open(folder/'edited.png').convert('RGB').resize((res,res),Image.Resampling.LANCZOS))
        # Keep known pixels exact at full resolution; only masked pixels are new.
        m=mask>0;image=np.where(m[...,None],image,view['rgb'])
        results[name]=image;previous=image
        _commit(atlas,known,geometry,frame,by_name[name],image,valid,points,normals)
    if editor is not None:
        from .qwen_edit_backend import GENERATOR,LICENSE
        return [results[v['name']] for v in views],dict(generator=GENERATOR+' sequential (CAP4D-style)',
            license=LICENSE,steps=steps,seed=seed,order=list(ORDER),seconds=seconds)
    return [results[v['name']] for v in views],dict(generator='Qwen-Image-2.1 sequential (CAP4D-style)',
        license='qwen-research',steps=steps,seed=seed,strength=strength,order=list(ORDER),seconds=seconds)


def grid(candidate, frame, views, out, *, steps=12, seed=317, strength=.9):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    names=('front','right','back','left');tiles=[next(v for v in views if v['name']==n) for n in names]
    sheet=np.zeros((1024,1024,3),np.uint8);mask=np.zeros((1024,1024),np.uint8)
    for i,v in enumerate(tiles):
        y,x=divmod(i,2)
        sheet[y*512:(y+1)*512,x*512:(x+1)*512]=np.asarray(Image.fromarray(v['rgb']).resize((512,512),Image.Resampling.LANCZOS))
        mask[y*512:(y+1)*512,x*512:(x+1)*512]=np.asarray(Image.fromarray(_mask(v)).resize((512,512),Image.Resampling.NEAREST))
    Image.fromarray(sheet).save(out/'grid_input.png');Image.fromarray(mask).save(out/'grid_mask.png')
    receipt=skin.qwen_generate(out/'grid_edited.png',PROMPT+' The image is a 2x2 sheet of the same head: front, '
        'right side, back, left side. Keep all four views consistent with each other.',
        reference=out/'grid_input.png',init_image=out/'grid_input.png',mask=out/'grid_mask.png',
        steps=steps,seed=seed,strength=strength,resolution=1024)
    edited=np.asarray(Image.open(out/'grid_edited.png').convert('RGB'))
    res=views[0]['depth'].shape[0];images={}
    for i,v in enumerate(tiles):
        y,x=divmod(i,2)
        tile=np.asarray(Image.fromarray(edited[y*512:(y+1)*512,x*512:(x+1)*512]).resize((res,res),Image.Resampling.LANCZOS))
        m=_mask(v)>0;images[v['name']]=np.where(m[...,None],tile,v['rgb'])
    return [images.get(v['name'],v['rgb']) for v in views],dict(generator='Qwen-Image-2.1 2x2 grid',
        license='qwen-research',steps=steps,seed=seed,strength=strength,seconds=receipt['seconds'])
