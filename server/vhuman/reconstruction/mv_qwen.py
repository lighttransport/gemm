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
import json
import time

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
    'dark hair stubble on the scalp where hair grows. The neck and shoulders are bare skin: the person wears no '
    'clothing in this render (no suit, shirt, collar or tie). Do not move, zoom, crop or change the camera; keep the '
    'silhouette, all textured regions and the plain gray background identical. Even diffuse light, no shadows, no '
    'highlights.')

SEQ_NEGATIVE=('blurry, hat, glasses, text, extra ears, shadows, highlights, clothing, suit, jacket, shirt, collar, '
              'tie, background, scenery, window, room, long hair, hair on the neck')


def matted_portrait(candidate):
    """Portrait reference without its photo context: BiRefNet foreground (keeps hair), cut just below the chin
    (face mask) to drop clothing, on 50% gray. The raw photo's background and shirt leaked into edits."""
    candidate=Path(candidate)
    img=Image.open(candidate/'portrait.png').convert('RGB');rgb=np.asarray(img,float)
    alpha=None
    try:
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
        net=AutoModelForImageSegmentation.from_pretrained('/mnt/disk01/models/BiRefNet',trust_remote_code=True).eval().float()
        x=transforms.Compose([transforms.Resize((1024,1024)),transforms.ToTensor(),
                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])(img)[None]
        with torch.no_grad():
            pred=net(x)[-1].sigmoid()[0,0].numpy()
        alpha=np.asarray(Image.fromarray(np.uint8(pred*255)).resize(img.size),float)/255
        del net
    except Exception as error:   # fall back to the face mask (drops hair) rather than failing the run
        print(f'[mv_qwen] BiRefNet matte unavailable ({error}); using face silhouette',flush=True)
    face=candidate/'parsing_0_silhouette.png'
    if face.exists():
        f=np.asarray(Image.open(face).convert('L').resize(img.size),float)/255
        if alpha is None:alpha=f
        ys,xs=np.where(f>.5)
        if len(ys):   # below mid-face keep only the (dilated) face: drops collar, tie and shirt beside the jaw
            from scipy.ndimage import binary_dilation
            mid=(ys.min()+ys.max())//2
            near=binary_dilation(f>.5,iterations=max(2,int(.02*rgb.shape[0])))
            alpha[mid:]*=near[mid:]
    if alpha is None:return np.uint8(rgb)
    out=rgb*alpha[...,None]+127.5*(1-alpha[...,None])
    ys,xs=np.where(alpha>.5)
    if len(ys):
        pad=int(.05*max(rgb.shape[:2]))
        out=out[max(ys.min()-pad,0):ys.max()+pad,max(xs.min()-pad,0):xs.max()+pad]
    return np.uint8(np.clip(out+.5,0,255))


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
    portrait=matted_portrait(candidate)
    res=views[0]['depth'].shape[0];by_name={v['name']:v for v in views};results={};seconds=0.;previous=None;previous_ref=None
    for index,name in enumerate(ORDER):
        t_render=time.time()
        _,current=cond.render_conditions(geometry,atlas,known,res,only=(name,))   # only the view being edited
        view=current[0];view['camera']=by_name[name]['camera']
        t_render=time.time()-t_render
        mask=_mask(view)
        if (mask>0).sum()<.02*view['valid'].sum():
            results[name]=view['rgb'];previous=previous_ref=view['rgb'];continue
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
            inputs=[portrait]+([np.asarray(Image.fromarray(previous_ref).resize((1024,1024)))] if previous is not None else [])+[big]
            edited,took=editor(inputs,EDIT_PROMPT+(' Picture 2 is the previously completed neighbouring view; keep skin '
                'tone and texture consistent with it.' if len(inputs)==3 else ''),steps=steps,seed=seed+index,
                negative=SEQ_NEGATIVE)
            Image.fromarray(edited).save(folder/'edited.png');seconds+=took
        image=np.asarray(Image.open(folder/'edited.png').convert('RGB').resize((res,res),Image.Resampling.LANCZOS))
        # Keep known pixels exact at full resolution; only masked pixels are new.
        m=mask>0;image=np.where(m[...,None],image,view['rgb'])
        results[name]=image;previous=image
        # The next view's 'previous' reference: this view inside the mesh silhouette only, on gray, so clothing or
        # scenery the editor painted outside/around the head cannot propagate along the chain.
        sil=by_name[name]['valid'][...,None]
        previous_ref=np.uint8(np.where(sil,image,127))
        t_commit=time.time()
        _commit(atlas,known,geometry,frame,by_name[name],image,valid,points,normals)
        print(f'[mv_qwen] {name}: render {t_render:.1f}s edit {took if editor is not None else receipt["seconds"]:.1f}s '
              f'commit {time.time()-t_commit:.1f}s',flush=True)
    if editor is not None:
        from .qwen_edit_backend import LICENSE
        return [results[v['name']] for v in views],dict(generator=editor.generator+' sequential (CAP4D-style)',
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


POLAR_PROMPT=('The LAST picture is a 3D render of the {where} of a bald human head, partly untextured (flat uniform '
    'beige). The other pictures are already-completed views of the same head. Edit only the last picture: fill the '
    'flat beige areas with {what}, matching the skin tone and texture of the other pictures. There is NO face in this '
    'view: no eyes, nose, mouth or lips. Do not move, zoom, crop or change the camera; keep the silhouette and gray '
    'background identical. Even diffuse light, no shadows, no highlights.')
POLAR_WHAT={'top':('top of the scalp seen from directly above','very short dark hair stubble over the whole scalp'),
            'bottom':('underside (chin, jaw and neck) seen from directly below','plain neck and under-chin skin')}


def regen_polar(candidate, frame, views, out, editor, *, steps=12, seed=917):
    """Re-edit top/bottom without the portrait (portrait-conditioned edits put faces on the crown/chin).

    References: the finished front and back raw edits; target render last (1024^2 alignment rule).
    Writes <view>/edited.png and polar_regen.json so the bake may fuse these raw outputs too.
    """
    out=Path(out);res=views[0]['depth'].shape[0];by_name={v['name']:v for v in views};report={}
    refs=[np.asarray(Image.open(out/n/'edited.png').convert('RGB').resize((1024,1024))) for n in ('front','back')]
    for index,name in enumerate(('top','bottom')):
        view=by_name[name]
        big=np.asarray(Image.fromarray(np.asarray(Image.open(out/f'view_{name}.png').convert('RGB'))).resize((1024,1024),Image.Resampling.LANCZOS))
        where,what=POLAR_WHAT[name]
        edited,took=editor(refs+[big],POLAR_PROMPT.format(where=where,what=what),steps=steps,seed=seed+index,
                           negative='face, eyes, nose, mouth, lips, teeth, ears on top, text, blurry, shadows, highlights')
        Image.fromarray(edited).save(out/name/'edited.png');report[name]=dict(seconds=took)
        print(f'[mv_qwen] regen {name}: {took:.1f}s',flush=True)
    (out/'polar_regen.json').write_text(json.dumps(dict(prompt=POLAR_PROMPT,references=['front','back'],steps=steps,
        seed=seed,views=report,generator=editor.generator),indent=2))
    return report
