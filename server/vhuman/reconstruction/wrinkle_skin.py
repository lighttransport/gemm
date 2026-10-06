"""Qwen-edited skin material baked in mesh coordinates without invented anatomy.

Whole-view diffusion may copy a flat render or hallucinate another ear. Edit an
anatomy-free material instead, and attach its detail to the captured GNM surface.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from . import generated_skin as skin
from .multiview_skin import crop_camera, detail_quality
from .observations import sha256

PROMPT=('Change this smooth skin material into mature neck skin with clearly visible thin creases and intersecting fine wrinkles. '
    'Macro photograph of a continuous medium-brown human skin texture. Several gently curving horizontal wrinkles and finer intersecting '
    'lines run across the entire image, with natural pores between the wrinkles. The square is completely filled by one flat skin '
    'surface texture. No body shape, no ear, no face, no hair, no objects. Even diffuse light, no cast shadows.')
NEGATIVE='smooth plastic, flat uniform color, airbrushed skin, ear, face, eye, nose, mouth, hair, objects, text, grid, deep scars'


def generate(work, prior, *, steps=10, seed=412):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    Image.open(prior).convert('RGB').resize((384,384),Image.Resampling.LANCZOS).save(work/'input.png')
    Image.new('L',(384,384),255).save(work/'mask.png')
    return skin.qwen_generate(work/'edited.png',PROMPT,reference=work/'input.png',init_image=work/'input.png',
        mask=work/'mask.png',steps=steps,strength=1.,cfg=4.,negative_prompt=NEGATIVE,resolution=384,seed=seed)


def material_field(points, normals, patch, period=.1):
    """Continuous world-space sampling: side/front folds follow head-up Y.

    Mirror addressing is continuous across repeats and UV chart cuts. The
    material adds fine/mid-scale luminance only, preserving the base complexion.
    """
    if not .02<=period<=.3:raise ValueError('material period must be 20..300 mm')
    band=skin.band_detail(patch,limit=.08,broad_sigma=24.)@np.array([.2126,.7152,.0722])
    weights=abs(normals)**4;weights/=np.maximum(weights.sum(1,keepdims=True),1e-12)
    field=np.zeros(len(points))
    for axis,pair in enumerate(((2,1),(0,2),(0,1))):
        uv=1-abs(np.mod(points[:,pair]/period,2)-1)
        field+=skin.sample(band,uv)*weights[:,axis]
    # Neck/jaw/ear creases taper off toward the upper scalp. This is a spatial
    # material prior, not a fitted anatomical wrinkle model.
    height=np.clip((.055-points[:,1])/.045,0,1)
    return field*height*height*(3-2*height)


def bake(candidate, work, out, *, period=.1):
    candidate,work,out=map(Path,(candidate,work,out));manifest=skin.validate_candidate(candidate)
    if out.exists() and any(out.iterdir()):raise ValueError('candidate output must be empty')
    receipt=json.loads((work/'edited.json').read_text());request=receipt['request']
    if (sha256(work/'edited.png')!=receipt['sha256'] or request['reference_sha256']!=sha256(work/'input.png')
            or request['init_image_sha256']!=sha256(work/'input.png') or request['mask_sha256']!=sha256(work/'mask.png')):
        raise ValueError('wrinkle material generation receipt mismatch')
    original=np.asarray(Image.open(work/'input.png').convert('RGB'))
    edited=np.asarray(Image.open(work/'edited.png').convert('RGB'))
    quality=detail_quality(original,edited,np.ones(original.shape[:2],bool))
    skin.write_json(work/'material_quality.json',quality)
    if not quality['passed']:raise ValueError('flat wrinkle material rejected: '+str(quality))
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:geometry=dict(z)
    base=np.asarray(Image.open(candidate/'skin_basecolor.png').convert('RGB'))
    observed=np.asarray(Image.open(candidate/'skin_coverage.png'))>0
    valid,points,normals=skin.atlas_surface(geometry,len(base))
    blend=skin.unseen_weight(points,observed[valid])
    scalar=material_field(points,normals,skin.srgb_to_linear(edited/255),period)
    linear=skin.srgb_to_linear(base[valid]/255)
    luminance=linear@np.array([.2126,.7152,.0722])
    delta=np.clip(scalar[:,None]*linear/np.maximum(luminance[:,None],.03),-.08,.08)*blend[:,None]
    result=skin.apply_detail(base,valid,delta,observed,limit=.08)
    out.mkdir(parents=True,exist_ok=True)
    for path in candidate.iterdir():
        if path.is_file():shutil.copyfile(path,out/path.name)
    distance,near=distance_transform_edt(~valid,return_indices=True)
    gutter=(~valid)&(distance<=4)&~observed
    result[gutter]=result[near[0][gutter],near[1][gutter]]
    Image.fromarray(result).save(out/'skin_basecolor.png')
    support=np.zeros(valid.shape);support[valid]=.1*blend*(abs(scalar)>1e-5)
    Image.fromarray(np.uint8(support*255+.5)).save(out/'skin_generated_support.png')
    report=dict(schema='vhuman.synthetic_skin_completion.v1',work=str(work.resolve()),synthetic=True,
        license='qwen-research',generator='Qwen-Image-2.1 material edit + mesh-space wrinkle transfer',
        basecolor_sha256=sha256(out/'skin_basecolor.png'),source_basecolor_sha256=sha256(candidate/'skin_basecolor.png'),
        source_geometry_sha256=manifest['geometry_sha256'],material_edit=receipt,material_quality=quality,
        previous_completion=manifest['material'].get('synthetic_completion'),period_m=period,
        max_linear_detail=float(abs(delta).max()),photographed_texels_changed=int(np.any(result[observed]!=base[observed],axis=-1).sum()),
        generated_texels=int((valid&np.any(result!=base,axis=-1)).sum()),
        limitations=['synthetic wrinkle color detail, not recovered wrinkle depth or anatomy',
            'one material field shared across views, not independent multiview photographic evidence',
            'generated texture stays outside measured confidence and permissive training'])
    manifest['material']['synthetic_completion']=report
    manifest['material_refinement']=dict(source=str(candidate.resolve()),geometry_unchanged=True)
    skin.write_json(out/'generated_skin.json',report);skin.write_json(out/'manifest.json',manifest)
    skin.write_json(out/'skin_material.json',manifest['material']);skin.validate_candidate(out)
    return report


def review(candidate, work, completed):
    candidate,work,completed=map(Path,(candidate,work,completed))
    manifest=skin.validate_candidate(completed)
    source=skin.validate_candidate(candidate)
    report=json.loads((completed/'generated_skin.json').read_text())
    if (source['geometry_sha256']!=report['source_geometry_sha256']
            or manifest['geometry_sha256']!=report['source_geometry_sha256']
            or sha256(candidate/'skin_basecolor.png')!=report['source_basecolor_sha256']
            or sha256(completed/'skin_basecolor.png')!=report['basecolor_sha256']):
        raise ValueError('review source changed')
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:geometry=dict(z)
    before=skin.srgb_to_linear(np.asarray(Image.open(candidate/'skin_basecolor.png').convert('RGB'))/255)
    after=skin.srgb_to_linear(np.asarray(Image.open(completed/'skin_basecolor.png').convert('RGB'))/255)
    observed=np.asarray(Image.open(candidate/'skin_coverage.png'))>0
    valid,points,_=skin.atlas_surface(geometry,len(before));blend=np.zeros(valid.shape)
    blend[valid]=skin.unseen_weight(points,observed[valid])
    p=geometry['captured'][0];head=p[p[:,1]>p[:,1].min()+.35*np.ptp(p[:,1])]
    rows=[];files=[];quality={}
    for name,yaw,pitch,box in [('ear_left',-85,10,(90,190,410,510)),('ear_right',85,10,(102,190,422,510)),
                               ('front',0,0,None),('rear',180,20,None)]:
        camera=skin.orbit_camera(head,yaw,pitch=pitch)
        if box:camera=crop_camera(camera,box)
        a,mask,_=skin.render_plate(geometry,before,blend,camera)
        b,_,_=skin.render_plate(geometry,after,blend,camera)
        if box:quality[name]=detail_quality(a,b,mask)
        canvas=Image.new('RGB',(1024,546),(25,27,30));draw=ImageDraw.Draw(canvas)
        for x,label,image in ((0,'Before',a),(512,'Baked wrinkle detail',b)):
            canvas.paste(Image.fromarray(image),(x,34));draw.text((x+12,10),name+' / '+label,fill='white')
        name=name+'.png';canvas.save(work/name);files.append(name)
        rows.append(f'<figure><a href="{name}"><img src="{name}" alt="Before and baked wrinkle detail"></a></figure>')
    skin.write_json(work/'baked_quality.json',quality)
    if not all(q['passed'] for q in quality.values()):raise ValueError('bake lost resolved wrinkle detail: '+str(quality))
    (work/'review.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Wrinkle material completion</title><style>body{background:#191b1e;color:#eee;font:16px system-ui;margin:24px}'
        'img{width:100%}figure{margin:24px 0}.patch{width:45%;max-width:384px}</style><h1>Wrinkle material completion</h1>'
        '<p>Qwen edits an anatomy-free skin material. Its wrinkle detail is attached to the fitted mesh in world coordinates. '
        'Photographed texels and geometry remain unchanged. These are synthetic color details, not measured wrinkle depth.</p>'
        '<h2>Actual Qwen material edit: before / after</h2><img class="patch" src="input.png"><img class="patch" src="edited.png">'
        '<h2>Final atlas on the fitted mesh</h2>'+''.join(rows))
    files+=['review.html','input.png','edited.png']
    skin.write_json(work/'review.json',dict(schema='vhuman.multiview_skin_review.v1',
        geometry_sha256=manifest['geometry_sha256'],basecolor_sha256=sha256(completed/'skin_basecolor.png'),
        files={name:sha256(work/name) for name in files}))
    return quality


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('generate','bake','review','all'))
    parser.add_argument('--candidate',required=True);parser.add_argument('--work',required=True)
    parser.add_argument('--out');parser.add_argument('--prior')
    parser.add_argument('--steps',type=int,default=10);parser.add_argument('--seed',type=int,default=412)
    parser.add_argument('--period',type=float,default=.1)
    args=parser.parse_args()
    if args.stage in ('generate','all') and not args.prior:parser.error('--prior required')
    if args.stage in ('bake','review','all') and not args.out:parser.error('--out required')
    if args.stage in ('generate','all'):generate(args.work,args.prior,steps=args.steps,seed=args.seed)
    if args.stage in ('bake','all'):print(json.dumps(bake(args.candidate,args.work,args.out,period=args.period),indent=2))
    if args.stage in ('review','all'):print(json.dumps(review(args.candidate,args.work,args.out),indent=2))


if __name__=='__main__':main()
