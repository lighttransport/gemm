"""Mesh-guided Qwen contact-sheet edits for ears, rear scalp and crown.

The native HIP backend accepts one image: a contact sheet provides multiple
calibrated views, while a separate full-resolution init image fixes the target.
Generated colors remain research-only appearance priors, never observations.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import gaussian_filter, binary_erosion

from . import generated_skin as skin
from .observations import sha256

VIEWS=(('ear_left',-85,10),('ear_right',85,10),
       ('rear_left',-140,25),('rear_right',140,25),
       ('crown_front',0,75),('crown_rear',180,65))


def crop_camera(camera, box, resolution=512):
    x0,y0,x1,y1=box
    if x1<=x0 or y1-y0!=x1-x0:raise ValueError('positive square crop required')
    scale=resolution/(x1-x0)
    return skin.Camera(camera.focal*scale,(camera.cx-x0)*scale,(camera.cy-y0)*scale,
                       camera.origin,camera.rotation,
                       camera.focal_y*scale if camera.focal_y else None,camera.skew*scale)


def detail_quality(original, edited, mask):
    """Reject flat/no-op edits using mid-scale contrast inside initially flat skin.

    This is a contrast gate, not evidence of anatomical wrinkle accuracy.
    Changed pixel counts alone can be satisfied by noise or tiny color shifts.
    """
    original,edited=np.asarray(original),np.asarray(edited)
    if original.shape!=edited.shape or original.ndim!=3 or original.shape[2]!=3:
        raise ValueError('matching RGB images required')
    bands=[]
    for rgb in (original,edited):
        gray=skin.srgb_to_linear(rgb.astype(float)/255)@np.array([.2126,.7152,.0722])
        bands.append(gaussian_filter(gray,1)-gaussian_filter(gray,12))
    interior=binary_erosion(np.asarray(mask)>0,iterations=12)
    flat=interior&(abs(bands[0])<.005)
    if flat.sum()<256:raise ValueError('insufficient flat interior skin to evaluate completion')
    before,after=(float(np.sqrt(np.mean(b[flat]**2))) for b in bands)
    residual=float(np.sqrt(np.mean((bands[1][flat]-bands[0][flat])**2)))
    return dict(flat_pixels=int(flat.sum()),before_rms=before,after_rms=after,residual_rms=residual,
                passed=bool(after>=.006 and residual>=.005 and after>=before*1.5),
                proves_wrinkle_accuracy=False)


def verify_inputs(work, record):
    work=Path(work).resolve()
    for name,digest in record['inputs'].items():
        path=(work/name).resolve()
        if not path.is_relative_to(work) or sha256(path)!=digest:
            raise ValueError('multiview input checksum mismatch: '+name)


def contact_sheet(target, portrait, neighbor, front):
    """One condition image, with the target always at upper left."""
    result=Image.new('RGB',(1024,1024),(118,118,118))
    for image,position in zip((target,portrait,neighbor,front),((0,0),(512,0),(0,512),(512,512))):
        result.paste(ImageOps.contain(image.convert('RGB'),(512,512),Image.Resampling.LANCZOS),position)
    return result


def prepare(candidate, work, *, prior_work=None, steps=20, seed=317):
    candidate,work=Path(candidate).resolve(),Path(work).resolve()
    if (work/'completion.json').exists():
        record=json.loads((work/'completion.json').read_text())
        if (record.get('method')!='gnm_multiview_contact_sheet' or record['candidate']!=str(candidate)
                or record['steps']!=steps or record['seed']!=seed
                or record['basecolor_sha256']!=sha256(candidate/'skin_basecolor.png')
                or record['geometry_sha256']!=skin.validate_candidate(candidate)['geometry_sha256']):
            raise ValueError('workspace belongs to another completion request')
        verify_inputs(work,record)
        return record
    if work.exists() and any(work.iterdir()):raise ValueError('prepare requires an empty workspace')
    work.mkdir(parents=True,exist_ok=True)
    if prior_work:
        # qwen_generate checks the copied receipt against the exact prompt,
        # seed, steps, model and output checksum before accepting the cache.
        for name in ('skin_prior.png','skin_prior.json'):
            shutil.copyfile(Path(prior_work)/name,work/name)
    record=skin.prepare(candidate,work,steps=steps,seed=seed)
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:geometry=dict(z)
    with np.load(work/'surface.npz',allow_pickle=False) as z:
        seeded,blend=z['seeded'],z['blend']
    points=geometry['captured'][0]
    # Crop the shoulder/low neck from the camera fit, retaining the exact mesh.
    head=points[points[:,1]>points[:,1].min()+.35*np.ptp(points[:,1])]
    front,_,_=skin.render_plate(geometry,seeded,blend,skin.orbit_camera(head,0))
    Image.fromarray(front).save(work/'front.png')
    shutil.copyfile(candidate/'portrait.png',work/'portrait.png')
    views=[]
    for name,yaw,pitch in VIEWS:
        folder=work/name;folder.mkdir(exist_ok=True)
        camera=skin.orbit_camera(head,yaw,pitch=pitch)
        rgb,mask,depth=skin.render_plate(geometry,seeded,blend,camera)
        Image.fromarray(rgb).save(folder/'input.png');Image.fromarray(mask).save(folder/'mask.png')
        np.save(folder/'depth.npy',depth)
        views.append(dict(name=name,yaw=yaw,pitch=pitch,camera=camera.as_dict(),mask_pixels=int((mask>0).sum())))
    for index,view in enumerate(views):
        # Side and rear views share a same-side guide; crown views share each other.
        neighbor=views[{0:2,1:3,2:0,3:1,4:5,5:4}[index]]
        folder=work/view['name']
        with Image.open(folder/'input.png') as target,Image.open(work/'portrait.png') as portrait,Image.open(work/neighbor['name']/'input.png') as other:
            contact_sheet(target,portrait,other,Image.fromarray(front)).save(folder/'reference.png')
        view['neighbor']=neighbor['name']
    record.update(method='gnm_multiview_contact_sheet',consistency='multiview',views=views,
                  reference_layout=['target','source portrait','adjacent mesh view','front mesh view'],
                  native_reference_images=1,geometry_source='fitted captured GNM mesh; camera orbit in mesh coordinates')
    inputs=['surface.npz','skin_prior.png','skin_prior.json','front.png','portrait.png']
    inputs += [f"{v['name']}/{name}" for v in views for name in ('input.png','mask.png','depth.npy','reference.png')]
    record['inputs']={name:sha256(work/name) for name in inputs}
    skin.write_json(work/'completion.json',record)
    return record


def edit(work, *, steps=12, seed=317):
    work=Path(work);record=json.loads((work/'completion.json').read_text());verify_inputs(work,record)
    for index,view in enumerate(record['views']):
        folder=work/view['name']
        prompt=('The reference is a four-panel guide to the same person: upper left is the target camera, '
            'upper right the identity portrait, lower left an adjacent mesh view, lower right the front mesh view. '
            'Output ONE image matching the upper-left target exactly, not a collage. Refine only its smooth '
            'untextured skin with subtle natural pores and fine creases, matching the portrait skin tone. '
            'Preserve the exact head pose, ear shape, scalp contour and silhouette. Soft even diffuse light; '
            'no added shadows, highlights, hair, clothing, jewelry, text or facial features on the scalp. '
            'Keep existing facial details and gray background unchanged.')
        print('Multiview Qwen edit: '+view['name'],flush=True)
        skin.qwen_generate(folder/'edited.png',prompt,reference=folder/'reference.png',
                           init_image=folder/'input.png',mask=folder/'mask.png',steps=steps,seed=seed+index+1)
        original=np.asarray(Image.open(folder/'input.png').convert('RGB'))
        edited=np.asarray(Image.open(folder/'edited.png').convert('RGB'))
        mask=np.asarray(Image.open(folder/'mask.png'))>0
        if original.shape!=edited.shape or not np.array_equal(original[~mask],edited[~mask]):
            raise ValueError('multiview editor altered protected pixels')


def fuse_views(colors, weights, tolerance=.012):
    """Reject conflicting projected residuals before blending overlapping views.

    Single-view texels get lower support; agreeing synthesized views are still
    not photographic evidence. Arrays are [views, surface texels, (RGB)].
    """
    colors,weights=np.asarray(colors,float),np.asarray(weights,float)
    if colors.shape!=weights.shape+(3,) or colors.ndim!=3 or not np.isfinite(colors).all() or not np.isfinite(weights).all() or (weights<0).any():
        raise ValueError('finite multiview colors and nonnegative weights required')
    # Weighted medoid: the supported view closest to all other supported views.
    # Unlike a componentwise median this cannot invent a new RGB direction.
    cost=np.zeros_like(weights)
    for i in range(len(colors)):
        for j in range(len(colors)):
            cost[i]+=np.max(abs(colors[i]-colors[j]),axis=-1)*weights[j]
    cost=np.where(weights>.001,cost,np.inf)
    centre=colors[np.argmin(cost,axis=0),np.arange(colors.shape[1])]
    eligible=weights>.001;agree=(np.max(abs(colors-centre),axis=-1)<=tolerance)&eligible
    accepted=weights*agree
    views=eligible.sum(0);votes=agree.sum(0)
    # If two views disagree, neither is corroborated: reject both.
    accepted[:,(views>=2)&(votes<2)]=0
    accepted[:,views==1]*=.35
    support=accepted.sum(0)
    result=(colors*accepted[...,None]).sum(0)/np.maximum(support[:,None],1e-12)
    # The final support map adds a .05 prior and caps at .25. Keep a singly
    # visible surface below half that cap even when it faces its camera head-on.
    support=np.where(views==1,np.minimum(support,.075),support)
    return result,support,dict(overlap_texels=int((views>=2).sum()),
        agreeing_overlap_texels=int(((views>=2)&(votes>=2)).sum()),
        rejected_overlap_texels=int(((views>=2)&(votes<2)).sum()),
        single_view_texels=int((views==1).sum()),tolerance_linear_rgb=tolerance)


def review(candidate, work, completed):
    """Render the final atlas through the same cameras, not just edited images."""
    candidate,work,completed=map(Path,(candidate,work,completed))
    record=json.loads((work/'completion.json').read_text());verify_inputs(work,record)
    manifest=skin.validate_candidate(completed)
    if manifest['geometry_sha256']!=record['geometry_sha256']:raise ValueError('review geometry mismatch')
    if sha256(candidate/'skin_basecolor.png')!=record['basecolor_sha256']:raise ValueError('review source material mismatch')
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:geometry=dict(z)
    before=skin.srgb_to_linear(np.asarray(Image.open(candidate/'skin_basecolor.png').convert('RGB'))/255)
    after=skin.srgb_to_linear(np.asarray(Image.open(completed/'skin_basecolor.png').convert('RGB'))/255)
    rows=[]
    for view in record['views']:
        camera=skin.Camera.from_dict(view['camera'])
        original,_,_=skin.render_plate(geometry,before,np.zeros(before.shape[:2]),camera)
        baked,_,_=skin.render_plate(geometry,after,np.zeros(after.shape[:2]),camera)
        edited=Image.open(work/view['name']/'edited.png').convert('RGB')
        canvas=Image.new('RGB',(1536,546),(25,27,30));draw=ImageDraw.Draw(canvas)
        for x,label,im in ((0,'Before',Image.fromarray(original)),(512,'Qwen masked edit',edited),(1024,'Baked atlas',Image.fromarray(baked))):
            canvas.paste(im,(x,34));draw.text((x+12,10),view['name']+' / '+label,fill='white')
        path=work/view['name']/'comparison.jpg';canvas.save(path,quality=94)
        rows.append(f'<figure><img src="{view["name"]}/comparison.jpg" alt="{view["name"]}: before, edited and baked"><figcaption>{view["name"]}</figcaption></figure>')
    (work/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Multiview skin completion</title>'
        '<style>body{background:#191b1e;color:#eee;font:16px system-ui;margin:24px}img{width:100%}figure{margin:24px 0}</style>'
        '<h1>Multiview skin completion</h1><p>Left: original atlas. Middle: masked Qwen edit. Right: final atlas rendered on the fitted mesh. '
        'Photographed texels are unchanged. Hidden detail is synthetic; broad generated lighting is removed before baking.</p>'+''.join(rows))
    files=['review.html']+[v['name']+'/comparison.jpg' for v in record['views']]
    skin.write_json(work/'review.json',dict(schema='vhuman.multiview_skin_review.v1',
        geometry_sha256=record['geometry_sha256'],basecolor_sha256=sha256(completed/'skin_basecolor.png'),
        files={name:sha256(work/name) for name in files}))
    return str(work/'review.html')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','edit','bake','review','all'))
    parser.add_argument('--candidate',required=True);parser.add_argument('--work',required=True)
    parser.add_argument('--out');parser.add_argument('--prior-work')
    parser.add_argument('--steps',type=int,default=20);parser.add_argument('--edit-steps',type=int,default=12)
    parser.add_argument('--seed',type=int,default=317)
    args=parser.parse_args()
    if not 1<=args.steps<=100 or not 1<=args.edit_steps<=100:parser.error('step counts must be 1..100')
    if args.stage in ('bake','review','all') and not args.out:parser.error('--out required')
    if args.stage in ('prepare','all'):prepare(args.candidate,args.work,prior_work=args.prior_work,steps=args.steps,seed=args.seed)
    if args.stage in ('edit','all'):edit(args.work,steps=args.edit_steps,seed=args.seed)
    if args.stage in ('bake','all'):print(json.dumps(skin.bake(args.candidate,args.work,args.out),indent=2))
    if args.stage in ('review','all'):print(review(args.candidate,args.work,args.out))


if __name__=='__main__':main()
