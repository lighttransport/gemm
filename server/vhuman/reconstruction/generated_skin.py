"""Research-only synthetic skin completion with immutable photographed texels.

Qwen T2I supplies a spatial detail prior; calibrated masked edits refine it.
Wan I2V supplies a temporal consistency gate, never a measured observation.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .observations import sha256
from .provenance import validate_candidate
from .reference import Camera, rasterize, srgb_to_linear, linear_to_srgb
from ..rig.bake import rasterize_uv
from ..rig.common import vertex_normals, normalize
from ..mobile.baking import sample

ROOT=Path(__file__).resolve().parents[3]


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2))


def qwen_generate(out, prompt, *, reference=None, mask=None, init_image=None, steps=20, seed=317,
                  strength=None, cfg=1., negative_prompt='', resolution=512,
                  model='/mnt/disk01/models/qimg-21', package='/mnt/disk01/models/qimg-21-fast/int8-smooth-a0.6'):
    """Use the installed HIP backend, releasing it before video generation."""
    from .. import gpu
    from ..qwen import _import_qimg21
    _import_qimg21()
    from qimg21_i23d.native import NativeBackend
    from qimg21_i23d.backends import GenRequest
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    receipt=out.with_suffix('.json')
    request=dict(prompt=prompt,steps=steps,seed=seed,model=str(Path(model).resolve()),
        package=str(Path(package).resolve()),reference_sha256=sha256(reference) if reference else None,
        mask_sha256=sha256(mask) if mask else None)
    if init_image is not None:request['init_image_sha256']=sha256(init_image)
    if resolution!=512:request['resolution']=resolution
    strength=(.45 if mask else 1.) if strength is None else strength
    if not 0<strength<=1 or cfg<1 or (cfg>1 and not negative_prompt):
        raise ValueError('valid strength and a negative prompt for guided generation required')
    if strength!=(.45 if mask else 1.) or cfg!=1. or negative_prompt:
        request.update(strength=strength,cfg=cfg,negative_prompt=negative_prompt)
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous['request']!=request or previous['sha256']!=sha256(out):
            raise ValueError('generation cache does not match requested inputs')
        return previous
    backend=NativeBackend(model=model,quant_package=package,backend='rocm',preset='low8',
        attention='sage',condition_resolution=512,python=sys.executable,resident=False,keep_work=True)
    with gpu.execution('rocm',0,Path(model).parent),gpu.device_session(10000):
        try:
            result=backend.generate(GenRequest(prompt=prompt,out=out,width=resolution,height=resolution,steps=steps,
                seed=seed,references=(reference,) if reference else (),init_image=(init_image or reference) if mask else None,
                mask=mask,mask_as_reference=False,strength=strength,true_cfg_scale=cfg,negative_prompt=negative_prompt))
        finally:backend.close()
    record=dict(request=request,sha256=sha256(out),generator='Qwen-Image-2.1',
        license='qwen-research',synthetic=True,geometry_evidence=False,
        backend=result.backend,seconds=result.seconds,details=result.details)
    write_json(receipt,record)
    return record


def band_detail(rgb, limit=.025, broad_sigma=9.):
    """Remove broad illumination; limit invented linear-light detail amplitude."""
    rgb=np.asarray(rgb,float)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or not np.isfinite(rgb).all():
        raise ValueError('finite RGB image required')
    return np.clip(gaussian_filter(rgb,(.65,.65,0))-gaussian_filter(rgb,(broad_sigma,broad_sigma,0)),-limit,limit)


def atlas_surface(geometry, resolution):
    uv=geometry['triangle_uvs'];tri=geometry['triangles'];p=geometry['captured'][0]
    ids,bary=rasterize_uv(uv.reshape(-1,2),np.arange(uv.size//2).reshape(-1,3),resolution)
    valid=ids>=0;faces=tri[ids[valid]];weights=bary[valid]
    points=(p[faces]*weights[...,None]).sum(1)
    normals=normalize((vertex_normals(p,tri)[faces]*weights[...,None]).sum(1))
    return valid,points,normals


def unseen_weight(points, observed, distance=.006):
    """Metric feather with exact protection of all photographed texels."""
    observed=np.asarray(observed,bool)
    if observed.shape!=(len(points),) or not observed.any():raise ValueError('observed surface anchors required')
    d,_=cKDTree(points[observed]).query(points)
    t=np.clip(d/distance,0,1)
    return np.where(observed,0,t*t*(3-2*t))


def apply_detail(base, valid, delta, observed, limit=.025):
    """Apply a bounded synthetic residual, preserving captured bytes exactly."""
    base=np.asarray(base,np.uint8);delta=np.asarray(delta,float)
    if delta.shape!=(int(valid.sum()),3) or not np.isfinite(delta).all():
        raise ValueError('finite covered-texel RGB residual required')
    result=base.copy()
    color=srgb_to_linear(base[valid]/255)+np.clip(delta,-limit,limit)
    result[valid]=np.uint8(np.clip(linear_to_srgb(np.clip(color,0,1))*255+.5,0,255))
    result[observed]=base[observed]
    return result


def triplanar_detail(points, normals, plate, period=.045):
    """World-space mirrored sampling prevents UV cuts from creating seams."""
    residual=band_detail(plate,limit=.012)
    weights=abs(normals)**4;weights/=np.maximum(weights.sum(1,keepdims=True),1e-12)
    result=np.zeros_like(points)
    for axis,pair in enumerate(((1,2),(0,2),(0,1))):
        uv=points[:,pair]/period
        uv=1-abs(np.mod(uv,2)-1)
        result+=sample(residual,uv)*weights[:,axis,None]
    return result


def orbit_camera(points, yaw, resolution=512, pitch=0):
    centre=(points.min(0)+points.max(0))/2
    extent=float(np.ptp(points,axis=0).max());distance=extent*3
    angle=np.deg2rad(yaw);elevation=np.deg2rad(pitch)
    back=np.array([np.sin(angle)*np.cos(elevation),np.sin(elevation),np.cos(angle)*np.cos(elevation)])
    right=np.array([np.cos(angle),0.,-np.sin(angle)]);up=np.cross(back,right)
    return Camera(.82*resolution*distance/extent,resolution/2,resolution/2,
                  centre+distance*back,np.stack((right,up,back)))


def render_plate(geometry, atlas, blend, camera, resolution=512):
    p=geometry['captured'][0];tri=geometry['triangles']
    ids,bary,depth=rasterize(p,tri,camera,(resolution,resolution));valid=ids>=0
    uv=(geometry['triangle_uvs'][ids[valid]]*bary[valid,...,None]).sum(1)
    rgb=np.full((resolution,resolution,3),.18);rgb[valid]=sample(atlas,uv)
    # Keep the silhouette and background exact even if the editor changes shape.
    from scipy.ndimage import binary_erosion
    eligible=np.zeros(valid.shape);eligible[valid]=sample(blend,uv)
    eligible*=binary_erosion(valid,iterations=3)
    return np.uint8(np.clip(linear_to_srgb(rgb)*255+.5,0,255)),np.uint8((eligible>.95)*255),depth


def temporal_consistency(reference, frames, *, return_consensus=False):
    """Flow-aligned I2V stability, rejecting motion/occlusion rather than baking it."""
    import cv2
    reference=np.asarray(reference,np.uint8)
    if len(frames)<3:raise ValueError('at least three I2V frames required')
    gray=cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY)
    h,w=gray.shape;yy,xx=np.mgrid[:h,:w];votes=[];errors=[]
    aligned_colors=[reference.astype(np.float32)] if return_consensus else []
    for frame in frames:
        if frame.shape!=reference.shape:raise ValueError('I2V crop/size changed')
        target=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        flow=cv2.calcOpticalFlowFarneback(gray,target,None,.5,3,21,3,5,1.2,0)
        backward=cv2.calcOpticalFlowFarneback(target,gray,None,.5,3,21,3,5,1.2,0)
        mx=(xx+flow[...,0]).astype(np.float32);my=(yy+flow[...,1]).astype(np.float32)
        aligned=cv2.remap(target,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        inverse=cv2.remap(backward,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        error=abs(aligned.astype(float)-gray)/255
        inside=(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1)
        good=inside&(np.linalg.norm(flow,axis=-1)<3)&(np.linalg.norm(flow+inverse,axis=-1)<1)&(error<.06)
        votes.append(good);errors.append(float(np.median(error)))
        if return_consensus:
            color=cv2.remap(frame,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
            aligned_colors.append(np.where(good[...,None],color,np.nan))
    support=np.mean(votes,axis=0)
    metrics=dict(frames=len(frames),
        stable_fraction=float((support>=.75).mean()),median_error=errors,
        method='forward/backward flow + bounded motion + photometric agreement',
        proves_observed_accuracy=False)
    gate=np.where(support>=.75,support,0)
    if return_consensus:
        consensus=np.uint8(np.clip(np.nanmedian(np.stack(aligned_colors),axis=0)+.5,0,255))
        consensus[gate==0]=reference[gate==0]
        return gate,metrics,consensus
    return gate,metrics


def prepare(candidate, out, *, steps=20, seed=317):
    candidate,out=Path(candidate).resolve(),Path(out).resolve();manifest=validate_candidate(candidate)
    out.mkdir(parents=True,exist_ok=True)
    binding=dict(candidate=str(candidate),geometry_sha256=manifest['geometry_sha256'],
                 basecolor_sha256=sha256(candidate/'skin_basecolor.png'),steps=steps,seed=seed)
    if (out/'binding.json').exists() and json.loads((out/'binding.json').read_text())!=binding:
        raise ValueError('completion workspace belongs to another candidate or generation setting')
    write_json(out/'binding.json',binding)
    prompt=('An evenly illuminated macro photograph of human medium brown skin, a flat continuous skin texture '
        'filling the entire square image. Natural fine pores and delicate skin creases, realistic subtle '
        'pigmentation. Cross-polarized diffuse white light, no highlights, no directional shadows, no hair, '
        'no eyes, no face, no lips, no objects, no text. This is a skin material reference patch.')
    print('Generating T2I skin-detail prior',flush=True)
    qwen_generate(out/'skin_prior.png',prompt,steps=steps,seed=seed)
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:geometry=dict(z)
    base=np.asarray(Image.open(candidate/'skin_basecolor.png').convert('RGB'))
    observed=np.asarray(Image.open(candidate/'skin_coverage.png'))>0
    valid,points,normals=atlas_surface(geometry,len(base))
    blend=np.zeros(valid.shape);blend[valid]=unseen_weight(points,observed[valid])
    linear=srgb_to_linear(base/255)
    prior=srgb_to_linear(np.asarray(Image.open(out/'skin_prior.png').convert('RGB'))/255)
    seed_detail=triplanar_detail(points,normals,prior)
    seeded=linear.copy();seeded[valid]=np.clip(linear[valid]+seed_detail*blend[valid,None],0,1)
    np.savez_compressed(out/'surface.npz',valid=valid,points=points,normals=normals,blend=blend,
                        seeded=seeded,seed_detail=seed_detail)
    views=[]
    for i,yaw in enumerate((-65,65,180)):
        folder=out/f'view_{i}';folder.mkdir(exist_ok=True)
        camera=orbit_camera(geometry['captured'][0],yaw)
        image,mask,depth=render_plate(geometry,seeded,blend,camera)
        Image.fromarray(image).save(folder/'input.png');Image.fromarray(mask).save(folder/'mask.png')
        np.save(folder/'depth.npy',depth)
        views.append(dict(name=folder.name,yaw=yaw,camera=camera.as_dict(),mask_pixels=int((mask>0).sum())))
    record=dict(schema='vhuman.generated_skin.v1',**binding,views=views,
        synthetic=True,license='qwen-research',photographed_texels_immutable=True)
    inputs=['surface.npz','skin_prior.png','skin_prior.json']
    inputs += [f"{view['name']}/{name}" for view in views for name in ('input.png','mask.png','depth.npy')]
    record['inputs']={name:sha256(out/name) for name in inputs}
    write_json(out/'completion.json',record)
    return record


def edit_views(out, *, steps=20, seed=317):
    out=Path(out);record=json.loads((out/'completion.json').read_text())
    for i,view in enumerate(record['views']):
        folder=out/view['name']
        prompt=('Refine only the plain skin areas of this exact head and neck material reference. '
            'Preserve the exact pose, head silhouette, camera, facial proportions and existing facial details. '
            'Add subtle realistic fine pores, delicate skin creases and natural skin pigmentation to the '
            'smooth untextured skin, matching the surrounding medium brown skin. Even diffuse lighting, '
            'no new shadows or highlights. No hair, no clothing, no jewelry, no tattoos, no text. '
            'Keep the gray background unchanged.')
        print('Masked Qwen edit: '+view['name'],flush=True)
        qwen_generate(folder/'edited.png',prompt,reference=folder/'input.png',mask=folder/'mask.png',
                      steps=steps,seed=seed+i+1)
        original=np.asarray(Image.open(folder/'input.png').convert('RGB'))
        edited=np.asarray(Image.open(folder/'edited.png').convert('RGB'))
        mask=np.asarray(Image.open(folder/'mask.png'))>0
        if edited.shape!=original.shape or not np.array_equal(original[~mask],edited[~mask]):
            raise ValueError('masked editor altered protected pixels')


def video_check(out, *, preset='fast5', frames=9):
    import cv2
    from ..video_backend import WanBackend
    out=Path(out);record=json.loads((out/'completion.json').read_text())
    # A short static-camera clip checks edited detail without inventing camera calibration.
    for view in record['views']:
        folder=out/view['name'];video=folder/'video'
        if not (video/'manifest.json').exists():
            print('Wan I2V consistency: '+view['name'],flush=True)
            WanBackend().generate(image=folder/'edited.png',out=video,preset=preset,frames=frames,
                width=512,height=512,allow_experimental=True,seed=811,
                progress=lambda step,total:print(f"Wan {view['name']}: {step}/{total}",flush=True),
                prompt='A completely still head and neck skin reference, fixed camera and pose, soft constant diffuse lighting. Preserve the fine skin texture. No talking, no turning, no camera motion.')
        manifest=json.loads((video/'manifest.json').read_text())
        if manifest.get('image_sha256')!=sha256(folder/'edited.png') or manifest.get('preset')!=preset or manifest.get('frames')!=frames:
            raise ValueError('I2V reference hash mismatch')
        cap=cv2.VideoCapture(str(video/'clip.mp4'));images=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            images.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release()
        reference=np.asarray(Image.open(folder/'edited.png').convert('RGB'))
        support,metrics,consensus=temporal_consistency(reference,images[1:],return_consensus=True)
        edit_mask=np.asarray(Image.open(folder/'mask.png'))>0
        metrics['editable_stable_fraction']=float((support[edit_mask]>0).mean()) if edit_mask.any() else 0.
        np.save(folder/'temporal_support.npy',support)
        Image.fromarray(consensus).save(folder/'temporal_consensus.png')
        write_json(folder/'temporal.json',dict(**metrics,video_sha256=sha256(video/'clip.mp4'),
                   consensus_sha256=sha256(folder/'temporal_consensus.png'),
                   support_sha256=sha256(folder/'temporal_support.npy'),
                   reference_sha256=sha256(folder/'edited.png')))


def bake(candidate, work, out):
    candidate,work,out=map(Path,(candidate,work,out));manifest=validate_candidate(candidate)
    record=json.loads((work/'completion.json').read_text())
    if record['geometry_sha256']!=manifest['geometry_sha256'] or record['basecolor_sha256']!=sha256(candidate/'skin_basecolor.png'):
        raise ValueError('completion source identity or material changed')
    for name,digest in record['inputs'].items():
        path=(work/name).resolve()
        if not path.is_relative_to(work.resolve()) or sha256(path)!=digest:
            raise ValueError('completion input checksum mismatch')
    if out.exists() and any(out.iterdir()):raise ValueError('candidate output must be empty')
    with np.load(work/'surface.npz',allow_pickle=False) as z:
        valid,points,normals,blend,seed_detail=(z[k] for k in ('valid','points','normals','blend','seed_detail'))
    base=np.asarray(Image.open(candidate/'skin_basecolor.png').convert('RGB'))
    observed=np.asarray(Image.open(candidate/'skin_coverage.png'))>0
    accum=np.zeros_like(points);weights=np.zeros(len(points));reports=[]
    multiview=record.get('consistency')=='multiview'
    view_colors=[];view_weights=[]
    from ..face_parsing import FaceParser
    from .occlusion import skin_bake_mask
    parser=FaceParser()
    for view in record['views']:
        folder=work/view['name'];camera=Camera.from_dict(view['camera'])
        edit_record=json.loads((folder/'edited.json').read_text())
        if sha256(folder/'edited.png')!=edit_record['sha256']:
            raise ValueError('edited image receipt mismatch')
        edited=srgb_to_linear(np.asarray(Image.open(folder/'edited.png').convert('RGB'))/255)
        source=srgb_to_linear(np.asarray(Image.open(folder/'input.png').convert('RGB'))/255)
        if multiview:
            request=edit_record['request']
            if (request.get('init_image_sha256')!=sha256(folder/'input.png')
                    or request['reference_sha256']!=sha256(folder/'reference.png')
                    or request['mask_sha256']!=sha256(folder/'mask.png')):
                raise ValueError('multiview edit input receipt mismatch')
            consensus=edited;temporal=None;gate=np.ones(edited.shape[:2])
            mask_bool=np.asarray(Image.open(folder/'mask.png'))>0
            original_bytes=np.asarray(Image.open(folder/'input.png').convert('RGB'))
            edited_bytes=np.asarray(Image.open(folder/'edited.png').convert('RGB'))
            if edited_bytes.shape!=original_bytes.shape or not np.array_equal(edited_bytes[~mask_bool],original_bytes[~mask_bool]):
                raise ValueError('multiview edit changed protected pixels')
        else:
            temporal=json.loads((folder/'temporal.json').read_text())
            if temporal['reference_sha256']!=edit_record['sha256']:raise ValueError('edited image receipt mismatch')
            if temporal['video_sha256']!=sha256(folder/'video/clip.mp4'):raise ValueError('video receipt mismatch')
            if temporal['consensus_sha256']!=sha256(folder/'temporal_consensus.png') or temporal['support_sha256']!=sha256(folder/'temporal_support.npy'):
                raise ValueError('temporal consensus checksum mismatch')
            consensus=srgb_to_linear(np.asarray(Image.open(folder/'temporal_consensus.png').convert('RGB'))/255)
            gate=np.load(folder/'temporal_support.npy')
        residual=band_detail((edited+consensus)*.5-source)
        mask=np.asarray(Image.open(folder/'mask.png'),float)/255
        # Exclude newly generated eyes/hair/accessories from skin detail.
        labels,confidence=parser.predict(np.uint8(linear_to_srgb(edited)*255+.5))
        if multiview:
            # Face parsers often call a rear scalp background. The calibrated
            # mesh/mask supplies the silhouette; reject confident foreign parts.
            excluded=np.isin(labels,[2,3,4,5,6,9,11,12,13,15,16,17,18])&(confidence>=.7)
        else:excluded=skin_bake_mask(labels,confidence)
        eligible=mask*(~excluded)*gate
        xy,z=camera.project(points);uv=xy/512
        depth=np.load(folder/'depth.npy');ix=np.clip(xy[:,0].astype(int),0,511);iy=np.clip(xy[:,1].astype(int),0,511)
        facing=np.maximum((normals*normalize(camera.origin-points)).sum(1),0)
        seen=(uv.min(1)>=0)&(uv.max(1)<1)&(abs(depth[iy,ix]-z)<.002)&(facing>.3)
        weight=sample(eligible,uv)*seen*facing**2
        color=sample(residual,uv)
        if multiview:view_colors.append(color);view_weights.append(weight)
        accum+=color*weight[:,None];weights+=weight
        reports.append(dict(view=view['name'],accepted_texels=int((weight>.05).sum()),temporal=temporal,
                            edit_sha256=edit_record['sha256']))
        if multiview:
            reports[-1]['projected_texels']=reports[-1].pop('accepted_texels')
            reports[-1]['edit_seconds']=edit_record['seconds']
            reports[-1]['condition_sha256']=edit_record['request']['reference_sha256']
    consistency=None
    if multiview:
        from .multiview_skin import fuse_views
        detail,weights,consistency=fuse_views(view_colors,view_weights)
    else:detail=accum/np.maximum(weights[:,None],1e-12)
    delta=seed_detail+detail
    delta=np.clip(delta,-.025,.025)*blend[valid,None]
    result=apply_detail(base,valid,delta,observed)
    # Keep all evidence maps and geometry unchanged; generated support is separate.
    out.mkdir(parents=True,exist_ok=True)
    for path in candidate.iterdir():
        if path.is_file():shutil.copyfile(path,out/path.name)
    from scipy.ndimage import distance_transform_edt
    distance,near=distance_transform_edt(~valid,return_indices=True);gutter=(~valid)&(distance<=4)&~observed
    result[gutter]=result[near[0][gutter],near[1][gutter]]
    Image.fromarray(result).save(out/'skin_basecolor.png')
    support=np.zeros(valid.shape);support[valid]=np.minimum(.05+weights,.25)*blend[valid]
    Image.fromarray(np.uint8(support*255+.5)).save(out/'skin_generated_support.png')
    report=dict(schema='vhuman.synthetic_skin_completion.v1',work=str(work.resolve()),
        synthetic=True,license='qwen-research',generator='Qwen-Image-2.1 mesh-guided multiview' if multiview else 'Qwen-Image-2.1 + Wan2.2',
        basecolor_sha256=sha256(out/'skin_basecolor.png'),
        inputs=record['inputs'],prior=json.loads((work/'skin_prior.json').read_text()),
        photographed_texels_changed=int(np.any(result[observed]!=base[observed],axis=-1).sum()),
        generated_texels=int((valid&np.any(result!=base,axis=-1)).sum()),
        edited_view_texels=int(((weights>.05)&(blend[valid]>0)).sum()),
        max_linear_detail=float(abs(delta).max()),views=reports,
        limitations=['generated detail is an appearance prior, not recovered anatomy',
                     'I2V agreement measures self-consistency, not photographic accuracy',
                     'no generated colors enter measured confidence or permissive training'])
    if multiview:
        report['multiview_consistency']=consistency
        report['limitations'][1]='multiview agreement measures synthetic self-consistency, not photographic accuracy'
    manifest['material']['synthetic_completion']=report
    manifest['material_refinement']=dict(source=str(candidate.resolve()),geometry_unchanged=True)
    write_json(out/'manifest.json',manifest);write_json(out/'skin_material.json',manifest['material'])
    write_json(out/'generated_skin.json',report)
    validate_candidate(out)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','edit','video','bake','all'))
    p.add_argument('--candidate',required=True);p.add_argument('--work',required=True);p.add_argument('--out')
    p.add_argument('--steps',type=int,default=20);p.add_argument('--seed',type=int,default=317)
    p.add_argument('--edit-steps',type=int,help='separate masked-edit schedule; defaults to --steps')
    p.add_argument('--video-preset',choices=('fast5','fast12','quality'),default='fast5')
    p.add_argument('--frames',type=int,default=9)
    a=p.parse_args()
    if not 1<=a.steps<=100 or (a.edit_steps is not None and not 1<=a.edit_steps<=100):
        p.error('step counts must be 1..100')
    if a.stage in ('bake','all') and not a.out:p.error('--out required for bake/all')
    if a.stage in ('prepare','all'):prepare(a.candidate,a.work,steps=a.steps,seed=a.seed)
    if a.stage in ('edit','all'):edit_views(a.work,steps=a.edit_steps or a.steps,seed=a.seed)
    if a.stage in ('video','all'):video_check(a.work,preset=a.video_preset,frames=a.frames)
    if a.stage in ('bake','all'):print(json.dumps(bake(a.candidate,a.work,a.out),indent=2))


if __name__=='__main__':main()
