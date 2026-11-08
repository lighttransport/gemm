"""Native-preserving offline skin lighting and eye-contact profile.

The native render remains the reference outside the authored face mask. This
screen-space approximation does not replace physical thin-part transmission.
"""
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
from .skin_scatter import (scatter_lighting, recombine_skin, preserve_native_regions,
                           suppress_indirect_spikes, smooth_scatter_correction)
from .eye_contact import soften_contact


def default_overrides(quality, ablate, overrides=None):
    """Enable the validated treatment for full skin; explicit values win."""
    result=dict(overrides or {})
    if quality=='full' and '2c' not in ablate:
        result.setdefault('skin_lighting_profile',1.)
        result.setdefault('eye_contact_blur',1.)
        if float(result['skin_lighting_profile'])>0:
            result.setdefault('eye_contact_strength',.8472126447375768)
            result.setdefault('eye_contact_skin_strength',1.)
            result.setdefault('eye_contact_skin_width_m',.0012)
            result.setdefault('skin_normal_weighted_blend',1.)
            result.setdefault('skin_micro_isotropic',1.)
            result.setdefault('skin_pore_scale',.5)
            result.setdefault('skin_normal_preserve_lids',1.)
    return result


def enabled(request):
    value=float((request.get('material_overrides') or {}).get('skin_lighting_profile',0.))
    if not np.isfinite(value) or value<0:raise ValueError('skin_lighting_profile must be finite and nonnegative')
    return value>0


def preflight(request):
    if not enabled(request):
        return
    if request.get('quality') != 'full' or '2c' in request.get('ablate', ()):
        raise ValueError('skin_lighting_profile requires the full skin material')
    try:
        import OpenEXR  # noqa: F401 -- fail before an expensive native render
    except ImportError:
        raise RuntimeError('skin_lighting_profile requires OpenEXR (requirements-render.txt)') from None
    mo=request.get('material_overrides') or {}
    guard=float(mo.get('skin_indirect_firefly_guard',0.))
    if not np.isfinite(guard) or guard not in (0.,1.,2.):
        raise ValueError('skin_indirect_firefly_guard must be 0, 1 or 2')
    denoise=float(mo.get('skin_scatter_denoise',0.))
    if not np.isfinite(denoise) or denoise not in (0.,1.,2.):
        raise ValueError('skin_scatter_denoise must be 0, 1 or 2')
    if guard and denoise:
        raise ValueError('skin_scatter_denoise and skin_indirect_firefly_guard cannot be combined')
    symmetric=float(mo.get('skin_scatter_symmetric',0.))
    if not np.isfinite(symmetric) or symmetric not in (0.,1.):
        raise ValueError('skin_scatter_symmetric must be 0 or 1')
    scale=float(mo.get('skin_scatter_scale',.5));weight=float(mo.get('skin_scatter_weight',1.))
    if not np.isfinite(scale) or scale<0 or not np.isfinite(weight) or not 0<=weight<=1:
        raise ValueError('finite nonnegative skin scale and weight in [0,1] required')
    correction_sigma=float(mo.get('skin_scatter_correction_sigma_m',0.))
    if not np.isfinite(correction_sigma) or correction_sigma<0:
        raise ValueError('skin_scatter_correction_sigma_m must be finite and nonnegative')
    for key, default in (('eye_contact_width_m',.0012),('eye_contact_sigma_m',.00025),
                         ('eye_contact_skin_width_m',.0012)):
        value=float(mo.get(key,default))
        if not np.isfinite(value) or value<=0:raise ValueError(f'{key} must be positive and finite')
    if 'eye_contact_skin_sigma_m' in mo:
        value=float(mo['eye_contact_skin_sigma_m'])
        if not np.isfinite(value) or value<=0:raise ValueError('eye_contact_skin_sigma_m must be positive and finite')
    for key in ('eye_contact_tint','eye_contact_skin_tint'):
        if key in mo:
            value=np.asarray(mo[key],np.float32)
            if value.shape!=(3,) or not np.isfinite(value).all() or (value<0).any():
                raise ValueError(f'{key} must contain three finite nonnegative values')
    strength=float(mo.get('eye_contact_strength',.35739718109334984))
    if not np.isfinite(strength) or not 0<=strength<=1:raise ValueError('eye contact strength must be in [0,1]')
    upper=mo.get('eye_contact_upper_strength');lower=mo.get('eye_contact_lower_strength')
    if (upper is None)!=(lower is None):raise ValueError('both upper and lower eye strengths are required')
    if upper is not None:
        upper,lower=float(upper),float(lower)
        if not np.isfinite(upper) or not np.isfinite(lower) or not 0<=upper<=1 or not 0<=lower<=1:
            raise ValueError('upper and lower eye strengths must be in [0,1]')
    skin_strength=float(mo.get('eye_contact_skin_strength',0.))
    if not np.isfinite(skin_strength) or not 0<=skin_strength<=1:
        raise ValueError('eye_contact_skin_strength must be in [0,1]')
    contact=float(mo.get('eye_contact_blur',0.))
    if not np.isfinite(contact) or contact<0:raise ValueError('eye_contact_blur must be finite and nonnegative')
    thin_aperture=float(mo.get('eye_contact_thin_aperture',0.))
    if not np.isfinite(thin_aperture) or thin_aperture not in (0.,1.):
        raise ValueError('eye_contact_thin_aperture must be 0 or 1')
    skin_coverage=float(mo.get('eye_contact_skin_coverage',0.))
    if not np.isfinite(skin_coverage) or skin_coverage not in (0.,1.):
        raise ValueError('eye_contact_skin_coverage must be 0 or 1')
    pore_scale=float(mo.get('skin_pore_scale',1.))
    if not np.isfinite(pore_scale) or pore_scale<0:raise ValueError('skin_pore_scale must be finite and nonnegative')


def compose(native, passes, metadata, overrides):
    """Compose exact material AOVs; independent of topology, pose and camera fit."""
    if 'skin_profile' not in passes or 'eye_mask' not in passes:
        raise ValueError('skin profile capture is missing anatomical/eye AOVs')
    profile=np.asarray(passes['skin_profile'][...,0],np.float32)
    skin=np.asarray(passes['skin_mask'][...,0],np.float32)
    eye=np.asarray(passes['eye_mask'][...,0],np.float32)
    if not all(np.isfinite(x).all() for x in (profile,skin,eye)) or (profile< -1e-5).any() or (profile>skin+1e-4).any():
        raise ValueError('invalid anatomical profile AOV')
    profile=np.clip(profile,0,1)
    if native.shape != passes['Combined'].shape:raise ValueError('native/capture image dimensions differ')
    h=native.shape[0];focal=h*float(metadata['lens_mm'])/float(metadata['sensor_height_mm'])
    depth=((passes['Position'][...,:3]-np.asarray(metadata['camera_origin']))*np.asarray(metadata['camera_forward'])).sum(-1)
    scale=float(overrides.get('skin_scatter_scale',.5));weight=float(overrides.get('skin_scatter_weight',1.))
    guard=float(overrides.get('skin_indirect_firefly_guard',0.))
    if not np.isfinite(guard) or guard not in (0.,1.,2.):
        raise ValueError('skin_indirect_firefly_guard must be 0, 1 or 2')
    denoise=float(overrides.get('skin_scatter_denoise',0.))
    if not np.isfinite(denoise) or denoise not in (0.,1.,2.):
        raise ValueError('skin_scatter_denoise must be 0, 1 or 2')
    if guard and denoise:
        raise ValueError('skin_scatter_denoise and skin_indirect_firefly_guard cannot be combined')
    symmetric=float(overrides.get('skin_scatter_symmetric',0.))
    if not np.isfinite(symmetric) or symmetric not in (0.,1.):
        raise ValueError('skin_scatter_symmetric must be 0 or 1')
    indirect=passes['DiffInd'][...,:3]
    direct=passes['DiffDir'][...,:3]
    guarded=0
    if denoise:
        if 'DiffIndDenoised' not in passes or (denoise==2 and 'DiffDirDenoised' not in passes):
            raise ValueError('skin scatter denoising requires the guided diffuse passes')
        indirect=passes['DiffIndDenoised'][...,:3]
        if denoise==2:direct=passes['DiffDirDenoised'][...,:3]
    if guard:
        if 'DiffIndDenoised' not in passes:
            raise ValueError('skin firefly guard requires the guided indirect pass')
        floor=.05
        if guard==2:
            luma=np.array([.2126,.7152,.0722],np.float32)
            floor=.02*np.maximum(passes['DiffDir'][...,:3] @ luma,0.)
        indirect,selected=suppress_indirect_spikes(indirect,passes['DiffIndDenoised'][...,:3],floor=floor)
        guarded=int((selected & (skin>.99)).sum())
    lighting=direct+indirect
    radii=np.array([1.05,.4069,.2625])*scale
    scattered=scatter_lighting(lighting,skin,depth,passes['skin_geometric_normal'][...,:3],focal,radii, symmetric=bool(symmetric))
    correction_sigma=float(overrides.get('skin_scatter_correction_sigma_m',0.))
    if not np.isfinite(correction_sigma) or correction_sigma<0:
        raise ValueError('skin_scatter_correction_sigma_m must be finite and nonnegative')
    correction_sigma_px=0.
    if correction_sigma:
        valid=(skin>.99)&(depth>0)
        if not valid.any():raise ValueError('skin correction smoothing requires visible positive-depth skin')
        correction_sigma_px=min(correction_sigma*focal/float(np.median(depth[valid])),8.)
        correction=smooth_scatter_correction(scattered-lighting,skin,depth,
                                             passes['skin_geometric_normal'][...,:3],correction_sigma_px)
        scattered=lighting+correction
    treated=recombine_skin(passes['Combined'],lighting,scattered,passes['DiffCol'][...,:3],skin,weight)
    result=preserve_native_regions(native,treated,profile)
    contact=float(overrides.get('eye_contact_blur',0.))>0
    thin_aperture=float(overrides.get('eye_contact_thin_aperture',0.))
    if not np.isfinite(thin_aperture) or thin_aperture not in (0.,1.):
        raise ValueError('eye_contact_thin_aperture must be 0 or 1')
    skin_coverage=float(overrides.get('eye_contact_skin_coverage',0.))
    if not np.isfinite(skin_coverage) or skin_coverage not in (0.,1.):
        raise ValueError('eye_contact_skin_coverage must be 0 or 1')
    if contact:
        result=soften_contact(result,eye,depth,focal,
            width_m=float(overrides.get('eye_contact_width_m',.0012)),
            sigma_m=float(overrides.get('eye_contact_sigma_m',.00025)),
            strength=float(overrides.get('eye_contact_strength',.35739718109334984)),
            tint=overrides.get('eye_contact_tint',(1.,.904661,.896269)),
            skin_mask=skin, skin_width_m=float(overrides.get('eye_contact_skin_width_m',.0012)),
            skin_strength=float(overrides.get('eye_contact_skin_strength',0.)),
            skin_sigma_m=float(overrides['eye_contact_skin_sigma_m']) if 'eye_contact_skin_sigma_m' in overrides else None,
            skin_tint=overrides.get('eye_contact_skin_tint'),
            upper_strength=float(overrides['eye_contact_upper_strength']) if overrides.get('eye_contact_upper_strength') is not None else None,
            lower_strength=float(overrides['eye_contact_lower_strength']) if overrides.get('eye_contact_lower_strength') is not None else None,
            thin_aperture=bool(thin_aperture),skin_coverage=bool(skin_coverage))
    record=dict(method='UE-inspired diffuse lighting profile; native thin parts/non-skin preserved',
                radius_mm=radii.tolist(),weight=weight,contact_blur=contact,contact_thin_aperture=contact and bool(thin_aperture),
                contact_skin_coverage=contact and bool(skin_coverage) and float(overrides.get('eye_contact_skin_strength',0.))>0,
                indirect_firefly_guard=bool(guard),indirect_firefly_guard_mode=int(guard),guarded_opaque_skin_pixels=guarded,
                scatter_denoise_mode=int(denoise),
                scatter_symmetric=bool(symmetric),
                correction_sigma_m=correction_sigma,correction_sigma_px=correction_sigma_px,
                anatomical_mask='interpolated skin_profile_weight point attribute',
                preserved_pixels=int((profile==0).sum()),eye_pixels=int((eye>.5).sum()),
                capture=metadata, overrides=dict(overrides),
                source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('skin_profile.py','skin_scatter.py','eye_contact.py')})
    return result,record


def finish(request, blender, *, env=None, hard_limit_mib=14336, memory_usage=None, cancel=None):
    """Capture unscattered lighting under the original camera, then finalize beauty.

    Caller holds the GPU execution lock. memory_usage returns whole-device MiB.
    The original beauty is retained; a failed profile never leaves a done marker.
    """
    preflight(request)
    from ..metahuman.peri_scatter import load_passes
    from ..metahuman.images import read_exr,write_exr,display_srgb
    from PIL import Image
    out=Path(request['out']);native=out/'beauty.exr';capture=out/'skin_profile_capture.exr'
    old_result=out/'render_result.json';result=json.loads(old_result.read_text())
    if result.get('skin_lighting_profile'):
        raise ValueError('skin lighting profile is already applied')
    old_result.replace(out/'native_render_result.json')
    started=time.monotonic();peak=0.
    worker=Path(__file__).with_name('skin_profile_blender.py')
    with (out/'skin_profile.log').open('w') as log:
        process=subprocess.Popen([str(blender),'-b','-noaudio','--factory-startup',str(out/'head.blend'),
            '--python-exit-code','1','--python',str(worker),'--',str(out/'request.json')],
            stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            while process.poll() is None:
                if cancel is not None and cancel.is_set():raise RuntimeError('skin profile cancelled')
                if memory_usage is not None:
                    used=float(memory_usage());peak=max(peak,used)
                    if used>hard_limit_mib:raise RuntimeError(f'skin profile exceeded {hard_limit_mib} MiB device limit')
                time.sleep(.1)
            if process.returncode:raise RuntimeError(f'skin profile capture failed; see {out/"skin_profile.log"}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
    metadata=json.loads(capture.with_suffix('.json').read_text())
    if metadata['native_sss_weight'] != 0:raise ValueError('profile capture must disable native skin SSS')
    native_image=read_exr(native);passes=load_passes(capture)
    image,record=compose(native_image,passes,metadata,request.get('material_overrides') or {})
    native_copy=out/'beauty_native.exr'
    pending=out/'beauty_profile.pending.exr';write_exr(pending,image)
    native.replace(native_copy)
    pending.replace(native)
    Image.fromarray(display_srgb(image[...,:3],request.get('exposure',-1.5))).save(out/'beauty.png')
    result['native_render_seconds']=result.get('seconds');result['skin_lighting_profile']=record
    result['profile_seconds']=time.monotonic()-started
    result['seconds']=float(result.get('seconds',0))+result['profile_seconds']
    result['blend_render']='native; regenerate final beauty through offline_render skin profile'
    old_result.write_text(json.dumps(result,indent=2))
    return peak
