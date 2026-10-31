"""Portable offline head scene and explicitly selected Cycles GPU rendering."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .. import gpu
from .offline_assets import prepare

DEFAULT_BLENDER = Path(os.environ.get('VHUMAN_BLENDER', str(Path.home()/'local/blender-5.2.2-linux-x64/blender')))


PRESETS = {'draft': (512, 32), 'final': (1024, 256), 'hero': (2048, 1024), 'crop': (1024, 1024)}
LIGHTING = ('studio', 'left', 'right', 'rim', 'hdri')
DEFAULT_HDRI = Path(__file__).resolve().parents[3] / 'ref/pixal3d/upstream/assets/hdri/interior.exr'   # CC0, Poly Haven hotel_room 1K
QUALITY = ('baseline', 'full')
ABLATIONS = ('2a', '2b', '2c', '2d', '2e', '2f')


def render(candidate, out, *, device='hip', preset='draft', accessories='keep',
           detail_preset='mature', blender=DEFAULT_BLENDER, gpu_index=0,
           hard_limit_mib=14336, cancel=None, motion=None, frame=1, appearance=None,
           lighting='studio', yaw=0., exposure=-1.5, sss_weight=.08,head_fit=None,parsing_model=None,
           include_hair=True,fit_eyes=False, camera='portrait', pitch=0., distance_scale=None, focal_scale=None,
           hdri=None, hdri_strength=1., hdri_rotation=0., samples=None, resolution=None, quality='baseline',
           ablate=(), noisy=True, eyes=None, chart=False, displacement_scale=1., micro_scale=1., save_blend=True,
           lip_coat_scale=1.):
    """camera: a render_cameras.CAMERAS preset; yaw/pitch/distance_scale/focal_scale override its values.
    lighting 'hdri' lights the scene with `hdri` (default DEFAULT_HDRI; its sha256 is recorded).
    preset hero = 2048^2 / 1024 spp; samples/resolution override any preset. noisy=True also writes the
    un-denoised beauty (beauty_noisy.exr) from the same render. quality/ablate select the material set
    ('full' = A/B study candidate B; ablate lists stages removed, e.g. ('2d',))."""
    from .render_cameras import CAMERAS
    if device not in ('hip', 'cuda', 'optix', 'cpu') or preset not in PRESETS:
        raise ValueError('invalid render device or preset')
    if not 1024 <= hard_limit_mib <= 14336:
        raise ValueError('hard GPU limit must be 1024..14336 MiB')
    if lighting not in LIGHTING or not -180<=yaw<=180 or not -89<=pitch<=89:
        raise ValueError('invalid lighting, yaw or pitch')
    if camera not in CAMERAS:
        raise ValueError('unknown camera preset: '+str(camera))
    if not -3<=exposure<=3 or not 0<=sss_weight<=1:
        raise ValueError('exposure must be -3..3 EV and SSS weight 0..1')
    if quality not in QUALITY or any(a not in ABLATIONS for a in ablate):
        raise ValueError('quality must be baseline/full; ablate entries must be stage ids 2a..2f')
    if samples is not None and not 1<=samples<=8192:raise ValueError('samples must be 1..8192')
    if resolution is not None and not (256<=resolution<=4096 and resolution%64==0):raise ValueError('resolution must be a multiple of 64 in 256..4096')
    hdri_record=None
    if lighting=='hdri':
        hdri=Path(hdri or DEFAULT_HDRI).resolve()
        if not hdri.is_file():raise FileNotFoundError(f'HDRI not found: {hdri}')
        if not 0<hdri_strength<=16:raise ValueError('hdri_strength must be in (0, 16]')
        hdri_record=dict(path=str(hdri),sha256=hashlib.sha256(hdri.read_bytes()).hexdigest(),strength=hdri_strength,
                         rotation_deg=hdri_rotation,license='CC0 (Poly Haven, via ref/pixal3d/upstream/assets/hdri/license.txt)')
    candidate, out = Path(candidate).resolve(), Path(out).resolve()
    blender = Path(blender).resolve()
    if not blender.is_file():
        raise FileNotFoundError(f'install the pinned Blender first: {blender}')
    if out.exists() and any(out.iterdir()):
        raise ValueError('use an empty output directory to preserve previous renders')
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 4 * 1024**3:
        raise RuntimeError('offline rendering requires 4 GiB free disk headroom')
    eyes_dir=None if (eyes is None or '2d' in ablate or quality!='full') else Path(eyes).resolve()
    prepare(candidate, out, accessories=accessories, detail_preset=detail_preset,head_fit=head_fit,
            parsing_model=parsing_model,include_hair=include_hair,fit_eyes=fit_eyes,eyes=eyes_dir)
    if quality=='full' and '2c' not in ablate:
        from .skin_templates import build as build_templates
        build_templates(candidate,out,res=2048 if preset=='hero' else 1024)
    if quality=='full' and '2f' not in ablate:
        from .wrinkle_library import augment
        augment(out/'skin_detail.npz')   # appends strain-driven wrinkle channels + drivers (wrinkle_library.json)
    if appearance:
        appearance=Path(appearance).resolve()
        from .observations import sha256
        metadata=json.loads((appearance/'expression_appearance.json').read_text())
        if metadata['candidate_geometry_sha256']!=sha256(candidate/'geometry.npz'):
            raise ValueError('appearance belongs to a different reconstruction')
        for name in ('expression_appearance.npz','expression_appearance.json'):
            shutil.copyfile(appearance/name,out/name)
    res_px,spp=PRESETS[preset]
    request = dict(out=str(out), device=device, preset=preset, gpu_index=gpu_index,lighting=lighting,yaw=yaw,exposure=exposure,sss_weight=sss_weight,
                   camera=camera,pitch=pitch,distance_scale=distance_scale,focal_scale=focal_scale,hdri=hdri_record,
                   samples=samples or spp,resolution=resolution or res_px,quality=quality,ablate=list(ablate),noisy=bool(noisy),
                   chart=bool(chart),displacement_scale=float(displacement_scale),micro_scale=float(micro_scale),save_blend=bool(save_blend),
                   lip_coat_scale=float(lip_coat_scale))
    if motion:
        motion=Path(motion).resolve()
        track=json.loads((motion/'motion.json').read_text())
        from .observations import sha256
        if track['candidate_geometry_sha256']!=sha256(candidate/'geometry.npz'):
            raise ValueError('motion belongs to a different reconstruction')
        if track.get('motion_sha256') and track['motion_sha256']!=sha256(motion/'motion.npz'):
            raise ValueError('motion file hash mismatch')
        if 'skin_min_oriented_area_ratio' not in track:
            raise ValueError('motion lacks the native skin topology gate; refit this legacy track')
        if not 1<=frame<=track['frames']:raise ValueError('render frame outside motion range')
        request.update(motion=str(motion),frame=frame)
    # A measured tooth shade travels with the candidate (cycles_scene.full_mouth_materials reads out/teeth.json).
    if (Path(candidate)/'teeth.json').is_file():
        shutil.copyfile(Path(candidate)/'teeth.json',out/'teeth.json');request['teeth']=str((Path(candidate)/'teeth.json').resolve())
    (out/'request.json').write_text(json.dumps(request, indent=2))
    started = time.monotonic()
    peak = 0
    usage_paths = sorted(Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_used'))
    usage_path = usage_paths[gpu_index] if gpu_index < len(usage_paths) else None
    if device == 'hip' and usage_path is None:
        raise RuntimeError('AMD VRAM monitor unavailable; refusing an unbounded HIP render')
    backend='rocm' if device=='hip' else 'cuda' if device in ('cuda','optix') else 'cpu'
    with gpu.execution(backend, gpu_index):
        session = gpu.device_session(4096, cancel) if backend!='cpu' else nullcontext()
        with session, (out/'cycles.log').open('w') as log:
            env = dict(os.environ, TMPDIR=str(out/'cache'), TEMP=str(out/'cache'), TMP=str(out/'cache'))
            # Blender's HIP may need a different ROCm runtime than the calling (PyTorch) process: on the RX 9070 XT
            # host core-10.0 aborts in rocclr thread creation inside Blender while core-10.1 works.
            if os.environ.get('VHUMAN_BLENDER_LD_LIBRARY_PATH'):
                env['LD_LIBRARY_PATH'] = os.environ['VHUMAN_BLENDER_LD_LIBRARY_PATH']
            (out/'cache').mkdir(exist_ok=True)
            process = subprocess.Popen([str(blender), '-b', '--factory-startup', '--python-exit-code', '1',
                '--python', str(Path(__file__).with_name('cycles_scene.py')), '--', str(out/'request.json')],
                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                next_cuda_poll=0.
                while process.poll() is None:
                    if cancel is not None and cancel.is_set():
                        raise gpu.Cancelled('offline render cancelled')
                    if usage_path is not None and device == 'hip':
                        used = int(usage_path.read_text()) / 1024**2
                        peak = max(peak, used)
                        if used > hard_limit_mib:
                            raise RuntimeError(f'AMD VRAM usage exceeded {hard_limit_mib} MiB')
                    if backend=='cuda' and time.monotonic()>=next_cuda_poll:
                        status=gpu.gpu_status(gpu_index,'cuda')
                        if status is None:raise RuntimeError('NVIDIA VRAM monitor unavailable')
                        used=status['total_mib']-status['free_mib'];peak=max(peak,used)
                        if used>hard_limit_mib:raise RuntimeError(f'NVIDIA VRAM usage exceeded {hard_limit_mib} MiB')
                        next_cuda_poll=time.monotonic()+.5
                    time.sleep(.05)
                if process.returncode:
                    raise RuntimeError(f'Cycles exited {process.returncode}; see {out / "cycles.log"}')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
    # Reopen in a fresh Blender process: live generated buffers can render
    # correctly yet disappear or lose signed values when a scene is saved.
    with (out/'cycles.log').open('a') as log:
        subprocess.run([str(blender),'-b',str(out/'head.blend'),'--python-exit-code','1',
            '--python',str(Path(__file__).with_name('cycles_validate.py')),'--',str(out)],
            stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    validation=json.loads((out/'asset_validation.json').read_text())
    result = json.loads((out/'render_result.json').read_text())
    if not save_blend:   # validated above; drop the ~30 MB scene for bulk renders (request.json regenerates it)
        (out/'head.blend').unlink(missing_ok=True);result['blend']=None
    result['float_map_storage']=validation['storage']
    result['asset_reload_validated']=validation['passed']
    result.update(elapsed_seconds=time.monotonic()-started, peak_device_used_mib=peak, vram_peak_mib=peak,
                  render_seconds=result.get('seconds'), hdri=hdri_record, hdri_sha256=hdri_record['sha256'] if hdri_record else None,
                  quality=quality, ablate=list(ablate),
                  target_12gib_met=peak <= 12288 if backend!='cpu' else None,
                  memory_measurement=('whole-device NVIDIA usage, sampled every 500 ms' if backend=='cuda' else
                    'whole-device AMD sysfs usage, sampled every 50 ms' if backend=='rocm' else 'CPU'),
                  hard_limit_mib=hard_limit_mib)
    (out/'render_result.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate'); parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=('hip','cuda','optix','cpu'), default='hip')
    parser.add_argument('--head-fit');parser.add_argument('--parsing-model')
    parser.add_argument('--no-hair',dest='include_hair',action='store_false')
    parser.add_argument('--fit-eyes',action='store_true')
    parser.add_argument('--preset', choices=tuple(PRESETS), default='draft')
    parser.add_argument('--samples',type=int);parser.add_argument('--resolution',type=int)
    parser.add_argument('--camera',default='portrait',help='render_cameras preset name')
    parser.add_argument('--pitch',type=float,default=0.);parser.add_argument('--distance-scale',type=float);parser.add_argument('--focal-scale',type=float)
    parser.add_argument('--hdri',type=Path,help='environment EXR for --lighting hdri (default: repo interior.exr)')
    parser.add_argument('--hdri-strength',type=float,default=1.);parser.add_argument('--hdri-rotation',type=float,default=0.)
    parser.add_argument('--quality',choices=QUALITY,default='baseline');parser.add_argument('--ablate',nargs='*',default=())
    parser.add_argument('--no-noisy',dest='noisy',action='store_false',help='skip the un-denoised beauty output')
    parser.add_argument('--eyes',type=Path,help='eye_personalize output directory (quality full only)')
    parser.add_argument('--chart',action='store_true',help='place a ColorChecker in front of the face (color_chart calibration)')
    parser.add_argument('--accessories', choices=('keep','omit'), default='keep')
    parser.add_argument('--detail-preset', choices=('source','mature'), default='mature')
    parser.add_argument('--blender', type=Path, default=DEFAULT_BLENDER)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--motion',type=Path,help='native GNM motion directory')
    parser.add_argument('--frame',type=int,default=1)
    parser.add_argument('--appearance',type=Path,help='gated expression appearance directory')
    parser.add_argument('--lighting',choices=LIGHTING,default='studio')
    parser.add_argument('--yaw',type=float,default=0)
    parser.add_argument('--exposure',type=float,default=-1.5,help='display exposure in EV')
    parser.add_argument('--sss-weight',type=float,default=.08,help='authored subsurface weight')
    args = parser.parse_args()
    print(json.dumps(render(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
