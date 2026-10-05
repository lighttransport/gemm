"""Generate and measure expression candidates with Wan, H3 or HV1.5 ROCm.

Candidates retain measured controls and shading-derived appearance details.
They never overwrite a fitted rig or assert identity review automatically.
"""
import argparse
import json
from pathlib import Path
import threading
import uuid
import numpy as np
from PIL import Image, ImageOps
from .. import gpu
from ..face_assets import asset_path, sha256
from ..face_parsing import FaceParser
from ..native_landmarks import FaceLandmarker
from ..video_backend import select
from .exprdata import EXPRESSIONS, KEEP, consistent_flow
from .wrinkles import shading_detail


def capture(clip, reference, name, out, *, cancel=None, mapping=None):
    import cv2
    from .gnm_expression import evaluate
    requested = EXPRESSIONS[name][1]
    tracker = FaceLandmarker(asset_path('mediapipe'))
    video = cv2.VideoCapture(str(clip))
    best, observations = None, []
    fps = video.get(cv2.CAP_PROP_FPS)
    if not video.isOpened() or not np.isfinite(fps) or fps <= 0:
        tracker.close(); video.release()
        raise ValueError('cannot decode expression clip')
    try:
        while True:
            if cancel and cancel.is_set():
                raise gpu.Cancelled('cancelled')
            ok, bgr = video.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            faces = tracker.detect(rgb)
            record = {'frame': len(observations), 'seconds': len(observations)/fps,
                      'visible': len(faces) == 1}
            if len(faces) == 1:
                controls = faces[0]['blendshapes']
                score = sum(controls.get(k, 0)*v for k, v in requested.items())/sum(requested.values())
                record.update(controls=controls, response=score)
                if best is None or score > best[0]:
                    best = score, rgb.copy(), record
            observations.append(record)
    finally:
        tracker.close(); video.release()
    if best is None or best[0] < .15:
        raise ValueError('no visible expression with measured response >= 0.15')
    score, rgb, selected = best
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    neutral = np.asarray(Image.open(reference).convert('RGB').resize((rgb.shape[1], rgb.shape[0])))
    parser = FaceParser()
    labels = parser(neutral)
    expression_labels = parser(rgb)
    F, confidence = consistent_flow(neutral, rgb)
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]].astype(np.float32)
    skin = np.isin(labels, [1, 10]).astype(np.float32)
    skin *= cv2.remap(np.isin(expression_labels, [1, 10]).astype(np.float32),
                      xx+F[..., 0], yy+F[..., 1], cv2.INTER_NEAREST)
    detail, mask = shading_detail(neutral, rgb, skin)
    registered = cv2.remap(rgb, xx+F[..., 0], yy+F[..., 1], cv2.INTER_LINEAR)
    Image.fromarray(rgb).save(out / 'expression.png')
    Image.fromarray(labels).save(out / 'parsing.png')
    Image.fromarray(registered).save(out / 'appearance.png')
    np.savez_compressed(out / 'detail.npz', shading_detail=detail,
                        confidence=confidence*mask, flow=F)
    result = {'file': 'expression.png', 'controls': selected['controls'],
              'frame_sha256': sha256(out/'expression.png'),
              'requested_controls': requested, 'wrinkle_group': EXPRESSIONS[name][2],
              'selected_frame': selected, 'observations': observations,
              'response': score, 'clip_sha256': sha256(clip),
              'texture': {'appearance': 'appearance.png', 'detail': 'detail.npz',
                          'space': 'neutral portrait pixels', 'drivers': requested,
                          'interpretation': 'registered lit appearance and shading-derived detail; not albedo or measured displacement'}}
    if mapping:
        result['gnm_coefficients'] = evaluate(mapping, selected['controls']).tolist()
    (out / 'capture.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def generate(*, portrait, out, backend='wan', model=None, preset='fast5', frames=None,
             names=None, seed=42, cancel=None, progress=None, mapping=None):
    names = list(names or EXPRESSIONS)
    if not names or len(set(names)) != len(names) or any(n not in EXPRESSIONS for n in names):
        raise ValueError('provide unique known expression names')
    selected = select(backend)
    if backend not in ('wan', 'h3', 'h3-fl2va', 'hv15-rocm') or preset not in selected.presets:
        raise ValueError('unsupported expression backend/preset')
    frames = frames if frames is not None else (9 if backend == 'wan' else 22 if backend.startswith('h3') else 81)
    if frames not in selected.frames:
        raise ValueError('unsupported expression frame count')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cancel = cancel or threading.Event()
    width, height = (480, 848) if backend == 'hv15-rocm' else (480, 832)
    rgba = ImageOps.exif_transpose(Image.open(portrait)).convert('RGBA')
    background = Image.new('RGBA', rgba.size, (96, 96, 96, 255))
    background.alpha_composite(rgba)
    rgb = background.convert('RGB')
    scale = max(width/rgb.width, height/rgb.height)
    resized = rgb.resize((round(rgb.width*scale), round(rgb.height*scale)), Image.Resampling.LANCZOS)
    left, top = (resized.width-width)//2, (resized.height-height)//2
    resized.crop((left, top, left+width, top+height)).save(out/'ref.png')
    manifest = {'schema': 'vhuman.video_expressions.v1', 'state': 'generating',
                'backend': backend, 'identity_conditioned': selected.identity_conditioned,
                'portrait_sha256': sha256(portrait), 'ref': 'ref.png', 'size': [width, height],
                'expressions': {}, 'review': {k: False for k in ('identity','expression','camera','visibility','artifacts')}}
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    try:
        with gpu.execution('rocm', 0):
            for i, name in enumerate(names):
                folder = out/name
                folder.mkdir()
                result = selected.generate(model=model, image=out/'ref.png', out=folder/'video',
                    prompt=KEEP+' '+EXPRESSIONS[name][0]+' Hold the expression, then relax.',
                    preset=preset, frames=frames, seed=seed+i, device=0,
                    allow_experimental=True, cancel=cancel,
                    progress=(lambda s, n: progress((i+s/max(n, 1))/len(names),
                                                    f'{name}: step {s}/{n}')) if progress else None)
                spec = capture(folder/'video/clip.mp4', out/'ref.png', name, folder,
                               cancel=cancel, mapping=mapping)
                spec['file'] = name+'/expression.png'
                for key in ('appearance', 'detail'):
                    spec['texture'][key] = name+'/'+spec['texture'][key]
                spec['generation'] = result
                manifest['expressions'][name] = spec
                save()
        manifest['state'] = 'candidate'
    except BaseException as error:
        manifest.update(state='cancelled' if cancel.is_set() else 'failed', error=str(error))
        save()
        raise
    save()
    return manifest


def expressions_job(service, request, progress, cancel):
    portrait = service.head_file(request.get('head_id'), 'portrait.png')
    rig = portrait.parent/'rig'/'rig.json'
    mapping = (json.loads(rig.read_text()).get('face_model_source', {}).get('expression_mapping')
               if rig.is_file() else None)
    return generate(portrait=portrait, out=portrait.parent/'rig'/'expression_candidates'/uuid.uuid4().hex,
                    backend=request.get('video_backend', 'wan'), model=request.get('model'),
                    preset=request.get('preset', 'fast5'), frames=request.get('frames'),
                    names=request.get('names'), seed=request.get('seed', 42), cancel=cancel, progress=progress,
                    mapping=mapping)


def export_reviewed(candidate, portrait, out):
    """Restore reviewed captures to portrait coordinates for the existing rig builder."""
    candidate, out = Path(candidate), Path(out)
    manifest = json.loads((candidate/'manifest.json').read_text())
    if manifest.get('state') != 'candidate' or not all(manifest.get('review', {}).get(k) is True
        for k in ('identity', 'expression', 'camera', 'visibility', 'artifacts')):
        raise ValueError('candidate requires identity/expression/camera/visibility/artifact review')
    if sha256(portrait) != manifest['portrait_sha256']:
        raise ValueError('candidate portrait hash mismatch')
    neutral = ImageOps.exif_transpose(Image.open(portrait)).convert('RGBA')
    bg = Image.new('RGBA', neutral.size, (96, 96, 96, 255)); bg.alpha_composite(neutral)
    neutral = bg.convert('RGB')
    width, height = manifest['size']
    scale = max(width/neutral.width, height/neutral.height)
    resized = (round(neutral.width*scale), round(neutral.height*scale))
    left, top = (resized[0]-width)//2, (resized[1]-height)//2
    out.mkdir(parents=True, exist_ok=False)
    neutral.save(out/'ref.png')
    exported = {'ref': 'ref.png', 'size': list(neutral.size), 'expressions': {},
                'candidate_manifest_sha256': sha256(candidate/'manifest.json'), 'review': manifest['review']}
    for name, spec in manifest['expressions'].items():
        if name not in EXPRESSIONS or sha256(candidate/name/'video/clip.mp4') != spec['clip_sha256']:
            raise ValueError('invalid expression provenance')
        if spec['file'] != name+'/expression.png' or sha256(candidate/spec['file']) != spec['frame_sha256']:
            raise ValueError('selected expression frame was modified')
        image = neutral.resize(resized, Image.Resampling.LANCZOS)
        image.paste(Image.open(candidate/spec['file']).convert('RGB'), (left, top))
        image.resize(neutral.size, Image.Resampling.LANCZOS).save(out/f'{name}.png')
        exported['expressions'][name] = {'file': f'{name}.png', 'controls': {
            k: spec['controls'][k] for k in spec['requested_controls'] if spec['controls'].get(k, 0) >= .05},
            'wrinkle_group': spec['wrinkle_group'], 'selected_frame': spec['selected_frame']['frame']}
    (out/'manifest.json').write_text(json.dumps(exported, indent=2)+'\n')
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portrait', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', choices=('wan','h3','h3-fl2va','hv15-rocm'), default='wan')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--preset', choices=('fast5','fast12','quality'), default='fast5')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--names', nargs='+', choices=tuple(EXPRESSIONS), default=['smile'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gnm-rig', type=Path, help='GNM rig.json containing geometric expression mapping')
    parser.add_argument('--reviewed-candidate', type=Path, help='export reviewed captures instead of generating')
    args = vars(parser.parse_args())
    rig = args.pop('gnm_rig')
    candidate = args.pop('reviewed_candidate')
    if candidate:
        export_reviewed(candidate, args['portrait'], args['out'])
        return
    mapping = json.loads(rig.read_text())['face_model_source']['expression_mapping'] if rig else None
    generate(**args, mapping=mapping, progress=lambda p, msg: print(f'{p:.1%} {msg}', flush=True))


if __name__ == '__main__':
    main()
