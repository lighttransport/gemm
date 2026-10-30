"""Reviewed, revisioned synthetic portrait inputs, independent of reconstructed heads."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import shutil
import time
import uuid
import zipfile

from PIL import Image

from . import gpu, qwen, portrait_generators as generators
from .service import ServiceError

FORMAT = 'vhuman.portrait_session.v1'
ANGLES = ((45, 'Left three-quarter'), (-45, 'Right three-quarter'),
          (90, 'Left profile'), (-90, 'Right profile'),
          (40, 'Left 40-degree view'), (-40, 'Right 40-degree view'))
# Joint sheets regenerate one of these four-angle sets together (2x2 panels in this order).
SHEET_SETS = ([45, -45, 90, -90], [40, -40, 90, -90])
# Apache-2.0 route: Qwen-Image-Edit-2511 (GGUF) with a camera-angle LoRA. Keys are request 'angle_lora' values.
ANGLE_LORAS = {
    'fal-2511': ('/mnt/nvme01/models/loras/qie2511-multiple-angles.safetensors:1.0:angles2511',),
    'dx-2509': ('/mnt/nvme01/models/loras/qie2509-multiple-angles.safetensors:1.0:angles2509',),
    'none': (),
}
EDIT_MIN_FREE_MIB = 4096
EDIT_NEGATIVE = ('blurry, illustration, plastic skin, beauty retouching, dramatic lighting, open mouth, '
                 'glasses, jewellery, hat, text, watermark, different person')
NEGATIVE = ('blur, illustration, plastic skin, beauty retouching, dramatic lighting, '
            'open mouth, glasses, jewellery, text, watermark')
PORTRAIT = ('Photorealistic studio head-and-neck portrait of {subject}. Camera centred at eye level, '
            'straight frontal view, no head roll, neutral relaxed expression, mouth closed, '
            'both eyes open looking straight ahead. Preserve natural facial asymmetry and skin texture. '
            'Entire head and upper neck visible with a little margin, ears visible where possible. '
            'Soft even diffuse neutral lighting, plain light grey background. No glasses, jewellery, '
            'makeup, beauty retouching, text or watermark.')
PRESERVE = ('Exactly the same person as the reference portrait: preserve facial identity, natural '
            'asymmetry, age, skin colour, eye colour, hairline, hairstyle and facial details. '
            'Head and upper neck only, neutral expression, closed mouth, open eyes, level camera. '
            'Same soft even lighting and light grey background. No beautification, text or labels.')


def sheet_prompt(angles):
    """2x2 joint sheet prompt for one SHEET_SETS entry (panel order: top-left, top-right, bottom-left, bottom-right)."""
    quarter = 40 if 40 in angles else 45
    return ('A precise 2 by 2 portrait reference sheet. Four separate panels, one head per panel, '
            f'no overlap, equal framing and size. Top left: the person\'s left three-quarter view turned {quarter} degrees. '
            f'Top right: right three-quarter view turned {quarter} degrees. Bottom left: left profile facing image left. '
            'Bottom right: right profile facing image right. '+PRESERVE)


def individual_prompt(angle):
    """Qwen-Image 2.1 reference-edit prompt for one ANGLES entry."""
    prompt = f'{dict(ANGLES)[angle]} of the person in the reference portrait. '+PRESERVE
    if abs(angle) == 90:
        prompt += ' Exact side profile; nose points to image '+('left.' if angle > 0 else 'right.')
    return prompt


def angle_prompt(angle, lora='fal-2511'):
    """Camera-angle instruction for Edit-2511. Positive yaw matches ANGLES ('Left ...': the subject's left side
    turns toward the camera, nose points to image left as in the Qwen 2.1 prompts)."""
    side = 'left' if angle > 0 else 'right'
    if lora == 'fal-2511':
        # The LoRA names the camera position from the viewer's side: 'front-left' puts the nose toward image
        # right (verified 2026-10-31), i.e. negative yaw in ANGLES, so the descriptor side is mirrored here.
        cam = 'right' if angle > 0 else 'left'
        azimuth = f'{cam} side view' if abs(angle) >= 90 else f'front-{cam} quarter view'
        return f'<sks> {azimuth} eye-level shot medium shot'
    if lora == 'dx-2509':
        zh = '左' if angle > 0 else '右'
        return f'将镜头向{zh}旋转{abs(angle)}度 (Rotate the camera {abs(angle)} degrees to the {side}, same eye level).'
    label = dict(ANGLES)[angle]
    return f'{label} of the person in the reference portrait, camera rotated {abs(angle)} degrees to the {side}.'


def _id(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{24}', value):
        raise ServiceError('valid portrait session/revision ID required')
    return value


def _root(service, sid):
    base = (service.work/'portrait_sessions').resolve()
    root = base/_id(sid)
    if not root.resolve().is_relative_to(base):
        raise ServiceError('no such portrait session')
    return root


def _save(root, doc):
    doc['updated'] = time.time()
    partial = root/'session.json.partial'
    partial.write_text(json.dumps(doc, indent=2)+'\n')
    partial.replace(root/'session.json')


def load(service, sid):
    try:
        return json.loads((_root(service, sid)/'session.json').read_text())
    except (OSError, ValueError) as exc:
        raise ServiceError('no such portrait session') from exc


def public(doc):
    doc = copy.deepcopy(doc)
    base = '/v1/portrait-sessions/'+doc['id']
    for row in doc['portraits']+[v for run in doc['view_runs'] for v in run['views']]:
        row['url'] = base+'/files/'+row['file']
    for run in doc['view_runs']:
        if run.get('sheet'):
            run['sheet_url'] = base+'/files/'+run['sheet']
        run['stale'] = run['revision'] != doc['active_revision']
    doc['bundle_url'] = base+'/bundle.zip'
    return doc


def listing(service):
    root = service.work/'portrait_sessions'
    docs = []
    for path in root.glob('*/session.json'):
        if re.fullmatch('[0-9a-f]{24}', path.parent.name):
            docs.append(public(load(service, path.parent.name)))
    return sorted(docs, key=lambda d: d['updated'], reverse=True)


def artifact(service, sid, name):
    doc = load(service, sid)
    allowed = {p['file'] for p in doc['portraits']}
    for run in doc['view_runs']:
        allowed.update(v['file'] for v in run['views'])
        if run.get('sheet'):
            allowed.add(run['sheet'])
    root = _root(service, sid)
    path = root/name
    if name not in allowed or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ServiceError('no such portrait file')
    return path


def bundle(service, sid):
    doc = load(service, sid)
    # The export is a self-contained reviewed input set, not an ambiguous mix
    # of every older candidate for the same angle.
    doc['portraits'] = [p for p in doc['portraits'] if p['id'] == doc['active_revision']]
    selected = set(doc['selection']['view_ids'])
    doc['view_runs'] = [dict(r, views=[v for v in r['views'] if v['id'] in selected])
                        for r in doc['view_runs'] if r['revision'] == doc['active_revision']
                        and any(v['id'] in selected for v in r['views'])]
    files = [p['file'] for p in doc['portraits']]
    for run in doc['view_runs']:
        files.extend(v['file'] for v in run['views'])
        if run.get('sheet'):
            files.append(run['sheet'])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('session.json', json.dumps(doc, indent=2))
        for name in files:
            archive.write(artifact(service, sid, name), name)
    return buf.getvalue()


def _selected(doc, request):
    rev = request.get('revision')
    if rev != doc['active_revision']:
        raise ServiceError('portrait revision changed; review the current portrait and views')
    ids = request.get('view_ids', [])
    if not isinstance(ids, list) or len(ids) > 4 or len(set(map(str, ids))) != len(ids):
        raise ServiceError('select at most one view per angle')
    views = {v['id']: v for r in doc['view_runs'] if r['revision'] == rev for v in r['views']}
    if any(not isinstance(i, str) or i not in views for i in ids):
        raise ServiceError('selected view does not belong to this portrait revision')
    selected = [views[i] for i in ids]
    if len({v['yaw_deg'] for v in selected}) != len(selected):
        raise ServiceError('select at most one view per angle')
    return rev, selected


def review(service, sid, request):
    with service._lock('portrait-'+_id(sid)):
        doc = load(service, sid)
        rev, views = _selected(doc, request)
        if type(request.get('portrait_reviewed')) is not bool:
            raise ServiceError('portrait_reviewed must be a boolean')
        doc['selection'] = dict(revision=rev, portrait_reviewed=request['portrait_reviewed'],
                                view_ids=[v['id'] for v in views])
        _save(_root(service, sid), doc)
        return public(doc)


def snapshot(service, request):
    """Copy reviewed inputs to an immutable, server-resolved reconstruction bundle."""
    sid = _id(request.get('session_id'))
    with service._lock('portrait-'+sid):
        doc = load(service, sid)
        rev, views = _selected(doc, request)
        selection = doc['selection']
        if not selection.get('portrait_reviewed') or selection.get('revision') != rev:
            raise ServiceError('approve the front portrait before reconstruction')
        if any(v['id'] not in selection['view_ids'] for v in views):
            raise ServiceError('review selected views before reconstruction')
        portrait = next(p for p in doc['portraits'] if p['id'] == rev)
        run_fields = ('id', 'method', 'reference_sha256', 'steps', 'cfg', 'negative_prompt', 'backend',
                      'portrait_model', 'generation',
                      'preset', 'attention', 'condition_resolution', 'model', 'license', 'synthetic')
        origins = {v['id']: {k: run[k] for k in run_fields if k in run}
                   for run in doc['view_runs'] for v in run['views']}
        target = _root(service, sid)/'snapshots'/uuid.uuid4().hex
        target.mkdir(parents=True)
        inputs = []
        try:
            for i, row in enumerate([dict(portrait, yaw_deg=0)]+views):
                source = artifact(service, sid, row['file'])
                if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
                    raise ServiceError('portrait input checksum mismatch')
                name = f'view_{i}.png'
                shutil.copyfile(source, target/name)
                copied = dict(row, file=name)
                if i:
                    copied['generation'] = origins[row['id']]
                inputs.append(copied)
            data = dict(format='vhuman.generated_face_inputs.v1', synthetic=True,
                        session_id=sid, revision=rev, subject=doc['subject'], views=inputs,
                        camera_parameters='requested angles only; cameras must be estimated')
            (target/'inputs.json').write_text(json.dumps(data, indent=2))
            return dict(portrait=str(target/'view_0.png'), generated_input=str(target/'inputs.json'))
        except BaseException:
            shutil.rmtree(target, ignore_errors=True)
            raise


def _parameters(request, default_steps=24):
    seed, steps = request.get('seed', 11), request.get('steps', default_steps)
    if type(seed) is not int or not 0 <= seed < 2**32 or type(steps) is not int or not 1 <= steps <= 100:
        raise ServiceError('seed must be a uint32; steps must be 1..100')
    preset = request.get('qwen_preset', 'low8')
    if preset not in ('low8', 'fast12'):
        raise ServiceError('qwen_preset must be low8 or fast12')
    return seed, steps, preset


def _record(path, root, **kwargs):
    with Image.open(path) as image:
        size = list(image.size)
    return dict(file=str(path.relative_to(root)), size=size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **kwargs)


def _check(cancel):
    if cancel.is_set():
        raise gpu.Cancelled('portrait generation cancelled')


def generate_job(service, request, progress, cancel, *, python=None, mock=False,
                 klein3_model=None, klein3_python=None):
    model = generators.validate(request.get('portrait_model', 'qwen-image21'))
    seed, steps, preset = _parameters(request, 16 if model == 'klein3' else 24)
    subject = ' '.join(str(request.get('subject', '')).split())
    if not subject or len(subject) > 400:
        raise ServiceError('describe the person in 1..400 characters')
    sid = _id(request['session_id']) if request.get('session_id') else uuid.uuid4().hex[:24]
    root = _root(service, sid)
    with service._lock('portrait-'+sid):
        if request.get('session_id'):
            load(service, sid)
        root.mkdir(parents=True, exist_ok=True)
        revision = uuid.uuid4().hex[:24]
        staging = root/'portraits'/('.'+revision+'.partial')
        staging.mkdir(parents=True)
        backend = None
        started = time.perf_counter()
        try:
            _check(cancel)
            if not mock:
                ready = generators.availability(python=python, klein3_model=klein3_model,
                                                klein3_python=klein3_python)[model]
                if not ready['available']:
                    raise ServiceError(ready['reason'])
            minimum = generators.KLEIN_MIN_FREE_MIB if model == 'klein3' else (
                gpu.QWEN_LOW8_MIN_FREE_MIB if preset == 'low8' else gpu.QWEN_MIN_FREE_MIB)
            progress(.02, 'waiting for image generation device')
            generation = {}
            with gpu.device_session(minimum, cancel, check_memory=not mock):
                prompt = PORTRAIT.format(subject=subject)
                progress(.1, ('Klein3' if model == 'klein3' else 'Qwen-Image 2.1')+': front portrait')
                if mock:
                    from .head.pipeline import mock_portrait
                    mock_portrait(staging/'portrait.png', seed)
                elif model == 'klein3':
                    generation = generators.generate_klein(service, staging/'portrait.png', prompt, seed,
                        steps, cancel, progress, model=klein3_model, python=klein3_python)
                else:
                    backend = qwen.make_backend(python, preset=preset, cancel=cancel)
                    qwen._import_qimg21()
                    from qimg21_i23d.backends import GenRequest
                    result = backend.generate(GenRequest(prompt, staging/'portrait.png', width=1024, height=1024,
                                                         steps=steps, seed=seed, negative_prompt=NEGATIVE))
                    generation = dict(details=result.details,
                        checkpoint=generators.receipt(gpu.model_path('qimg-21-fast/int8-smooth-a0.6/blocks.bin')),
                        package_manifest=generators.receipt(gpu.model_path('qimg-21-fast/int8-smooth-a0.6/manifest.json')))
                if backend:
                    backend.close()
                    backend = None
            _check(cancel)
            final = root/'portraits'/revision
            staging.replace(final)
            row = _record(final/'portrait.png', root, id=revision, seed=seed, steps=steps,
                          prompt=prompt, portrait_model=model, generation=generation,
                          negative_prompt=NEGATIVE if model == 'qwen-image21' else None,
                          cfg=4. if model == 'qwen-image21' else None,
                          preset=preset if model == 'qwen-image21' else 'fp8',
                          attention='mock' if mock else 'native' if model == 'klein3' else
                                    'flash' if gpu.backend() == 'cuda' else 'sage',
                          backend='mock' if mock else gpu.backend(),
                          license=generators.KLEIN_LICENSE if model == 'klein3' else qwen.LICENSE,
                          synthetic=True, model='FLUX 3 Image Klein' if model == 'klein3' else 'Qwen-Image 2.1',
                          created=time.time(), seconds=round(time.perf_counter()-started, 2))
            doc = load(service, sid) if (root/'session.json').exists() else dict(
                format=FORMAT, id=sid, created=time.time(), portraits=[], view_runs=[], reconstructions=[])
            doc.update(subject=subject, active_revision=revision,
                       selection=dict(revision=revision, portrait_reviewed=False, view_ids=[]))
            doc['portraits'].append(row)
            _save(root, doc)
            return dict(session_id=sid, revision=revision, session=public(doc))
        except InterruptedError:
            _check(cancel)
            raise
        finally:
            if backend:
                backend.close()
            shutil.rmtree(staging, ignore_errors=True)


def views_job(service, request, progress, cancel, *, python=None, mock=False):
    if request.get('portrait_model', 'qwen-image21') != 'qwen-image21':
        raise ServiceError('side views require Qwen-Image 2.1')
    seed, steps, preset = _parameters(request)
    sid = _id(request.get('session_id'))
    method = request.get('method', 'individual')
    if method not in ('individual', 'sheet', 'edit2511'):
        raise ServiceError('view method must be individual, sheet or edit2511')
    angles = request.get('angles', SHEET_SETS[0])
    if not isinstance(angles, list) or not angles or any(type(a) is not int or a not in dict(ANGLES) for a in angles) or len(set(angles)) != len(angles):
        raise ServiceError('angles must be distinct members of -90, -45, -40, 40, 45, 90')
    if method == 'sheet' and angles not in SHEET_SETS:
        raise ServiceError('a joint sheet regenerates one four-angle set together: 45/-45/90/-90 or 40/-40/90/-90')
    held_out = request.get('held_out', [])
    if not isinstance(held_out, list) or any(a not in angles for a in held_out):
        raise ServiceError('held_out must list requested angles that are evaluation-only')
    angle_lora = request.get('angle_lora', 'fal-2511')
    if angle_lora not in ANGLE_LORAS:
        raise ServiceError('angle_lora must be one of ' + ', '.join(ANGLE_LORAS))
    offload_blocks = request.get('offload_blocks', 1)
    if type(offload_blocks) is not int or offload_blocks < 0:
        raise ServiceError('offload_blocks must be a non-negative integer')
    with service._lock('portrait-'+sid):
        doc = load(service, sid)
        revision, _ = _selected(doc, dict(request, view_ids=[]))
        if not doc['selection']['portrait_reviewed']:
            raise ServiceError('approve the front portrait before generating views')
        front = next(p for p in doc['portraits'] if p['id'] == revision)
        reference = artifact(service, sid, front['file'])
        root = _root(service, sid)
        rid = uuid.uuid4().hex[:24]
        staging = root/'views'/('.'+rid+'.partial')
        staging.mkdir(parents=True)
        backend = None
        started = time.perf_counter()
        records = []
        try:
            _check(cancel)
            if not mock and method != 'edit2511':
                ready = qwen.availability(python=python)
                if not ready['available']:
                    raise ServiceError(ready['reason'])
            progress(.02, 'waiting for image generation device')
            minimum = EDIT_MIN_FREE_MIB if method == 'edit2511' else (
                gpu.QWEN_LOW8_MIN_FREE_MIB if preset == 'low8' else gpu.QWEN_MIN_FREE_MIB)
            editor_metadata = {}
            with gpu.device_session(minimum, cancel, check_memory=not mock):
                if not mock and method != 'edit2511':
                    backend = qwen.make_backend(python, preset=preset, cancel=cancel)
                    from qimg21_i23d.backends import GenRequest
                if method == 'edit2511':
                    from .reconstruction import qwen_edit_backend
                    editor = None
                    if not mock:
                        progress(.05, 'Qwen-Image-Edit-2511: loading GGUF editor' + (f' + LoRA {angle_lora}' if ANGLE_LORAS[angle_lora] else ''))
                        editor = qwen_edit_backend.make_editor('gguf', offload_blocks=offload_blocks,
                                                               loras=ANGLE_LORAS[angle_lora] or None)
                        editor_metadata = editor.metadata
                    for i, angle in enumerate(angles):
                        _check(cancel)
                        label = dict(ANGLES)[angle]
                        prompt = angle_prompt(angle, angle_lora)
                        if request.get('preserve', True):
                            prompt += ' ' + PRESERVE
                        progress(.1+.8*i/len(angles), 'Qwen-Image-Edit-2511: '+label.lower())
                        view_seed = (seed+[a for a, _ in ANGLES].index(angle)) % 2**32
                        metrics = None
                        if mock:
                            shutil.copyfile(reference, staging/f'{angle}.png')
                        else:
                            with Image.open(reference) as im:
                                result, _ = editor([im.convert('RGB')], prompt, steps=steps, seed=view_seed, cfg=4.,
                                                   size=1024, negative=EDIT_NEGATIVE)
                            Image.fromarray(result).save(staging/f'{angle}.png')
                            metrics = {k: v for k, v in editor.last_metrics.items() if k != 'prompt_image_sizes'}
                        records.append(dict(id=uuid.uuid4().hex[:24], yaw_deg=angle, label=label,
                                            prompt=prompt, seed=view_seed, angle_lora=angle_lora, editor_metrics=metrics))
                    del editor
                elif method == 'sheet':
                    prompt = sheet_prompt(angles)
                    progress(.1, 'Qwen-Image 2.1: joint four-view sheet')
                    if mock:
                        sheet = Image.new('RGBA', (1024, 1024))
                        with Image.open(reference) as im:
                            for i in range(4):
                                sheet.paste(im.resize((512, 512)), ((i % 2)*512, (i//2)*512))
                        sheet.save(staging/'sheet.png')
                    else:
                        backend.generate(GenRequest(prompt, staging/'sheet.png', width=1024, height=1024,
                                                    steps=steps, seed=seed, references=(reference,), negative_prompt=NEGATIVE))
                    _check(cancel)
                    with Image.open(staging/'sheet.png') as sheet:
                        if sheet.size != (1024, 1024):
                            raise ServiceError('joint sheet has unexpected dimensions')
                        for i, angle in enumerate(angles):
                            label = dict(ANGLES)[angle]
                            box = ((i % 2)*512, (i//2)*512, (i % 2+1)*512, (i//2+1)*512)
                            sheet.crop(box).save(staging/f'{angle}.png')
                            records.append(dict(id=uuid.uuid4().hex[:24], yaw_deg=angle, label=label,
                                                prompt=prompt, seed=seed, sheet_box=list(box)))
                else:
                    for i, angle in enumerate(angles):
                        _check(cancel)
                        label = dict(ANGLES)[angle]
                        prompt = individual_prompt(angle)
                        progress(.1+.8*i/len(angles), 'Qwen-Image 2.1: '+label.lower())
                        view_seed = (seed+[a for a, _ in ANGLES].index(angle)) % 2**32
                        if mock:
                            shutil.copyfile(reference, staging/f'{angle}.png')
                        else:
                            backend.generate(GenRequest(prompt, staging/f'{angle}.png', width=1024, height=1024,
                                                        steps=steps, seed=view_seed, references=(reference,), negative_prompt=NEGATIVE))
                        records.append(dict(id=uuid.uuid4().hex[:24], yaw_deg=angle, label=label,
                                            prompt=prompt, seed=view_seed))
                if backend:
                    backend.close()
                    backend = None
            _check(cancel)
            final = root/'views'/rid
            staging.replace(final)
            for r in records:
                r['held_out'] = r['yaw_deg'] in held_out
            records = [_record(final/f"{r['yaw_deg']}.png", root, **r) for r in records]
            edit = method == 'edit2511'
            run = dict(id=rid, revision=revision, reference_sha256=front['sha256'], method=method,
                       portrait_model='qwen-image-edit-2511' if edit else 'qwen-image21',
                       backend='mock' if mock else gpu.backend(),
                       generation={} if mock else editor_metadata if edit else dict(
                           checkpoint=generators.receipt(gpu.model_path('qimg-21-fast/int8-smooth-a0.6/blocks.bin')),
                           package_manifest=generators.receipt(gpu.model_path('qimg-21-fast/int8-smooth-a0.6/manifest.json'))),
                       steps=steps, cfg=4., negative_prompt=EDIT_NEGATIVE if edit else NEGATIVE,
                       preset='gguf-q4_k_m' if edit else preset, views=records, held_out=held_out,
                       attention='mock' if mock else 'sdpa' if edit else 'flash' if gpu.backend() == 'cuda' else 'sage',
                       condition_resolution=1024,
                       model='Qwen-Image-Edit-2511' if edit else 'Qwen-Image 2.1',
                       license='apache-2.0' if edit else qwen.LICENSE,
                       angle_lora=angle_lora if edit else None,
                       sheet=str((final/'sheet.png').relative_to(root)) if method == 'sheet' else None,
                       seconds=round(time.perf_counter()-started, 2), synthetic=True,
                       camera_parameters='requested angles, not calibrated cameras')
            doc['view_runs'].append(run)
            _save(root, doc)
            return dict(session_id=sid, revision=revision, session=public(doc))
        except InterruptedError:
            _check(cancel)
            raise
        finally:
            if backend:
                backend.close()
            shutil.rmtree(staging, ignore_errors=True)
