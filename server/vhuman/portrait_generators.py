"""Front portrait generators; Qwen reference editing remains a separate stage."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image

from server.python_env import environment_path
from . import gpu, qwen
from .service import ROOT, ServiceError

MODELS = ('qwen-image21', 'klein3')
KLEIN_MODEL = Path('/mnt/nvme02/model/flux3-klein')
KLEIN_PYTHON = environment_path(ROOT/'tmp/flux3-text-venv')/'bin/python'
KLEIN_MIN_FREE_MIB = 9216
KLEIN_LICENSE = 'FLUX Kommunity License v.1.0 (model); see checkpoint LICENSE.md and NOTICE.md'


def validate(model):
    if model not in MODELS:
        raise ServiceError('portrait_model must be qwen-image21 or klein3')
    return model


@lru_cache(maxsize=8)
def _text_runtime(python, environment_mtime):
    code = ('from importlib.metadata import version; import json; '
            'print(json.dumps({n:version(n) for n in ("torch","transformers","safetensors")}))')
    result = subprocess.run([python, '-B', '-c', code], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise ValueError('Klein3 CPU encoder dependencies missing: '+result.stderr[-500:])
    return json.loads(result.stdout)


def text_runtime(python):
    python = Path(python).absolute()
    # Package installation changes this directory even when the interpreter is a symlink.
    stamps = [(p.stat().st_mtime_ns) for p in (python.parent.parent/'lib').glob('python*/site-packages')]
    return _text_runtime(str(python), tuple(stamps))


def availability(*, python=None, mock=False, klein3_model=None, klein3_python=None):
    q = dict(qwen.availability(python=python, mock=mock), label='Qwen-Image 2.1', default_steps=24,
             supports_views=True, min_free_mib=gpu.QWEN_LOW8_MIN_FREE_MIB)
    k = dict(available=True, label='Klein3', backend='mock' if mock else 'native-rocm',
             default_steps=16, supports_views=False, min_free_mib=KLEIN_MIN_FREE_MIB)
    model = Path(klein3_model or KLEIN_MODEL)
    py = Path(klein3_python or KLEIN_PYTHON)
    required = ['variants/fp8r/model.safetensors', 'vae/diffusion_pytorch_model.safetensors',
                'text_encoder/config.json', 'text_encoder/tokenizer.json', 'text_encoder/chat_template.json',
                'text_encoder/model.safetensors.index.json', 'LICENSE.md', 'NOTICE.md']
    if not mock:
        if gpu.backend() != 'rocm':
            k.update(available=False, reason='Klein3 requires the native ROCm backend')
        elif not (ROOT/'rdna4/flux3/test_hip_flux3').is_file():
            k.update(available=False, reason='build make -C rdna4/flux3')
        elif not py.is_file():
            k.update(available=False, reason='Klein3 CPU text interpreter missing; run rdna4/flux3/setup.sh')
        else:
            try:
                missing = [name for name in required if not (model/name).is_file()]
                if not missing:
                    index = json.loads((model/required[5]).read_text())
                    missing.extend('text_encoder/'+name for name in set(index['weight_map'].values())
                                   if not (model/'text_encoder'/name).is_file())
                if missing:
                    k.update(available=False, reason='Klein3 assets missing: '+', '.join(sorted(missing)))
                else:
                    k['text_runtime'] = text_runtime(py)
            except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
                k.update(available=False, reason='invalid Klein3 checkpoint: '+str(exc))
    return {'qwen-image21': q, 'klein3': k}


@lru_cache(maxsize=16)
def _digest(path, size, mtime):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def receipt(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, sha256=_digest(str(path), stat.st_size, stat.st_mtime_ns))


def run(command, cancel, log, progress=None):
    from server.pixal3d.app import run_command, JobCancelled
    try:
        result = run_command(list(map(str, command)), 7200, cancel=cancel, progress=progress)
    except JobCancelled as exc:
        raise gpu.Cancelled('portrait generation cancelled') from exc
    log.write_text(result.stdout+'\n'+result.stderr)
    if result.returncode:
        raise ServiceError('Klein3 worker failed: '+result.stderr[-1600:])
    return result


def generate_klein(service, output, prompt, seed, steps, cancel, progress, *, model=None, python=None):
    from rdna4.flux3 import prompt_cache
    model, python = Path(model or KLEIN_MODEL), Path(python or KLEIN_PYTHON)
    cache = service.work/'cache/klein3-text'
    cache.mkdir(parents=True, exist_ok=True)
    descriptor = prompt_cache.identity(model, prompt, runtime=text_runtime(python))
    features = cache/(prompt_cache.key(descriptor)+'.f32')
    scratch = service.work/'portrait-work'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='klein3-', dir=scratch) as temp:
        temp = Path(temp)
        with gpu.file_lock(features.with_suffix('.lock'), 900, cancel):
            manifest = prompt_cache.cached(features, descriptor)
            if manifest is None:
                progress(.12, 'Klein3: encoding portrait prompt on CPU')
                prepared = temp/'text.f32'
                run([python, ROOT/'rdna4/flux3/prepare_text.py', '--model-dir', model,
                     '--prompt', prompt, '--output', prepared, '--threads', 16], cancel, temp/'encoder.log')
                manifest = prompt_cache.publish(prepared, features, descriptor)
        progress(.2, 'Klein3: native ROCm front portrait')
        ppm = temp/'portrait.ppm'
        def update(line):
            if cancel.is_set():
                return
            if line.startswith('flux3: step '):
                try:
                    step, total = map(int, line.split()[2].split('/'))
                    progress(.2+.7*step/total, 'Klein3: '+line.removeprefix('flux3: '))
                except (ValueError, IndexError):
                    pass
        run([ROOT/'rdna4/flux3/test_hip_flux3', '--model-dir', model, '--text', features,
             '--ntxt', manifest['tokens'], '--width', 1024, '--height', 1024, '--steps', steps,
             '--seed', seed, '--device', gpu.device_index(), '--reserve-mib', 1024,
             '--vae-wmma', 0, '--output', ppm], cancel, temp/'native.log', update)
        if cancel.is_set():
            raise gpu.Cancelled('portrait generation cancelled')
        with Image.open(ppm) as image:
            if image.size != (1024, 1024):
                raise ServiceError('Klein3 portrait has unexpected dimensions')
            image.convert('RGB').save(output)
        # Logs are retained with the revision, while large temporary arrays are removed.
        for log in temp.glob('*.log'):
            (output.parent/log.name).write_text(log.read_text())
    return dict(checkpoint=receipt(model/'variants/fp8r/model.safetensors'),
                vae_checkpoint=receipt(model/'vae/diffusion_pytorch_model.safetensors'),
                model_license=receipt(model/'LICENSE.md'), model_notice=receipt(model/'NOTICE.md'),
                text_encoder_identity=descriptor, reserve_mib=1024, vae_wmma=0)
