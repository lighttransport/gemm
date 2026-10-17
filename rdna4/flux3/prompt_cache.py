"""Content/settings identity and validation for exported CPU prompt features."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path


def runtime_identity():
    return {name: version(name) for name in ('torch', 'transformers', 'safetensors')}


def identity(model, prompt, threads=16, runtime=None):
    encoder = Path(model).resolve() / 'text_encoder'
    files = []
    for path in sorted(encoder.iterdir()):
        if path.is_file():
            stat = path.stat()
            row = [path.name, stat.st_size, stat.st_mtime_ns]
            if path.suffix == '.json' and stat.st_size < 16 * 1024**2:
                row.append(hashlib.sha256(path.read_bytes()).hexdigest())
            files.append(row)
    return dict(encoder=str(encoder), files=files, prompt=prompt, threads=threads,
                runtime=runtime if runtime is not None else runtime_identity(),
                implementation=hashlib.sha256(Path(__file__).with_name('prepare_text.py').read_bytes()).hexdigest())


def key(descriptor):
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()


def cached(path, descriptor):
    path = Path(path)
    try:
        manifest = json.loads(path.with_suffix('.f32.json').read_text())
        tokens = manifest['tokens']
        if (manifest.get('cache_identity') == descriptor and type(tokens) is int and 0 < tokens <= 4096
                and manifest['channels'] == 20480 and path.stat().st_size == tokens * 20480 * 4):
            with path.open('rb') as stream:
                if manifest.get('sha256') == hashlib.file_digest(stream, 'sha256').hexdigest():
                    return manifest
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def publish(source, target, descriptor):
    source, target = Path(source), Path(target)
    manifest = json.loads(source.with_suffix('.f32.json').read_text())
    tokens = manifest['tokens']
    if (type(tokens) is not int or not 0 < tokens <= 4096 or manifest.get('channels') != 20480
            or source.stat().st_size != tokens * 20480 * 4):
        raise ValueError('invalid Klein3 prompt features')
    manifest['cache_identity'] = descriptor
    with source.open('rb') as stream:
        manifest['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    source.with_suffix('.f32.json').write_text(json.dumps(manifest, indent=2)+'\n')
    source.replace(target)
    source.with_suffix('.f32.json').replace(target.with_suffix('.f32.json'))
    return manifest
