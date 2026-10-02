"""Neutral identity generation with the repository FLUX.2 Klein executable.

Reference-conditioned expression generation still needs a separate native port.
No ML framework is imported and no alternate inference backend is selected.
"""
import json
from pathlib import Path
import subprocess
import time
from .provenance import sha256, verify_files

ROOT = Path(__file__).resolve().parents[5]
MODEL = 'black-forest-labs/FLUX.2-klein-4B'


def load_assets(config):
    config = Path(config).resolve()
    assets = json.loads(config.read_text())
    if assets.get('format') != 'vhuman.native_flux2_assets.v1' or assets.get('model') != MODEL:
        raise ValueError('expected native FLUX.2 Klein 4B asset manifest')
    if not isinstance(assets.get('revision'), str) or not assets['revision']:
        raise ValueError('native model source revision required')
    paths = {}
    for name in ('dit', 'vae', 'encoder', 'tokenizer'):
        receipt = assets['components'][name]
        path = (config.parent / receipt['path']).resolve()
        files = receipt.get('files', {})
        if not files:
            raise ValueError('component hashes required: ' + name)
        if name == 'encoder':
            if not path.is_dir():
                raise ValueError('encoder must be a safetensors directory')
            present = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
            if present != set(files):
                raise ValueError('encoder receipt must cover every asset file')
            for relative, digest in files.items():
                # HF snapshots contain symlinks into their content-addressed cache.
                # Permit those read-only targets, while checking every content hash
                # and rejecting absolute/parent-traversing receipt names.
                filename = Path(relative)
                target = path / filename
                if filename.is_absolute() or '..' in filename.parts or sha256(target) != digest:
                    raise ValueError('encoder checksum/path mismatch')
        elif set(files) != {path.name} or sha256(path) != files[path.name]:
            raise ValueError('component checksum mismatch: ' + name)
        paths[name] = path
    return assets, paths


def create_assets(output, revision, *, dit, vae, encoder, tokenizer):
    """Record the caller's explicitly selected checkpoint files; never fetch weights."""
    paths = dict(dit=Path(dit).resolve(), vae=Path(vae).resolve(),
                 encoder=Path(encoder).resolve(), tokenizer=Path(tokenizer).resolve())
    components = {}
    for name, path in paths.items():
        if name == 'encoder':
            if not path.is_dir():
                raise ValueError('encoder must be a directory')
            files = {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob('*')) if p.is_file()}
        else:
            files = {path.name: sha256(path)}
        components[name] = {'path': str(path), 'files': files}
    spec = dict(format='vhuman.native_flux2_assets.v1', model=MODEL,
                revision=revision, components=components)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2) + '\n')
    load_assets(output)
    return output


def generate(output, config, seed, prompt, *, expressions=False, resume=False,
             runner=None, device=0):
    if expressions:
        raise ValueError('native FLUX.2 currently supports neutral generation only; reference-conditioned expressions require --identity-backend torch-reference')
    if config is None:
        raise ValueError('native identity generation requires --native-assets with verified FLUX.2 component paths')
    if type(seed) is not int or not 0 <= seed < 2**63 or type(device) is not int or device < 0:
        raise ValueError('invalid seed/device')
    runner = Path(runner) if runner else ROOT / 'cuda/flux2/test_cuda_flux2'
    if not runner.is_file():
        raise RuntimeError('Build native identity generation with make -C cuda/flux2')
    assets, paths = load_assets(config)
    asset_hash = sha256(config)
    runner_hash = sha256(runner)
    output = Path(output)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if (not resume or old.get('backend') != 'native_flux2' or
                old.get('assets_sha256') != asset_hash or old.get('seed') != seed or
                old.get('runner_sha256') != runner_hash or
                old.get('references', [{}])[0].get('prompt') != prompt):
            raise ValueError('identity exists or native asset/seed/runner/prompt receipt differs')
        verify_files(old['provenance'], output)
        return dict(output=str(output), images=len(old['references']), model=MODEL, backend='native_flux2')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('native identity output directory must be empty')
    ppm = output.resolve() / 'neutral.ppm'
    command = [str(runner.resolve()), '--generate', '--dit', str(paths['dit']), '--vae', str(paths['vae']),
               '--enc', str(paths['encoder']), '--tok', str(paths['tokenizer']), '--prompt', prompt,
               '--height', '512', '--width', '512', '--steps', '4', '--seed', str(seed),
               '--weight-type', 'bf16', '--gemm', 'repo',
               '--device', str(device), '--no-dumps', '--out', str(ppm)]
    started = time.monotonic()
    with (output / 'native.log').open('w') as log:
        subprocess.run(command, cwd=output.resolve(), stdout=log, stderr=subprocess.STDOUT, check=True)
    from PIL import Image
    with Image.open(ppm) as image:
        if image.size != (512, 512):
            raise ValueError('unexpected native identity image dimensions')
        image.convert('RGB').save(output / 'neutral.png')
    ppm.unlink()
    receipt = dict(path='neutral.png', source='https://huggingface.co/' + MODEL, revision=assets['revision'],
                   license='Apache-2.0', license_scope='model and original project-generated artifact',
                   sha256=sha256(output / 'neutral.png'), roles=['appearance-training', 'rig-training'],
                   seed=seed, conditioning=[], generator='native_flux2', assets_sha256=asset_hash)
    manifest = dict(format='vhuman.clean_identity.v1', model=MODEL, revision=assets['revision'], seed=seed,
                    backend='native_flux2', assets_sha256=asset_hash, assets=assets, runner_sha256=runner_hash,
                    provenance=[receipt], requires_manual_identity_and_pose_QA=True,
                    references=[dict(name='neutral', path='neutral.png', controls={}, prompt=prompt,
                                     generation_seconds=time.monotonic()-started, controls_are_approximate=True)],
                    parity='unverified', precision='BF16 DiT weights (dequantized from selected checkpoint)',
                    gemm='repo')
    partial = manifest_path.with_suffix('.json.partial')
    partial.write_text(json.dumps(manifest, indent=2) + '\n')
    partial.replace(manifest_path)
    return dict(output=str(output), images=1, model=MODEL, backend='native_flux2')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create a hashed native FLUX.2 asset manifest')
    for name in ('dit', 'vae', 'encoder', 'tokenizer', 'revision', 'output'):
        parser.add_argument('--' + name, required=True)
    print(create_assets(**vars(parser.parse_args())))
