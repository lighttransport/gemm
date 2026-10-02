"""Identity and reference-conditioned expressions with native FLUX.2 Klein.

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
    from PIL import Image
    import numpy as np
    from ....rig.exprdata import EXPRESSIONS, KEEP
    if config is None:
        raise ValueError('native identity generation requires --native-assets with verified FLUX.2 component paths')
    specs = [('neutral', prompt, {})]
    if expressions:
        specs += [(name, KEEP + ' ' + text, controls) for name, (text, controls, _) in EXPRESSIONS.items()]
    if type(seed) is not int or not 0 <= seed <= 2**63-len(specs) or type(device) is not int or device < 0:
        raise ValueError('invalid seed/device')
    runner = Path(runner) if runner else ROOT / 'cuda/flux2/test_cuda_flux2'
    if not runner.is_file():
        raise RuntimeError('Build native identity generation with make -C cuda/flux2')
    assets, paths = load_assets(config)
    asset_hash, runner_hash = sha256(config), sha256(runner)
    output = Path(output).resolve()
    manifest_path = output / 'manifest.json'
    recipe = 'diffusers_512_right_padding_reference_t10'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (not resume or manifest.get('backend') != 'native_flux2' or
                manifest.get('assets_sha256') != asset_hash or manifest.get('seed') != seed or
                manifest.get('runner_sha256') != runner_hash or manifest.get('conditioning_recipe') != recipe or
                manifest.get('neutral_prompt') != prompt):
            raise ValueError('identity exists or native asset/seed/runner/prompt receipt differs')
        if manifest['provenance']:
            verify_files(manifest['provenance'], output)
        # A changed expression recipe must not silently reuse old supervision.
        available = {name: (text, controls) for name, text, controls in specs}
        for record in manifest['references']:
            if record['name'] in available and (record['prompt'], record['controls']) != available[record['name']]:
                raise ValueError('expression prompt/control receipt differs')
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError('native identity output directory must be empty')
        manifest = dict(format='vhuman.clean_identity.v1', model=MODEL, revision=assets['revision'], seed=seed,
                        backend='native_flux2', assets_sha256=asset_hash, assets=assets, runner_sha256=runner_hash,
                        neutral_prompt=prompt, conditioning_recipe=recipe, references=[], provenance=[],
                        requires_manual_identity_and_pose_QA=True, parity='unverified',
                        precision='F16 DiT weights converted from selected checkpoint', gemm='repo')
    def checkpoint():
        partial = manifest_path.with_suffix('.json.partial')
        partial.write_text(json.dumps(manifest, indent=2) + '\n')
        partial.replace(manifest_path)
    checkpoint()  # A failed subprocess can be resumed explicitly.
    for index, (name, text, controls) in enumerate(specs):
        if any(record['name'] == name for record in manifest['references']):
            continue
        ppm, reference = output / (name + '.ppm'), output / 'reference-image.f32'
        command = [str(runner.resolve()), '--generate', '--dit', str(paths['dit']), '--vae', str(paths['vae']),
                   '--enc', str(paths['encoder']), '--tok', str(paths['tokenizer']), '--prompt', text,
                   '--height', '512', '--width', '512', '--steps', '4', '--seed', str(seed + index),
                   '--weight-type', 'f16', '--gemm', 'repo', '--conditioning', 'diffusers',
                   '--device', str(device), '--no-dumps', '--out', str(ppm)]
        if index:
            with Image.open(output / 'neutral.png') as neutral:
                pixels = np.asarray(neutral.convert('RGB'), np.float32) / 127.5 - 1
            np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype='<f4').tofile(reference)
            command += ['--reference-image-f32', str(reference)]
        started = time.monotonic()
        try:
            with (output / (name + '.log')).open('w') as log:
                subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
            with Image.open(ppm) as image:
                if image.size != (512, 512):
                    raise ValueError('unexpected native identity image dimensions')
                image.convert('RGB').save(output / (name + '.png'))
        finally:
            ppm.unlink(missing_ok=True)
            reference.unlink(missing_ok=True)
        receipt = dict(path=name + '.png', source='https://huggingface.co/' + MODEL, revision=assets['revision'],
                       license='Apache-2.0', license_scope='model and original project-generated artifact',
                       sha256=sha256(output / (name + '.png')), roles=['appearance-training', 'rig-training'],
                       seed=seed + index, conditioning=[] if not index else [manifest['provenance'][0]['sha256']],
                       generator='native_flux2', assets_sha256=asset_hash)
        manifest['provenance'].append(receipt)
        manifest['references'].append(dict(name=name, path=receipt['path'], controls=controls, prompt=text,
                                          generation_seconds=time.monotonic()-started, controls_are_approximate=True))
        checkpoint()
    return dict(output=str(output), images=len(manifest['references']), model=MODEL, backend='native_flux2')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create a hashed native FLUX.2 asset manifest')
    for name in ('dit', 'vae', 'encoder', 'tokenizer', 'revision', 'output'):
        parser.add_argument('--' + name, required=True)
    print(create_assets(**vars(parser.parse_args())))
