"""Native identity subprocess/receipt contract without an inference framework."""
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import numpy as np
from unittest.mock import patch
from PIL import Image
from .src.avatar import native_identity as native


class NativeIdentityTests(unittest.TestCase):
    def setUp(self):
        (native.ROOT / 'tmp').mkdir(exist_ok=True)
        work = tempfile.TemporaryDirectory(dir=native.ROOT / 'tmp')
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        components = {name: self.root / name for name in ('dit', 'vae', 'tokenizer')}
        for path in components.values():
            path.write_bytes(b'fixture')
        components['encoder'] = self.root / 'encoder'
        components['encoder'].mkdir()
        (components['encoder'] / 'weights.safetensors').write_bytes(b'fixture')
        self.config = native.create_assets(self.root / 'assets.json', 'fixture-revision', **components)
        self.runner = self.root / 'runner'
        self.runner.write_bytes(b'fixture executable')

    def fake_generate(self, command, **kwargs):
        self.assertIn('--no-dumps', command)
        self.assertIn('--gpu-enc', command)
        self.assertIn('--no-text-cache', command)
        self.assertNotIn('--keep-gpu-enc', command)
        self.assertEqual(command[command.index('--weight-type') + 1], 'f16')
        self.assertEqual(command[command.index('--conditioning') + 1], 'diffusers')
        self.assertEqual(command[command.index('--gemm') + 1], 'repo')
        self.assertTrue(kwargs['check'])
        if '--reference-image-f32' in command:
            pixels = np.fromfile(command[command.index('--reference-image-f32') + 1], dtype='<f4').reshape(3, 512, 512)
            np.testing.assert_allclose(pixels[:, 0, 0], np.array([80, 110, 140])/127.5-1, atol=1e-7)
        Image.new('RGB', (512, 512), (80, 110, 140)).save(command[command.index('--out') + 1])

    def test_manifest_resume_and_modified_artifact(self):
        output = self.root / 'identity'
        with patch.object(native.subprocess, 'run', side_effect=self.fake_generate) as run:
            result = native.generate(output, self.config, 7, 'neutral', runner=self.runner)
            self.assertEqual(result['images'], 1)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['parity'], 'unverified')
            self.assertEqual(manifest['text_backend'], 'cuda_f32_kv')
            self.assertEqual(manifest['provenance'][0]['conditioning'], [])
            native.generate(output, self.config, 7, 'neutral', runner=self.runner, resume=True)
            run.assert_called_once()
            for changes in ({'prompt': 'different'}, {'seed': 8}):
                args = dict(output=output, config=self.config, seed=7, prompt='neutral', runner=self.runner, resume=True)
                args.update(changes)
                with self.assertRaisesRegex(ValueError, 'receipt differs'):
                    native.generate(**args)
            (output / 'neutral.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum differs'):
                native.generate(output, self.config, 7, 'neutral', runner=self.runner, resume=True)

    def test_unverified_assets_fail_before_subprocess(self):
        with patch.object(native.subprocess, 'run') as run:
            (self.root / 'encoder/extra.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'every asset file'):
                native.generate(self.root / 'output', self.config, 7, 'neutral', runner=self.runner)
            run.assert_not_called()

    def test_extend_neutral_to_expressions_and_resume_failure(self):
        output = self.root / 'identity'
        with patch.object(native.subprocess, 'run', side_effect=self.fake_generate):
            native.generate(output, self.config, 7, 'neutral', runner=self.runner)
        def fail(command, **kwargs):
            raise subprocess.CalledProcessError(1, command)
        with patch.object(native.subprocess, 'run', side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                native.generate(output, self.config, 7, 'neutral', expressions=True, resume=True, runner=self.runner)
        self.assertFalse((output / 'reference-image.f32').exists())
        with patch.object(native.subprocess, 'run', side_effect=self.fake_generate) as run:
            result = native.generate(output, self.config, 7, 'neutral', expressions=True, resume=True, runner=self.runner)
            self.assertEqual(result['images'], 13)
            self.assertEqual(run.call_count, 12)
            native.generate(output, self.config, 7, 'neutral', expressions=True, resume=True, runner=self.runner)
            self.assertEqual(run.call_count, 12)
        spec = json.loads((output / 'manifest.json').read_text())
        for i, receipt in enumerate(spec['provenance'][1:], 1):
            self.assertEqual(receipt['seed'], 7+i)
            self.assertEqual(receipt['conditioning'], [spec['provenance'][0]['sha256']])
        with patch.dict(native.EXPRESSION_PROMPTS, {'smile': 'Changed expression recipe'}):
            with self.assertRaisesRegex(ValueError, 'prompt/control receipt differs'):
                native.generate(output, self.config, 7, 'neutral', expressions=True, resume=True, runner=self.runner)

    def test_hf_cache_symlinks_are_hash_checked(self):
        weights = self.root / 'encoder/weights.safetensors'
        blob = self.root / 'cached-blob'
        weights.rename(blob)
        weights.symlink_to(blob)
        native.load_assets(self.config)
        blob.write_bytes(b'changed cache contents')
        with self.assertRaisesRegex(ValueError, 'checksum/path mismatch'):
            native.load_assets(self.config)


if __name__ == '__main__':
    unittest.main()
