"""Native identity subprocess/receipt contract without an inference framework."""
import json
from pathlib import Path
import tempfile
import unittest
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
        self.assertNotIn('--gpu-enc', command)
        self.assertEqual(command[command.index('--weight-type') + 1], 'bf16')
        self.assertEqual(command[command.index('--gemm') + 1], 'repo')
        self.assertTrue(kwargs['check'])
        self.assertEqual(command[command.index('--seed') + 1], '7')
        Image.new('RGB', (512, 512), (80, 110, 140)).save(command[command.index('--out') + 1])

    def test_manifest_resume_and_modified_artifact(self):
        output = self.root / 'identity'
        with patch.object(native.subprocess, 'run', side_effect=self.fake_generate) as run:
            result = native.generate(output, self.config, 7, 'neutral', runner=self.runner)
            self.assertEqual(result['images'], 1)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['parity'], 'unverified')
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

    def test_unverified_assets_and_expression_mode_fail_before_subprocess(self):
        with patch.object(native.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'neutral generation only'):
                native.generate(self.root / 'output', self.config, 7, 'neutral', expressions=True)
            (self.root / 'encoder/extra.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'every asset file'):
                native.generate(self.root / 'output', self.config, 7, 'neutral', runner=self.runner)
            run.assert_not_called()

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
