"""Conditioned runner mode validation before weights or GPU allocation."""
import unittest
from pathlib import Path
import sys
import hashlib
import json
import tempfile
from contextlib import nullcontext
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rdna4.minimax_h3 import generate as runner
generate = runner.generate


class ConditioningValidationTests(unittest.TestCase):
    def test_image_modes_fail_before_loading(self):
        base = dict(out='unused', prompt='smile', allow_experimental=True)
        for options in ({'variant':'other'}, {'variant':'fl2va','reference_images':['face.png']},
                        {'variant':'ref2va','first_frame':'face.png'},
                        {'reference_images':['face.png']*10},
                        {'reference_images':['face.png'],'conditioning_dir':'prepared'},
                        {'backend':'cuda','reference_images':['face.png']}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                generate(**base, **options)

    def test_invalid_bundle_cannot_start_gpu_and_preserves_source(self):
        scratch = ROOT/'tmp/video-rocm/tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root = Path(folder)
            model, bundle = root/'model', root/'bundle'
            bundle.mkdir()
            components = {}
            for name in runner.COMPONENTS:
                path = model/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
                components[name] = {'bytes': 7, 'sha256': hashlib.sha256(b'fixture').hexdigest()}
            recipe = dict(schema='h3.image_conditioning.v1', variant='ref2va', prompt='test',
                          width=64, height=64, frames=5, seed=42, files={})
            for binding, message in ((None, 'weights mismatch'), ({}, 'weights mismatch'),
                                      (components, 'omits a required file')):
                receipt = {**recipe, 'verified_components': binding}
                (bundle/'manifest.json').write_text(json.dumps(receipt))
                with patch.object(runner.video, 'run_process') as process, \
                        patch.object(runner.video, 'MemorySampler'), \
                        patch.object(runner.video, 'device_lock', return_value=nullcontext()):
                    with self.assertRaisesRegex(ValueError, message):
                        generate(model=model, out=root/'video', prompt='test', width=64,
                                 height=64, frames=5, steps=6, conditioning_dir=bundle,
                                 allow_experimental=True)
                    process.assert_not_called()
                self.assertTrue((bundle/'manifest.json').is_file())
                self.assertFalse((root/'video.partial').exists())


if __name__ == '__main__':
    unittest.main()
