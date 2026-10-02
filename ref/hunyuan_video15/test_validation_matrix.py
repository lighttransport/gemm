import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ref.hunyuan_video15.validation_matrix import matrix, audit, digest, COMPONENTS
from ref.hunyuan_video15.convert_native import convert
from ref.hunyuan_video15.verify_vae_decode import load_native_latent

ROOT = Path(__file__).resolve().parents[2]


class ValidationMatrixTests(unittest.TestCase):
    def test_missing_stale_wrong_backend_and_wrong_precision_never_pass(self):
        (ROOT / 'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as work:
            root = Path(work)
            portraits = [root / f'{i}.png' for i in range(3)]
            for i, path in enumerate(portraits):
                path.write_bytes(bytes([i]))
            plan = matrix(portraits)
            self.assertEqual(len(plan['cases']), 144)
            result = audit(plan, root)
            self.assertEqual(result['missing'], 144)
            self.assertFalse(result['passed_all'])
            case = plan['cases'][0]
            output = root / case['id']; output.mkdir()
            manifest = dict(case, backend='hv15n_cuda_experimental')
            mpath = output / 'manifest.json'
            mpath.write_text(json.dumps(manifest))
            report = dict(scope='independent_official_component_chain_with_matched_noise',
                          generation_manifest_sha256=digest(mpath),
                          reference_precision={'qwen': 'float32_cpu', 'google_siglip': 'float32_cpu'},
                          results={name: {'pass': True} for name in COMPONENTS},
                          frame_errors=[{'frame': i, 'pass': True} for i in range(case['frames'])])
            report['pass'] = True
            (output / 'parity.json').write_text(json.dumps(report))
            self.assertEqual(audit(plan, root)['passed'], 1)
            self.assertEqual(audit(plan, root, 'float16')['failed'], 1)
            self.assertEqual(audit(plan, root, backend='legacy')['failed'], 1)
            manifest['seed'] += 1
            mpath.write_text(json.dumps(manifest))
            self.assertEqual(audit(plan, root)['failed'], 1)

    def test_canonical_native_dump_conversion_checks_byte_count(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as work:
            root = Path(work)
            array = np.arange(60, dtype='<f4').reshape(1, 3, 4, 5)
            array.tofile(root / 'vae_encoded.f32')
            meta = {'dtype': 'float32', 'layout': 'NCTHW', 'shape': [1, 3, 1, 4, 5]}
            (root / 'vae_encoded.json').write_text(json.dumps(meta))
            convert(root)
            np.testing.assert_array_equal(np.load(root / 'vae_encoded.npy'), array[:, :, None])
            meta['shape'][-1] += 1
            (root / 'vae_encoded.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'byte count'):
                convert(root)

    def test_decode_accepts_both_lengths_and_refuses_missing_temporal_frames(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as work:
            root = Path(work)
            for frames, length in ((81, 21), (121, 31)):
                shape = [1, 32, length, 53, 30]
                np.zeros(shape, '<f4').tofile(root / 'latent_final.f32')
                (root / 'latent_final.json').write_text(json.dumps(
                    dict(shape=shape, dtype='float32', layout='NCTHW')))
                self.assertEqual(load_native_latent(root, frames).shape, tuple(shape))
                if frames == 81:
                    with self.assertRaisesRegex(ValueError, 'exceed'):
                        load_native_latent(root, 121)


if __name__ == '__main__':
    unittest.main()
