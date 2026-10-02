"""Reference reuse rejects changed inputs and non-native GEMM execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from . import verify_hunyuan_backend as backend
from .verify_hunyuan_backend import RECIPE, matching_recipe


class ReferenceReuseTests(unittest.TestCase):
    def test_changed_recipe_and_weights_rejected(self):
        baseline = {key: 'fixture' for key in RECIPE}
        candidate = dict(baseline, gemm='repo', gemm_fallback='error',
                         metrics=dict(repo_gemm_calls=42, cublas_gemm_calls=0, fallback_gemm_calls=0))
        matching_recipe(candidate, baseline)
        for key in RECIPE:
            changed = copy.deepcopy(candidate)
            changed[key] = 'different'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'differs'):
                matching_recipe(changed, baseline)
        for key in ('cublas_gemm_calls', 'fallback_gemm_calls'):
            changed = copy.deepcopy(candidate); changed['metrics'][key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'strict repository'):
                matching_recipe(changed, baseline)
        del candidate['model']
        with self.assertRaisesRegex(ValueError, 'model'):
            matching_recipe(candidate, baseline)

    def test_capture_and_reference_integrity(self):
        # Small spatial fixtures exercise receipt binding and frame gates;
        # production shape/profile validation is retained in the real verifier.
        scratch = backend.ROOT / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            paths = {key: root/key for key in ('candidate_run', 'candidate_actual',
                     'baseline_run', 'baseline_actual', 'baseline_reference')}
            for path in paths.values(): path.mkdir()
            reference = paths['baseline_reference']/'reference'
            reference.mkdir()
            baseline = {key: 'fixture' for key in RECIPE}
            baseline['task'] = 't2v'
            candidate = dict(baseline, gemm='repo', gemm_fallback='error',
                             metrics=dict(repo_gemm_calls=42, cublas_gemm_calls=0, fallback_gemm_calls=0))
            for key, value in (('baseline_run', baseline), ('candidate_run', candidate)):
                (paths[key]/'manifest.json').write_text(json.dumps(value))
            manifest_hash = backend.digest(paths['baseline_run']/'manifest.json')
            values = dict(noise_input=np.ones((1, 2), np.float32),
                          vae_decoded=np.ones((1, 3, 81, 1, 1), np.float32))
            receipts, native_hashes, reference_hashes = {}, {}, {}
            for name, value in values.items():
                for key in ('candidate_actual', 'baseline_actual'):
                    value.tofile(paths[key]/(name+'.f32'))
                    (paths[key]/(name+'.json')).write_text(json.dumps(dict(dtype='float32', shape=value.shape)))
                np.save(reference/(name+'.npy'), value)
                native_hashes[name] = backend.digest(paths['baseline_actual']/(name+'.f32'))
                reference_hashes[name] = backend.digest(reference/(name+'.npy'))
                receipts[name] = dict(sha256=reference_hashes[name], provenance=dict(
                    generation_manifest_sha256=manifest_hash, weights={'weight':'hash'}, upstream_revision='fixture'))
            (reference/'receipts.json').write_text(json.dumps(receipts))
            accepted = dict(scope='pipeline', generation_manifest_sha256=manifest_hash,
                            results={name:{'pass':True} for name in values},
                            decoded_frames=[{'pass':True} for _ in range(81)],
                            reference_weights={'weight':{'sha256':'hash'}}, reference_dtype='fixture',
                            native_capture_sha256=native_hashes, reference_capture_sha256=reference_hashes)
            accepted['pass'] = True
            acceptance = paths['baseline_reference']/'compare_parity.json'
            acceptance.write_text(json.dumps(accepted))
            args = dict(paths, output=root/'report.json')
            with patch.object(backend, 'pipeline_names', return_value=list(values)), \
                    patch.object(backend, 'validate_pipeline_shapes'):
                self.assertTrue(backend.verify(**args)['passed'])
                broken = values['vae_decoded'].copy()
                broken[0, 0, 40, 0, 0] = 99
                broken.tofile(paths['candidate_actual']/'vae_decoded.f32')
                self.assertFalse(backend.verify(**args)['decoded_frames'][40]['pass'])
                np.save(reference/'vae_decoded.npy', broken)
                with self.assertRaisesRegex(ValueError, 'stale baseline'):
                    backend.verify(**args)
                np.save(reference/'vae_decoded.npy', values['vae_decoded'])
                (values['noise_input']*2).tofile(paths['candidate_actual']/'noise_input.f32')
                with self.assertRaisesRegex(ValueError, 'noise differs'):
                    backend.verify(**args)
                accepted['decoded_frames'].pop()
                acceptance.write_text(json.dumps(accepted))
                with self.assertRaisesRegex(ValueError, 'complete independent reference'):
                    backend.verify(**args)


if __name__ == '__main__':
    unittest.main()
