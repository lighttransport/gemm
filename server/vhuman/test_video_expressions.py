"""ROCm expression capabilities, geometric mapping and review boundaries."""
import unittest
from unittest.mock import patch
import numpy as np
from .rig.gnm_expression import project, evaluate
from .video_backend import select
from .rig.video_expressions import export_reviewed
from . import video


class ExpressionMappingTests(unittest.TestCase):
    def test_geometry_projection_and_neutral(self):
        basis = np.zeros((2, 4, 3))
        basis[0, 0, 0] = .01
        basis[1, 1, 1] = .02
        targets = np.stack([basis[0]*.7+basis[1]*.2, basis[1]*.5])
        mapping = project(basis, targets, ['smile', 'blink'], ['region_0', 'region_1'], ridge=0)
        np.testing.assert_allclose(evaluate(mapping, {}), [0, 0])
        np.testing.assert_allclose(evaluate(mapping, {'smile': 1}), [.7, .2], atol=1e-5)
        self.assertLess(max(mapping['rms_residual_metres']), 1e-7)
        with self.assertRaises(ValueError):
            evaluate(mapping, {'smile': float('nan')})

    def test_nonrepresentable_geometry_records_residual(self):
        basis = np.zeros((1, 4, 3)); basis[0, 0, 0] = .01
        target = np.zeros((1, 4, 3)); target[0, 1, 1] = .02
        mapping = project(basis, target, ['smile'])
        self.assertGreater(mapping['rms_residual_metres'][0], .001)

    def test_backend_specific_short_runs(self):
        wan, h3, hv = (select(name) for name in ('wan', 'h3', 'hv15-rocm'))
        self.assertIn(9, wan.frames)
        self.assertIn(22, h3.frames)
        self.assertEqual(hv.frames, (81,))
        self.assertTrue(h3.identity_conditioned)
        for adapter in (wan, h3, hv):
            self.assertTrue(adapter.manages_device_lock)
            self.assertEqual(adapter.hardware, 'rocm')
        request = video.validate({'head_id': 'head', 'frames': 9, 'preset': 'fast5'},
                                 supported_frames=wan.frames, presets=wan.presets)
        self.assertEqual(request['frames'], 9)

    def test_hv_rocm_runner_and_no_fallback(self):
        adapter = select('hv15-rocm')
        result = {'metrics': {'hipblas_gemm_calls': 0, 'fallback_gemm_calls': 0, 'repo_gemm_calls': 5}}
        with patch.object(adapter.module, 'generate', return_value=result) as runner:
            adapter.generate(image='portrait.png', out='clip', prompt='smile')
            self.assertEqual(runner.call_args.kwargs['runner'], adapter.RUNNER)
            self.assertEqual(runner.call_args.kwargs['backend'], 'rocm')
            self.assertEqual(runner.call_args.kwargs['gemm_fallback'], 'error')
        with patch.object(adapter.module, 'generate', return_value={'metrics': {'hipblas_gemm_calls': 1}}):
            with self.assertRaisesRegex(RuntimeError, 'fallback'):
                adapter.generate(image='portrait.png', out='clip', prompt='smile')

    def test_h3_conditions_portrait_and_uses_five_updates(self):
        adapter = select('h3')
        with patch.object(adapter.module, 'generate', return_value={}) as runner, \
                patch.object(adapter.module.video, 'digest', return_value='portrait-hash'):
            result = adapter.generate(image='portrait.png', out='clip', prompt='smile', frames=22, preset='fast5')
            self.assertNotIn('image', runner.call_args.kwargs)
            self.assertEqual(runner.call_args.kwargs['reference_images'], ['portrait.png'])
            self.assertEqual(runner.call_args.kwargs['steps'], 6)
            self.assertTrue(result['source_portrait_used_for_conditioning'])

    def test_h3_fl2va_anchors_first_frame(self):
        adapter = select('h3-fl2va')
        with patch.object(adapter.module, 'generate', return_value={}) as runner, \
                patch.object(adapter.module.video, 'digest', return_value='portrait-hash'):
            adapter.generate(image='portrait.png', out='clip', prompt='smile', frames=22, preset='fast5')
            self.assertEqual(runner.call_args.kwargs['variant'], 'fl2va')
            self.assertEqual(runner.call_args.kwargs['first_frame'], 'portrait.png')
            self.assertNotIn('reference_images', runner.call_args.kwargs)

    def test_unreviewed_capture_cannot_become_rig_data(self):
        with patch('pathlib.Path.read_text', return_value='{"state":"candidate","review":{}}'):
            with self.assertRaisesRegex(ValueError, 'review'):
                export_reviewed('candidate', 'portrait.png', 'out')


if __name__ == '__main__':
    unittest.main()
