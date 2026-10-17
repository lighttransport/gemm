"""Generator selection, mixed provenance, cache integrity and worker cleanup."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from rdna4.flux3 import prompt_cache
from . import gpu, portrait, portrait_generators as generators
from .service import ROOT, EyeService, ServiceError
from .telemetry import ROCmTelemetry

SCRATCH = ROOT/'tmp/test-portrait-generators'


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = EyeService(self.root/'work')
        self.cancel = threading.Event()
        session = gpu.device_session
        lock = patch.object(gpu, 'device_session', lambda minimum, cancel, **kwargs:
                            session(minimum, cancel, lock_path=self.root/'device.lock', **kwargs))
        lock.start()
        self.addCleanup(lock.stop)

    def test_mixed_session_snapshot_and_switch_invalidates_review(self):
        with gpu.execution('cpu'):
            doc = portrait.generate_job(self.service, dict(subject='test', portrait_model='klein3'),
                                       lambda *_: None, self.cancel, mock=True)['session']
            front = doc['portraits'][0]
            self.assertEqual((front['portrait_model'], front['steps']), ('klein3', 16))
            self.assertIsNone(front['cfg'])
            self.assertIsNone(front['negative_prompt'])
            self.assertIn('FLUX', front['license'])
            req = dict(session_id=doc['id'], revision=doc['active_revision'], portrait_reviewed=True, view_ids=[])
            portrait.review(self.service, doc['id'], req)
            doc = portrait.views_job(self.service, dict(req, method='sheet'), lambda *_: None,
                                     self.cancel, mock=True)['session']
            req['view_ids'] = [doc['view_runs'][0]['views'][0]['id']]
            portrait.review(self.service, doc['id'], req)
            snapshot = portrait.snapshot(self.service, req)
            inputs = json.loads(Path(snapshot['generated_input']).read_text())
            self.assertEqual(inputs['views'][0]['portrait_model'], 'klein3')
            self.assertEqual(inputs['views'][1]['generation']['model'], 'Qwen-Image 2.1')
            changed = portrait.generate_job(self.service, dict(subject='test', session_id=doc['id']),
                                           lambda *_: None, self.cancel, mock=True)['session']
            self.assertEqual(changed['portraits'][-1]['portrait_model'], 'qwen-image21')
            self.assertFalse(changed['selection']['portrait_reviewed'])
            self.assertTrue(changed['view_runs'][0]['stale'])

    def test_klein_dispatch_does_not_load_qwen(self):
        def draw(service, output, *args, **kwargs):
            Image.new('RGB', (1024, 1024)).save(output)
            return {'reserve_mib': 1024}
        with gpu.execution('rocm'), \
             patch.object(generators, 'availability', return_value={'klein3': {'available': True}}), \
             patch.object(gpu, 'gpu_status', return_value={'free_mib': 12000}), \
             patch.object(generators, 'generate_klein', side_effect=draw) as klein, \
             patch.object(portrait.qwen, 'make_backend') as qwen:
            portrait.generate_job(self.service, dict(subject='test', portrait_model='klein3'),
                                  lambda *_: None, self.cancel)
            klein.assert_called_once()
            qwen.assert_not_called()

    def test_klein_admission_precedes_cpu_encoding(self):
        with gpu.execution('rocm'), \
             patch.object(generators, 'availability', return_value={'klein3': {'available': True}}), \
             patch.object(gpu, 'gpu_status', return_value={'free_mib': generators.KLEIN_MIN_FREE_MIB-1}), \
             patch.object(generators, 'generate_klein') as launch:
            with self.assertRaises(gpu.GpuBusy):
                portrait.generate_job(self.service, dict(subject='test', portrait_model='klein3'),
                                      lambda *_: None, self.cancel)
            launch.assert_not_called()
            self.assertEqual(portrait.listing(self.service), [])

    def test_invalid_generator_and_klein_side_views_are_rejected(self):
        with self.assertRaises(ServiceError):
            portrait.generate_job(self.service, dict(subject='test', portrait_model='unknown'),
                                  lambda *_: None, self.cancel, mock=True)
        with self.assertRaisesRegex(ServiceError, 'side views require'):
            portrait.views_job(self.service, {'portrait_model': 'klein3'}, lambda *_: None, self.cancel, mock=True)
        with gpu.execution('cuda'):
            self.assertFalse(generators.availability()['klein3']['available'])

    def test_cancelled_preparation_removes_scratch_and_does_not_publish_cache(self):
        with patch.object(prompt_cache, 'identity', return_value={'prompt': 'test'}), \
             patch.object(generators, 'text_runtime', return_value={}), \
             patch.object(generators, 'run', side_effect=gpu.Cancelled('cancelled')):
            with self.assertRaises(gpu.Cancelled):
                generators.generate_klein(self.service, self.root/'portrait.png', 'test', 11, 16,
                                          self.cancel, lambda *_: None)
        self.assertFalse(list((self.service.work/'portrait-work').iterdir()))
        self.assertFalse(list((self.service.work/'cache').rglob('*.f32')))

    def test_worker_cancellation_terminates_child_process_group(self):
        child = self.root/'child.pid'
        code = ('import subprocess,sys,time,pathlib; '
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                f'pathlib.Path({str(child)!r}).write_text(str(p.pid)); time.sleep(60)')
        timer = threading.Timer(.5, self.cancel.set)
        timer.start()
        try:
            with self.assertRaises(gpu.Cancelled):
                generators.run([sys.executable, '-c', code], self.cancel, self.root/'worker.log')
        finally:
            timer.cancel()
        pid = child.read_text()
        state = Path('/proc')/pid/'stat'
        self.assertTrue(not state.exists() or state.read_text().split()[2] == 'Z')

    def test_cache_rejects_incomplete_and_different_encoder_settings(self):
        source, target = self.root/'source.f32', self.root/'cached.f32'
        source.write_bytes(bytes(20480*4))
        source.with_suffix('.f32.json').write_text(json.dumps({'tokens': 1, 'channels': 20480}))
        descriptor = {'prompt': 'test', 'encoder': 'v1'}
        prompt_cache.publish(source, target, descriptor)
        self.assertIsNotNone(prompt_cache.cached(target, descriptor))
        self.assertIsNone(prompt_cache.cached(target, dict(descriptor, encoder='v2')))
        target.write_bytes(bytes([1]) + bytes(20480*4-1))
        self.assertIsNone(prompt_cache.cached(target, descriptor))
        target.write_bytes(b'partial')
        self.assertIsNone(prompt_cache.cached(target, descriptor))

    def test_rocm_telemetry_uses_pci_identity_not_card_order(self):
        drm = self.root/'drm'
        drm.mkdir()
        for index, bus, used in [(0, '0000:04:00.0', 800), (7, '0000:0b:00.0', 200)]:
            device = self.root/bus
            device.mkdir()
            for name, value in [('mem_info_vram_used', used), ('mem_info_vram_total', 1000), ('gpu_busy_percent', 12)]:
                (device/name).write_text(str(value))
            card = drm/f'card{index}'
            card.mkdir()
            (card/'device').symlink_to(device)
        monitor = ROCmTelemetry(status={'pci_bus_id': '00000000:0b:00.0'}, drm_root=drm)
        self.assertEqual(monitor.sample()['memory_used_bytes'], 200)


if __name__ == '__main__':
    unittest.main()
