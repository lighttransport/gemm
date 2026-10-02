"""Replay gates must reject bad tensors and misleading timing summaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
HERE=Path(__file__).resolve().parent
def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
replay=module('replay');report=module('report_replays')
SCRATCH=replay.ROOT/'tmp/hv15-native/tests'
SCRATCH.mkdir(parents=True,exist_ok=True)


class ReplayGates(unittest.TestCase):
    def test_chunked_metric_and_zero(self):
        a=np.arange(700000,dtype=np.float32).reshape(1000,700)/700000
        result=replay.compare_arrays(a,a*1.001)
        self.assertTrue(result['pass_all'])
        self.assertAlmostEqual(result['relative_l2'],.001,places=6)
        self.assertTrue(replay.compare_arrays(np.zeros(3),np.zeros(3))['pass_all'])
        self.assertFalse(replay.compare_arrays(np.zeros(3),np.ones(3))['pass_all'])

    def test_fail_closed_tensors(self):
        for a,b in ((np.zeros(0),np.zeros(0)),(np.ones(3),np.ones(4)),(np.array([np.nan]),np.ones(1))):
            with self.assertRaises(ValueError):replay.compare_arrays(a,b)
        self.assertFalse(replay.compare_arrays(np.ones(3),np.array([-1.,1.,1.]))['pass_all'])

    def test_warm_samples_exclude_first(self):
        self.assertEqual(report.samples([10.,2.,3.],True)['median'],2.5)
        for values,warm in (([],False),([1.,0.,2.],True),([float('nan')],False),([1.,2.],True)):
            with self.assertRaises(ValueError):report.samples(values,warm)


class PerformanceArtifacts(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(directory.cleanup)
        base=Path(directory.name)
        self.folders=dict(native=base/'native',reference=base/'reference')
        for folder in self.folders.values():
            folder.mkdir()
            (folder/'output.f32').write_bytes(np.ones(3,dtype=np.float32).tobytes())
            (folder/'output.json').write_text(json.dumps(dict(shape=[3],dtype='float32')))
            (folder/'timing.json').write_text(json.dumps(dict(wall_seconds=[3.,1.,1.],metrics={})))
            folder.with_suffix('.json').write_text(json.dumps(dict(status='pass',elapsed_seconds=1.,sampled_peak_vram_mib=None)))
        self.entry=dict(label='audit',**{name:str(folder) for name,folder in self.folders.items()})
        self.parity=dict(pass_all=True,full_pipeline_acceptance=False,results={'output':{'pass':True}},
            outputs=dict(output={name:report.digest(folder/'output.f32') for name,folder in self.folders.items()}),
            output_metadata=dict(output={name:report.digest(folder/'output.json') for name,folder in self.folders.items()}),
            timing_sha256={name:report.digest(folder/'timing.json') for name,folder in self.folders.items()},
            receipt_sha256={name:report.digest(folder.with_suffix('.json')) for name,folder in self.folders.items()})
        self.write_parity()

    def write_parity(self):
        (self.folders['native']/'parity.json').write_text(json.dumps(self.parity))

    def test_valid_report_allows_missing_vram_sample(self):
        self.assertEqual(report.pair(self.entry)['native_over_reference'],1.)

    def test_changed_accepted_artifacts_are_rejected(self):
        for folder in self.folders.values():
            for path in [folder/name for name in ('output.f32','output.json','timing.json')]+[folder.with_suffix('.json')]:
                with self.subTest(path=path):
                    original=path.read_bytes()
                    path.write_bytes(original+b' ')
                    with self.assertRaises(ValueError):report.pair(self.entry)
                    path.write_bytes(original)
        receipt=self.folders['native'].with_suffix('.json');receipt.unlink()
        with self.assertRaises(ValueError):report.pair(self.entry)

    def test_old_or_inconsistent_parity_cannot_authorize_timing(self):
        del self.parity['outputs'];self.write_parity()
        with self.assertRaises(ValueError):report.pair(self.entry)
        self.parity['results']['output']['pass']=False;self.write_parity()
        with self.assertRaises(ValueError):report.pair(self.entry)

    def test_gpu_receipt_required_and_cpu_reference_explicit(self):
        for backend in ('native','reference'):
            path=self.folders[backend].with_suffix('.json');original=path.read_bytes()
            path.unlink();self.parity['receipt_sha256'][backend]=None;self.write_parity()
            with self.assertRaises(ValueError):report.pair(self.entry)
            path.write_bytes(original);self.parity['receipt_sha256'][backend]=report.digest(path)
        self.folders['reference'].with_suffix('.json').unlink()
        self.parity['receipt_sha256']['reference']=None
        timing=self.folders['reference']/'timing.json'
        values=json.loads(timing.read_text());values['device']='cpu_fp32'
        timing.write_text(json.dumps(values));self.parity['timing_sha256']['reference']=report.digest(timing)
        self.write_parity()
        self.assertIsNone(report.pair(self.entry)['receipts'][1])


if __name__=='__main__':unittest.main()
