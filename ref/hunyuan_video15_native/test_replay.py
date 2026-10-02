"""Replay gates must reject bad tensors and misleading timing summaries."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
HERE=Path(__file__).resolve().parent
def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
replay=module('replay');report=module('report_replays')


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


if __name__=='__main__':unittest.main()
