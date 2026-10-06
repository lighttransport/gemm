"""Native portrait refinement: withheld-evidence and anatomy safeguards."""
import unittest
import numpy as np
from .reconstruction.refine_fit import acceptance_gate,landmark_metrics,MOUTH


class NativeFitTests(unittest.TestCase):
    def test_gate_rejects_training_only_gain_folds_and_nonmouth_regression(self):
        before=dict(all_px=4.,heldout_px=4.,mouth_px=6.,heldout_mouth_px=7.,nonmouth_px=3.)
        after=dict(all_px=2.,heldout_px=2.6,mouth_px=1.9,heldout_mouth_px=2.5,nonmouth_px=2.3)
        self.assertTrue(acceptance_gate(before,after,.16))
        for bad in (dict(heldout_px=4.1),dict(heldout_mouth_px=7.1),dict(nonmouth_px=3.4),dict(mouth_px=float('nan'))):
            self.assertFalse(acceptance_gate(before,dict(after,**bad),.16))
        self.assertFalse(acceptance_gate(before,after,.01))

    def test_heldout_metric_does_not_count_training_fit(self):
        target=np.zeros((468,2));predicted=np.zeros_like(target)
        weights=np.ones(468);heldout=np.arange(468)%10==0
        predicted[heldout,0]=3.;predicted[~heldout,0]=100.
        report=landmark_metrics(predicted,target,weights,heldout)
        self.assertAlmostEqual(report['heldout_px'],3.)
        self.assertAlmostEqual(report['heldout_mouth_px'],3.)
        self.assertGreater(report['all_px'],80.)


if __name__=='__main__':unittest.main()
