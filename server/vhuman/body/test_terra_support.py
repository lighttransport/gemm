"""Heel lifts must not erase real raised supports or ramps."""
import unittest
import numpy as np
from .terra_support import flat_support_evidence


class NativeSupportTest(unittest.TestCase):
    def source(self):
        return dict(times=np.arange(12)/30,positions=np.zeros((12,17,3)),
                    sole_vertices=np.zeros((12,4,3)),sole_sides=np.array([0,0,1,1]))

    def test_toe_support_keeps_a_raised_heel_on_flat_ground(self):
        source = self.source()
        source['sole_vertices'][:,0,1] = np.linspace(0,.12,12)
        source['positions'][:,11,1] = np.linspace(.07,.19,12)
        report = flat_support_evidence(source)
        self.assertTrue(report['qualified'])
        self.assertEqual(report['stance_max_abs_clearance_m'],0)

    def test_platform_and_ramp_are_not_flattened(self):
        for elevation in (.12,np.linspace(0,.1,12)[:,None]):
            with self.subTest(elevation=str(elevation)):
                source = self.source()
                source['sole_vertices'][:,:,1] += elevation
                self.assertFalse(flat_support_evidence(source)['qualified'])

    def test_missing_and_unobservable_support_do_not_force_flat(self):
        self.assertFalse(flat_support_evidence({})['qualified'])
        source = self.source()
        source['positions'][:,:,0] = np.arange(12)[:,None]*.1
        self.assertFalse(flat_support_evidence(source)['qualified'])


if __name__ == '__main__':
    unittest.main()
