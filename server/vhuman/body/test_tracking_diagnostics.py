"""Track global/core errors separately and respect quaternion sign ambiguity."""
import unittest
import numpy as np
from .tracking_diagnostics import measure


class TrackingDiagnosticTest(unittest.TestCase):
    def test_core_failure_is_not_hidden_by_other_sites(self):
        reference = dict(positions=np.zeros((2,4,3)), qpos=np.tile([0,0,0,1,0,0,0],(2,1)), qvel=np.zeros((2,6)))
        rollout = {k:v.copy() for k,v in reference.items()}
        rollout['positions'][1,:2,0] = .2
        rollout['qpos'][:,3:] *= -1  # the same physical root orientation
        arrays, report = measure(reference,rollout,core_sites=[0,1],
            limits=dict(global_mpjpe_m=.15,core_mpjpe_m=.15,root_angle_rad=1.))
        self.assertAlmostEqual(arrays['global_mpjpe_m'][-1],.1)
        self.assertAlmostEqual(arrays['core_mpjpe_m'][-1],.2)
        np.testing.assert_allclose(arrays['root_angle_rad'],0)
        self.assertEqual(report['final_violations'],dict(global_mpjpe_m=False,core_mpjpe_m=True,root_angle_rad=False))


if __name__ == '__main__':
    unittest.main()
