import unittest
import numpy as np
from .reconstruction.sequence_clearance import constrain_sequence


class SequenceClearanceTests(unittest.TestCase):
    def test_shared_weights_prevent_a_fold_and_leave_unrelated_vertices_exact(self):
        base=np.array([[[0.,0,0],[1,0,0],[0,1,0],[2,2,2]]]*3,dtype=np.float32)
        desired=base.copy();desired[1,2,1]=-1
        corrected,weight,history=constrain_sequence(base,desired,np.array([[0,1,2]]))
        self.assertGreaterEqual(float(corrected[1,2,1]),.2)
        self.assertLess(weight[2],1)
        np.testing.assert_array_equal(corrected[:,3],base[:,3])
        self.assertEqual(history[-1]['constrained_vertices'],0)

    def test_noop_and_invalid_sequences(self):
        base=np.array([[[0.,0,0],[1,0,0],[0,1,0]]],dtype=np.float32)
        corrected,weight,history=constrain_sequence(base,base,np.array([[0,1,2]]))
        np.testing.assert_array_equal(corrected,base);self.assertEqual(history,[])
        for tri in (np.array([[0,1,3]]),np.array([[0.,1,2]])):
            with self.assertRaises(ValueError):constrain_sequence(base,base,tri)

    def test_midpoint_collapse_is_limited_even_when_saved_faces_keep_orientation(self):
        base = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]]] * 2,
                        dtype=np.float32)
        desired = base.copy()
        desired[1, 1] = [-1, 0, 0]
        desired[1, 2] = [0, -1, 0]
        corrected, weight, history = constrain_sequence(base, desired, np.array([[0, 1, 2]]))
        self.assertEqual(history[0]['minimum_relative_area'], 0)
        self.assertLess(weight[1], 1)
        midpoint = corrected.mean(axis=0)
        normal = np.cross(midpoint[1] - midpoint[0], midpoint[2] - midpoint[0])
        self.assertGreaterEqual(float(normal[2]), .2)


if __name__=='__main__':unittest.main()
