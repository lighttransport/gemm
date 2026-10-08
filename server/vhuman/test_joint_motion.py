import unittest
import numpy as np
from .reconstruction.joint_motion import shifted_joint_positions


def yaw(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


class ShiftedJointTests(unittest.TestCase):
    def setUp(self):
        self.joint=np.array([.03,.02,.01])
        self.offset=np.array([-.002,.0004,.0002])
        self.center=self.joint+self.offset
        self.points=self.center+np.array([[0,0,0],[.012,0,0],[0,.012,0],[0,0,.012]])

    def test_gaze_rotates_surface_but_does_not_orbit_fitted_center(self):
        rot=np.array([yaw(-.5),yaw(0),yaw(.5)])
        actual=shifted_joint_positions(self.points,self.joint,np.tile(self.joint,(3,1)),
            rot,self.offset,np.tile(np.eye(3),(3,1,1)))
        np.testing.assert_allclose(actual[:,0],np.tile(self.center,(3,1)),atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(actual[:,1:]-actual[:,:1],axis=-1),.012,atol=1e-15)
        self.assertGreater(np.linalg.norm(actual[0,1]-actual[2,1]),.001)

    def test_head_carries_offset_with_independent_eye_rotation(self):
        head=yaw(.4);eye=head@yaw(-.2);translation=np.array([.1,-.03,.2])
        actual=shifted_joint_positions(self.points,self.joint,
            (head@self.joint+translation)[None],eye[None],self.offset,head[None])[0]
        np.testing.assert_allclose(actual[0],head@self.center+translation,atol=1e-15)
        expected=(self.points-self.center)@eye.T+head@self.center+translation
        np.testing.assert_allclose(actual,expected,atol=1e-15)

    def test_zero_offset_matches_original_joint_formula(self):
        rot=np.array([yaw(.3)]);posed=self.joint[None]+.01
        actual=shifted_joint_positions(self.points,self.joint,posed,rot,np.zeros(3),rot)
        expected=np.einsum('fij,vj->fvi',rot,self.points-self.joint)+posed[:,None]
        np.testing.assert_array_equal(actual,expected)

    def test_invalid_shape_nonfinite_and_nonrotation_are_rejected(self):
        args=[self.points,self.joint,self.joint[None],np.eye(3)[None],self.offset,np.eye(3)[None]]
        for index,value in [(4,[0,0]),(4,[0,np.nan,0]),(3,(np.eye(3)*2)[None]),(5,np.eye(3))]:
            changed=args.copy();changed[index]=value
            with self.assertRaises(ValueError):shifted_joint_positions(*changed)


if __name__=='__main__':unittest.main()
