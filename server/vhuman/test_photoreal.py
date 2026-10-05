"""Calibration and physical detail invariants for the offline head pipeline."""
import unittest
import numpy as np
from .reconstruction.reference import Camera
from .reconstruction.temporal import crop_camera
from .reconstruction.skin_detail import driver_matrix,evaluate
from .reconstruction.offline_assets import bound_tubes,attachment_frames,curved_cap


class PhotorealTests(unittest.TestCase):
    def test_curved_cap_preserves_outline_and_is_closed(self):
        camera=Camera(1400,255,299,np.array([.02,.03,1.5]),np.eye(3))
        angle=np.linspace(0,2*np.pi,40,endpoint=False)
        contour=np.array([255,140])+np.stack((90*np.cos(angle),70*np.sin(angle)),-1)
        vertices,faces,split=curved_cap(camera,contour,.03)
        pixels,_=camera.project(vertices[:len(contour)])
        np.testing.assert_allclose(pixels,contour,atol=1e-10)
        edges=np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
        _,counts=np.unique(edges,axis=0,return_counts=True)
        np.testing.assert_equal(counts,2)
        triangles=vertices[faces]
        area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
        self.assertGreater(area.min(),1e-10)
        volume=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2])).sum()/6
        self.assertGreater(volume,0)
        self.assertGreater(np.ptp(vertices[:,2]),.01)

    def test_concave_cap_fan_does_not_fold(self):
        camera=Camera(1400,255,299,np.array([.02,.03,1.5]),np.eye(3))
        contour=np.array([[0,0],[4,0],[4,4],[2,2],[0,4]],float)*30+[200,70]
        vertices,faces,split=curved_cap(camera,contour,.03)
        pixels,_=camera.project(vertices)
        t=pixels[faces[:split]]
        a=t[:,1]-t[:,0];b=t[:,2]-t[:,0]
        signed=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        self.assertTrue((signed>1e-8).all() or (signed<-1e-8).all())

    def test_generated_crop_preserves_camera_rays(self):
        camera=Camera(1400,255,299,np.array([.02,.03,1.5]),np.eye(3))
        target=np.array([480,832]);source=np.array([510,598])
        cropped=crop_camera(camera,source,target)
        points=np.array([[0,0,0],[.03,-.04,.02],[-.04,.01,-.05]])
        original,_=camera.project(points);actual,_=cropped.project(points)
        scale=max(target/source);offset=(source*scale-target)/2
        np.testing.assert_allclose(actual,original*scale-offset,atol=1e-10)
        np.testing.assert_allclose(cropped.rays(actual),camera.rays(original),atol=1e-10)

    def test_metric_wrinkle_drivers_have_neutral_and_compression_gauges(self):
        vertices=np.array([[0.,0,0],[.01,0,0],[0,.01,0]])
        tri=np.array([[0,1,2]])
        basis=np.stack((-vertices*.01,vertices*.01,np.tile([0,0,.001],(3,1))))
        drivers=driver_matrix(vertices,tri,basis,np.ones((1,3)))
        np.testing.assert_allclose(drivers,[[.12,-.12,0]],atol=1e-7)
        reference=np.array([1,2,3])
        np.testing.assert_equal(evaluate(drivers,reference,reference),[0])
        np.testing.assert_equal(evaluate(drivers,reference+[100,0,0],reference),[1])
        with self.assertRaises(ValueError):evaluate(drivers,[np.nan,0,0],reference)

    def test_lid_bound_tubes_follow_rigid_motion_without_double_transform(self):
        full=np.array([[0.,0,0],[.01,0,0],[0,.01,0]])
        roots=np.tile([0,1,2],(2,1));weights=np.tile([.4,.3,.3],(2,1))
        path=np.array([[.003,.003,0],[.003,.003,.004]])
        vertices,tri,binding=bound_tubes([path],[roots],[weights],full,.000025)
        angle=.4;r=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        translation=np.array([.03,-.04,.01]);moved=full@r.T+translation
        centres=(moved[binding['ids']]*binding['weights'][:,:,None]).sum(1)
        frames=attachment_frames(moved[binding['ids'][:,:3]])
        actual=centres+np.einsum('vij,vj->vi',frames,binding['offsets'])
        np.testing.assert_allclose(actual,vertices@r.T+translation,atol=1e-12)
        self.assertEqual(tri.shape,(16,3))


if __name__=='__main__':unittest.main()
