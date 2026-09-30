"""Coordinate convention and public fixture import checks."""
import tempfile
import struct
import unittest
from pathlib import Path
import numpy as np
from .reconstruction.reference import Camera
from .reconstruction.prepare_datasets import multiface_camera, load_krt, exr_header


class DatasetPreparationTests(unittest.TestCase):
    def test_anisotropic_skew_projection_and_rays(self):
        camera=Camera(710.,320.,240.,np.array([0.,0.,1.]),np.eye(3),695.,2.)
        points=np.array([[.02,.01,0.],[-.1,.04,.1]])
        xy,depth=camera.project(points)
        rays=camera.rays(xy)
        target=points-camera.origin;target/=np.linalg.norm(target,axis=1,keepdims=True)
        np.testing.assert_allclose(rays,target,atol=1e-12)
        np.testing.assert_allclose(camera.scaled(.5).project(points)[0],xy*.5)
        np.testing.assert_allclose(Camera.from_dict(camera.as_dict()).project(points)[0],xy)
        self.assertTrue((depth>0).all())
        invalid=camera.as_dict();invalid['focal_y']=float('nan')
        with self.assertRaises(ValueError):Camera.from_dict(invalid)

    def test_multiface_projection_composes_head_pose_and_millimetres(self):
        from scipy.spatial.transform import Rotation
        k=np.array([[780.,1.,320.],[0.,790.,240.],[0.,0.,1.]])
        head=np.column_stack((Rotation.from_euler('xyz',[.1,-.2,.03]).as_matrix(),[3.,8.,1000.]))
        rt=np.column_stack((Rotation.from_euler('xyz',[-.04,.05,.1]).as_matrix(),[2.,4.,40.]))
        points=np.array([[30.,20.,0.],[-20.,-10.,4.],[0.,0.,10.]])
        cv=(points@head[:,:3].T+head[:,3])@rt[:,:3].T+rt[:,3]
        pixels=cv@k.T;pixels=pixels[:,:2]/pixels[:,2:]
        camera=multiface_camera(k,rt,head)
        np.testing.assert_allclose(camera.project(points*.001)[0],pixels,atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(camera.rotation),1.)

    def test_distortion_rejected_and_exr_headers_bounded(self):
        scratch=Path(__file__).resolve().parents[2]/'tmp/test-dataset-preparation'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            path=Path(folder)/'KRT'
            text='camera\n700 0 320\n0 701 240\n0 0 1\n0 0 0 0 0\n1 0 0 0\n0 1 0 0\n0 0 1 1000\n'
            path.write_text(text);self.assertEqual(len(load_krt(path)),1)
            path.write_text(text.replace('0 0 0 0 0','0.1 0 0 0 0'))
            with self.assertRaisesRegex(ValueError,'distortion'):load_krt(path)
            path.write_bytes(b'not an EXR')
            with self.assertRaisesRegex(ValueError,'magic'):exr_header(path)
            data=struct.pack('<4i',0,0,63,31)
            path.write_bytes(struct.pack('<II',20000630,2)+b'dataWindow\0box2i\0'+struct.pack('<I',len(data))+data+b'\0')
            self.assertEqual(exr_header(path)['data_window'],[0,0,63,31])


if __name__=='__main__':unittest.main()
