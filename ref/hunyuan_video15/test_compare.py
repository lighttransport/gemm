import unittest
from pathlib import Path
import tempfile
import numpy as np
from ref.hunyuan_video15.compare import compare, compare_frames
ROOT=Path(__file__).resolve().parents[2]

class FrameGateTest(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)

    def test_motion_frame_failure_survives_global_average(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            r,a=Path(directory)/'r',Path(directory)/'a'
            r.mkdir(); a.mkdir()
            reference=np.ones((1,3,81,2,2),dtype=np.float32)
            actual=reference.copy()
            actual[:,:,39]*=1.03
            np.save(r/'vae_decoded.npy',reference)
            np.save(a/'vae_decoded.npy',actual)
            self.assertTrue(compare(r,a,['vae_decoded'])['vae_decoded']['pass'])
            frames=compare_frames(r,a)
            self.assertEqual([v['frame'] for v in frames if not v['pass']],[39])
            actual[:,:,39]=np.nan
            np.save(a/'vae_decoded.npy',actual)
            with self.assertRaisesRegex(ValueError,'non-finite'):
                compare_frames(r,a)

    def test_bad_video_shape_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            p=Path(directory)
            np.save(p/'vae_decoded.npy',np.ones((1,3,4,4)))
            with self.assertRaisesRegex(ValueError,'shapes'):
                compare_frames(p,p)

if __name__=='__main__':
    unittest.main()
