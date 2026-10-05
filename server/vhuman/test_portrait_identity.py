"""Regression gates for subject mismatch and pre-fit accessory exclusion."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from .reconstruction.provenance import portrait_record, verify_rig_portrait
from .reconstruction.occlusion import parsing_masks

ROOT = Path(__file__).resolve().parents[2]/'tmp/test-portrait-identity'


class IdentityTests(unittest.TestCase):
    def setUp(self):
        ROOT.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_fixture_rig_cannot_be_used_for_another_person(self):
        rig = self.root/'rig';rig.mkdir()
        (rig/'rig.json').write_text('{}')
        Image.new('RGB',(32,32),(100,100,100)).save(self.root/'portrait.png')
        Image.new('RGB',(32,32),(200,100,100)).save(self.root/'hopper.png')
        with self.assertRaisesRegex(ValueError,'rebuild'):
            verify_rig_portrait(rig,self.root/'hopper.png')
        verify_rig_portrait(rig,self.root/'portrait.png')

    def test_lossless_reencoding_keeps_identity(self):
        rig = self.root/'rig';rig.mkdir()
        image = self.root/'original.bmp'
        Image.new('RGB',(32,32),(120,80,60)).save(image)
        Image.open(image).convert('RGB').save(self.root/'portrait.png')
        (rig/'rig.json').write_text(json.dumps(dict(portrait_provenance=portrait_record(image))))
        verify_rig_portrait(rig,self.root/'portrait.png')

    def test_pre_fit_masks_suppress_glasses_and_hat_without_geometry(self):
        image=self.root/'portrait.png';Image.new('RGB',(32,32)).save(image)
        class Parser:
            def predict(self, rgb):
                labels=np.ones((32,32),np.uint8)
                labels[:6]=18;labels[12:16,8:24]=6
                return labels,np.ones((32,32),np.float32)
        view=dict(image_path=str(image))
        parsing_masks(view,self.root/'mask',Parser())
        exclusion=np.asarray(Image.open(view['exclusion_mask_path']))
        self.assertEqual(exclusion[2,16],255)
        self.assertEqual(exclusion[14,16],255)
        self.assertEqual(exclusion[25,16],0)
        manual=self.root/'manual.png';Image.new('L',(32,32),0).save(manual)
        view['exclusion_mask_path']=str(manual)
        parsing_masks(view,self.root/'second',Parser())
        self.assertEqual(view['exclusion_mask_path'],str(manual))
