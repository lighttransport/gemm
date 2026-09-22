from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image

from editing_inputs import prepare_image, write_layout


class EditingInputsTest(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[2]/"tmp"
        root.mkdir(exist_ok=True)
        self.work=tempfile.TemporaryDirectory(dir=root,prefix="qimg21-edit-inputs-")
        self.path=Path(self.work.name)

    def tearDown(self):
        self.work.cleanup()

    def test_rgba_preprocessing(self):
        source=self.path/"image.png"
        Image.new("RGBA",(64,64),(20,80,160,200)).save(source)
        self.assertEqual(prepare_image(source,self.path/"prepared",64),(4,4))
        actual=np.load(self.path/"prepared/image.npy")
        expected=np.array([20,80,160,200],np.float32)/np.float32(255)*2-1
        np.testing.assert_allclose(actual[:,0,0],expected,rtol=0,atol=1e-7)
        self.assertEqual(actual.shape,(4,64,64))
        with Image.open(self.path/"prepared/resized.png") as image:
            self.assertEqual(image.mode,"RGBA")

    def test_text_helper_passes_image_list_for_both_branches(self):
        import torch
        from test_cuda_qimg21 import _dump_prompt
        from unittest.mock import Mock, patch
        pipe=Mock()
        pipe.encode_prompt.return_value=(torch.zeros(1,7,4096),None,torch.zeros(1,7,dtype=torch.bool))
        image=Image.new("RGBA",(64,64))
        with patch("test_cuda_qimg21._torch", return_value=torch):
            _dump_prompt(pipe,"edit",self.path,image,negative_prompt="")
        self.assertEqual(pipe.encode_prompt.call_count,2)
        for call in pipe.encode_prompt.call_args_list:
            self.assertEqual(call.kwargs["image"],[image])

    def fixtures(self, prefix="", native=False):
        shape=(7,4096) if native else (1,7,4096)
        np.save(self.path/f"{prefix}prompt_embeds.npy",np.zeros(shape,np.float32))
        np.save(self.path/f"{prefix}image_pad_mask.npy",np.array([[0,1,1,1,1,0,0]],bool))
        np.save(self.path/f"{prefix}prompt_mask.npy",np.ones((1,7),bool))

    def test_layout_native_reader_and_negative_branch(self):
        for native in (False,True):
            for negative in (False,True):
                self.fixtures("negative_" if negative else "", native=native)
                output=self.path/(f"{'native' if native else 'batched'}-{'negative' if negative else 'positive'}.txt")
                write_layout(self.path,output,(4,4),(4,4),negative)
                binary=Path(__file__).with_name("test_joint_layout")
                result=subprocess.run([str(binary),str(output)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(result.stdout.splitlines()[0],"35 19 32")

    def test_reject_mismatched_or_split_vision_slots(self):
        self.fixtures()
        with self.assertRaises(ValueError):
            write_layout(self.path,self.path/"bad.txt",(8,4),(4,4))
        np.save(self.path/"image_pad_mask.npy",np.array([[1,1,0,1,1,0,0]],bool))
        with self.assertRaises(ValueError):
            write_layout(self.path,self.path/"bad.txt",(4,4),(4,4))


if __name__=="__main__":
    unittest.main()
