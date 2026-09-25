"""Reject misleading layer comparisons with wrong shape, layout or provenance."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from compare_qwen35_prefill_layer import STAGES, compare


class LayerComparisonTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent/'tmp'
        root.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='layer-comparison-',dir=root)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runner=self.root/'runner';self.runner.mkdir()
        self.reference=self.root/'reference';self.reference.mkdir()
        for index,(ref,ours,axis) in enumerate(STAGES):
            self.assertEqual(axis,1)
            path=self.reference/f'{index:04d}-{ref}-0.bin'
            np.arange(6,dtype='<f4').tofile(path)
            Path(str(path)+'.json').write_text(json.dumps({'type':0,'shape':[2,3,1,1],'strides':[4,8,24,24]}))
            np.array([4,5],dtype='<f4').tofile(self.runner/f'runner-{ours}-00.bin')

    def test_exact_and_mismatch(self):
        self.assertTrue(all(x['bit_identical'] for x in compare(self.runner,self.reference,0)))
        np.array([4,6],dtype='<f4').tofile(self.runner/'runner-attn-norm-00.bin')
        result=compare(self.runner,self.reference,0)[0]
        self.assertFalse(result['bit_identical']);self.assertEqual(result['max_abs'],1)

    def test_wrong_shape(self):
        np.zeros(3,dtype='<f4').tofile(self.runner/'runner-attn-norm-00.bin')
        with self.assertRaisesRegex(ValueError,'token row'):compare(self.runner,self.reference,0)

    def test_noncontiguous(self):
        path=self.reference/'0000-attn_norm-0.bin.json'
        m=json.loads(path.read_text());m['strides'][1]=16;path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'noncontiguous'):compare(self.runner,self.reference,0)

    def test_ambiguous_chunk(self):
        (self.reference/'9999-attn_norm-0.bin').write_bytes(b'')
        with self.assertRaisesRegex(ValueError,'one reference tensor'):compare(self.runner,self.reference,0)

    def test_nonfinite(self):
        np.array([4,np.nan],dtype='<f4').tofile(self.runner/'runner-attn-norm-00.bin')
        with self.assertRaisesRegex(ValueError,'nonfinite'):compare(self.runner,self.reference,0)


if __name__=='__main__':
    unittest.main()
