#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = None
CHECKER = None

class StreamTest(unittest.TestCase):
    def test_cross_layout_frontend(self):
        # Reuse the NumPy fixture generator on development hosts; the shipped
        # comparator and its Fugaku invocation require only the standard library.
        from test_compare_glm53f_fields import capture
        from compare_glm53f_fields_stream import compare
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            a, b = str(Path(directory) / 'tp12'), str(Path(directory) / 'pp')
            capture(a, False); capture(b, True)
            result = compare(a, b, CHECKER, bit_exact=True)
            self.assertTrue(result['pass'])
            self.assertEqual(result['layer_worst']['layer0']['relative_l2'], 0)
            path = Path(b + '.rank00.layer00.kda')
            with path.open('r+b') as f:
                f.write(struct.pack('<f', 2))
            self.assertFalse(compare(a, b, CHECKER)['pass'])
            with path.open('r+b') as f:
                f.write(struct.pack('<f', 1))
            # Sparse FP32 data are replicated: even a tiny single-rank change
            # must fail the exact replica gate rather than the L2 threshold.
            path = Path(b + '.rank01.layer03.sparse')
            with path.open('r+b') as f:
                f.seek(20); f.write(struct.pack('<f', 1.0001))
            self.assertFalse(compare(a, b, CHECKER)['pass'])
            with path.open('r+b') as f:
                f.seek(20); f.write(struct.pack('<f', 1))
            path = Path(b + '.rank00.layer00.kda')
            path.write_bytes(path.read_bytes()[:-4])
            with self.assertRaises(ValueError):
                compare(a, b, CHECKER)

    def test_numeric_metadata_and_request_guards(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            a,b=Path(directory)/'a',Path(directory)/'b'
            def run(kind,count,limit=0,exact=False,oa=0,ob=0):
                command=[CHECKER]+(['--bit-exact'] if exact else [])
                request='\t'.join(map(str,(kind,'field',a,oa,b,ob,count,limit)))+'\n'
                r=subprocess.run(command,input=request,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                return r,json.loads(r.stdout) if r.stdout else None
            a.write_bytes(struct.pack('<4f',1,-1,.5,0));b.write_bytes(struct.pack('<4f',1.0001,-1,.5,0))
            self.assertTrue(run('F',4)[1]['pass']);self.assertFalse(run('F',4,exact=True)[1]['pass'])
            b.write_bytes(struct.pack('<4f',1.01,-1,.5,0));self.assertFalse(run('F',4)[1]['pass'])
            for value in (float('nan'),float('inf'),-float('inf')):
                b.write_bytes(struct.pack('<4f',value,-1,.5,0));self.assertEqual(run('F',4)[1]['reason'],'nonfinite')
            a.write_bytes(struct.pack('<f',0));b.write_bytes(struct.pack('<f',-0.0));self.assertEqual(run('F',1)[1]['reason'],'zero_norm_requires_exact')
            a.write_bytes(struct.pack('<16f',*range(16)));b.write_bytes(a.read_bytes());self.assertTrue(run('M',4,exact=True)[1]['pass'])
            b.write_bytes(b.read_bytes()+b'x');self.assertTrue(run('B',64)[1]['pass']);self.assertFalse(run('B',65)[1]['pass'])
            a.write_bytes(struct.pack('<3i',0,2,3));b.write_bytes(struct.pack('<3i',0,1,3));self.assertEqual(run('S',3,4)[1]['changes'],1)
            b.write_bytes(struct.pack('<3i',0,0,3));self.assertEqual(run('S',3,4)[1]['reason'],'duplicate_indices')
            b.write_bytes(struct.pack('<3i',0,1,4));self.assertEqual(run('S',3,4)[1]['reason'],'invalid_indices')
            a.write_bytes(struct.pack('<8i8f',*range(8),*[.125]*8));b.write_bytes(a.read_bytes());self.assertTrue(run('R',1,exact=True)[1]['pass'])
            b.write_bytes(struct.pack('<8i8f',0,0,2,3,4,5,6,7,*[.125]*8));self.assertEqual(run('R',1)[1]['reason'],'duplicate_routes')
            self.assertFalse(run('F',4,oa=2**63)[1]['pass'])
            self.assertEqual(run('F',2**64-1)[0].returncode,2)
            # Independent cached cursors are required even for the same file.
            request='\t'.join(map(str,('F','self',a,0,a,0,16,0)))+'\n'
            result=subprocess.run([CHECKER,'--bit-exact'],input=request,universal_newlines=True,stdout=subprocess.PIPE)
            self.assertEqual(result.returncode,0)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checker');parser.add_argument('temporary_root');args=parser.parse_args()
    CHECKER=args.checker;ROOT=args.temporary_root
    unittest.main(argv=['test_field_stream'],verbosity=2)
