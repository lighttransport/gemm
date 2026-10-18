"""Independent reference checks of every supported decode tile and reduction."""
from pathlib import Path
import json
import subprocess
import unittest


class DecodeTests(unittest.TestCase):
    def run_case(self,m,n,k,data="random",extra=(),library="libdecode_e0m3.so",fmt="e0m3"):
        result = subprocess.run(["../fp4/cublas_fp4_gemm","--cutlass",f"./build/{library}",
            "--format",fmt,"--output","bf16","-m",str(m),"-n",str(n),"-k",str(k),
            "--data",data,"--full-check","--warmup","1","--iters","2","--rounds","1",*extra],
            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout)
        self.assertIn("PASS",result.stdout)
        return result.stdout

    def test_patch_manifest(self):
        manifest = json.loads(Path("build/libdecode_e0m3.patch.json").read_text())
        self.assertFalse(manifest["ptx_fallback"])
        self.assertEqual(manifest["instruction_count"],manifest["changed_byte_count"])
        self.assertGreater(manifest["instruction_count"],200)

    def test_all_code_pairs(self):
        self.run_case(16,16,64,"codebook",("--compare-cublas",))

    def test_single_token_and_tails(self):
        self.run_case(1,24,320,extra=("--compare-cublas",))

    def test_partial_m_n_k_tiles(self):
        self.run_case(17,40,576,extra=("--compare-cublas",))

    def test_multiple_m_tiles(self):
        self.run_case(63,72,192,extra=("--compare-cublas",))

    def test_graph_cold_global_reduction(self):
        self.run_case(1,128,576,extra=("--candidate","28","--graph","--flush-mb","64","--compare-cublas"))

    def test_graph_local_reduction(self):
        self.run_case(17,40,320,extra=("--candidate","49","--graph","--compare-cublas"))

    def test_dp4a_reference(self):
        self.run_case(1,128,320,extra=("--candidate","36","--graph","--compare-cublas"))

    def test_fp4_reuse(self):
        self.run_case(1,32,128,extra=("--compare-cublas",),fmt="nvfp4",library="libdecode_base.so")


if __name__ == "__main__":
    unittest.main()
