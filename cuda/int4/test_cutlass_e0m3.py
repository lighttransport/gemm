"""Exercise the optimized E0M3 adapter, independent references and negative controls."""
from pathlib import Path
import subprocess
import unittest


class CutlassE0M3Tests(unittest.TestCase):
    runner = str(Path("../fp4/cublas_fp4_gemm"))
    patched = "./build/libcutlass_e0m3.so"
    baseline = "./build/libcutlass_e0m3_base.so"

    def invoke(self, library, fmt="e0m3", m=128, n=256, k=128, data="random", extra=()):
        return subprocess.run([self.runner,"--format",fmt,"--output","bf16",
            "--cutlass",library,"-m",str(m),"-n",str(n),"-k",str(k),
            "--data",data,"--full-check","--iters","2","--rounds","1",*extra],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def check_case(self, **kwargs):
        result = self.invoke(self.patched,extra=("--compare-cublas",),**kwargs)
        print(result.stdout, end="")
        self.assertEqual(result.returncode,0,result.stdout)
        self.assertIn("bit_mismatches=0",result.stdout)

    def test_all_256_code_pairs(self):
        self.check_case(m=16,n=16,k=64,data="codebook")

    def test_asymmetric_random(self):
        self.check_case()

    def test_m_and_k_tails(self):
        self.check_case(m=384,n=128,k=320)

    def test_unit_scales(self):
        self.check_case(m=16,n=16,k=64,data="ones")

    def test_fp4_baseline_control(self):
        result = self.invoke(self.baseline,fmt="nvfp4",m=16,n=16,k=64,
                             data="codebook",extra=("--compare-cublas",))
        print(result.stdout, end="")
        self.assertEqual(result.returncode,0,result.stdout)
        self.assertIn("bit_mismatches=0",result.stdout)

    def test_unpatched_library_rejected_by_reference(self):
        result = self.invoke(self.baseline,m=16,n=16,k=64,data="ones")
        self.assertNotEqual(result.returncode,0)
        self.assertIn("verify(",result.stdout)

    def test_unverified_e0m3_rejected(self):
        result = self.invoke(self.patched,extra=("--skip-verify",))
        self.assertNotEqual(result.returncode,0)


if __name__ == "__main__":
    unittest.main()
