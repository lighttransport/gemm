"""Extract the pinned Apache-2.0 gfx12 device implementation without PyTorch."""
from pathlib import Path
import subprocess, sys
REVISION = '30c94f510a0e0c54532ea3f940a69fac3b9fcd3e'
root = Path(__file__).resolve().parents[2]
source = root / 'tmp/sageattention-gfx12'
if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() != REVISION:
    raise SystemExit('SageAttention checkout does not match the pinned revision')
s = (source/'csrc/qattn/qk_int_sv_gfx12_native.cu').read_text()
s = s[:s.index('} // namespace')+len('} // namespace')]
s = s.replace('#include <torch/extension.h>', '').replace('#include <ATen/hip/HIPContext.h>', '')
s = s.replace('#include "../reduction_utils.cuh"', '#include "sage_reduction.cuh"')
a=s.index('void hip_kernel_launch_check()'); b=s.index('__device__',a);s=s[:a]+s[b:]
a=s.index('template <typename OutT, bool ToFp8>\ntorch::Tensor');b=s.index('template <typename T>\n__global__ void transpose_value_fp8',a);s=s[:a]+s[b:]
assert 'torch::' not in s and 'TORCH_CHECK' not in s
out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
(out.parent/'sage_reduction.cuh').write_text((source/'csrc/reduction_utils.cuh').read_text())
(out.parent/'SAGE_LICENSE').write_text((source/'LICENSE').read_text())
