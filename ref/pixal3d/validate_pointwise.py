"""Check host vector paths, precision boundaries and scalar tails vs PyTorch."""
import ctypes as C
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
root=Path(__file__).resolve().parent.parent.parent
lib=C.CDLL(str(root/'cpu/pixal3d/libpixal3d_validation.so'))
fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
lib.px_test_pointwise.argtypes=[fp,fp,fp,fp,C.c_int,C.c_int,C.c_int,C.c_int]
rng=np.random.default_rng(814)
for rows,channels in [(3,29),(47,1536)]:
    x=rng.normal(0,3,(rows,channels)).astype(np.float32)
    h=rng.normal(0,.4,x.shape).astype(np.float32)
    params=rng.normal(0,.3,2*channels).astype(np.float32)
    for operation in range(5):
        for precision,dtype in [(0,torch.float32),(1,torch.bfloat16),(2,torch.float16)]:
            if operation>=3 and precision==2:continue
            values=torch.tensor(x);other=torch.tensor(h);mod=torch.tensor(params)
            round_dtype=lambda t:t.to(dtype).float()
            if operation==0:expected=round_dtype(values)
            elif operation in [1,2]:expected=round_dtype(F.gelu(values,approximate='tanh' if operation==1 else 'none'))
            elif operation==3:expected=round_dtype(values+round_dtype(other*mod[:channels]))
            else:expected=round_dtype(round_dtype(round_dtype(values)*round_dtype(1+mod[channels:]))+mod[:channels])
            actual=np.empty_like(x)
            assert lib.px_test_pointwise(actual,x,h,params,rows,channels,operation,precision)==0
            expected=expected.numpy()
            if operation in [0,3,4]:np.testing.assert_array_equal(actual,expected)
            else:np.testing.assert_allclose(actual,expected,atol=1e-5 if not precision else .005,rtol=1e-4 if not precision else .005)
            print(f'Pointwise op={operation} precision={precision} rows={rows} channels={channels}: PASS',flush=True)
