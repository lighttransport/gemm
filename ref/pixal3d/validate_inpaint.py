"""Bit-exact native OpenCV 4.12 inpainting parity on structured/random holes."""
import ctypes as C
from pathlib import Path
import cv2
import numpy as np
root=Path(__file__).resolve().parent.parent.parent
lib=C.CDLL(str(root/'cpu/pixal3d/libpixal3d_validation.so'))
ptr=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
lib.px_test_inpaint.argtypes=[ptr,ptr,C.c_int,C.c_int,C.c_int]
rng=np.random.default_rng(4612)
for size in [63,128,257]:
    yy,xx=np.indices((size,size))
    for channels in [1,3]:
        pixels=rng.integers(0,256,(size,size,channels),dtype=np.uint8)
        for radius in [1,3]:
            mask=(((xx-size*.5)**2+(yy-size*.5)**2)<(size*.3)**2).astype(np.uint8)
            mask[3:7,:size//2]=1;mask[-6:,-9:]=1
            reference=cv2.inpaint(pixels,mask,radius,cv2.INPAINT_TELEA).reshape(pixels.shape)
            actual=pixels.copy()
            assert lib.px_test_inpaint(actual,mask,size,channels,radius)==0
            np.testing.assert_array_equal(actual,reference)
            print(f'Inpaint {size}x{size}, channels={channels}, radius={radius}: exact PASS',flush=True)
