"""Actual BF16 tie regression; opt in with H3_GPU_TESTS=1 in a ROCm environment."""
import ctypes
import json
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get('H3_GPU_TESTS') == '1', 'requires AMD GPU opt-in')
class ConvrotGpuTests(unittest.TestCase):
    def test_bf16_rounding_boundary_matches_pytorch(self):
        import torch
        fixture = json.loads((Path(__file__).parent/'fixtures/convrot_bf16_boundary.json').read_text())
        self.assertTrue(torch.version.hip)
        torch.cuda.set_device(0)
        x = torch.zeros((fixture['rows'], 256), device='cuda', dtype=torch.bfloat16)
        x[fixture['row']] = torch.tensor(fixture['bf16_bits'], dtype=torch.uint16).view(torch.bfloat16).to('cuda')
        h = torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], dtype=torch.float32)
        rotation = h
        for _ in range(3):
            rotation = torch.kron(rotation, h)
        rotation = (rotation/16).to('cuda', torch.bfloat16)
        previous = torch.backends.cuda.preferred_blas_library()
        library = ctypes.CDLL(str(ROOT/'tmp/video-rocm/h3-build/libvideo_aotriton.so'))
        library.video_bf16_lt_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.video_bf16_lt_destroy.argtypes = [ctypes.c_void_p]
        library.video_bf16_lt_forward.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3 + [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        state = ctypes.c_void_p()
        self.assertEqual(library.video_bf16_lt_create(ctypes.byref(state)), 0)
        try:
            torch.backends.cuda.preferred_blas_library('hipblaslt')
            expected = x @ rotation
            output = torch.empty_like(expected)
            workspace = torch.empty(32*1024*1024, dtype=torch.uint8, device='cuda')
            status = library.video_bf16_lt_forward(state, output.data_ptr(), rotation.data_ptr(),
                x.data_ptr(), len(x), 256, 256, workspace.data_ptr(), workspace.numel(),
                torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            self.assertEqual(status, 0)
            self.assertTrue(torch.equal(output, expected))
            self.assertEqual(float(expected[fixture['row'], fixture['column']]), -0.054931640625)
        finally:
            torch.cuda.synchronize()
            library.video_bf16_lt_destroy(state)
            torch.backends.cuda.preferred_blas_library(previous)


if __name__ == '__main__':
    unittest.main()
