"""Wan GGUF projections on the repository HIP kernels, with PyTorch ownership."""
import ctypes
from pathlib import Path
import types
import weakref

ROOT = Path(__file__).resolve().parents[2]


class HipRunner:
    def __init__(self, library=None, gemm="blaslt"):
        import torch
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("Wan requires PyTorch ROCm and an accessible AMD GPU")
        self.device = torch.cuda.current_device()
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
        if arch != "gfx1201":
            raise RuntimeError(f"This HIP build targets gfx1201; found {arch}")
        self.library = ctypes.CDLL(str(library or ROOT / "tmp/video-rocm/wan22-build/libwan22_hip.so"))
        self.projection = self.library.wan22_projection
        self.projection.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        self.projection.restype = ctypes.c_int
        if gemm not in ("wmma", "blaslt"):
            raise ValueError("GEMM must be wmma or blaslt")
        self.gemm = gemm
        self.context = ctypes.c_void_p()
        self.workspaces = {}
        if gemm == "blaslt":
            self.library.wan22_blas_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            self.library.wan22_blas_create.restype = ctypes.c_int
            self.library.wan22_blas_destroy.argtypes = [ctypes.c_void_p]
            self.library.wan22_blas_destroy.restype = None
            status = self.library.wan22_blas_create(ctypes.byref(self.context))
            if status:
                raise RuntimeError(f"Wan hipBLASLt initialization failed: {status}")
            self._cleanup = weakref.finalize(self, self.library.wan22_blas_destroy, self.context)
            self.blas_projection = self.library.wan22_projection_blas
            self.blas_projection.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_size_t] + [ctypes.c_int] * 3 + [ctypes.c_void_p]
            self.blas_projection.restype = ctypes.c_int
        self.calls = 0

    def linear(self, inputs, weight, bias=None):
        import torch
        from gguf import GGMLQuantizationType
        if weight.quant_type != GGMLQuantizationType.Q8_0:
            raise ValueError("HIP projection requires Q8_0 weights")
        n, k = weight.quant_shape
        if inputs.shape[-1] != k or k % 32:
            raise ValueError("Invalid Q8_0 projection shape")
        if not inputs.is_cuda or inputs.device != weight.device:
            raise ValueError("Input and weights must be on the same AMD GPU")
        if inputs.device.index != self.device:
            raise ValueError("Input GPU must match the HIP runner's device")
        if bias is not None and (bias.device != inputs.device or bias.ndim != 1 or bias.numel() != n):
            raise ValueError("Bias must be a vector on the input GPU with one value per output column")
        x = inputs.to(torch.float16).contiguous().reshape(-1, k)
        packed = weight.as_tensor().contiguous()
        if packed.dtype != torch.uint8 or packed.numel() != n * k // 32 * 34:
            raise ValueError("Invalid Q8_0 packed storage")
        scratch = torch.empty((n, k), dtype=torch.float16, device=x.device)
        stream = torch.cuda.current_stream(x.device)
        if self.gemm == "blaslt" and inputs.dtype == torch.float16:
            # One workspace per stream prevents races in asynchronous callers.
            key = (x.device.index, stream.cuda_stream)
            if key not in self.workspaces:
                self.workspaces[key] = torch.empty(32 * 1048576, dtype=torch.uint8, device=x.device)
            workspace = self.workspaces[key]
            out = torch.empty((x.shape[0], n), dtype=torch.float16, device=x.device)
            bias_half = bias.to(torch.float16).contiguous() if bias is not None else None
            status = self.blas_projection(self.context, out.data_ptr(), x.data_ptr(), packed.data_ptr(),
                                          scratch.data_ptr(), bias_half.data_ptr() if bias_half is not None else 0,
                                          workspace.data_ptr(), workspace.numel(), x.shape[0], n, k,
                                          stream.cuda_stream)
            if status:
                raise RuntimeError(f"Wan hipBLASLt projection failed: status {status}")
            self.calls += 1
            return out.reshape(*inputs.shape[:-1], n)
        out = torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)
        status = self.projection(out.data_ptr(), x.data_ptr(), packed.data_ptr(), scratch.data_ptr(),
                                 x.shape[0], n, k, stream.cuda_stream)
        if status:
            raise RuntimeError(f"Wan HIP projection failed: HIP error {status}")
        self.calls += 1
        # Round the accumulation before the separate activation-dtype bias add.
        out = out.to(inputs.dtype).reshape(*inputs.shape[:-1], n)
        if bias is not None:
            out = out + bias.to(inputs.dtype)
        return out

    def install(self, transformer):
        from diffusers.quantizers.gguf.utils import GGUFLinear
        from gguf import GGMLQuantizationType
        count = 0
        runner = self
        for module in transformer.modules():
            if isinstance(module, GGUFLinear):
                if module.weight.quant_type != GGMLQuantizationType.Q8_0:
                    raise ValueError("Checkpoint has a non-Q8_0 quantized projection")
                def forward(layer, inputs):
                    return runner.linear(inputs, layer.weight, layer.bias)
                module.forward = types.MethodType(forward, module)
                count += 1
        if count == 0:
            raise ValueError("No Q8_0 projections found; refusing a silent dense fallback")
        return count
