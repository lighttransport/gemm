"""Fail early if a reference environment uses the wrong PyTorch build/device."""
import json
import sys
import torch

backend = sys.argv[1]
if backend == "rocm":
    assert torch.version.hip, "ROCm environment resolved a non-HIP torch"
elif backend == "cuda":
    assert torch.version.cuda and not torch.version.hip, "Expected CUDA torch"
else:
    assert not torch.version.cuda and not torch.version.hip, "Expected CPU torch"
device = "cpu" if backend == "cpu" else "cuda"
torch.manual_seed(0)
x = torch.randn(32, 32, device=device)
assert torch.isfinite(x @ x.T).all()
print(json.dumps({"backend": backend, "torch": torch.__version__,
                  "cuda": torch.version.cuda, "hip": torch.version.hip,
                  "device": "CPU" if device == "cpu" else torch.cuda.get_device_name(0)}, indent=2))
