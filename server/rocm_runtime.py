"""Locate a complete installed ROCm library set for the pinned PyTorch wheel."""
from pathlib import Path


def library_path():
    # Newer core alternatives can omit the SMI ABI required by ROCm 7.2 Torch.
    candidates = ('/opt/rocm/core/lib', '/opt/rocm/core-10.0/lib', '/opt/rocm/lib')
    for directory in candidates:
        path = Path(directory)
        if all((path/name).is_file() for name in ('libamdhip64.so', 'libroctx64.so.4', 'librocm_smi64.so.1')):
            return directory
    return '/opt/rocm/core/lib'


if __name__ == '__main__':
    print(library_path())
