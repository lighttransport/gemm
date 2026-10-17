"""Compatibility entry point for the shared ROCm library resolver."""
from server.rocm_runtime import library_path

if __name__ == '__main__':
    print(library_path())
