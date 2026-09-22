# Pixal3D RDNA4 backend

This directory builds the ROCm shared backend used by `server/pixal3d`:

```sh
make -C rdna4/pixal3d
```

The library targets `gfx1201` by default and exposes the versioned resident
ABI from `common/pixal3d_device.h`. Resident linear layers use BF16 WMMA when
the caller selects `--gpu-kernels mma`; attention uses the RDNA4 fused WMMA
path for head dimension 128 and otherwise falls back to the HIP BLAS/scalar
implementation. `--gpu-kernels auto` enables the same path when the operation
shape is supported.

The workspace budget is enforced inside the backend, including pooled
resident buffers, so the web server can run the 1024-cascade within the
configured VRAM limit. The backend is loaded by the Pixal3D server through
`rdna4/pixal3d/libpixal3d_rocm.so`.
