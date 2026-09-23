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

Validate the resident kernels against PyTorch ROCm with:

```sh
ref/pixal3d/run.sh rocm ref/pixal3d/validate_resident.py --backend rocm
```

The suite includes a 1,024-token, 12-head BF16 attention case and asserts
that the WMMA attention counter advances, so a scalar fallback cannot pass
unnoticed. On the RX 9070 XT this case has cosine 0.999999475 and normalized
RMSE 0.001025 against PyTorch ROCm. The shared test also passes on the RTX
5060 Ti CUDA backend.
