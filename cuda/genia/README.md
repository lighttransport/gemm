# Native GenIA CUDA primitives

Driver API + NVRTC implementation. No Torch, cuBLAS, gsplat or model downloads.
This is not yet an end-to-end GenIA runner. See [integration status](../../doc/genia.md).

```sh
make -C cuda/genia all test-host
XDG_CACHE_HOME="$PWD/tmp/genia-cache" TMPDIR="$PWD/tmp/genia-runtime" \
  ./cuda/genia/test_genia --gpu
```

`genia_cuda.h` exposes visibility, observation fusion, clean-latent estimation,
velocity correction, and free Gaussian forward rendering/vector-Jacobian
products. Host arrays are copied into a private CUDA context. Calls on a context
must be serialized; prior caller context is restored after each call. Allocation
budgets apply to explicit device buffers, not driver/module/kernel scratch.
Peak allocation is queryable; errors return `-1` with `genia_cuda_error()`.

Gaussians use the repository 17-float PLY property order: XYZ, unused normal,
SH DC, opacity logit, three log scales, and a wxyz quaternion. Render cameras
are R3 world-to-camera affine matrices with separate pixel-space intrinsics.
Input loss gradients may include RGB, alpha and expected depth. Output gradients
cover Gaussian attributes and all 12 camera matrix entries. The renderer uses
16×16 tiles, stable front-to-back sorting and a bounded 8M tile-overlap list.
Camera gradients are affine-matrix derivatives; a pose-token chain is still needed.

The tests exercise single/multiple Gaussian compositing, expected depth,
background transmission, finite-difference Gaussian/camera derivatives,
zero/nonfinite guidance, visibility edge cases and allocation-budget rejection.
The renderer still needs independent upstream gsplat parity before production use.
