# Virtual-human face deformer (CUDA)

GPU backend of `ryzen/vhuman_deformer.h` for the rig packages written by
`server/vhuman/rig` (`<head>/rig/rig_deformer.safetensors`). The host
evaluates the rig per frame (`vh_deformer_prepare`: controls, correctives,
ML MLP, skinning matrices). The device holds the rest shape, all morphs and the
skin weights, and one fused kernel per batch blends the morphs and applies
4-influence LBS. The morph blend is register-tiled over 8 frames, so the
morph matrix is read once per tile. The driver API goes through cuew and
kernels are NVRTC-compiled at run time (no nvcc).

```sh
make -C cuda/vhuman
./cuda/vhuman/bench_vhuman_deformer_cuda <head>/rig/rig_deformer.safetensors 1024
```

RTX 5060 Ti, reference head (12.6k vertices), 1024 frames: 2.3 µs/frame in
the kernel with 117 morphs, and 6.4 µs/frame with 149 morphs (region-split ML
basis) while another compute process shared the GPU. The CPU deformer takes
1.9 ms single frame and 0.57 ms/frame batched. Host-side rig preparation takes
~9 µs/frame. The results equal the CPU deformer
(`test_rig`), and the downloaded positions are 50 MB per 1024 frames. The
Python wrapper is `server/vhuman/rig/native.py` (`NativeGPU`).
