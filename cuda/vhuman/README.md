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

A second kernel (`vh_contacts`, one block per frame) applies the exact
post-skinning contact projection (`server/vhuman/rig/contacts.py`). It is on by
default when the package has contact tensors.

Idle RTX 5060 Ti, reference head (12.6k vertices, 149 morphs), 1024 frames:
2.2 µs/frame deform, 6.7 µs/frame with exact contacts. Host preparation (MLP as
two `sgemm_avx2` calls) takes 4.5 µs/frame, and downloading 50 MB takes ~22 ms. The results equal the CPU deformer
(`test_rig`), and the downloaded positions are 50 MB per 1024 frames. The
Python wrapper is `server/vhuman/rig/native.py` (`NativeGPU`).
