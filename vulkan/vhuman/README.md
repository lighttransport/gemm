# Virtual-human face deformer (Vulkan compute)

Vulkan backend of `ryzen/vhuman_deformer.h` for the rig packages written by
`server/vhuman/rig` (`<head>/rig/rig_deformer.safetensors`). It uses the same
design as `cuda/vhuman`: the host evaluates the rig per frame
(`vh_deformer_prepare`), and one compute shader (`shaders/vh_deform.comp`,
SPIR-V embedded at build time with `glslc -mfmt=c`) blends the morphs,
tiled over 8 frames, and applies 4-influence LBS. Vulkan is loaded at run
time through `../deps/vkew`, and the runner is `../deps/vulkan-runner`.

```sh
make -C vulkan/vhuman
./vulkan/vhuman/bench_vhuman_deformer_vk <head>/rig/rig_deformer.safetensors 1024
```

Memory: per-frame inputs (morph weights, skinning matrices) go to
device-local host-visible memory (resizable BAR) when the driver offers it.
Without it, every workgroup reads them over the bus, which was ~2x slower
here. Readback goes through a host-cached staging buffer. On the tested
NVIDIA driver that copy ran at only ~0.6 GB/s, so keep outputs on the GPU
(`out = NULL`) when rendering.

RTX 5060 Ti, reference head (12.6k vertices, 149 morphs), 1024 frames:
8–12 µs/frame for submit-to-completion. The results equal the CPU deformer
(`server/vhuman/test_rig`). The GPU was shared with another compute process
during these measurements.
