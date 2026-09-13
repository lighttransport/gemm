# Pixal3D reference and native adaptations

Pinned repository URLs and revisions are in `sources.json`. The native implementation
adapts algorithms from the Pixal3D main release; paper/Direct3D-S2 code is excluded.
Reference checkouts, Python environments and weights are local build artifacts.

- TencentARC/Pixal3D: Copyright (c) 2026 Tencent, MIT. Native model orchestration,
  projection, flow and sparse decoder implementations are C++ adaptations.
  See [license](licenses/Pixal3D-MIT.txt).
- valeoai/NAF: Apache-2.0. `cpu/pixal3d/conditioning.cc` replaces PyTorch convolutions
  and NATTEN execution with native convolution and projected-query neighborhood
  attention. See [license](licenses/NAF-Apache-2.0.txt). Its RoPE source also carries
  the DINOv3 attribution and license notice below.
- JeffreyXiang/CuMesh: Copyright (c) 2025 Jianfeng XIANG, MIT. Native `remesh.cc`,
  `simplify.cc`, `unwrap.cc` and hole filling adapt its UDF dual contouring,
  midpoint QEM collapse scheduling, chart aggregation and boundary-loop algorithms.
  See [license](licenses/CuMesh-MIT.txt).
- microsoft/TRELLIS.2: MIT. FDG extraction, PBR channel layout, sparse sampling and
  export conventions follow its o-voxel implementation. See
  [license](licenses/TRELLIS.2-MIT.txt).
- JeffreyXiang/FlexGEMM: MIT. Sparse trilinear sampling follows its CUDA boundary
  and valid-neighbor normalization rules. See [license](licenses/FlexGEMM-MIT.txt).
- DINOv3: Copyright (c) Meta Platforms, Inc. and affiliates. Model weights and
  associated DINOv3 code, including the RoPE module used by NAF, are covered by
  the [DINOv3 License Agreement](licenses/DINOv3-LICENSE.txt).
- Shared `common/xatlas.*`, `common/stb_image*.h`, `common/lightrt.*`, OpenBLAS and
  OpenCV retain their own license notices. The shared xatlas implementation has
  the same public API as CuMesh's namespace-renamed copy; atlas packing need not
  be byte-identical because chart aggregation and floating-point reductions differ.

The native ports change implementation language, memory management and execution
backend. No upstream weights are embedded or redistributed in this source tree.

`cpu/pixal3d/inpaint_opencv.cc` retains OpenCV 4.12.0's complete file license notice.
It adapts [the pinned upstream file](https://github.com/opencv/opencv/blob/4.12.0/modules/photo/src/inpaint.cpp)
by changing includes, helper linkage and the exported name; tracing is removed.
It uses the 4.12 stable priority heap and numerical behavior even when linked to
older system OpenCV core/imgproc libraries. This avoids the quadratic queue cost
of the system 4.6 photo implementation and matches the Python reference version.
