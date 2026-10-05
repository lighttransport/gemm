# ROCm expression candidates and GNM v3

Assets are stored under `/mnt/disk01/data/vhuman`. Run the idempotent,
checksum-pinned downloader with `python -m server.vhuman.face_assets`.
It installs Google's GNM v3 head (Apache-2.0), Google's MediaPipe Face
Landmarker task and its detector/landmark/blendshape weights (Apache-2.0),
and the author's BiSeNet ResNet18 face parser (MIT). `download.json` records
source URLs, revisions, byte counts and hashes. MediaPipe graphs run through
the repository CPU GEMM executor; the parser explicitly uses CPU ONNX Runtime.
Install `requirements-face-parsing.txt` in the ROCm environment and build
`make -C cpu/vhuman libvhuman_landmarks.so`.

The vhuman server and `video` CLI accept `--video-backend wan`, `h3`, `h3-fl2va` or
`hv15-rocm` with `--backend rocm` and experimental opt-in. Model directories
default to `/mnt/disk01/models/wan22`, `/mnt/disk01/models/h3/weights` and
`/mnt/disk01/models/hv15`. Each child owns the shared AMD device lock; the
server does not take it a second time. No CUDA GPU is required.

Wan uses Q8_0 weights and the HIP projection runner with hipBLASLt. Short
iterations use `fast5` and 9 frames (legal counts are 4*n+1).
H3 `fast5` uses six sigma points/five Euler updates, normally 22 frames
(17*n+5). `h3` conditions Ref2VA on the portrait as `<Picture 1>`;
`h3-fl2va` anchors FL2VA's first frame to the portrait. Both use a default
12,288 MiB budget, staged PyTorch ROCm image encoders, and the native HIP
language model, DiT and decoder. See the
[H3 setup and validation instructions](../../rdna4/minimax_h3/README.md).
HV1.5 supports 81 frames with `fast12` or `quality`.

Generate one candidate expression:

```sh
LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib \
  tmp/vhuman-rocm-venv/bin/python -m server.vhuman.rig.video_expressions \
  --portrait /path/to/portrait.png --out tmp/expression-candidate \
  --backend wan --preset fast5 --frames 9 --names smile
```

Use `--backend h3 --preset fast5 --frames 22`,
`--backend h3-fl2va --preset fast5 --frames 22`, or
`--backend hv15-rocm --preset fast12 --frames 81` for the other runners.
The server's existing `expressions` job also accepts `video_backend`,
`names`, `model`, `preset`, `frames` and `seed`; it writes a new directory
under the head's `rig/expression_candidates`.

Each clip is decoded and measured by native MediaPipe. The selected frame
maximizes the requested controls' measured response and must score at least
0.15. Missing/ambiguous faces are excluded. Capture reports retain all frame
observations, actual control weights, requested weights and clip hashes.
Parsing masks exclude non-skin features from registered appearance/detail
captures. Texture detail is inferred from shading, not measured displacement;
appearance contains lighting and is not an albedo estimate.

GNM rig builds export `face_model_source.expression_mapping` in `rig.json`:
a 383-by-control matrix, original regional coefficient names and residuals
in metres. The mapping projects actual transferred geometric morph deltas
into the aligned GNM basis with ridge regularization and clipping. GNM's
regional PCA channels do not share MediaPipe's semantic names. The final rig
mapping includes all 51 semantic controls and their joint motion, including
jaw opening. Combining coefficients is a linear approximation to nonlinear
joint rotations and correctives. Pass `--gnm-rig /path/to/rig.json` to record selected-frame GNM
coefficients; server jobs use the head's existing mapping when available.

Generated captures remain candidates. After reviewing identity, expression,
camera, visibility and artifacts, set those five `review` values to `true`
in the candidate manifest. Export to a new expression directory:

```sh
tmp/vhuman-rocm-venv/bin/python -m server.vhuman.rig.video_expressions \
  --portrait /path/to/portrait.png --out /path/to/new-rig/expressions \
  --reviewed-candidate tmp/expression-candidate
```

Then build the head with that new rig directory as `--out` and
`--face-model gnm_v3`. Export restores portrait coordinates and uses measured
driver weights. The rig builder produces grouped wrinkle atlases, individual
`wm_expr_<name>.png` normal perturbations and `appearance_<name>.png` lit color
atlases with confidence alpha. `wrinkles.expression_maps` links the observed
texture samples to their control drivers; paired expressions are not asserted
to be independent left/right measurements. The viewer uses `wm_side.png`
derived from geometry for GNM UVs. Live rigs are not overwritten by generation.

Validation on RX 9070 XT: Wan HIP, 480x832, nine frames, five steps took
153.583 s total, 29.997 s preparation/denoising, 8412.47 MiB peak allocated,
3000 HIP projections. Native MediaPipe found one face in all nine frames;
the selected smile weights were 0.837/0.800. Five-step quality is a smoke test,
not a final expression quality assessment. A portrait-conditioned H3 Ref2VA
candidate at 480x832, 22 frames and five updates completed with 5129.20 MiB
sampled peak and 309.914 s generation time excluding model verification.
MediaPipe found a face in all 22 frames and selected smile weights 0.949/0.950.
A first-frame FL2VA candidate at the same resolution and update count
completed in 202.508 s, with 5214.73 MiB sampled peak and a visible face in all
22 frames. Its selected smile weights were 0.910/0.870. Both H3 modes also
passed independent five-update, 64x64 diagnostics with exact latent updates.
Full-trajectory parity, identity review and HV1.5 expression capture remain
unvalidated. These single-run timings are not comparative throughput benchmarks.

The small synthetic GNM build exported a 383x51 matrix, jaw-driven coefficients,
the UV side mask and individual expression atlases. Its largest single-control
projection residual was 4.079 mm; this is an approximation, not an exact GNM
equivalent of the procedural rig. The synthetic portrait produced no detected
skin wrinkle signal and therefore flat maps. The real Wan capture also evaluated
successfully through this GNM mapping. Rig regression: 22 tests, one skipped;
video/mapping/native-observer/asset tests: 24 tests, one skipped. Validation data
and logs are under `tmp/video-rocm/wan22-build/expression-validation`.
