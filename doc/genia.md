# GenIA integration: native foundations

This is an **incomplete implementation of the static GenIA plan**. Prepared
observations and native CUDA primitives work; end-to-end guided reconstruction
and test-time refinement are not implemented. The vhuman health endpoint exposes
that distinction, and reconstruction jobs fail rather than publishing an
unguided SAM3D result as GenIA. Existing head, body, and SAM3D paths remain the
default.

The reference is [GenIA](https://facebookresearch.github.io/GenIA/), release
`6fbd691ac90248eedb16e1099f2c7bef6831c69c`, with SAM3D source
`f91db411c50efee93d8db7aeb323885650f6f722`. See
[`ref/genia/lock.json`](../ref/genia/lock.json). GenIA's code is CC BY-NC 4.0;
SAM3D and the other components retain their own licenses. No upstream code or
model weights are vendored into the production modules.

## Implemented

- Versioned static observations: 1–16 RGB images with object masks, float32
  camera-Z depth, pinhole pixel intrinsics, and rigid camera-to-world transforms.
  Metres/centimetres/millimetres are converted consistently to metres. Depth
  unprojection follows upstream integer pixel centres, without a half-pixel shift.
- Bounded, validated ZIP uploads and a CLI preflight; HTTP validation reuses the
  existing vhuman queue, backend/device configuration, progress and cancellation.
  Uploads reject traversal, symlinks, extra files, duplicates, truncated archives,
  invalid calibration, empty masks and invalid foreground depth.
- CPU/CUDA/HIP SAM3D contexts retain their Stage-1 pose tokens. New
  `*_get_ss_pose_tokens` accessors return normalized 6D rotation, translation,
  log-scale and translation-scale; existing configuration structs and sampling
  entry points are unchanged. These are raw tokens, not decoded world poses.
- The separate SLAT conditioning fuser can be exported from the existing original
  checkpoint without overwriting SS weights. All 11 shared fuser tensors differ
  between the inspected SS and SLAT checkpoints. The 69,235,832-byte export and
  hash manifest are in `tmp/genia-native-models/`; this export is not wired into
  the production sampler yet.
- Shared host/device DDA visibility, stable floored observation fusion,
  passive-stream compensation equation, detached-velocity clean-latent estimate,
  and normalized/raw rendering-gradient velocity correction.
- A standalone CUDA library with device allocation budgets, explicit device
  selection and private contexts that restore the caller's CUDA context.
- Free anisotropic Gaussian projection, tiled depth-sorted RGB/alpha/expected-depth
  compositing, and analytic derivatives for Gaussian attributes and the camera
  affine matrix. The Gaussian input is the existing 17-float PLY layout. This is
  a renderer prototype: gsplat/upstream rendering parity is **not** established.

Projection uses scalar forward-mode dual numbers, one Jacobian column at a time.
This avoids the large per-thread scratch allocation of a 26-wide dual at every
covariance intermediate. It does not approximate derivatives with finite
differences. Finite differences are used only to test those derivatives.

## Prepared observation contract

Place this `scene.json` beside the assets:

```json
{
  "format": "vhuman.genia_observations.v1",
  "units": "m",
  "views": [{
    "image": "image.png",
    "mask": "mask.png",
    "depth": "depth.npy",
    "K": [[500, 0, 256], [0, 500, 256], [0, 0, 1]],
    "c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  }]
}
```

Each view uses R3 camera axes (+X right, +Y down, +Z forward), an undistorted
pinhole camera without skew, and a common world coordinate system. `mask.png`
is foreground above grayscale value 127. Image/mask/depth dimensions must agree;
images are limited to 4M pixels. Optional `image_sha256`, `mask_sha256` and
`depth_sha256` fields verify supplied asset hashes. All asset paths stay inside
the bundle. Background depth may be invalid; foreground depth must be finite
and positive.

CLI preflight:

```sh
python3 -m server.vhuman.cli --models-root /mnt/nvme01/models --backend cuda \
  genia --observations path/to/scene.json --check
```

Open `/genia` on the existing vhuman server to upload and validate a ZIP containing
`scene.json` and exactly its referenced files. Upload limit: 64 MiB compressed,
256 MiB expanded, 64 entries. `POST /v1/genia/uploads` accepts ZIP bytes and returns
an upload ID. Submit `{"kind":"genia_validate","upload_id":"..."}` to
`POST /v1/jobs`. `genia_reconstruct` is reserved and explicitly unavailable.
Validation does not create candidate geometry or modify accepted avatars.

## Native validation

```sh
make -C cuda/genia all test-host
XDG_CACHE_HOME="$PWD/tmp/genia-cache" TMPDIR="$PWD/tmp/genia-runtime" \
  ./cuda/genia/test_genia --gpu
python3 ref/genia/dump_math.py --upstream tmp/genia-upstream
XDG_CACHE_HOME="$PWD/tmp/genia-cache" TMPDIR="$PWD/tmp/genia-runtime" \
  python3 ref/genia/verify_math.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  server.vhuman.genia.test_observations server.vhuman.test_runtime -v
```

HTTP checks bind a local ephemeral port:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  server.vhuman.test_app.ServerTest.test_genia_prepared_upload_and_capability_gate \
  server.vhuman.test_app.ServerTest.test_rig_page_and_routes -v
```

On RTX 5060 Ti, the 1,019-voxel, six-camera pinned upstream DDA fixture matched
exactly. Fusion maximum absolute error was `1.1920929e-7`; clean-latent estimates
and normalized guidance matched exactly. The DDA uses reciprocal multiplication
and disables fused multiply-add to preserve upstream boundary tie behavior.
These results establish primitive parity, not model or reconstruction quality.
Receipts and build logs are under repository `tmp/genia-ref/` and
`tmp/genia-runtime/`. No performance claim is made for the shared GPU.

The library allocation budget covers its explicit buffers, excluding driver,
module and kernel scratch memory. Full pipeline memory residency and the 16 GiB
acceptance criterion remain unvalidated.

## Remaining static-plan milestones

1. Establish the full isolated PyTorch/CUDA 12.8 reference environment and capture
   prepared-scene stages, exact RNG draws, active losses and optimizer updates.
   Audit checkpoint variants and separately export SLAT conditioning/mesh/LPIPS
   assets where needed. The installed MoGe-2 camera-only path is not equivalent
   to GenIA's MoGe v1 or MapAnything preparation.
2. Extend native conditioning to match cropped/full image and mask streams,
   prenorm/register token layout, pointmap normalization and sampler-specific
   CFG. Implement fixed-shape pose sampling, SSI pose decoding, entropy-weighted
   shape fusion, depth refit, ICP and shared world-space poses.
3. Connect on-demand visibility bias to sparse cross-attention. Implement
   Gaussian-decoder backward and native SSIM/LPIPS/depth/normal/silhouette losses;
   verify the renderer against the upstream oracle before connecting guidance.
4. Implement guided SLAT denoising and per-scene rank-4 DoRA/latent/pose refinement,
   visibility gradient masks and unseen-opacity anchors. Preserve frozen base
   weights. Establish per-stage numerical gates and real held-out quality results.
5. Add the production runner, staged weight residency, memory guards, candidate
   publication, canonical/world placement exports, learned mesh exports and
   reconstruction previews. Enable health capabilities only after qualification.
6. Port and qualify the complete path on ROCm. Current HIP changes expose pose
   tokens only; no ROCm GenIA inference runner exists. A pre-existing duplicated
   `gemm_wmma` struct member was removed to restore the SAM3D HIP build.

Automatic preparation, external mesh inversion, MHR/head candidate promotion and
video remain follow-on work, as chosen in the plan.
