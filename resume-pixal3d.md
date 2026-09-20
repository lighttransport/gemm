# Resume Pixal3D work

## Current state

Work on the Pixal3D **main** release is on branch `pixal3d`. The native CPU and
CUDA pipelines, single-view and posed multiview inference, mixed/FP32 precision
modes, 7 GiB and 12 GiB CUDA memory paths, pinned PyTorch reference runners,
RMBG/MoGe preparation, GLB/PLY export, and the Python web demo are implemented.
The worktree was clean before this handoff file was added.

Recent commits:

- `aadd125c` documents RMBG/BiRefNet mask-parity provenance.
- `a62a785c` limits RMBG setup to required inference files and hardens validation.
- `01534c1c` adds real RMBG-2.0 CUDA validation.
- `a0bb8a63` adds matched render and silhouette diagnostics.
- `0b8fc6bd` reduces byte-identical CPU boundary-processing memory.
- `f1db5126` fits the complete four-view PyTorch reference run on a 16 GB CUDA GPU.
- `c90aa5f9` adds the repeated CUDA reliability soak.
- `dda25263` validates complete four-view generation at the 7 and 12 GiB budgets.
- `9550f93b` through `2b4a34ba` cover web lifecycle hardening and functional browser tests.

Validated hardware and model locations:

- NVIDIA GeForce RTX 5060 Ti, 16 GB, CUDA/PyTorch 2.7.1+cu128.
- Pixal3D weights: `/mnt/disk2/models/Pixal3D`.
- DINOv3: `/mnt/disk2/models/dinov3-vitl16/model.safetensors`.
- RMBG-2.0: `/mnt/disk2/models/RMBG-2.0`.
- Exact upstream BiRefNet comparison checkpoint: `/mnt/disk2/models/BiRefNet`.
- MoGe-2: `/mnt/disk2/models/moge-2-vitl/model.pt`.
- NAF: `ref/pixal3d/weights/naf_release.safetensors`.

Important measured results:

- Four-view native output is byte-identical at 7168 and 12288 MiB budgets.
- Four-view native versus pinned PyTorch symmetric Chamfer RMS: `0.005548`.
- Absolute normal cosine means: `0.9533` and `0.9586`.
- Matched preview PSNR: `26.39–29.09 dB`; silhouette IoU: `0.9885–0.9949`.
- RMBG-2.0 mask IoU against the bundled house alpha: `0.994194`.
- Exact upstream BiRefNet with normal semantics: `0.998310`.
- Asset-tuned BiRefNet/Lanczos/threshold-122 diagnostic: `0.999004`; do not
  present this threshold-tuned result as general production quality.
- Byte-identical 4K/1M postprocessing replay: 92.6 s, SHA-256
  `6d8c267b006df8cf60e3c5ea71d470e89c3cb734f61b91b5e1c622e1f711a8b7`.

## Remaining work, in priority order

### 1. Run a real web-demo end-to-end CUDA test

`server/pixal3d/test_browser.py` drives the real HTTP handler and Chrome, but
uses `FakePixal` for inference. Add an opt-in test or documented harness that
runs one real RTX 5060 Ti generation through the queued HTTP API and verifies:

- upload, job polling, progress, and file-backed artifact download;
- native GLB validation and viewer loading;
- an automatic RMBG input path;
- a paired PyTorch reference result when memory permits;
- cancellation followed by successful queue/GPU recovery.

Keep the normal browser test fast and fake-backed. Gate the real test behind an
explicit command-line option and record peak process VRAM/RSS and elapsed time.

### 2. Support explicit masks in paired PyTorch comparison

The native server accepts RGB plus `mask_upload`, but `PixalServer.reference`
currently rejects any request containing a separate mask. Route the already
prepared RGBA image into the single-view reference runner so native and
reference paths consume the same alpha and resolved camera values. Add unit and
real CUDA checks proving that enabling `reference` does not invoke RMBG again
and produces both artifacts.

### 3. Surface quantitative comparison in the web job result

The browser currently displays side-by-side/overlay meshes, while the HTTP
result exposes only vertex/triangle deltas and bounds differences. Integrate a
bounded version of the existing `compare_outputs.py` geometry metrics into
paired jobs, or run it asynchronously after both GLBs are ready. Display
Chamfer RMS, directional p95, and orientation-independent normal agreement.
Render metrics should remain optional because four CPU previews add latency.
Do not hold decoded meshes or images in the long-lived job dictionary.

### 4. Consolidate the current validation record

`ref/pixal3d/validation-results.json` predates several recent results. Add
machine-readable entries for:

- 7168/12288 MiB four-view native runs;
- full four-view pinned PyTorch comparison;
- matched geometry, normal, PSNR, and silhouette metrics;
- RMBG-2.0 and exact-upstream BiRefNet provenance results;
- the latest CUDA reliability soak;
- the 92.6-second byte-identical postprocessing replay.

Record commands, commit IDs, checkpoint revisions, artifact hashes, device and
driver identity, and whether a number is isolated or affected by concurrent
load. Avoid duplicating large logs or model files.

### 5. Continue byte-identical CPU postprocessing optimization

The latest 4K/1M replay still spends about 17.8 s in hole filling plus original
mesh BVH construction, 16.2 s in unwrap/normals, and 14.3 s in inpainting.
Profile those phases separately before changing them. Preserve face/vertex
ordering and require the established GLB SHA-256 after every optimization.
Promising bounded work includes compact BVH build inputs, allocation reuse in
UV chart construction, and removal of avoidable image repacking. Do not change
the 4K/1M reference-quality defaults.

### 6. Evaluate remaining CUDA memory/performance opportunities

`cpu/pixal3d/OPTIMIZATION.md` still identifies packed flow activation storage
and additional GEMM tiling as opportunities. Measure full-stage and complete
generation behavior, not only microbenchmarks. Preserve mixed-trajectory error
(`<0.001` NRMSE for all four stages), the 7168 MiB path, and byte-identical
outputs between the 7 and 12 GiB budget runs. Remove experiments that do not
improve end-to-end time or peak memory.

### 7. Refresh user-facing performance wording

Some older documentation describes the initial host-offloaded implementation
and 28–97 minute runs under concurrent load, while the resident path and newer
complete runs are substantially different. Reconcile `cpu/pixal3d/README.md`,
`cpu/pixal3d/OPTIMIZATION.md`, and `ref/pixal3d/README.md` so each timing names
its execution mode, fixture, concurrency conditions, and whether serialization
is included. Keep historical results only when they explain a regression or
tradeoff.

### 8. Optional service durability

Queued artifacts are file-backed, but job metadata remains in memory and is
discarded on server restart. If the demo is promoted beyond a workstation
tool, add an atomic on-disk job manifest and startup recovery for terminal
results. Preserve current TTL/deletion behavior and never resurrect running
jobs as active; mark interrupted work failed with a clear reason. This is lower
priority than inference and validation work.

## Explicitly out of scope

- Direct3D-S2 and the paper-version Pixal3D pipeline.
- HIP/ROCm-specific implementation or optimization work.
- Previously excluded backlog items 1 and 15.
- Git push without a new explicit user instruction.
- Threshold tuning against the bundled house alpha as a substitute for general
  RMBG quality.

## Required workflow

- Follow `AGENTS.md`; use the project-local `uv` environments and repository
  `tmp/`, never `/tmp`.
- Keep native inference independent of Python/PyTorch.
- Validate math and kernels against the pinned PyTorch implementation.
- Preserve the 7168 MiB minimum and 12288 MiB target CUDA paths.
- For postprocessing changes, replay saved decoder outputs and require
  byte-identical output unless an intentional quality change is approved.
- Run focused tests first, then the server unit and functional browser suites
  for web changes.
- Commit coherent units, report hashes and validation evidence, and do not push.

## Baseline commands

```sh
make -C cpu/pixal3d -j4 all validation test
make -C cuda/pixal3d

ref/pixal3d/run_reference_cuda310.sh cuda \
  ref/pixal3d/validate_reference_attention.py

ref/pixal3d/run.sh cuda ref/pixal3d/validate_rmbg.py \
  --model /mnt/disk2/models/RMBG-2.0 --device cuda

ref/pixal3d/run.sh cpu -m unittest server.pixal3d.test_app
ref/pixal3d/run.sh cpu server/pixal3d/test_browser.py

ref/pixal3d/run.sh cpu ref/pixal3d/replay_postprocess.py \
  --dump-dir tmp/pixal3d/resident-runs/cuda-jester/dumps \
  --output tmp/pixal3d/postprocess-profile/jester.glb \
  --profile-json tmp/pixal3d/postprocess-profile/jester.json
```

GPU access and the localhost/Chrome browser test may require running outside
the filesystem sandbox. The real PyTorch multiview environment is
`ref/pixal3d/.venv-reference-cuda310` and must retain its source-built
Torch-ABI extensions for `sm_120`.

## Resumption prompt

```text
Resume Pixal3D work in /mnt/nvme02/work/gemm/pixal3d on branch pixal3d.
Read AGENTS.md and resume-pixal3d.md first. Treat the current worktree and
artifacts as authoritative. Work through the prioritized remaining tasks,
starting with the real queued web-demo CUDA end-to-end test, then explicit-mask
paired PyTorch comparison and bounded quantitative metrics in web results.
Keep scope on Pixal3D main. Exclude Direct3D-S2/paper work, all HIP/ROCm-specific
work, previously excluded items 1 and 15, and git push. Use the per-project uv
environments and repository tmp/ only. Preserve the verified 7168 MiB minimum
and 12288 MiB target CUDA paths, mixed-precision quality, and byte-identical
postprocessing outputs. Validate against pinned PyTorch where applicable, run
focused tests plus server/browser regressions, update documentation and
validation-results.json with reproducible evidence, and commit each coherent
unit without pushing.
```
