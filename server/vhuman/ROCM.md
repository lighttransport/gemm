# Virtual humans on RX 9070 XT

The vhuman workflow and the Qwen/Pixal3D demos support native ROCm on RDNA4
(`gfx1201`, wave32). The default model root is `/mnt/disk1/models`.
Backend selection, device selection, Python subprocesses and device locks are
shared across the workflow. CPU geometry, texture baking, export and browser
playback remain portable.

## Setup and launch

The environment and runners have been built on this machine. Launch with:

```sh
sh server/vhuman/run.sh --backend rocm --device 0 --models-root /mnt/disk1/models
```

Open `http://127.0.0.1:8790/`. The same global arguments work with
`python -m server.vhuman.cli` before the subcommand. `--backend auto` probes
CUDA first, then ROCm. An explicit backend checks that Python uses the matching
PyTorch build. `--qwen-python` and `--rig-python` override the isolated interpreter.

To reproduce the environment and build from the repository root:

```sh
sh server/vhuman/setup_rocm.sh
sh ref/pixal3d/setup_native.sh
make -C cpu/pixal3d -j4
make -C rdna4/pixal3d HIPCC=/opt/rocm/core/bin/hipcc GPU_ARCH=gfx1201
make -C rdna4/qimg21 all fast rocew-probe HIPCC=/opt/rocm/core/bin/hipcc GPU_ARCH=gfx1201
make -C rdna4/sam3 -j4
make -C rdna4/sam3d_body -j4
make -C rdna4/vhuman -j4
make -C speech rocm -j4
python3 -B -m server.vhuman.check_rocm_assets
```

Setup pins the ROCm PyTorch/torchvision/Triton wheels, diffusers, transformers,
MoGe, gfx12 SageAttention and rocWMMA revisions. NAF, GNM and ICT downloads use
checksums pinned in the repository. CUDA and ROCm rotary tables are separate.
Build products, caches and temporary files use repository `tmp/` directories.
Always create a directory before setting `TMPDIR`: a nonexistent directory
causes COMGR to report invalid kernel images during PyTorch kernel loading.

The isolated environment is `tmp/vhuman-rocm-venv`. On this machine that path,
`tmp/uv-cache`, public rig assets and validation artifacts are symlinks to
`/mnt/disk1/vhuman-rocm/` to preserve root filesystem space. Existing model
weights were not copied or changed. Allow several GB for the environment,
compiler caches and generated rigs.

The existing demos can also run with the new backend:

```sh
sh server/pixal3d/run.sh --backend rocm
sh server/qwen_image21/run.sh
```

Select ROCm in the standalone Qwen page. Its default GPU is device 0.

## Coverage and limits

| Module | ROCm path |
| --- | --- |
| Qwen Image 2.1 | Native text/vision encoders, VAE encode/decode, BF16 and INT8 WMMA, fast12/low8, resident denoising, image conditioning and editing |
| SageAttention | Pinned AMD gfx12 INT8 QK / FP8 PV device kernels; F32 PV accumulation |
| Pixal3D | Native HIP reconstruction, DINOv3 conditioning, geometry and texture decoders |
| Input preparation | ROCm PyTorch RMBG-2.0 and MoGe-2 |
| SAM 3 | Full text-prompted segmentation, including CLIP tokenization and mask export |
| SAM 3D Body | DINOv3 + decoder, GLB and decoded MHR sidecar export |
| Body assembly/motion | ROCm PyTorch MHR pose/identity decoding; portable assembly/export |
| Facial rig | ROCm registration, expression fitting and ML deformer training |
| Deformer playback | Native HIP blendshapes, skinning, ML and contact kernels; existing browser rendering |
| Speech | Native HIP Qwen3-TTS and Japanese wav2vec2 alignment |
| Video/soft-deformer fitting | Selected ROCm PyTorch device; CPU observation/geometry and LightGeom teacher retained |
| Optional portrait depth | Selected ROCm PyTorch device when the verified Small checkpoint is installed |

NVFP4 remains CUDA-only and is rejected on ROCm. The historical
`--int8-gemm cutlass` spelling selects the HIP WMMA plugin on ROCm; the CUDA
cuBLAS fallback is rejected. The `flash` and `exact` modes use the native BF16
online-softmax attention plugin. Sage is an approximate quantized path; it is
not bitwise equivalent to BF16 attention. Both Sage accumulation spellings use
F32 accumulation on AMD. The ROCm VAE decoder currently runs one-shot; the
transformer remains resident. Encoders run before residency to limit memory.
Use `NativeBackend.prepare()` for references that will be sent to the resident
image-conditioned denoiser. Shared per-backend/device locks serialize jobs;
free-memory checks report contention rather than assuming an idle GPU.

## Weight audit

All required assets for the tested main workflow are present. The audit lists
paths and returns a nonzero status if a required asset is absent. These optional
weights are missing on this machine:

- `tmp/vhuman-rig/models/depth-anything-v2-small/depth_anything_v2_vits.pth`
  (also needs its verified installation manifest and pinned source).
- `tmp/vhuman-emotion/sensevoice-small-q8.gguf`
  (also needs the optional SenseVoice runtime).
- `tmp/vhuman-rig/models/face_landmarker.task`
  (also needs MediaPipe for uploaded face-video observation; see
  `server/vhuman/rig/setup_face_video.sh`).

The optional CPU LightGeom teacher is also absent (`third_party/LightGeom`);
its private source and facial runner are needed for soft-tissue simulation.
See `server/vhuman/rig/setup_face_sources.sh`.

Their dependent features report missing assets. They do not prevent image,
head, body, rig, speech or native reconstruction jobs.

## Validation on this machine

These are bounded smoke tests, not quality or production throughput benchmarks.
Artifacts and exact runner output are in `tmp/rocm-validation/`.

- Two-step 256×256 Qwen generation with Sage: 17.66 s denoising, 6969 MiB peak.
  Low8: 6553 MiB peak; eight streamed transfers verified with zero mismatches;
  output latents exactly match fast12 with the same fixture.
- Two image requests reused one resident denoiser without fallback. Prepared
  image-conditioned editing also used a resident request with zero fallbacks.
- Pixal3D textured GLB: 222.22 s, 188917 vertices, 298372 triangles,
  4.63 GB device peak. RMBG background removal and MoGe camera inference pass.
- SAM 3 produced text-prompted masks. SAM 3D Body produced a valid GLB,
  finite 204 decoded parameters and 45 identity coefficients. Full body assembly
  produced GLB/USD with 127 body and 12 face joints; MHR parity error 1e-6 m.
- Real facial rig self-test completed registration, 52 shapes, 60 controls,
  ML training, expressions and two LODs with valid topology.
- HIP deformer on that rig: 32 frames, maximum CPU difference 1.49e-8 m;
  ML+contacts 58.22 us/frame, ML without contacts 6.41 us/frame.
- Japanese TTS: 0.88 s audio in 1.85 s plus 0.17 s alignment (5.3 s loading).
- Independent BF16/INT8 GEMM and rectangular/masked attention checks pass.
  INT8 matches its reference exactly, including tails. Sage max error <0.073
  against F32 attention in these fixtures.

Verification commands:

```sh
mkdir -p tmp/vhuman-runtime
export TMPDIR="$PWD/tmp/vhuman-runtime"
export LD_LIBRARY_PATH="/opt/rocm/core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
tmp/vhuman-rocm-venv/bin/python -B rdna4/qimg21/test_fast_plugins.py
make -C cpu/pixal3d test
tmp/vhuman-rocm-venv/bin/python -B -m unittest \
 server.vhuman.test_runtime server.vhuman.test_app server.vhuman.test_rig \
 server.vhuman.test_speech server.vhuman.test_reconstruction \
 server.vhuman.body.test_body server.vhuman.body.test_motion \
 server.qwen_image21.test_app server.qwen_image21.test_form server.pixal3d.test_app
```

The last regression run passed 178 tests (12 optional tests skipped); the dedicated studio suite passed another
10 tests. Local
HTTP/socket tests and GPU validation require access outside a restricted sandbox.
