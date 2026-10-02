# Repository-owned HunyuanVideo 1.5 CUDA port

This directory implements the standalone C/C++ runtime and adapter. It does not
depend on ggml, stable-diffusion.cpp, or PyTorch for generation. PyTorch is used
only by the independent reference and benchmark tools. No shared server/UI, existing
model port, GEMM source, or live rig needs modification.

The current runtime remains **experimental**. A complete fast12 I2V run has
passed independent comparisons for all 12 steps and all 81 decoded frames,
including strict repository GEMM. The quality T2V/I2V profiles remain under
validation. Generation requires `--allow-experimental`; generation receipts
retain `parity: unverified`, with accepted comparisons stored in separate
reference reports. Never infer acceptance of another profile from a bounded
test or the fast12 result.

Measured component results and exact checks are in [VALIDATION.md](VALIDATION.md).
Measured native/PyTorch timing comparisons are in [PERFORMANCE.md](PERFORMANCE.md).
The optimized build has passed production-shape block, attention, encoder and
VAE tile checks. Its full-video performance and quality remain provisional.
Use a separate `BUILD=../../tmp/hv15-native/opt-build` while a frozen baseline
or quality campaign is running; do not rebuild that campaign's runner.

Supported requests are 480×848, 81 frames, 24 fps: quality T2V and I2V use
50 Euler steps, CFG 6, shift 5; distilled I2V uses 12 steps, CFG 1, shift 7 and
next-timestep conditioning. The initial backend requires NVIDIA compute
capability 12.x, CUDA/NVRTC, at least 64 GiB host RAM, and FFmpeg for MP4 packaging.

## Build and test

From the repository root:

The build needs a C++17 compiler and PCRE2 headers/library. The Python wrapper
and tests need NumPy and Pillow; MP4 packaging needs FFmpeg.

```sh
make -C cuda/hunyuan_video15_native -j4
make -C cuda/hunyuan_video15_native test
make -C cuda/hunyuan_video15_native compile-kernels
make -C cuda/hunyuan_video15_native ../../tmp/hv15-native/build/test_gpu
mkdir -p tmp/pixal3d/device-locks
flock -w 30 tmp/pixal3d/device-locks/cuda-0.lock tmp/hv15-native/build/test_gpu
flock -w 30 tmp/pixal3d/device-locks/cuda-0.lock tmp/hv15-native/build/test_gpu repo-only
flock -w 30 tmp/pixal3d/device-locks/cuda-0.lock tmp/hv15-native/build/test_gpu cublas
flock -w 30 tmp/pixal3d/device-locks/cuda-0.lock tmp/hv15-native/build/test_gpu memory
```

All generated files default to `tmp/hv15-native/` in the repository. GPU access
may require running outside an agent's filesystem/device sandbox. `--validate`
and `--compile-kernels` do not require a GPU. Test failures never count as skips
or successful parity checks.

The native API is declared in `hv15_native.h`; the build produces `libhv15n.so`,
`hv15n`, `component_probe`, and `tokenizer_probe`. Calls on one context must be
serialized. Cancellation is supported by a callback or `hv15n_cancel` from a
second thread; free the context only after generation returns. Callback RGB
memory is valid for the duration of the callback. The C API takes already
prepared RGB portraits and CHW float32 SigLIP inputs; `generate.py` prepares them.

## Stage pinned weights separately

Use a Python environment with `huggingface_hub`, `numpy`, and `safetensors`:

```sh
tmp/qimg21-ref-venv/bin/python cuda/hunyuan_video15_native/stage_models.py \
  --out /mnt/nvme02/data/models/hv15 --reuse tmp/hunyuan-video15-model --transport aria2
```

Omit `--reuse` for fresh downloads; `--transport hub` is the default.
`--profiles quality_t2v quality_i2v fast12_i2v` is the default set. Completed
existing assets are verified and hardlinked, then treated as immutable. Existing
manifests and files are never rewritten. Downloaded assets have pinned repository
revisions, byte counts, and SHA256 receipts. Google SigLIP SO400M/14/384 is the
explicit vision profile. The exporter keeps its vision weights in FP16. ByT5's
configuration is pinned separately; quoted text uses its UTF-8 byte tokenizer,
deduplicated ASCII/curved quotations, EOS, and the upstream 256-token limit.
The public prompt interface follows the upstream plain glyph formatter; custom
font/color control tokens are not an exposed generation option.

The local model package is `/mnt/nvme02/data/models/hv15`, including quality
I2V/T2V and Fast12 I2V checkpoints, shared encoders/VAE, tokenizer and pinned
reference configurations. The C API, native runner, generator, vHuman adapter
and bounded replay tool default to this directory. Use `--model DIR` (or
`hv15n_config.model_dir`) to select another installation; on b550, use
`--model /mnt/disk01/data/models/hv15`. The old `tmp/hv15-native/model` path is a
compatibility symlink so the frozen running quality campaign can finish.

## Generate and publish

```sh
python3 cuda/hunyuan_video15_native/generate.py \
  --model /mnt/nvme02/data/models/hv15 --out tmp/hv15-native/t2v-run \
  --task t2v --preset quality --prompt 'A person smiles naturally.' \
  --dump-dir tmp/hv15-native/t2v-captures --allow-experimental

python3 cuda/hunyuan_video15_native/vhuman_adapter.py generate \
  --work tmp/vhuman --head HEAD_ID --model /mnt/nvme02/data/models/hv15 \
  --preset quality --prompt 'The same person smiles gently, fixed frontal camera.' \
  --seed 42 --allow-experimental
```

The adapter acquires the existing CUDA device lock and publishes complete clips
atomically in `heads/HEAD_ID/videos/RUN_ID/`. Each run contains `clip.mp4`,
`poster.png`, `manifest.json`, `metrics.json`, and its runner log. The existing
video listing can discover this layout without a server change. T2V uses the
standalone generator because it does not have an input head portrait.

`--gemm repo` selects the repository's FP16-input/FP32-accumulate PTX GEMM.
Compatible large projections use repository v7 with padded grids and guarded
tails. Other shapes retain the private original GEMM paths. Shared GEMM and
FlashAttention2 sources are read-only inputs. DiT uses FP16 FlashAttention2
with FP32 softmax reductions, fused head normalization/RoPE/packing,
normalization/modulation and residual gates. Unsupported dimensions, masks,
GQA and encoder attention retain the existing paths. Attention tensors are
released before the large MLP. Quality processes both CFG branches per block
so they share each weight upload; its CFG-6 arithmetic and Euler schedule remain
unchanged.
`--gemm-fallback cublas` explicitly permits IEEE FP32 cuBLAS for the encoders.
`--gemm-fallback error` uses the native IEEE kernel for those operations, avoiding
cuBLAS entirely. `--gemm cublas` selects the comparison backend. Call counts and
fallback counts are recorded. Encoder projections use IEEE FP32 arithmetic,
including a private register-tiled kernel in strict repo mode. Residual states
and reductions stay FP32; DiT attention and fused VAE norm/SiLU intermediates
use FP16. Attention uses bounded tiles rather than an N×N score allocation.
Weights are mapped on the host, transferred through bounded pinned staging,
and cached by immutable file identity. DiT overlaps next-block transfer with
compute, retaining a fixed six-block prefix; its other blocks stream through
the same reusable buffer pool. VAE convolutions use an implicit-GEMM gather
derived privately from repository v7, with a bounded FP16 im2col fallback.
VAE spatial tiles retain full temporal context and only two completed rows on
the host. The C API and vHuman adapter accept the same arguments and outputs;
select an isolated candidate with the existing `--runner` argument.

`--vram-budget-mib` defaults to 14336 and caps managed allocations at the budget
minus 3072 MiB for driver/vendor workspace. The Python wrapper also samples
process VRAM/RSS and refuses a measured budget overrun. Missing NVIDIA sampling
is explicitly `unverified`. Noise uses `mt19937_64_box_muller_v1`; its seed does
not claim equivalence to PyTorch's RNG. Use `--noise-file` with canonical
NCTHW float32 noise for comparisons. SIGTERM and callback cancellation stop at
kernel/stage boundaries and remove the wrapper's partial output.
When captures were requested, a failed native run keeps a diagnostic
`failure.json` beside them, including sampled peaks and its error. Those captures
cannot satisfy complete pipeline acceptance.

## Independent reference validation

See [the reference procedure](../../ref/hunyuan_video15_native/README.md).
Every required tensor must meet cosine ≥0.9999 and relative L2 ≤0.02. Every
decoded frame is also compared separately. Missing captures, missing components,
nonfinite values, or an incomplete 81-frame video fail validation.

`validate_quality.py` runs both full 50-step quality profiles with strict repo
GEMM and no vendor fallback, then executes the independent encoder, denoising,
decoder and final comparison stages for each. It takes the shared GPU lock for
CUDA stages; CPU encoder checks run without that lock. Choose a fresh campaign
directory and a Torch-capable reference interpreter:

```sh
python3 cuda/hunyuan_video15_native/validate_quality.py \
  --model /mnt/nvme02/data/models/hv15 --image tmp/hv15-native/prepared/input.png \
  --out tmp/hv15-native/quality-campaign \
  --reference-python tmp/qimg21-ref-venv/bin/python
```

`campaign.json` records the controller PID, current task/stage, fixed inputs and
binary hashes, errors, and completed report hashes. A failed or cancelled stage
stops the campaign. `pass_both_quality_pipelines` becomes true only after both
complete independent gates pass; this never publishes a clip or promotes a rig.
The current quality run uses the same prompt, seed 42 and portrait as the fast12
validation. Per-run `runner.log` files report generation progress.

Send SIGTERM to the controller to cancel its work. Re-run the same command with
`--resume` to reuse completed native generation, check frozen inputs/binaries,
and repeat the final gate. Failed native runs retain diagnostic captures and
need fresh output/capture directories for a retry. To supervise an already
running I2V generator, add `--adopt-run RUN --adopt-captures CAPTURES --adopt-pid PID`.
Use the Python generator PID, rather than its native child PID. Adoption binds
the PID's start time and output directory; cancellation sends SIGINT to that
wrapper so its normal child cleanup runs. A completed adopted run needs no PID.

## Reviewed synthetic candidate training

`vhuman_adapter.py train-candidate` consumes a hand-reviewed dataset. Each clip
needs the exact `clip.mp4` SHA256, explicit review flags, a partition, and native
I2V provenance. The reviewed generation manifest's SHA256 binds the seed,
portrait and model receipts to that review. A minimal entry is:

```json
{
  "schema": "hv15n.synthetic_dataset.v1",
  "clips": [
    {"clip": "path/to/train-run", "clip_sha256": "SHA256", "split": "train",
     "generation_manifest_sha256": "MANIFEST_SHA256",
     "review": {"identity": true, "expression": true, "camera": true,
                "visibility": true, "artifacts": true},
     "expression_frames": {"smile": 40}},
    {"clip": "path/to/validation-run", "clip_sha256": "SHA256", "split": "validation",
     "generation_manifest_sha256": "MANIFEST_SHA256",
     "review": {"identity": true, "expression": true, "camera": true,
                "visibility": true, "artifacts": true}},
    {"clip": "path/to/test-run", "clip_sha256": "SHA256", "split": "test",
     "generation_manifest_sha256": "MANIFEST_SHA256",
     "review": {"identity": true, "expression": true, "camera": true,
                "visibility": true, "artifacts": true}}
  ]
}
```

Paths are relative to the dataset file. Use distinct seeds between partitions;
all clips must come from the selected head's identical portrait. Select expression
frames only from training clips. The existing modal trainer supports one
validation take; keep at most eight train/validation takes and one independent
test take.

```sh
python3 cuda/hunyuan_video15_native/vhuman_adapter.py train-candidate \
  --work tmp/vhuman --head HEAD_ID --dataset reviewed_dataset.json \
  --out tmp/hv15-native/candidate --python tmp/vhuman-rig-venv/bin/python
```

The command copies assets into its own candidate workspace, fits the videos,
requires ≥95% observed frames and normalized landmark error ≤0.03, restores
the generated videos and selected expression crops to the portrait's original coordinates, and supplies
the **fitted controls at that frame** to the existing expression refinement.
It rebuilds candidate geometry, re-fits all clips, runs the existing physics
teacher, then trains the existing compact soft deformer. The test partition is
reserved for independent teacher-error evaluation after model selection.
`candidate.json` records synthetic provenance and geometry hashes; a candidate
must improve on the test baseline to be marked reviewable. The active rig is
never replaced. This training workflow needs real reviewed clips and has not
been validated by the kernel/component tests.
