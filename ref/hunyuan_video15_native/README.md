# Independent native-port references

`verify.py` reads only the clean upstream revision
`60783e704160023913bee78f0b47036d393d4dfa`, pinned model receipts, and the native
float32 captures. It uses FP32 CPU references for Qwen, SigLIP, and ByT5, and
block-offloaded FP16 CUDA references for the official DiT/VAE. SigLIP uses the
PIL processor matching its pinned resize/normalization configuration. Qwen uses
hidden state 26, including the assistant suffix after the system-prefix crop.
The ByT5 reference uses Transformers' byte tokenizer and encoder independently.

Generation never imports these PyTorch modules. Each validation phase writes its
own receipt and exits nonzero on failure. A bounded diagnostic is labeled as such
and cannot satisfy the pipeline acceptance check. Reference tensor hashes are
bound to the exact generation manifest, upstream revision, and verified weight
and configuration hashes. Reusing a reference from another run fails validation.

After a native run with `--dump-dir CAPTURES`, run the following phases under the
existing CUDA device lock. Use a reference environment with torch, transformers,
diffusers, safetensors, and the pinned upstream's dependencies. Set `TMPDIR` to
an existing repository scratch directory before invoking it.

```sh
MODEL=tmp/hv15-native/model
ACTUAL=tmp/hv15-native/i2v-captures
OUT=tmp/hv15-native/i2v-reference
RUN=tmp/hv15-native/i2v-run/manifest.json
PY=tmp/qimg21-ref-venv/bin/python
mkdir -p tmp/hv15-native/reference-tmp
export TMPDIR=$PWD/tmp/hv15-native/reference-tmp

$PY ref/hunyuan_video15_native/verify.py --model "$MODEL" --actual "$ACTUAL" --out "$OUT" \
  --generation-manifest "$RUN" --phase encoders \
  --components qwen_hidden qwen_negative_hidden byt5_hidden siglip_hidden \
  --image tmp/hv15-native/i2v-run/input.png --vision-pixels tmp/hv15-native/i2v-run/vision_pixels.f32

flock -w 900 tmp/pixal3d/device-locks/cuda-0.lock $PY ref/hunyuan_video15_native/verify.py \
  --model "$MODEL" --actual "$ACTUAL" --out "$OUT" \
  --generation-manifest "$RUN" --phase encoders --components vae_encoded \
  --image tmp/hv15-native/i2v-run/input.png --vision-pixels tmp/hv15-native/i2v-run/vision_pixels.f32

flock -w 900 tmp/pixal3d/device-locks/cuda-0.lock $PY ref/hunyuan_video15_native/verify.py \
  --model "$MODEL" --actual "$ACTUAL" --out "$OUT" --generation-manifest "$RUN" --phase denoise
flock -w 900 tmp/pixal3d/device-locks/cuda-0.lock $PY ref/hunyuan_video15_native/verify.py \
  --model "$MODEL" --actual "$ACTUAL" --out "$OUT" --generation-manifest "$RUN" --phase decode
$PY ref/hunyuan_video15_native/verify.py --model "$MODEL" --actual "$ACTUAL" --out "$OUT" \
  --generation-manifest "$RUN" --phase compare
```

CPU encoder checks can run while another job holds the GPU lock. The VAE phase
merges its tensor receipts with those CPU results in the same output directory;
run these phases sequentially to avoid concurrent receipt writes.
For T2V omit `siglip_hidden`, the VAE encoder command, `--image`, and `--vision-pixels`.
For fast12 omit `qwen_negative_hidden`. The denoiser consumes the independently
computed reference conditioning and matched native noise, uses the official
Euler scheduler, and compares every saved step. The decoder consumes the
independent reference's final latent. The final `compare` phase requires every
component for the chosen task/preset and all 81 decoded frames.
It also requires all 12 or 50 intermediate latents, the prescribed generation
recipe, and a measured memory-budget pass. Only this final phase has pipeline
acceptance scope; individual phases remain component checks.

`component_probe MODEL qwen|byt5 PROMPT OUT` captures complete encoders;
`component_probe MODEL siglip PIXELS.f32 OUT` captures the vision encoder.
`component_probe MODEL vae_encode|vae_decode THWC.f32 OUT T H W` exercises VAE
graphs with smaller shapes. `bounded-decode` compares a diagnostic decoded
capture to `OUT/reference/latent_final.npy` without claiming full-video parity.
`component_probe MODEL dit INPUT_DIR OUT T H W [PROFILE]` executes all 54
blocks with smaller spatial/temporal dimensions and zero image conditioning.
Its input directory contains `latent_thwc.f32`, Qwen captures, and optional
ByT5/SigLIP captures. `bounded-dit` compares it to the official graph using
independent conditioning previously saved in `OUT/reference/`; omit vision
captures for the zero-vision diagnostic. These tests are useful for bring-up,
but production acceptance still requires both complete quality pipelines.

## Reference performance

`benchmark.py` measures the same pinned official graphs with FP32 CPU encoders,
FP16 CUDA DiT/VAE, block offload, Torch FlexAttention, and 128-pixel VAE tiles.
It records synchronized wall times for loading/inference, each denoising step,
VAE decode, RGB/MP4 packaging, and sampled process VRAM/RSS. Receipt verification
and the final numerical comparison are outside the generation timer.

```sh
tmp/qimg21-ref-venv/bin/python ref/hunyuan_video15_native/benchmark.py \
  --model tmp/hv15-native/model --out tmp/hv15-native/reference-performance \
  --fast-native-run tmp/hv15-integration-review/full-repo-run \
  --fast-native-captures tmp/hv15-integration-review/full-repo-captures \
  --quality-native-run tmp/hv15-native/full-quality-i2v-v1 \
  --quality-native-captures tmp/hv15-native/full-quality-i2v-captures-v1 \
  --quality-measure-steps 4
```

Fast12 runs all 12 steps, decodes/packages all 81 frames, and rechecks complete
numerical parity. Quality uses its prescribed **50-step** schedule and CFG 6,
but defaults to measuring only its first four steps. Its first step is reported
separately; the remaining three give warmed throughput. A 50-step projection is
labeled as an estimate, with `full_quality_generation_measured: false`.
Set `--quality-measure-steps 50` to measure the entire quality denoising loop;
the quality branch does not decode/package a video.

Normal runs acquire the shared device lock. To benchmark while an owned native
quality job holds that lock, `--suspend-native-pid PID` binds its start time and
output path, checks the parent family's lock ownership, suspends it, and waits
for **that process's** GPU work to drain. The benchmark resumes it in `finally`,
including on cancellation/failure. Native GPU memory stays resident throughout;
memory figures cover the reference process separately. Desktop graphics remain
active and their sampled utilization is recorded. SIGTERM cancels at stage/block
boundaries. Only use this option with a native job you own.

Every suspension is appended to the native run's `performance_pause.json`.
Native capture intervals exclude those pauses, while the original generation
manifest keeps its observed wall time. Timing tests check cold/warm separation,
pause subtraction and rejection of missing/invalid timing samples:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s ref/hunyuan_video15_native -p 'test_benchmark.py' -v
```

## Bounded optimization replays

Build the candidate separately from a frozen quality campaign:

```sh
make -C cuda/hunyuan_video15_native BUILD=../../tmp/hv15-native/opt-build -j4 \
  all ../../tmp/hv15-native/opt-build/replay_probe ../../tmp/hv15-native/opt-build/test_gpu

REPLAY_DIR=tmp/hv15-native/replay-example
python3 ref/hunyuan_video15_native/replay.py prepare --case gemm \
  --fixture "$REPLAY_DIR/fixture" --rows 33390
python3 ref/hunyuan_video15_native/short_run.py --out "$REPLAY_DIR/native.json" -- \
  tmp/hv15-native/opt-build/replay_probe "$REPLAY_DIR/fixture" "$REPLAY_DIR/native" candidate
python3 ref/hunyuan_video15_native/short_run.py --out "$REPLAY_DIR/reference.json" -- \
  tmp/qimg21-ref-venv/bin/python ref/hunyuan_video15_native/replay.py reference \
  --fixture "$REPLAY_DIR/fixture" --out "$REPLAY_DIR/reference"
OPENBLAS_NUM_THREADS=1 python3 ref/hunyuan_video15_native/replay.py compare \
  --fixture "$REPLAY_DIR/fixture" --out "$REPLAY_DIR/reference" --actual "$REPLAY_DIR/native"
```

Each receipt/output directory must be fresh. `short_run.py` defaults to a
55-second budget and terminates the child process group before resuming its
baseline, including descendants left after a successful leader exit. It reserves
time for cleanup and limits each GPU-idle monitor call to two seconds.
Imports, loading, compilation, reservations and
capture writes count toward that budget. It serializes short experiments,
samples process VRAM/RSS, rejects VRAM above 14,336 MiB, and records executable,
private/shared source hashes. Scratch and compiler caches stay in the repository.
Normal checks take the shared CUDA lock. To borrow an owned baseline's existing
reservation, add `--native-pid PID --native-run RUN` to each short-run command;
the launcher verifies the PID's start time, output path and family lock ownership.
Pause intervals are appended to that baseline's separate performance sidecar.

`prepare --case attention` uses 34,138 tokens, 16 heads and dimension 128 by
default. `prepare --case dit_block --profile fast12_i2v|quality_i2v|quality_t2v`
uses the official first-block hook and independent encoder/noise captures in
`--captures`; run preparation itself through the bounded launcher and reference
interpreter. `--negative` selects negative Qwen conditioning. Only prefix
weights are loaded. `prepare --case dit_pair --actual POSITIVE_FIXTURE
--negative-fixture NEGATIVE_FIXTURE` combines previously captured CFG states.
Choose `--blocks 4` or `8` for streamed block measurements. Each repeated
forward restarts from its fixture inputs.

`advance --actual PREVIOUS_OUTPUT --out NEXT_FIXTURE` carries that backend's
image/text states into the next segment. Advance native and reference outputs
independently to detect accumulated drift. `replay_chain.py --native-fixture
FIXTURE --reference-fixture FIXTURE --out CHAIN` automates consecutive segments
through block 53. A whole chain can take several minutes; each GPU command has
its own 55-second watchdog. `finish` creates a final-projection/CFG/first-Euler
fixture after block 53 using `noise_input.npy` in `--captures`. These are first-step
checks with matched official prefix inputs, rather than complete generations.

`prepare --case vae_decode --captures REFERENCE` crops a 21×8×8 latent tile from
`latent_final.npy`, retaining all temporal context and decoding all 81 frames.
Use `--tile-row 48 --tile-column 24` for the 5×6 edge tile. `conv` isolates a
production convolution selected by `--prefix`. `qwen`, `siglip` and `byt5`
prepare one encoder layer from its `NAME_embedding.f32/.json` diagnostics in
`--captures`; these independent Transformers references run FP32 on CPU.
Encoder GEMM/attention semantics are preserved.

The native probe records CUDA-event and synchronized wall times, setup time,
buffer reuse, weight transfers/cache hits, kernel calls and managed VRAM.
`compare` reads bounded chunks of raw float32 outputs, fails on missing,
empty, mismatched or nonfinite data, applies cosine ≥0.9999 and relative L2
≤0.02, and checks every decoded frame separately. Fixture hashes, checkpoint
identity, raw output/metadata, timing and execution receipt hashes bind the gate
to its inputs and measured run. No replay receipt can
satisfy full-pipeline acceptance. Final speed/quality still require the normal
complete-video gates when that work is authorized.

`report_replays.py --spec PAIRS.json --out REPORT.json` summarizes accepted
pairs. Each entry provides `label`, `native`, `reference`, and optional `warm`
(default true). Warm comparisons require a cold plus at least two warm samples;
staged one-forward segments use `warm: false`. Reports retain the remaining
performance gaps. Changed or unbound artifacts are rejected; rerun `compare`
with the original fixtures to refresh older parity reports. GPU timings require
a successful bounded execution receipt. Existing CPU encoder references are
explicitly identified as `cpu_fp32` and may have no launcher receipt.
See the measured
[candidate results](../../cuda/hunyuan_video15_native/PERFORMANCE.md#optimized-bounded-replays).

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s ref/hunyuan_video15_native -p 'test_*.py'
python3 ref/hunyuan_video15_native/short_run.py --out tmp/hv15-native/math-check.json -- \
  tmp/hv15-native/opt-build/test_gpu repo-only
```
