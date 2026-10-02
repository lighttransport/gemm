# vhuman neural avatar runtime

This module adds a local, sample-timed neural rendering PoC to the existing
vhuman rig. It does **not** convert the old research portraits into commercially
cleared training data. Old heads can be loaded explicitly as diagnostic geometry;
appearance training uses new, independently cleared references.

Current validated assets, updated commands and remaining quality/latency limits are
in [VALIDATION.md](VALIDATION.md). The initial two-take motion checkpoint was
trained on nearly silent truncated audio; use the complete-corpus student below.

Implemented paths:

```
native Qwen talker -> hidden state + 16 RVQ tokens -> causal GRU -> named controls
                  -> stateful CPU codec -> timestamped PCM -> native audio ring
named controls -> native CUDA rig/contact -> borrowed vertices -> native CUDA Gaussian renderer
PortAudio DAC sample clock -> silence-aware speech sample position -> interpolation
```

`live` requires trained appearance and motion bundles. `--diagnostic` explicitly
allows diagnostic bundles. A trained flag records optimization, not a quality
certification. No pretrained direct-Qwen motion adapter is bundled.

## Setup

Inference now requires only the packages in `requirements-runtime.txt`, the
CUDA driver/NVRTC, and the native binaries. The default Gaussian renderer does
not import Torch, gsplat, ONNX or another model runtime. Build it with:

```sh
make -C cpu/vhuman libvhuman_motion.so libvhuman_training.so
make -C cuda/vhuman libvhuman_runtime.so libvhuman_splat.so
# Use a NumPy/Pillow/SciPy interpreter; the existing framework-free environment:
tmp/mhr-native/runtime/bin/python -m server.vhuman.realtime doctor
```

The following larger environment is needed only for offline training and the
explicit `renderer.gsplat_reference` oracle. `run.sh` remains a convenience
wrapper for that environment; direct module invocation needs no Torch settings.

Run from repository root on Linux. The tested machine has Python 3.12,
PyTorch **2.14.0+cu130**, CUDA toolkit **13.2**, driver **615.71.09** and an RTX
5060 Ti (sm_120). The existing rig environment is reused. Setup/build caches and
all model/data artifacts stay under repository `tmp/`.

```sh
mkdir -p tmp/vhuman-realtime/upstream tmp/vhuman-realtime/cache/tmp
git clone --recursive https://github.com/nerfstudio-project/gsplat.git tmp/vhuman-realtime/upstream/gsplat
git -C tmp/vhuman-realtime/upstream/gsplat checkout 512d366b67073d77ca099ede742683c165dfc23b
git -C tmp/vhuman-realtime/upstream/gsplat submodule update --init --recursive

# Install into an isolated target rather than replacing the rig environment's Torch.
TMPDIR="$PWD/tmp/vhuman-realtime/cache/tmp" \
UV_CACHE_DIR="$PWD/tmp/vhuman-realtime/cache/uv" \
uv pip install --python tmp/vhuman-rig-venv/bin/python \
  --target tmp/vhuman-realtime/deps --no-deps \
  ninja jaxtyping wadler-lindig nvidia-ml-py \
  diffusers==0.37.1 transformers==4.57.1 tokenizers==0.22.1 \
  huggingface-hub==0.36.0 accelerate safetensors pyyaml regex tqdm \
  filelock fsspec hf-xet requests packaging importlib_metadata zipp \
  httpx==0.28.1 httpcore==1.0.9 anyio sniffio h11

make -C cpu/vhuman libvhuman_motion.so
make -C cuda/vhuman libvhuman_runtime.so

# Native playback requires libportaudio development headers (portaudio19-dev on Ubuntu).
make -C speech BUILD=../tmp/vhuman-realtime/speech \
  ../tmp/vhuman-realtime/speech/qwen3_tts_cuda \
  ../tmp/vhuman-realtime/speech/test_codec_stream
sh server/vhuman/realtime/run.sh doctor
```

The live speech-to-expression adapter runs its two-layer GRU through
`cpu/vhuman/libvhuman_motion.so`, with repository CPU GEMM and no Torch/ONNX
inference. Motion training also runs in native C++ with repository CPU GEMM,
GRU backpropagation, gradient clipping and AdamW. It writes safetensors and JSON
directly to a `.native/` bundle. Convert older Torch checkpoints once with a
Torch-capable interpreter:

```sh
python -m server.vhuman.realtime.src.animation.export_native \
  tmp/vhuman-realtime/motion-v3.pt
```

`--adapter` accepts the exported directory or the original `.pt` path (resolved
to its sibling `.native/` directory with a source-hash check). Missing or stale
exports fail explicitly. For new training, prefer a `.native` directory output.
A file-shaped output such as `motion.pt` is now a JSON training receipt with a
sibling `.native/` bundle, rather than a Torch checkpoint. Existing revision,
diagnostic/production and epoch/sequence checks remain in force. Motion and
Gaussian appearance training are native CPU paths. Explicit gsplat reference
comparisons retain optional Torch; training and live inference are framework-free.

For scheduling, audio replay, native model adapters, geometry and receipt tools,
install only `requirements-runtime.txt` in a separate interpreter and run
`python -m server.vhuman.realtime ...`. The broader setup above is for the
gsplat compatibility renderer and offline reference generation.
`doctor` probes the CUDA driver through the native library without importing
Torch. Held-out `evaluate-motion` uses the native GRU; `--reference-parity`
explicitly enables the optional Torch full-sequence oracle.

`run.sh` restricts the JIT build to 3DGS/RGB/sm_120. The initial RGB build took
about 144 seconds here. The default all-feature build spent more than 25 minutes
compiling unrelated templates and was stopped. Build flags are dependency build
settings; runtime policy remains explicit configuration/CLI arguments.
Optional UI dependencies are `pygame==2.6.1`, `pyvirtualcam==0.15.0`, and `ffmpeg`.
Linux virtual cameras additionally need an installed/configured v4l2loopback device;
this module does not install kernel modules or change system devices.

## Diagnostic render and replay

```sh
RIG=tmp/vhuman-independent/heads/291dfa911553/rig
sh server/vhuman/realtime/run.sh bind --rig "$RIG" --count 50000 \
  --output tmp/vhuman-realtime/diagnostic50k.npz
sh server/vhuman/realtime/run.sh render --rig "$RIG" \
  --avatar tmp/vhuman-realtime/diagnostic50k.npz --animate --res 720 --frames 100

# A take produced by vhuman's existing speech path remains compatible.
sh server/vhuman/realtime/run.sh replay --audio speech.wav --motion animation.json \
  --sink offline --output tmp/vhuman-realtime/replay.jsonl
# Actual audio device playback, optionally rendering the rig too:
sh server/vhuman/realtime/run.sh replay --audio speech.wav --motion animation.json \
  --sink device --rig "$RIG" --avatar tmp/vhuman-realtime/diagnostic50k.npz
```

`replay` reads a complete WAV/take but feeds incremental 20 ms PCM chunks with
bounded read-ahead. It tests transport/synchronization, not causal waveform motion
inference. Offline sink advances a simulated clock; device sink uses DAC time.
The renderer covers the welded skin and mouth surfaces. Rigid carried eyeballs,
hair, separate teeth and their materials still need additional Gaussian/mesh
components; the diagnostic point cloud is not a photographic avatar.

## Incremental native TTS

```sh
tmp/vhuman-realtime/speech/qwen3_tts_cuda \
  --backend cuda --threads 4 \
  --model /mnt/nvme01/models/speech/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Ono_Anna --language Japanese --text 'こんにちは。今日はいい天気ですね。' \
  --streaming --max-frames 64 \
  --features-out tmp/vhuman-realtime/features.bin \
  --pcm-out tmp/vhuman-realtime/pcm.bin --out tmp/vhuman-realtime/speech.wav
```

`--streaming` selects Qwen's incremental text feed. **`--pcm-out` actually emits
incremental decoded audio**. Its decoder maintains absolute-position RoPE, a
72-frame per-layer KV window and per-convolution history. It never re-encodes
waveform or re-runs the whole prefix. The current decoder is a CPU correctness
reference, with an allocation-free native audio callback downstream. It still
allocates decoder temporaries on the model worker and needs GPU optimization.
In PCM mode CUDA holds talker weights; unused full-batch GPU codec weights are
not uploaded. The final WAV is independently decoded for comparison.

`qtts_generate_ex` preserves `qtts_generate` and adds a borrowed hidden/token
callback. Returning nonzero cancels after committing that frame. Callback buffers
must be copied before return. Exactly **1920 samples at 24 kHz = 80 ms = 12.5 Hz**
separate successive codec frames. The motion adapter emits eight 10 ms control
samples per codec frame. Future audio is not required for the GRU.

Binary formats are little-endian on supported Linux x86 hosts:

* Features: `VHFEAT1\0`, i32 H; then repeated i64 sample start, i32 tokens[16], f32 hidden[H].
* PCM: `VHPCM1\0\0`, i32 rate; then repeated i64 sample start, i32 count (1920), f32 PCM[count].

The Python subprocess adapter uses separate pipes and bounded reader queues.
The one-shot adapter uses EOF and reloads weights per utterance. `ResidentTTS`
uses `--serve-stdin` to retain weights/context across requests. It waits for
`VHTTSRDY`, writes one JSON line `{"text":"こんにちは。"}` per request, and reads
`VHTTSBEG` request-start acknowledgement before the ordinary PCM header and an
ordinary feature header per utterance. PCM EOS is sample=-1,count=0;
feature EOS is sample=-1 followed by zero tokens/hidden. Missing EOS is an error.
`submit(text, epoch)` requires increasing epochs and drained previous queues;
`cancel()` waits for the request-start acknowledgement before sending SIGUSR1,
discards/drains both streams through EOS, joins readers,
and leaves the worker reusable. `close()` stops the process. Use `--resident`
for live warm serving; the CLI handles one turn, while the Python API can reuse
one worker across many turns. Model identity is the actual local
`model.safetensors` SHA256, verified before starting the worker; codec/tokenizer
hashes should also become part of future full-model receipts.
Rebuild the native runner when updating this module: resident framing now includes
the request-start acknowledgement, and older resident binaries are incompatible.

## Clean identity and appearance training

After staging the hashed assets described below, generate native references:

```sh
python -m server.vhuman.realtime generate-identity \
  --native-assets tmp/vhuman-realtime/flux2-assets.json \
  --output tmp/vhuman-realtime/native-identity --expressions
# Uses only the newly generated neutral image as expression conditioning.
tmp/vhuman-rig-venv/bin/python -m server.vhuman.cli \
  --work tmp/vhuman-realtime/clean-heads \
  --rig-python tmp/vhuman-rig-venv/bin/python portrait-reconstruct \
  --portrait tmp/vhuman-realtime/native-identity/neutral.png \
  --face-model gnm_v3 --profile full --res 512 --gaussians 0
```

The default `--identity-backend native` invokes the existing C/CUDA FLUX.2
runner for a neutral portrait and, with `--expressions`, twelve image-conditioned
expression portraits. `--identity-backend torch-reference` explicitly selects
the optional offline Torch oracle. Native generation requires hashed, locally
staged assets:

```sh
make -C cuda/flux2
python ref/vhuman/export_flux2_weights.py \
  --source /path/to/FLUX.2-klein-4B/transformer/diffusion_pytorch_model.safetensors \
  --output tmp/vhuman-realtime/flux2-dit-bf16.safetensors
python ref/vhuman/export_flux2_tokenizer.py \
  --source /path/to/FLUX.2-klein-4B/tokenizer \
  --output tmp/vhuman-realtime/flux2-tokenizer.gguf
python -m server.vhuman.realtime.src.avatar.native_identity \
  --dit tmp/vhuman-realtime/flux2-dit-bf16.safetensors --vae /path/to/vae.safetensors \
  --encoder /path/to/text_encoder --tokenizer tmp/vhuman-realtime/flux2-tokenizer.gguf \
  --revision SOURCE_REVISION --output tmp/vhuman-realtime/flux2-assets.json
python -m server.vhuman.realtime generate-identity \
  --native-assets tmp/vhuman-realtime/flux2-assets.json \
  --output tmp/vhuman-realtime/native-identity --expressions
```

The tokenizer exporter uses only Python's standard library and writes source/output
hash receipts. Its token IDs, including the non-thinking chat template, were
checked against the pinned Qwen tokenizer for English and Japanese prompts.
The adapter selects F16 DiT computation and repository GEMM for the 16 GB
budget. The exporter preserves source BF16 bytes while changing tensor names,
concatenating QKV and permuting the final scale/shift rows; the runner converts
weights to F16 during upload. HF snapshot
symlinks are accepted only after verifying their content hashes.

The native adapter never downloads weights or switches to Torch. It uses
Diffusers' 512 right-padded text queries, masks padding keys, normalizes VAE
reference latents and assigns reference tokens time coordinate 10. The generated
neutral portrait is the sole image reference for every expression. Each completed
image is checkpointed with its prompt, controls, seed and conditioning hash;
`--resume` can extend a neutral-only identity or continue interrupted expressions.
Approximate expression labels still require visual review.

The native adapter uses CUDA text encoding with FP32 KV storage, releases that
encoder before loading the DiT and disables sidecar text-weight caches so verified
model directories remain unchanged. Its full padded features match independent
FP32 Qwen with relative L2 `2.35e-5` (`1.29e-5` for the optional CPU encoder).
Full four-step distilled-4B comparisons passed using original BF16 DiT weights
and independently computed FP32 Qwen/DiT/VAE references:

| CUDA text conditioning | Final latent relative L2 | Decoded RGB relative L2 |
|---|---:|---:|
| Text-to-image, 512 square | 0.001882 | 0.001363 |
| One reference image, 512 square | 0.004880 | 0.001700 |
| Text-to-image, 256×384 | 0.001267 | 0.001003 |
| One reference image, 256×384 | 0.005067 | 0.002101 |

Every intermediate velocity and Euler latent passed the 2% relative-L2 gate.
The image encoder alone had relative L2 `0.006803` against FP32. Validation uses
`ref/vhuman/verify_flux2.py`, outside inference; receipts and tensors are under
`tmp/vhuman-flux2-parity/`. The optional CPU text path also passed both full
pipelines (decoded relative L2 `0.002299` / `0.001673`). CUDA Qwen uses corrected
IEEE half weight conversion. The old converter discarded half subnormals, and old F16 sidecar
caches are invalidated by a format-version bump. The converter passes exhaustive
BF16/F16 and half-rounding-boundary checks against x86 F16C hardware. Earlier compact
text/FP8 smoke outputs are not parity or quality evidence for this recipe.
These results cover distilled Klein 4B, four steps and one image reference;
base/9B checkpoints and multiple references are not validated by these checks.

On RTX 5060 Ti, a real native adapter neutral-to-smile run and resume passed
without importing Torch/ONNX/MediaPipe. Peak process VRAM was 8,636 / 9,008 MiB,
host RSS 22,210 / 22,125 MiB, and generation took 61.8 / 71.0 seconds. The
complete twelve-expression scheduling, interrupted generation, changed-prompt
rejection and hash-checked resume have separate unit coverage.

Native Klein prompts for smile, raised brows and jaw opening were refined using
one fixed reference portrait and seeds 8/9/10. The smile now keeps lips closed
(jaw-open score 0.036 → 0.0014); raised-brow scores improved from 0.313 inner /
0.041 and 0.074 outer to 0.790 / 0.915 and 0.846. The jaw-opening image increased
its jaw score from 0.629 to 0.736 while reducing mean smile score from 0.698 to
0.276 without introducing strong lip pursing. A second identity also produced a
closed-mouth smile in the adapter smoke test. These native landmark scores and
visual checks are diagnostics; residual expression mixing and approximate
control labels still require review before training. The other nine expression
prompts have not received this native quality check. Final contact sheets and
receipts are in `tmp/vhuman-flux2-parity/expressions-final/` and `adapter-smoke/`.

Reference generator: [Apache FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
at `e7b7dc27f91deacad38e78976d1f2b499d76a294`. Model CPU offload limits peak VRAM.
Generated expression control labels are approximate and require identity/pose QA.
The provider writes checksums, seeds, conditioning origins and a manifest.
Receipt validation is an engineering provenance guard, not proof of legal rights.
The operator must ensure all input/model permissions; metadata cannot relicense
restricted inputs. Known Qwen-Image-2.1 sources are explicitly rejected.

Appearance corpus JSON:

```json
{
  "format": "vhuman.appearance_corpus.v1",
  "purpose": "production",
  "data": "frames.npz",
  "control_names": ["jawOpen"],
  "provenance": [{
    "path": "frames.npz", "source": "original cleared reference corpus",
    "revision": "corpus-v1", "license": "Apache-2.0",
    "sha256": "REPLACE_WITH_ACTUAL_64_HEX_DIGEST", "roles": ["appearance-training"]
  }]
}
```

`frames.npz` (no pickle): vertices[F,V,3] in metres, triangles[T,3] integer,
images[F,H,W,3] in **linear RGB** over black, controls[F,C], view[F,4,4],
intrinsics[F,3,3]. Cameras must be calibrated to the final rig after contact
projection. Native splatting uses camera +Z forward; vhuman's reconstruction camera
uses -Z forward, so flip Y/Z in its world-to-camera transform. Appearance fitting
currently requires pinhole intrinsics without skew.
`export-neutral` exports the calibrated clean portrait and final posed rig with
an explicit Y/Z conversion and projection parity check. It uses a projected
convex mask and marks the result **diagnostic**. Expression registration,
photographic component masks and held-out views remain necessary.
Do not substitute screenshots of the conventional mesh renderer for real targets.

```sh
sh server/vhuman/realtime/run.sh fit-appearance --manifest appearance.json \
  --output tmp/vhuman-realtime/avatar.npz --count 50000 --steps 1000
```

The fitter optimizes RGB, opacity, local diagonal covariance, bounded normal
offset and an eight-component expression color basis. Triangle/barycentric
attachments remain fixed. Runtime covariance is transported by deformed triangle
edges/normal. New fits use shared differentiable trace bounds with the renderer;
legacy bundles retain their versioned eigenvalue policy. Explicit mask/alpha
supervision prevents a bright transparent face from satisfying RGB loss. Format `vhuman.gaussian_avatar.v1` preserves
control order, topology SHA256, provenance, fixed-light assumption and purpose.
There is no relighting model, densification or automatic quality acceptance yet.

Appearance fitting now uses `cpu/vhuman/libvhuman_training.so`, with no Torch,
gsplat or CUDA requirement. The CPU renderer follows the native CUDA pinhole/EWA,
tile ordering, visibility thresholds and front-to-back alpha rules. Its analytic
compositing reverse pass and four local projection derivatives update RGB,
opacity, diagonal covariance, normal offset, color basis and expression matrix.
Repository GEMM computes expression projections; native AdamW with zero decay
implements Adam. The renderer uses FP64 local math with FP32 parameters and
outputs. Sorting, clipping and tile membership are piecewise constant. The
appearance rasterizer currently runs serially; `--threads` controls GEMM work.
Large 50k-splat training speed and GPU backward execution remain unvalidated.
Start CPU integration checks with a small count and step budget. Checkpoints
retain `vhuman.gaussian_avatar.v1` and `trace-v1` covariance semantics.

```sh
make -C cpu/vhuman libvhuman_training.so test
python -m server.vhuman.realtime fit-appearance --manifest appearance.json \
  --output tmp/vhuman-realtime/native-avatar.npz --count 512 --steps 50 --threads 4
python -m unittest server.vhuman.test_native_image_training \
  server.vhuman.realtime.test_training -v
# Optional CPU Torch math oracle; no gsplat or GPU needed:
python ref/vhuman/verify_image_training.py \
  --output tmp/vhuman-native-image-training/parity.json
```

Synthetic full training/export/load and gradient checks pass with model
frameworks blocked. Independent CPU checks cover CNN training and Gaussian
projection, compositing and optimizer updates, including several tiles and
visibility/clipping branches. GPU kernel parity and visual quality of newly
trained assets remain deferred. Earlier appearance quality figures refer to
the previous fitted assets.

## Direct-TTS motion training and live integration

`animation.corpus.capture` resamples an existing `vhuman.performance.v1` teacher
take at feature sample start + {0,240,...,1680}. This interpolation is an offline
training operation, not future-dependent live inference.
`animation.train.train` reads `vhuman.motion_corpus.v1` JSON with `names`, `ranges`,
`tts_revision`, `purpose`, `provenance`, and `takes:[{path,split}]`. Each checksum
verified NPZ contains hidden[T,H], codes[T,16], controls[T,8,C]. Use distinct
utterances for `train` and `validation`; changing exact TTS weights invalidates
adapter compatibility. Features from different hidden sizes cannot be mixed.

```sh
sh server/vhuman/realtime/run.sh train-motion --manifest motion-corpus.json \
  --output tmp/vhuman-realtime/motion.native --epochs 20
sh server/vhuman/realtime/run.sh live --rig "$RIG" \
  --avatar tmp/vhuman-realtime/avatar.npz --adapter tmp/vhuman-realtime/motion.native \
  --model /mnt/nvme01/models/speech/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --revision ACTUAL_MODEL_SAFETENSORS_SHA256 \
  --text 'こんにちは。' --sink device --resident --tts-threads 4 --display
# Optional: --video out.mp4 or --virtual-camera. Synthetic tests require --diagnostic.
```

The student uses training-only per-channel hidden normalization, then hidden projection128 +16 token embeddings16 +2-layer GRU128,
with recurrent state reset per utterance epoch. Training uses active-control weighted
Huber loss and temporal velocity; validation features never set normalization. Named retargeting handles tongueOut in
the extra controls. Missing controls become neutral. No LLM is integrated.

Native training currently uses CPU, at most 32 feature records per truncated
backpropagation chunk, and an explicit thread budget (default 4). Optimizer resume
is not implemented. Initialization follows the reference distributions with a
NumPy seed; it does not reproduce Torch's random stream. CPU reference checks
passed with maximum forward error 5.96e-8, gradient error 1.49e-8, and AdamW error
4.77e-7 when supplied identical gradients. GPU and visual quality validation of
newly trained assets are deferred.

```sh
python -m unittest server.vhuman.test_native_training \
  server.vhuman.realtime.test_training.MotionTrainingTests -v
# Optional CPU Torch oracle, isolated from the training runtime:
python ref/vhuman/verify_motion_training.py --output tmp/motion-training-parity.json
```


## Reproduce the exercised clean PoC

The following generated assets exist locally under ignored `tmp/`. They are not
bundled pretrained production models. Recreating the identity can produce a
slightly different portrait across library/hardware changes; manifests record
seeds, revisions and image hashes. This section records the earlier assets and
their legacy training run; its quality results do not validate the new native
trainer.

```sh
HEAD=tmp/vhuman-realtime/clean-heads/heads/f17ebe3fd0e6
RIG="$HEAD/reconstruction/d4f19d9cf4534f84/rig"
sh server/vhuman/realtime/run.sh export-neutral --head "$HEAD" \
  --identity tmp/vhuman-realtime/clean-identity-001 \
  --output tmp/vhuman-realtime/clean-neutral-corpus-v3
sh server/vhuman/realtime/run.sh fit-appearance \
  --manifest tmp/vhuman-realtime/clean-neutral-corpus-v3/manifest.json \
  --output tmp/vhuman-realtime/clean-neutral-avatar-v3.npz --count 50000 --steps 2000
sh server/vhuman/realtime/run.sh render --rig "$RIG" \
  --avatar tmp/vhuman-realtime/clean-neutral-avatar-v3.npz --frames 100 \
  --output tmp/vhuman-realtime/clean-neutral-render-srgb.png
sh server/vhuman/realtime/run.sh train-motion \
  --manifest tmp/vhuman-realtime/motion-corpus-v2/manifest.json \
  --output tmp/vhuman-realtime/motion-v3.pt --epochs 150
sh server/vhuman/realtime/run.sh live --rig "$RIG" \
  --avatar tmp/vhuman-realtime/clean-neutral-avatar-v3.npz \
  --adapter tmp/vhuman-realtime/motion-v3.pt \
  --model /mnt/nvme01/models/speech/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --revision bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb \
  --text 'こんにちは。今日は、このアバターの動きと音声の同期を確認します。' --sink offline --resident --tts-threads 4 \
  --max-frames 256 --diagnostic --video tmp/vhuman-realtime/improved.mp4
```

Replace the head/reconstruction IDs with CLI results for a newly generated
identity. The current motion corpus has16 complete utterances (69.68s;12 train/4 validation),
Qwen hidden/token features and existing Apache Japanese CTC/original-viseme
teacher takes. `collect-motion` rejects low-energy audio, incomplete codec records
and neutral teacher takes. The normalized student beats neutral on held-out
sentences; perceptual animation quality remains unvalidated. See VALIDATION.md
for corpus capture/evaluation commands. No old research portrait was used as an
appearance target or identity-generation condition. GNM geometry comes from its
existing cleared model path. Generated expression photographs are present but
have not yet been registered/used for appearance training.

The complete 1.2-second offline-sink live run produced 73 ticks, **55.5 elapsed
FPS**, warm first PCM **97.5 ms** from request submission, and sampled NVML memory
**492 MiB avatar + 1788 MiB TTS** (two samples each, not certified peaks). Stage
wall p50/p95: motion **1.53/2.13 ms**, rig submission **0.20/0.33 ms**,
render completion **6.29/10.90 ms**, compositor **1.22/4.53 ms**. Buffer maximum
was **140 ms**. Final partial-tick silence was 320 samples; p95 underflow was zero.
Two separate resident requests had first PCM107/108ms; the final integration
regression under concurrent GPU tests measured116/116ms. Cancellation/reuse passed.
The renderer/compositor are prewarmed before TTS and audio start.

**Limits:** the neutral fit has visible eye, hair and surface artifacts. It was
trained on one frontal photograph, without held-out quality evidence. Teeth,
tongue, profiles, expressions and lighting are not validated. The initial MP4 is silent; newer headless recordings mux actual playout PCM,
including inserted silence, and retain an exact PCM16 WAV sidecar. The offline sink simulates playout, so audible phoneme-to-mouth
latency, DAC latency, sustained real-time codec performance and actual window/
virtual-camera operation are not established by this run. Use `--sink device`
only with an available audio device; it fails clearly when none is configured.

## Timing and ownership

PCM ring capacity2s, nominal startup80ms, high400ms/low160ms; motion capacity200
samples at100Hz. The live producer stops accepting audio at high water; native
and Python bounded pipes/queues propagate backpressure. The low-water setting is
reserved for a future hysteretic scheduler. Renderer output is synchronous and
does not accumulate stale queued frames.

All media timestamps are int64 sample positions. Epoch/sequence validation
rejects stale or reordered packets. DAC playout and utterance speech are separate
timelines: inserted underrun silence advances playout and holds speech position.
Motion interpolates at the DAC speech position; it holds then eases to neutral
after stale motion. Integer rate conversion avoids long-session rounding drift.
An overflow stops the session instead of overwriting unplayed audio/motion.

The C CUDA runtime retains the primary context and owns a nonblocking stream.
Native rig submission returns a borrowed
`DeviceView`, without importing Torch. Pinned host rig staging has an upload
fence; one fused deformation/contact operation is enqueued per frame. DLPack
exports retain the owner and accept only the same stream. Views are overwritten
by the next submit and must be released before close; copy on the same stream
for persistent geometry. Completion/timing events, RGB/alpha conversion and
pinned output download use the native library. Rendering deforms and projects
Gaussians on CUDA, bins/sorts their tile intersections on the host, then blends
pixels on CUDA. Frames own independent device allocations. The renderer checks
frame/event/rig lifetimes before closing. TTS runs in a separate process/private
context. Earlier gsplat FPS figures below describe the previous integration.

Both `trace-v1` and `eigen-v1` covariance policies are supported, with fixed-light
RGB, pinhole cameras and 16x16 tiles. This is an inference renderer, not a
differentiable training backend. It has no third-party GEMM or rasterizer dependency.
The eight expression coefficients and 3x3 transforms use direct fixed-size math.
Output is bounded to 4096x4096 and eight million tile intersections; exceeding
the overlap budget fails explicitly. Host sorting limits speed relative to the
previous gsplat backend.

## Verification and measured limits

With GPU permissions enabled, the new path was exercised on RTX 5060 Ti 16 GB:

- Synthetic geometry matched NumPy for both covariance policies; native/gsplat
  maximum RGBA error was 1.73e-6. Retained frames, clipped/degenerate geometry,
  native output conversion and close guards passed on CUDA.
- The trained 50k-Gaussian neutral avatar passed four pose comparisons. Worst
  mean RGBA error was 2.03e-7 and relative L2 2.78e-5. A few pixels differed by
  up to 0.00572 near discrete rasterization/ordering thresholds.
- A framework-free 120-frame animated CLI render measured 91 FPS at 512x512.
  Renderer-managed peak allocation was 22.5 MiB, excluding rig, CUDA context,
  output staging and TTS allocations; this is not whole-process VRAM.
- Native resident TTS → GRU → rig → render → MP4 completed 211 frames/5.6 seconds
  of speech at 36.5 presentation FPS. Render completion p95 was 23.65 ms. Some
  underrun silence occurred; this does not certify lip-sync or appearance quality.

Reports and outputs: `tmp/vhuman-native-renderer/{clean-parity,live-report.json,live.mp4}`.
Reproduce independent real-avatar comparison with a Torch/gsplat oracle environment:

```sh
python ref/vhuman/verify_renderer.py --rig PATH_TO_RIG --avatar PATH_TO_AVATAR \
  --out tmp/vhuman-renderer-check --reference --frames 60
# Omit --reference to run entirely without Torch/gsplat.
python -m unittest server.vhuman.realtime.test_native_renderer -v
# Optional training + oracle environment: 4 tests passed on the 5060 Ti.
python -m unittest server.vhuman.realtime.test_training.AppearanceTrainingTests \
  server.vhuman.realtime.test_native_renderer -v
```

The final framework-free regression command passed38 tests with one optional
Torch oracle skipped:

```sh
TMPDIR="$PWD/tmp/vhuman-native-runtime" tmp/mhr-native/runtime/bin/python -m unittest \
  ref.hunyuan_video15.test_compare ref.hunyuan_video15.test_validation_matrix \
  server.vhuman.realtime.test_native_runtime server.vhuman.realtime.test_native_identity \
  server.vhuman.realtime.test_lifecycle server.vhuman.realtime.test_output \
  server.vhuman.test_video server.vhuman.test_asset_dependencies ref.vhuman.test_flux2_tokenizer
```

Initial dependency-removal checks (2026-10-02, before enabling sandbox GPU access):

```sh
TMPDIR="$PWD/tmp/vhuman-native-runtime" tmp/mhr-native/runtime/bin/python -m unittest \
  ref.hunyuan_video15.test_compare ref.hunyuan_video15.test_validation_matrix \
  server.vhuman.realtime.test_native_runtime server.vhuman.realtime.test_native_identity \
  server.vhuman.realtime.test_lifecycle server.vhuman.realtime.test_output \
  server.vhuman.test_video server.vhuman.test_asset_dependencies
# 37 tests: 36 passed, optional Torch oracle skipped.

TMPDIR="$PWD/tmp/vhuman-native-runtime" tmp/vhuman-rig-venv/bin/python -m unittest \
  server.vhuman.test_asset_dependencies server.vhuman.test_rig_backend \
  server.vhuman.realtime.test_training.MotionTrainingTests
# 8 tests: 7 passed, ROCm hardware test skipped.

make -C cuda/hunyuan_video15_native \
  BUILD="$PWD/tmp/hv15-integration-review/build" test compile-kernels
# Host math/tokenizer/request checks, Python orchestration tests and compute_120 compilation passed.
```

The native presentation CPU math matched an independent NumPy formula within
one RGB8 level; DLPack ownership and close guards passed host tests. The 12
local HTTP tests passed with loopback socket access. No GPU stream-lifetime,
rendering-performance or new model-parity claim follows from these host checks.
Earlier GPU measurements follow below.

```sh
tmp/vhuman-rig-venv/bin/python -m unittest server.vhuman.realtime.test_runtime -v
tmp/vhuman-rig-venv/bin/python -m unittest server.vhuman.realtime.test_training.MotionTrainingTests -v
OMP_NUM_THREADS=8 tmp/vhuman-realtime/speech/test_codec_stream \
  /mnt/nvme01/models/speech/Qwen3-TTS-12Hz-0.6B-CustomVoice/speech_tokenizer
make -C speech BUILD=../tmp/vhuman-realtime/speech test
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  server/vhuman/realtime/test_audio_device.c -lportaudio -lm \
  -o tmp/vhuman-realtime/tests/test_audio_device
tmp/vhuman-realtime/tests/test_audio_device
# Requires the real model and GPU; verifies two turns plus cancel/reuse:
tmp/vhuman-rig-venv/bin/python -m unittest server.vhuman.realtime.test_resident -v
# GPU checks use the same environment variables as run.sh:
# python -m unittest server.vhuman.realtime.test_gpu server.vhuman.realtime.test_training
```

Observed on the RTX 5060 Ti, 100-frame diagnostic runs after warmup:

| Workload | Wall FPS | CUDA event ms/frame | Torch allocated peak / reserved peak |
|---|---:|---:|---:|
| 5k static, 512 square | 450 | 2.22 | 58 /78 MiB |
| 50k static, 512 square | 167 | 5.97 | 254 /264 MiB |
| 50k animated native rig, 720 square | 136 | 7.33 | 258 /464 MiB |
| Clean trained neutral, 50k, 512 square | 175 | 5.73 | 254 /264 MiB |

Square720 is not full1280×720. These measurements include Gaussian deformation
and rasterization; animated runs also submit the native rig each frame. They
exclude display/audio/LLM/TTS and photoreal appearance quality. Torch allocation
does not include native buffers or CUDA context overhead. Live NVML counters
report separate avatar/TTS process VRAM and global utilization when available.

Short native streaming run: first PCM134ms after generation started, excluding
2s-class cold load; optimized longer run first PCM117ms excluding0.95s load,
31 frames/2.48s audio generated in2.85s including CPU codec (RTF1.15).
Earlier unoptimized longer run was RTF4.20. Concurrency and short silent prefixes
make these smoke timings insufficient to guarantee audible speech TTFA or p95.
Persistent serving is implemented and exercised; stateful GPU codec and sustained
concurrency/quality benchmarks are still required. Rendering FPS does not establish animation latency.

The 77-frame codec test crosses the72-frame KV window and resets state;
after optimization maximum waveform error was2.50e-6, RMSE9.26e-8 versus full
decoding. Native GPU contacts agree with the CPU rig within2e-6 absolute /2e-5
relative tolerances. CPU tests cover fragmented pipes, contention, cancellation,
queue overflow, covariance transport, named retargeting, checksum refusal and an
hour of integer-clock accounting. Synthetic training checks verify gradients and
streaming checkpoint loading; they provide no speech realism evidence.

See [VALIDATION.md](VALIDATION.md) for the improvement pass, commands and measured
quality/latency limits. See [TASKS.md](TASKS.md) for remaining independently testable production work and
[upstreams.json](upstreams.json) for reviewed upstream revisions.
