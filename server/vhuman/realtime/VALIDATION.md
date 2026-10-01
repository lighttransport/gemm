# Improvement and validation pass — 2026-10-01

The direct-TTS pipeline now has a better neutral appearance fit, a speech-bearing
motion corpus and independent sentence-level student evaluation. It remains a
diagnostic PoC: rendered speech still has mouth/eye surface artifacts, and the
actual audible-startup goal is not met.

## Changes and evidence

* Appearance fitting and rendering now share differentiable geometry. New bundles
  use `covariance_policy=trace-v1`; legacy bundles keep `eigen-v1`. This removes the
  training/runtime bounds mismatch and the per-frame covariance eigensolve.
  Repeated-eigenvalue gradient tests and CPU/GPU parity pass.
* Initialize Gaussian colors from the cleared photograph and supervise foreground
  alpha as well as premultiplied linear RGB. The initial fit could reproduce RGB
  using a bright, transparent face, which looked washed out over a background.
* Headless audio follows elapsed monotonic time, including renderer stalls.
  Expired video deadlines are dropped rather than accelerating simulated audio.
  Recording uses playout sample positions, duplicates missed CFR slots and muxes
  actual PCM—including underrun silence. An exact PCM16 `.playout.wav` is retained.
* Complete-utterance capture rejects low-energy audio, truncated/mismatched codec
  frames and nearly neutral teacher controls. Offline Japanese CTC is used only
  for training supervision; live inference still consumes direct TTS features.
* The old two truncated diagnostic takes were almost entirely silence. The first
  replacement student also barely beat neutral. Training-only feature normalization
  fixes hidden-channel outliers (observed values up to100); active-control weighted
  Huber and temporal velocity losses replace the original loss. Validation features
  never set normalization. Legacy raw-feature checkpoints remain compatible.
* First PCM and first speech-bearing chunk are separate counters. The latter uses
  a20ms RMS threshold0.003, not a phoneme detector or a calibrated audible timestamp.
* TTS text-feed mode is explicit. `full` still emits incremental PCM/features; it
  conditions on the complete supplied text segment. Checkpoints record the mode,
  and live rejects mismatched modes. The comparison below does not justify switching
  the current `incremental` adapter.

## Appearance and renderer results

RTX5060Ti16GB, driver615.71.09, Torch2.14.0+cu130, CUDA13.2, Linux/Python3.12.
50k splats,512 square, fixed frontal calibration, one clean photograph.

| Reconstruction metric | Initial1000-step fit | Corrected2000-step, mask/alpha fit |
|---|---:|---:|
| Foreground linear-RGB L1 | 0.01210 | 0.01109 |
| Full-image linear-RGB PSNR | 30.08dB | 30.33dB |
| Mean foreground alpha | 0.459 | 0.898 |

These are **training-image reconstruction metrics**, not held-out photorealism.
The corrected appearance is `tmp/vhuman-realtime/clean-neutral-avatar-v3.npz`;
its report/render are in `tmp/vhuman-realtime/eval-v3/`.
The fit's final training L1 was0.006873; exported runtime L1 was0.006869.

300 warmed frames with animated native rig: **409 wall FPS**,2.45ms CUDA perframe,
Torch peak63MiB allocated/84MiB reserved. This excludes TTS, display and recording.
The prior eigensolve-based static50k benchmark was175FPS; the workloads differ,
so this is not a controlled percentage speedup. Full720p remains unmeasured here.

## Motion evaluation

16 complete Japanese utterances,69.68s generated audio:12 train,4 validation.
Same native Qwen0.6B weights/Ono_Anna/incremental text feed throughout.
The normalized student was trained150 epochs on the12 training sentences.

| Held-out teacher agreement | Neutral baseline | Student |
|---|---:|---:|
| Active-control MAE | 0.04674 | 0.01767 |
| JawOpen MAE | 0.15304 | 0.03967 |

Per-sentence jaw correlation:0.923–0.946; best cross-correlation lag0–10ms.
Stepwise/full-sequence maximum difference<=4.18e-7. Equal-utterance weighting
prevents longer takes from hiding failures. These compare against an offline
CTC/viseme teacher; they do not establish actual rendered lip-sync or motion realism.
Checkpoints/corpus/reports: `motion-v3.pt`, `motion-corpus-v2/`,
`motion-v3-evaluation.json` under `tmp/vhuman-realtime/`.

## Live and TTS results

The improved appearance/student ran together for15.04s of generated speech:
**59.89 presentation FPS**,922 rendered ticks. Sampled NVML avatar268MiB,
TTS1788MiB,16 samples perprocess. These include process context/native allocations
but are not allocation peaks. Torch live allocation44.7MiB.

| Wall measurement | p50 | p95 |
|---|---:|---:|
| Motion inference | 1.33ms | 2.02ms |
| Native rig submission | 0.25ms | 0.34ms |
| Render completion | 3.52ms | 8.05ms |
| Compositor/record write | 1.66ms | 4.75ms |
| Presentation interval | 17.13ms | 22.41ms |

Buffer depth maximum323ms. First PCM124ms; first speech-bearing chunk571ms;
the waveform begins with roughly400ms low energy. Recorded playout has8229
inserted silence samples (~343ms total, including terminal padding). Underflow
p95 waszero but p99 was400 samples: sustained jitter is **not solved**.
An earlier15s run had only224 inserted samples; contention/timing vary.
The Japanese CTC recognizer recovered the long spoken sentence. The MP4 now has
AAC audio and an exact PCM sidecar, both on the same simulated playout timeline.

100 resident requests produced338.4s of speech-bearing audio. Post-request NVML
was1788MiB throughout, with **zero sampled growth**. FirstPCM p50/p95/p99:
121/174/662ms, maximum784ms. Portions ran concurrently with CPU training,
another avatar/TTS worker and a text-feed comparison; external contention was not
controlled. This is a residency/correctness soak, not an isolated latency guarantee.
Cancellation/reuse also passes a separate real-model integration test.

Decoder thread profiling, three complete requests per setting:

| Threads | CPU codec mean perframe (range across requests) | First PCM p50 | First speech chunk p50 |
|---:|---:|---:|---:|
| 1 | 132–133ms | 191ms | 2432ms |
| 2 | 73–75ms | 124ms | 1471ms |
| 4 | 53–56ms | 102ms | 1138ms |
| 8 | 48–52ms | 105ms | 1106ms |

Short requests have different low-energy prefixes than the long live sentence.
Full-text conditioning/4threads measured106ms firstPCM and1141ms first speech
chunk p50 across three short requests, so it did not materially help.
Stateful GPU codec and prefix behavior are the next audible-latency priorities.

## Expression/input and device limits

All13 clean reference images were audited against calibrated final rig landmarks
with their **approximate** authored controls. Neutral mouth RMS3.33px; generated
expressions range3.94–49.35px. Jaw-open49.35px and stretch20.03px show that pose
labels are not registered well enough to train as exact poses. None is automatically
accepted for expression training. The rendered talking mouth/eyes visibly become
grainy/deformed; a neutral-only appearance fit does not solve expression appearance.
Next: fit expression poses/camera drift, verify anatomy/component attachments,
then train and evaluate photographic expression radiance.

No `/dev/snd` exists here. Actual DAC/presentation calibration, physical audio
device restart and long-session hardware drift remain untested. Desktop/virtual
camera integration also remains untested. The headless clock and native callback
underrun tests are not substitutes for those checks.

## Reproduction

Run from repository root with the setup in [README.md](README.md):

```sh
HEAD=tmp/vhuman-realtime/clean-heads/heads/f17ebe3fd0e6
RIG="$HEAD/reconstruction/d4f19d9cf4534f84/rig"
MODEL=/mnt/nvme01/models/speech/Qwen3-TTS-12Hz-0.6B-CustomVoice
REV=bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb
sh server/vhuman/realtime/run.sh export-neutral --head "$HEAD" \
  --identity tmp/vhuman-realtime/clean-identity-001 --output tmp/vhuman-realtime/clean-neutral-corpus-v3
sh server/vhuman/realtime/run.sh fit-appearance \
  --manifest tmp/vhuman-realtime/clean-neutral-corpus-v3/manifest.json \
  --output tmp/vhuman-realtime/clean-neutral-avatar-v3.npz --steps 2000 --count 50000
sh server/vhuman/realtime/run.sh evaluate-appearance \
  --manifest tmp/vhuman-realtime/clean-neutral-corpus-v3/manifest.json \
  --avatar tmp/vhuman-realtime/clean-neutral-avatar-v3.npz --output tmp/vhuman-realtime/eval-v3
make -C speech BUILD=../tmp/vhuman-realtime/speech ../tmp/vhuman-realtime/speech/ja_align_cuda
sh server/vhuman/realtime/run.sh collect-motion --rig "$RIG" --model "$MODEL" \
  --runner tmp/vhuman-realtime/speech/qwen3_tts_cuda \
  --aligner tmp/vhuman-realtime/speech/ja_align_cuda \
  --align-model /mnt/nvme01/models/speech/japanese-wav2vec2-large-hiragana-ctc/ja_align.safetensors \
  --sentences server/vhuman/realtime/examples/japanese_motion_sentences.json \
  --output tmp/vhuman-realtime/motion-corpus-v2 --threads 4
sh server/vhuman/realtime/run.sh train-motion --manifest tmp/vhuman-realtime/motion-corpus-v2/manifest.json \
  --output tmp/vhuman-realtime/motion-v3.pt --epochs 150 --threads 4
sh server/vhuman/realtime/run.sh evaluate-motion --manifest tmp/vhuman-realtime/motion-corpus-v2/manifest.json \
  --checkpoint tmp/vhuman-realtime/motion-v3.pt --output tmp/vhuman-realtime/motion-v3-evaluation.json
sh server/vhuman/realtime/run.sh live --rig "$RIG" --model "$MODEL" --revision "$REV" \
  --avatar tmp/vhuman-realtime/clean-neutral-avatar-v3.npz --adapter tmp/vhuman-realtime/motion-v3.pt \
  --text 'こんにちは。今日は、このアバターの動きと音声の同期を確認します。' \
  --sink offline --resident --tts-threads 4 --max-frames 256 --diagnostic \
  --video tmp/vhuman-realtime/improved.mp4
sh server/vhuman/realtime/run.sh stress-tts --model "$MODEL" \
  --runner tmp/vhuman-realtime/speech/qwen3_tts_cuda --requests 100 --threads 4 \
  --output tmp/vhuman-realtime/tts-soak.json
sh server/vhuman/realtime/run.sh audit-references --head "$HEAD" \
  --identity tmp/vhuman-realtime/clean-identity-001 --output tmp/vhuman-realtime/expression-audit.json
tmp/vhuman-rig-venv/bin/python -m unittest server.vhuman.realtime.test_runtime \
  server.vhuman.realtime.test_training.MotionTrainingTests server.vhuman.realtime.test_output \
  server.vhuman.realtime.test_lifecycle -v
# Real-GPU tests use run.sh's gsplat build environment:
# python -m unittest server.vhuman.realtime.test_gpu server.vhuman.realtime.test_training -v
tmp/vhuman-rig-venv/bin/python -m unittest server.vhuman.realtime.test_resident -v
```

CPU/output/lifecycle tests27 passed; CUDA/deformation/appearance tests4 passed;
both real resident request/cancel/reuse tests passed, including cancellation while
the worker was paused before request setup. Cancellation waits for the native
request-start acknowledgement; appearance inference requires a trained bundle
outside diagnostic mode; cleanup attempts all releases and preserves the original
runtime error. CPU and CUDA speech runners build with `-Wall -Wextra -Werror`.
Strict `-Wpedantic -Werror` also diagnoses existing NVRTC overlength strings and
POSIX dynamic-loader function-pointer casts in cuew; these were not changed here.
MediaPipe's native runtime stalled inside the execution sandbox; the identified
audit was stopped and the same command succeeded outside it. No old research
portrait was used in the new fitting or capture paths.
