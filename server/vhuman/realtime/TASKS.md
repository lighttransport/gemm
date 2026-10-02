# Engineering tasks and acceptance gates

Each task should be one reviewable change. Do not count synthetic fitting tests
as appearance quality, or rendering FPS as conversational latency.

## Implemented and exercised

The table retains earlier Torch/gsplat measurements as historical evidence.
The default inference renderer now uses the repository C++/CUDA implementation;
see [README.md](README.md#verification-and-measured-limits) for current native
measurements. Training and explicit oracle checks can still use Torch/gsplat.

| Task | Success criterion / evidence |
|---|---|
| Typed sample-positioned audio/motion/features | Invalid shape/range/timestamps fail; records own immutable data |
| Native bounded PCM SPSC ring | Wrap, backpressure, silence and10k-sample concurrent ordering tests pass |
| Native PortAudio callback and DAC bridge | Warning-clean build; synthetic callback verifies queue and physical host underruns without moving speech clock |
| Audio-master sample clock | One-hour44.1k/24k rational conversion stays exact; silence holds speech position |
| Epoch reset and control retargeting | Stale epochs rejected; tongueOut mapped by name; capacity never overwrites unplayed motion |
| Native Qwen feature hook | CPU/GPU runner build; seven real codec frames carry hidden[1024], tokens16 and0..11520 sample starts |
| Stateful CPU codec | Absolute RoPE,72-frame KV eviction, convolution histories;77-frame max error2.50e-6 vs full decode; reset matches |
| Timestamped PCM/features IPC | Independent bounded pipes; arbitrary read fragmentation tested; cancellation joins readers |
| Causal direct-TTS adapter + trainer | Full-sequence vs stepwise GRU parity; independently held-out synthetic take trains and loads |
| Corpus resampling API | `animation.corpus.capture` maps sample positions onto teacher controls; two real truncated diagnostic teacher takes exercised; larger quality corpus pending |
| Versioned Gaussian bundle | Topology/order/provenance/finite SPD/bounds checks; atomic save/load; rigid covariance parity |
| CUDA primary-context rig and DLPack view | Four poses/contacts match CPU; borrowed view retains owner and prevents premature free |
| gsplat RGB build on sm_120 | Pinned Apache upstream builds with CUDA13.2 and Torch2.14; rasterization finite and visible |
| Native CUDA Gaussian inference and output | Both covariance policies match NumPy; four real-avatar poses match gsplat with worst mean RGBA error2.03e-7; framework-free512 render91FPS; native TTS-to-MP4 run36.5FPS |
| Appearance fitter | Synthetic gradient test plus clean neutral50k/1000-step fit; photographic quality unvalidated |
| Static/dynamic render benchmarks | 50k splats:167FPS512 static;136FPS720-square animated; no appearance claim |
| Incremental offline/device replay + live wiring | Code paths implemented; replay/pipe tests exercised; clean trained diagnostic live run55.5FPS; hardware audio/quality validation pending |
| Optional display/record/virtual-camera adapters | Implemented; actual graphical/device integration remains to be exercised |
| Clean identity provider | Pinned Apache FLUX provider; new neutral only; 13 new neutral/expression images generated with receipts; old conditioning excluded |
| Persistent native TTS | Two requests retain process/weights;107/108ms warm firstPCM; cancellation/reuse exercised |
| Clean neutral registration | New GNM reconstruction; corrected native mouth namespace; calibrated projection error0.000101px |
| Shared fit/runtime geometry and alpha | Versioned trace policy, finite repeated-eigenvalue gradients, CPU/GPU parity; neutral alpha0.46->0.90 |
| Complete motion corpus and held-out evaluation | 16 complete takes/69.68s,12 train/4 validation; reject silent or motionless captures; normalized causal student beats neutral |
| Actual playout recording | Monotonic-time offline sink; exact PCM sidecar; CFR missed-frame duplication and AAC mux tests |
| TTS residency soak | 100 complete requests/338.4s audio, sampled process VRAM1788MiB throughout; p95 firstPCM174ms under varying contention |
| Expression-input audit | 13 landmark audits; jaw-open mouth residual49px, so approximate labels are not accepted as registered training targets |
| Telemetry | Stage wall times,percentiles,maxima,buffer depth,Torch and optional per-process NVML memory; latency calibration pending |

## Next tasks, in dependency order

1. **Complete expression-reference QA.** The pinned provider generated all13 images;
   neutral and jaw-open were inspected; all13 have landmark audits. Review identity/lighting and fit pose labels before using them as expression targets. Success: checksum receipts for every output, no old
   portrait/rig conditioning, consistent identity, eye openness and mouth closure.
2. **Audit the clean GNM rig receipts.** The separate reconstruction and rig are
   built; inspect all component provenance and anatomical outputs. Success: isolated head directory; correct welded topology/control
   order; every required rig asset has a provenance chain; existing heads untouched.
3. **Register photographic references to final rig poses.** Add a corpus exporter
   on top of the implemented neutral camera exporter; register landmarks and
   expressions with uncertainty estimates. Success: round-trip camera projection
   within0.5px on synthetic tests; measured landmark errors on every real frame;
   low-confidence expressions excluded, not assigned fabricated exact labels.
4. **Mask/crop targets and handle mouth components.** Export linearRGB targets over
   black and explicit component masks, with static background separately. Success:
   teeth/tongue/eyes have correct attachment/component IDs and no face-mask leakage.
5. **Improve the first50k appearance bundle.** Neutral1000-step fitting is done;
   fit registered expressions and preserve a held-out frontal image. Success: successful bundle load and distinct held-out image
   metrics/visuals; lip closure and eye highlights pass inspection. No quality
   claims based on training loss alone.
6. **Capture a Japanese motion corpus.** Generate at least100 varied utterances,
   short/long pauses and vowel/consonant coverage, using exact pinned TTS weights.
   Use an independently cleared offline teacher (existing Apache Japanese CTC
   + original viseme mapping is a starter; LAM remains a separately audited option).
   Success: feature/PCM sample counts agree, takes have model/data checksums,
   sentence-level train/validation separation and independently checked closures.
7. **Expand/evaluate the real direct-TTS student.** The normalized150-epoch student
   beats neutral on4 held-out sentences (active MAE0.0177 vs0.0467); expand coverage
   and compare to an energy baseline. Success: causal step parity, held-out jaw/lip/closure
   errors, signed head/gaze range compliance and no future-context dependency;
   mouth latency measured against actual audible phonetic landmarks.
8. **Stress persistent native TTS serving.** Weight/context reuse, request framing
   and cancellation are implemented; test long sessions and failures.
   The100-request test passed with no sampled VRAM growth. Next success criteria: cancel/restart produces
   no stale PCM/features; warm TTFA p50/p95 reported separately from cold load.
9. **Port stateful codec to CUDA.** Reuse the tested CPU state machine, preserve
   absolute RoPE and convolution state, reserve fixed buffers and skip unused
   full-batch codec weights. Success:77-frame/long-utterance parity across KV wrap,
   tail/reset/cancel, bounded VRAM, steady codec RTF<=0.5 and measured p95<40ms/frame.
10. **Extend live PoC acceptance.** Prerecorded/chunked replay and native streaming
    TTS run; extend beyond the current short diagnostic offline-sink session. Success:512 minimum output at>=30FPS, avatar NVML peak<=8GB,
    audible mouth alignment<100ms where verified; p50/p95/p99 and failures recorded.
11. **Verify native audio hardware.** Test24k direct and driver-resampled output,
    physical underruns, device restart and a30-minute session. Success: no drift,
    no callback allocation/locks/Python/model work; calibrated DAC/presentation offset.
12. **Add head/eye components and motion overlays.** Blink/gaze/head/emotion API must
    preserve sample timestamps and clamp supported poses. Success: symmetric blink
    closure, controlled gaze, teeth/tongue visibility and no contact intersection
    over a labelled pose sweep; photographic upper-face targets needed.
13. **Add renderer/display stream overlap.** Separate motion/render streams only
    where dependencies allow; pass explicit completion events and retain tensors.
    Success: no global synchronizations in steady rendering, no stale-frame queue,
    race/lifetime stress tests and improved p95 under TTS contention.
14. **Optimize renderer with measured memory budgets.** Shared trace bounds remove
    the per-frame eigensolve;50k animated512 measured409FPS without TTS. Profile triangle gather,
    covariance eigensolve,deformation and rasterization separately; preallocate
    staging and use CUDA graphs where shapes are fixed. Success: same image/pose
    tolerances, actual process VRAM<=4GB desired,>=60FPS720p desired under TTS load.
15. **Validate output integrations.** Exercise desktop window,ffmpeg output and
    configured virtual camera. Success: correct sRGB/alpha, expected dimensions,
    timestamped frames, graceful close, video/audio mux sample alignment. Headless playout mux and sidecar tests pass; hardware/device recording remains.
16. **Measure photorealism limits.** Held-out speech,lip closures,teeth/tongue,
    blinks,modest profiles and temporal flicker. Success: reference clips and
    failure cases checked in as reports; pose/lighting support is accurately bounded.
17. **Integrate the LLM last.** Add a text segment source and bounded streaming
    sentence/clause scheduler into the persistent TTS API. Success: interruptible
    turn playback, first-segment latency metrics and no audio-generation regression.

Commercial production acceptance requires the clean subject/data path and
measured latency/quality gates. The current diagnostic runtime is useful for
bring-up, but those acceptance conditions are not yet achieved.
