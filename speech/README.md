# speech — Qwen3-TTS runner + Japanese CTC aligner (C / CUDA)

A standalone project with two parts. Neither needs PyTorch at runtime.

- **`qwen3_tts/`** runs Qwen3-TTS (12 Hz, CustomVoice) and turns text into a 24 kHz waveform.
  - It uses our own GEMM code: AVX2 on the CPU, NVRTC kernels on CUDA.
  - It has been validated against the PyTorch reference, and the waveforms match bit-for-bit to within float rounding.
- **`ja_align/`** is a dependency-free C99 module (optional OpenMP). It takes Japanese speech and produces:
  - phoneme and kana timings,
  - frame-level phoneme posteriors,
  - prosody (energy, F0),
  - viseme curves for facial animation.

Licensing: all code here is MIT. See `NOTICE` for the upstream work that was followed and the parts written clean-room.

## Models

```
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir /mnt/nvme01/models/speech/Qwen3-TTS-12Hz-1.7B-CustomVoice
hf download sakasegawa/japanese-wav2vec2-large-hiragana-ctc --local-dir /mnt/nvme01/models/speech/japanese-wav2vec2-large-hiragana-ctc
python ja_align/convert_ckpt.py --out /mnt/nvme01/models/speech/japanese-wav2vec2-large-hiragana-ctc/ja_align.safetensors
```

The TTS runner reads the following files straight from the model directory:
- `config.json`
- `model.safetensors`, in BF16, read in place
- `vocab.json` and `merges.txt`
- `tokenizer_config.json`
- `speech_tokenizer/`

## Build

```
make -C speech            # CPU tools -> speech/build/
make -C speech cuda       # CUDA runner (driver API + NVRTC, loaded with dlopen)
make -C speech test       # unit tests that need no weights
```

## Qwen3-TTS runner

```
speech/build/qwen3_tts --model <dir> --text "今日はいい天気ですね。" --speaker Ono_Anna \
    --language Japanese [--instruct "..."] [--seed N | --greedy] --out out.wav
```

How the runner follows the reference implementation (`qwen_tts`, Apache-2.0):

- **Talker.** A 28-layer Qwen3 model (hidden size 2048 for 1.7B) with q/k-norm and GQA.
  - The interleaved MRoPE reduces to 1-D RoPE because the T/H/W positions are always equal.
  - Its input is the sum of:
    - `text_projection(text_embedding(tok))`, a SiLU MLP from 2048 to the hidden size;
    - the codec embeddings of all 16 codebooks from the previous frame;
    - one trailing text embedding per step (`tts_pad` in non-streaming mode, which is the CustomVoice default).
- **Prompt, CustomVoice.** The token layout is:
  - `[instruct]`
  - `<|im_start|>assistant\n`
  - `tts_pad×k + tts_bos`, added element-wise to the codec tags `[think, think_bos, lang, think_eos, speaker, pad]`
  - `text + tts_eos`, added to `codec_pad`
  - `tts_pad`, added to `codec_bos`
- **Sampling.** Default settings match `generation_config.json`:
  - temperature 0.9, top-k 50;
  - repetition penalty 1.05 on codebook 0;
  - codec ids from 2048 upward are suppressed, except EOS;
  - EOS is blocked for the first 2 steps.
- **Code predictor.** A 5-layer Qwen3 model (hidden 1024). For each frame it prefills `[talker hidden, codec_emb(code0)]` through `small_to_mtp_projection`, then predicts codebooks 1–15 autoregressively using 15 separate LM heads.
- **Codec decoder.** Converts codes to a waveform in these stages:
  1. Split RVQ: 1 + 15 codebooks, 256-d.
  2. A causal k3 convolution.
  3. An 8-layer transformer with sliding window 72 and LayerScale.
  4. Two stages of 2× transposed convolution + ConvNeXt.
  5. A SEANet decoder: rates 8/5/4/3, SnakeBeta activations, residual units with dilation 1/3/9.
  6. Clamp to [-1, 1].

  Decoding is chunked into 300-frame chunks with 25 frames of left context, as in `chunked_decode`.

### Validation

Fixtures are in `tmp/speech/`.

```
# PyTorch reference (FP32, CPU, greedy) -> npy fixtures
python speech/ref/qtts_reference.py --dump-dir tmp/speech/ref_greedy0 --max-new-tokens 80
# C runner fed the same token ids, greedy
speech/build/qwen3_tts --greedy --max-frames 80 --ids tmp/speech/ref_greedy0/input_ids.npy \
    --dump-dir tmp/speech/c_greedy0 --out tmp/speech/c_greedy0.wav
python speech/ref/compare.py --reference-dir tmp/speech/ref_greedy0 --runner-dir tmp/speech/c_greedy0
# codec only, with random codes
python speech/ref/codec_reference.py --dump-dir tmp/speech/codec_rand
speech/build/test_codec <model>/speech_tokenizer tmp/speech/codec_rand/codes.npy tmp/speech/codec_rand_c
```

| Check (1.7B CustomVoice, Ono_Anna, 「今日はいい天気ですね。散歩に行きましょう。」) | Result |
|---|---|
| token ids (C BPE vs HF processor) | identical |
| prefill embeddings | cos 1.0, max abs 2.4e-7 |
| talker logits, 80 steps | cos 1.0, max abs 3.2e-5 |
| greedy codes, 79 frames × 16 | **identical** |
| codec stages (RVQ → final conv) | cos 1.0, rel L2 ≤ 2e-5 |
| waveform | **SNR 113 dB** |
| codec only (random codes) | SNR 106.8 dB |

CPU speed on a Threadripper 1950X with 16 threads: talker RTF 3.1, codec RTF 0.6. The CPU path is a reference path; the CUDA backend is the one meant for fast generation.

### CUDA backend

```
make -C speech cuda
speech/build/qwen3_tts_cuda --backend cuda --text "…" --out out.wav
speech/build/qwen3_tts_cuda --backend cuda --codes-in codes.npy --out out.wav   # decode only
```

`qwen3_tts/qtts_cuda.h` and `qtts_cuda_kernels.h` use the driver API with NVRTC, loaded through `cuda/cuew`. Nothing links against the CUDA toolkit, and every kernel is our own; there is no cuBLAS.

- **Talker and code predictor.**
  - BF16 weights stay resident on the GPU. Q/K/V and gate/up are each fused into a single matrix.
  - Decode (M ≤ 4) uses a warp-per-row BF16 GEMV with 16-byte loads. Prefill uses a 64×64 tiled GEMM.
  - Each head gets RMSNorm and RoPE in one fused kernel.
  - Attention is warp-split with an online softmax.
  - The KV cache is F32 and lives on the device.
- **Sampling stays on the host** and uses the same sampler code and Philox stream as the CPU backend. Only the logits come back over PCIe: 16 × ≤12 KB per frame.
- **Codec decoder.** All convolutions are an implicit-GEMM kernel with a causal gather. Transposed convolutions are a GEMM followed by overlap-add. SnakeBeta, ConvNeXt, LayerNorm and the sliding-window attention (reusing the LM kernels) are elementwise or small kernels. Weights are F32.
- Kernels are compiled without fast-math so results track the CPU reference.

| Check (same text as above) | Result |
|---|---|
| greedy codes vs PyTorch, 79 × 16 | **identical** |
| talker logits / hidden | max abs 4.2e-5 / 4.8e-5 |
| codec stages / waveform vs PyTorch | cos 1.0 / **SNR 109 dB** |
| sampled run, seed 7, CUDA vs CPU runner | **identical codes** (96 frames) |

Speed on an RTX 5060 Ti (16 GB, sm_120), with the GPU shared with another busy process: talker RTF 0.24–0.28, codec RTF 0.01–0.09, end to end ≈ 0.26–0.31.

### Voice cloning (Base model)

```
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir /mnt/nvme01/models/speech/Qwen3-TTS-12Hz-1.7B-Base
speech/build/qwen3_tts_cuda --backend cuda --model <Base dir> --ref-wav ref.wav \
    --ref-text "参照音声の書き起こし" --text "合成したい文" --out out.wav      # in-context (ICL)
speech/build/qwen3_tts_cuda ... --ref-wav ref.wav --xvec-only ...                  # speaker embedding only
```

The reference wav can be at any sample rate; it is resampled to 24 kHz. Both modes run on the CPU backend too, and `tts_ja` accepts the same options. New components, all running once per reference clip on the CPU:

- **`qtts_spk.h`: speaker encoder.**
  - It computes a log-mel spectrogram with an FFT, a Slaney mel filterbank and reflect padding, all written from the definitions. torch.stft and librosa served only as numeric oracles.
  - An ECAPA-TDNN then produces a 2048-d x-vector: TDNN, 3 SE-Res2Net blocks, attentive statistics pooling.
  - This follows `Qwen3TTSSpeakerEncoder` (Apache-2.0).
- **`qtts_codec_enc.h`: tokenizer encoder.** It encodes the reference audio into 12.5 Hz codes and follows the Apache-2.0 HF `MimiModel.encode` path. It consists of:
  - a causal SEANet encoder with strides 4/5/6/8;
  - an 8-layer causal transformer;
  - a stride-2 downsample with replicate padding;
  - split-RVQ nearest-centroid encoding (1 semantic + 15 acoustic codebooks).
- **Prompt construction (`qtts_talker.h`, `qtts_voice_clone`).**
  - The x-vector takes the speaker slot of the codec prefix.
  - ICL adds `(reference text + target text) + tts_eos` on the text track and `codec_bos + summed reference codes` on the codec track. It has streaming and non-streaming layouts, following `generate_icl_prompt`.
  - Output is decoded as `[reference codes + generated codes]` and the reference part is cut (`qtts_clone.h`).
  - Voice cloning defaults to the streaming text feed, as `generate_voice_clone` does.

**Validation.** The reference clip is JSUT BASIC5000_0002 resampled to 24 kHz, and decoding is greedy. The comparison is `speech/ref/qtts_clone_reference.py` vs `qwen3_tts_cuda --dump-dir`.

| Check | ICL, streaming | ICL, non-streaming | x-vector only |
|---|---|---|---|
| log-mel vs torch.stft + librosa | max abs 1.8e-4 | same | same |
| speaker embedding | cos 1.0 (max abs 1.9e-6) | same | same |
| encoder stages (SEANet / transformer / downsample) | cos 1.0 | same | — |
| reference codes, 62 × 16 | **identical** | **identical** | — |
| prompt embeddings | max abs 9.5e-7 | 9.5e-7 | 9.5e-7 |
| generated greedy codes | **identical** (39) | **identical** (24) | **identical** (24) |
| waveform | SNR 114.3 dB | 113.3 dB | 113.7 dB |

**Quality** (`speech/ref/clone_eval.py`, 10 sentences, sampling).
- Speaker similarity is the cosine of SpeechBrain ECAPA-VoxCeleb embeddings between the reference clip and the output. SpeechBrain is Apache-2.0 and used only as an evaluation oracle.
- CER uses the hiragana CTC model as in the intelligibility eval.

| Output | Speaker similarity to the reference | Kana CER |
|---|---|---|
| another recording of the same JSUT speaker (ceiling) | 0.744 | — |
| **clone, ICL** | **0.634** | 0.4% |
| clone, x-vector only | 0.575 | 1.3% |
| CustomVoice Ono_Anna (a different speaker) | 0.239 | 3.6% |

`speech/build/tts_ja_cuda --text "…" --out out.wav --aux aux.json` runs the whole pipeline: text → Qwen3-TTS (CUDA) → ja_align (CPU) → wav plus `aux.json`.

## ja_align — Japanese CTC aligner and speech features for facial animation

```
speech/build/ja_align --model ja_align.safetensors --wav speech.wav \
    [--kana "こんにちわ、きょーわ…"] [--phonemes "k o N n i ch i w a"] [--fps 30] \
    [--out aux.json] [--posteriors post.npy] [--dump-dir dir]
```

- **Model.** [`sakasegawa/japanese-wav2vec2-large-hiragana-ctc`](https://huggingface.co/sakasegawa/japanese-wav2vec2-large-hiragana-ctc), Apache-2.0.
  - A wav2vec2-large encoder pretrained on 35k hours of ReazonSpeech.
  - It has two CTC heads: a 42-phoneme InterCTC head at layer 12 (OpenJTalk-style symbols) and a kana CTC head at layer 24.
  - It runs at 20 ms frames.
  - `ja_align/convert_ckpt.py` turns the `.pt` checkpoint into safetensors and folds in the positional-conv weight norm.
- **Self-contained.** `ja_align/` has no dependencies outside itself: its own safetensors reader, GEMM, WAV I/O and DSP. It is plain C99 plus optional OpenMP and AVX2, with a scalar fallback.
- **Alignment modes.**
  - **Free** (the default): greedy CTC spans for phonemes and kana. This is the mode for TTS output, whose content is already known and clean.
  - **Forced**: when a reading is given, the phonemes and kana are aligned to it with Viterbi, and a CTC-segmentation confidence score is computed.
  - The reading has to be a *pronunciation* kana string (は→わ, long vowels as ー or a repeated vowel). Kanji-to-reading conversion is deliberately left out of the C module.
- **Outputs** (`aux.json`, `"format": "ja_align.v1"`):
  - `phones` and `kana`: CTC spans `{s, start, end, conf}` in seconds. Phones also carry a `viseme`.
  - `intervals`: contiguous phone intervals running from one onset to the next.
    - A phone holds while the energy stays above the loudest frame − 35 dB.
    - A quiet gap of 120 ms or more becomes `sil`.
    - `cl` (a geminate) takes the viseme of the following consonant.
  - `visemes`: `fps`, and `frames[n][15]` weights over `sil PP FF TH DD kk CH SS nn RR aa E ih oh ou`.
    - The weights are built from the intervals with ±35 ms linear co-articulation ramps and normalized to sum to 1.
    - Devoiced vowels (uppercase `U` / `I`) are weighted at half strength.
  - `prosody`: at a 10 ms hop, `rms_db` (dBFS, 25 ms window), `f0_hz` (YIN, 60–600 Hz, 0 when unvoiced) and `aperiodicity` (the minimum of the YIN CMNDF).
  - `--posteriors`: 50 Hz phoneme posteriors `[T, 43]`, as `.npy`. These are useful for soft, learned lip-sync models.

### Validation

These use fixtures in `tmp/speech/`. The test audio is Qwen3-TTS output from the C runner.

```
python speech/ref/align_reference.py --wav x.wav --dump-dir ref --phonemes "…" --kana "…"   # HF + torchaudio + ctc-segmentation
speech/build/test_w2v2 ja_align.safetensors ref/input.npy c && python speech/ref/compare.py --reference-dir ref --runner-dir c
speech/build/test_ja_ctc ref
make -C speech test
```

| Check | Result |
|---|---|
| encoder stages vs HF `Wav2Vec2Model`, same 16 kHz input | cos 1.0; log-posterior max abs 4e-5 |
| forced alignment vs `torchaudio.functional.forced_align` (black-box oracle), phoneme + kana | **0 / 383 frames differ** |
| CTC-segmentation vs the `ctc-segmentation` package (the method ESPnet uses) | timings exact; `char_probs` exact; confidence identical to 1e-6 |
| Viterbi vs brute-force enumeration (200 random problems) | optimal in every case |
| end to end with our resampler vs a torchaudio-resampled reference | posterior cos 0.99997, argmax agreement 99.2% |
| resampler 24→16 kHz / YIN / RMS | 80 dB tone SNR, −64 dB stopband / 0.8% F0 error / 0.15 dB |

Speed: CPU RTF ≈ 0.2 (7.7 s of audio in 1.6 s on 16 threads, Threadripper 1950X).

### CUDA encoder

```
make -C speech cuda
speech/build/ja_align_cuda --cuda --model ja_align.safetensors --wav x.wav --out aux.json
```

- `ja_align/ja_cuda.h` keeps the module self-contained. It `dlopen`s `libcuda.so.1` and `libnvrtc.so` and declares only the ~25 driver/NVRTC entry points it uses, so it needs no CUDA headers and links only `-ldl`.
- The kernels are:
  - a tiled SGEMM whose A operand has a row stride, so strided convolutions run as a plain GEMM;
  - the grouped positional conv, as a gathered GEMM with one grid z-slice per group;
  - the CNN front end (conv0 plus per-channel GroupNorm + GELU);
  - LayerNorm, and bidirectional multi-head attention (online softmax, 4 warps per query).
- It plugs into `ja_align_run_ex` through a `ja_encoder_fn` hook. CTC decoding, prosody and visemes stay on the CPU.
- `tts_ja_cuda` uses it automatically with `--backend cuda`.

| Check | Result |
|---|---|
| every encoder stage and both heads' log-posteriors vs the CPU encoder (7.7 s utterance) | cos 1.0, log-posterior max abs 5e-5; decoded phonemes and kana identical |
| speed, 7.7 s of audio (RTX 5060 Ti vs 16 CPU threads) | 0.26 s vs 1.97 s (RTF 0.034 vs 0.26) |

**Edge padding.** wav2vec2 CTC misses speech that begins right at the start of the buffer, and TTS output starts speaking almost immediately. `ja_align` therefore pads 0.5 s of silence on both sides (`ja_align_opts.pad`) and drops the padded frames afterwards. `align_reference.py --pad 0.5` does the same, so end-to-end comparisons stay like-for-like.

## Intelligibility eval (Qwen3-TTS, Japanese)

```
python speech/ref/asr_eval.py --out-dir tmp/speech/eval1 --seeds 1 2
```

- **Test set:** 20 everyday sentences (`tests/ja_sentences.txt`) × 2 seeds, speaker Ono_Anna, default sampling.
- **Systems compared:** our C/CUDA runner vs the official `qwen_tts` package (BF16, CUDA).
- **Scoring:** both are transcribed by the hiragana-ctc model and scored with kana CER against pyopenjtalk pronunciation readings. pyopenjtalk is used only as an evaluation oracle.

| Recognizer input | C/CUDA runner | PyTorch qwen_tts |
|---|---|---|
| raw wav | 16.5% (12/40 above 20%) | 17.8% (17/40 above 20%) |
| wav padded with 0.5 s silence | **4.5%** (median 3.7%) | **4.5%** (median 3.6%) |

The C runner is exactly as intelligible as the reference implementation.

## Pitch-accent check

```
python speech/ref/accent_eval.py --eval-dir tmp/speech/eval1 --jsut jsut_ver1.1 --jsut-n 100 \
    --ja-align speech/build/ja_align_cuda --extra-args=--cuda
```

**Method**
1. pyopenjtalk full-context labels (used only as an evaluation oracle) give each utterance's accent phrases, with mora count *n* and accent type *k*.
2. `ja_align` force-aligns the labelled phonemes.
3. Each mora gets the median YIN F0 of its vowel, in semitones.

**What is scored, per accent phrase**
- **Accented phrases** (1 ≤ *k* < *n*): the largest F0 drop must fall right after the nucleus. One mora of lag is allowed, because the fall is realized late. The drop must be at least 1 semitone.
- **Unaccented phrases**: no drop of 1.5 semitones or more inside the phrase.
- **Chance**: the same F0 scored against accent types shuffled among phrases of the same length.

**Human reference.** The same pipeline was run on 99 JSUT recordings of read speech by a native speaker. They are used only as test data and are not redistributed.

| Speech | Nucleus placement | Chance | Above chance | Unaccented phrases flat |
|---|---|---|---|---|
| JSUT (human, 99 utterances) | 61.6% | 41.1% | +20.5 | 76.0% |
| Qwen3-TTS, C/CUDA runner (40) | 46.3% | 29.6% | +16.7 | 94.4% |
| Qwen3-TTS, PyTorch qwen_tts (40) | 39.6% | 25.7% | +13.9 | 95.8% |

**Findings**
- The metric is noisy even for human speech. Human speech is only 20 points above chance, and the likely causes are OpenJTalk accent-prediction errors and phrase-boundary and F0 noise.
- Measured against that ceiling, Qwen3-TTS (Ono_Anna) realizes about 70–80% of the accent-nucleus signal a human reader shows.
- Qwen3-TTS keeps unaccented phrases flatter than the human reader does, which suggests a compressed pitch range.
- The difference between the C runner and PyTorch is sampling variance: both run the same model, and greedy codes are identical.
- The phrase-initial rise was also measured. It is at chance even for the human reader, so it is left out as uninformative.
- One mispronounced accent seen by inspection: 二時間 rises where the dictionary puts a fall after じ. The remaining errors are mostly small-kana and long-vowel confusions (ひい for ひー, じぎょー for じゅぎょー), which the model card lists as its known weak spots.
