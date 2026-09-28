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
