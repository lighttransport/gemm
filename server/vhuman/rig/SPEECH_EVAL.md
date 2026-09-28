# Japanese speech-motion evaluation preparation

## Why ReazonSpeech fits a pilot

[ReazonSpeech](https://research.reazon.jp/projects/ReazonSpeech/) supplies
natural Japanese broadcast speech across varied speaking styles. The
[dataset card](https://huggingface.co/datasets/reazon-research/reazonspeech)
describes 16 kHz FLAC clips with transcriptions; its `tiny` configuration is
about 600 MB / 8.5 hours. These are useful out-of-domain audio inputs for
`ja_align` and for judging how the rig responds to pauses, consonants and
pitch. The corpus has **no paired face animation or video labels**, so it
cannot yield a ground-truth facial-motion error. Its transcription also does
not provide phoneme timings; inspect the inferred alignment before scoring.

The corpus is gated separately from the Audio2Emotion model. Its publisher
labels the corpus CDLA-Sharing-1.0 and limits use to information analysis
under Japanese Copyright Act Article 30-4. Keep downloaded clips and rendered
reviews local, and use the current dataset terms when obtaining access. The
library/model Apache-2.0 license is distinct from the corpus terms. The
publisher's dataset script uses `trust_remote_code=True`; the
[dataset card](https://huggingface.co/datasets/reazon-research/reazonspeech/blob/main/README.md)
shows this loading path. Use `datasets<4` for that script (there is an
[upstream compatibility issue](https://github.com/reazon-research/ReazonSpeech/issues/64)
with `datasets>=4`).

## Make a reproducible pilot set

After accepting the **corpus** access conditions, authenticate Hugging Face
with a cache inside the repository's `tmp/` directory:

```sh
python3 -m venv tmp/reazon-env
tmp/reazon-env/bin/pip install 'datasets>=2,<4' soundfile
HF_HOME="$PWD/tmp/hf-reazon" tmp/reazon-env/bin/hf auth login
HF_HOME="$PWD/tmp/hf-reazon" tmp/reazon-env/bin/python \
  speech/ref/prepare_reazonspeech_eval.py --out tmp/reazon-motion-pilot --seed 7
```

The script hash-ranks candidates by original clip name, selects six 2–4 s
and six 4–8 s clips, rejects silence/clipping and repeated transcriptions,
and writes PCM WAVs, `manifest.json` (including the dataset fingerprint),
and a blank `review.csv`. Selection is deterministic for the same dataset
revision. Listen to all selected clips and log exclusions for overlapping
speakers, background music, truncated words or erroneous transcripts.
Do not treat a clean, small pilot as a population score.

For each accepted WAV, use the same head and settings:

```sh
python3 -m server.vhuman.cli --work tmp/vhuman-independent rig-speech \
  --head <head-id> --wav tmp/reazon-motion-pilot/clips/rs_00.wav --backend cpu
```

Record the returned take ID in `review.csv`. First review the `align.json`
phone intervals and audio together; discard or mark failed alignments before
judging animation. Open `/rig`, load the take, and play/seek with audio.
Give each of lip sync, consonant closures, prosody, naturalness and secondary
motion a 1–5 rating (1 = distracting, 3 = usable, 5 = convincing). Review at
least two playback passes per clip, and have two people rate independently if
the result will be used to compare changes. Note whether a low rating comes
from ASR/CTC alignment, phoneme-to-viseme mapping, rig deformation, or motion
timing. Keep the head and rendering conditions fixed.

Check `m/p/b` closure versus `n/N` tongue contact, onset and release around
pauses, vowel aperture changes across loud/quiet syllables, blink timing,
and whether gaze/head movement distracts. Do a second pass with
`--secondary-strength 0` on the same audio/align via `--source-take` to judge
secondary motion separately. The CSV is a human review template, not an
automatic realism score. ReazonSpeech has no emotion labels, so evaluate
audio-emotion suggestions separately and do not score them as ground truth.
