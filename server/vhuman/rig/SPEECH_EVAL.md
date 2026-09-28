# Japanese speech-motion evaluation preparation

## Why ReazonSpeech fits a pilot

[ReazonSpeech](https://research.reazon.jp/projects/ReazonSpeech/) supplies
natural Japanese broadcast speech across varied speaking styles. The
[dataset card](https://huggingface.co/datasets/reazon-research/reazonspeech)
describes 16 kHz FLAC clips with transcriptions; its `tiny` configuration is
about 600 MB / 8.5 hours. These are useful natural audio inputs for judging
how the rig responds to pauses, consonants and pitch. The `ja_align` encoder
was pretrained on ReazonSpeech, so these clips are **not** an independent
aligner generalization benchmark. The corpus has **no paired face animation
or video labels**, so it cannot yield a ground-truth facial-motion error. Its
transcription also does not provide phoneme timings; inspect the inferred
alignment before scoring.

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

After accepting the **corpus** access conditions, use the publisher's `tiny`
index and audio archive (the same URLs used by its dataset loader). Keep the
downloads under the repository's `tmp/` directory:

```sh
mkdir -p tmp/reazon-source
curl -fL -o tmp/reazon-source/tiny.tsv \
  https://corpus.reazon-research.org/reazonspeech-v2/tsv/tiny.tsv
curl -fL -C - -o tmp/reazon-source/000.tar \
  https://corpus.reazon-research.org/reazonspeech-v2/data/000.tar
python3 speech/ref/prepare_reazonspeech_eval.py \
  --tsv tmp/reazon-source/tiny.tsv --tar tmp/reazon-source/000.tar \
  --out tmp/reazon-motion-pilot --seed 7
```

This route uses the installed FFmpeg to decode only the top 256 hash-ranked
candidate names while scanning the archive. The `datasets` loader remains
available when preferred:

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
  --head <head-id> --wav tmp/reazon-motion-pilot/clips/rs_00.wav \
  --transcript '<manifest transcription>' --backend cpu
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

Run the take diagnostics on any saved pilot take:

```sh
python3 speech/ref/eval_rig_motion.py \
  tmp/vhuman-independent/heads/<head-id>/rig/takes/<take-id> \
  --out tmp/vhuman/motion-diagnostics.json
```

The report checks aligned-phone confidence, bilabial closure coverage,
residual jaw/lip press at those closures, nonlabial nasal behavior,
interior-silence jaw motion, blink/gaze/head ranges,
neutral ending and JSON/LightRig face-track agreement. These checks find
pipeline failures; they do not replace listening or visual review. High energy
outside aligned speech can indicate background audio or missed phones, so it
is a review flag rather than proof of an alignment error. A known kana reading
can be supplied to `rig-speech --wav ... --kana ...` for forced alignment.

### Local ReazonSpeech pilot (2026-09-29)

Using the publisher's `tiny` archive with seed 7, we aligned 24 hash-ranked
eligible clips of 2–4 seconds and 24 of 4–8 seconds. The diagnostics marked
13 short and 14 long clips with no warnings. Across all 48 takes, all 72
aligned bilabial closures met the 0.7 target, and JSON/LightRig face tracks
agreed exactly. A rendered /m/ frame initially showed a thin teeth strip even
with a 0.8 closure control; reducing jaw opening and lip press to zero at the
peak removed it. All 72 peaks now have zero residual jaw and press. The first
six in each bucket form a 12-clip review queue. All 48 alignments, warnings, take
IDs and source names remain in `tmp/reazon-motion-screen/report.json`; the
queue and blank rating sheet are `shortlist.md` and `shortlist.csv` there.
To open its take links, run:

```sh
python3 -m server.vhuman.app --work tmp/vhuman-independent
```

The viewer accepts `?head=<id>&take=<id>` to open a saved take, with optional
`&at=<seconds>` to seek to a specific frame. It plays the saved audio and
animation together. Machine screening is only triage: a reviewer must listen
for overlap, music, transcript errors and unnatural facial motion before
assigning the 1–5 ratings. The 48 candidates and 12-clip queue are local
research artifacts; do not redistribute corpus audio with the code.

### Local JSUT pilot (2026-09-29)

The workspace already has [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/corpus)
BASIC5000 recordings. We ran 12 clips through the same `--wav` path on one
rigged head (575 aligned phones). The first pass found 26/29 bilabial
intervals with `mouthClose >= 0.7`; the three misses were 20 ms phones falling
between 30 fps frames. After nearest-frame closure snapping, all 29/29 passed,
with zero diagnostic warnings and exact JSON/LightRig face-track agreement.
The local reports and blank review sheet are in
`tmp/vhuman/jsut_motion_pilot/`. JSUT is controlled read speech by one speaker,
so this result checks timing and export behavior; it does not establish
naturalness on broadcast speech or perceptual facial quality. Human ratings
have not been filled in.
