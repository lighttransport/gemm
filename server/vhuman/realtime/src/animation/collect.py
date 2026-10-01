"""Capture complete native-TTS utterances and an offline Japanese CTC teacher.

CTC is used only for corpus supervision; the live student consumes direct TTS
features. Silent or motionless takes fail before entering the training manifest.
"""
import json
from pathlib import Path
import queue
import struct
import subprocess
import time
import wave
import numpy as np
from .corpus import capture
from ..avatar.provenance import sha256
from ..avatar.rig import RigAvatar
from ..tts.resident import ResidentTTS
from ....rig.speech import build_frames


def collect(rig_path, model, runner, aligner, align_model, sentences, output, work, threads=4, text_feed="incremental"):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(sentences).read_text())
    if not isinstance(items, list) or len(items) < 2: raise ValueError("sentence list required")
    if {item["split"] for item in items} != {"train", "validation"}: raise ValueError("train/validation sentences required")
    if len({item["text"] for item in items}) != len(items): raise ValueError("duplicate sentences across corpus")
    rig = RigAvatar(rig_path, Path(work) / "native")
    try: names, ranges = list(rig.names), rig.ranges.tolist()
    finally: rig.close()
    revision = sha256(Path(model) / "model.safetensors")
    worker = ResidentTTS(runner, model, revision, output / "worker", max_frames=256, threads=threads, text_feed=text_feed)
    receipts, takes = [], []
    try:
        for epoch, item in enumerate(items):
            stem = f"take-{epoch:03d}"
            worker.submit(item["text"], epoch)
            audio, features = [], []
            deadline = time.monotonic() + 60
            while True:
                worker.check()
                for destination, result in ((worker.audio, audio), (worker.features, features)):
                    while True:
                        try: result.append(destination.get_nowait())
                        except queue.Empty: break
                if worker.done_audio.is_set() and worker.done_features.is_set() and worker.audio.empty() and worker.features.empty(): break
                if time.monotonic() > deadline: raise TimeoutError("corpus TTS request exceeded60s")
                time.sleep(.001)
            if not audio or len(audio) != len(features) or len(audio) >= 255:
                raise ValueError("empty, truncated or mismatched corpus utterance")
            pcm = np.concatenate([frame.pcm for frame in audio])
            rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
            if rms < .003: raise ValueError(f"{stem}: nearly silent audio, RMS={rms}")
            wav_path, feature_path = output / (stem+".wav"), output / (stem+".features")
            with wave.open(str(wav_path), "wb") as wav:
                wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                wav.writeframes((np.clip(pcm, -1, 1)*32767).round().astype("<i2").tobytes())
            with feature_path.open("wb") as stream:
                stream.write(b"VHFEAT1\0" + struct.pack("<i", len(features[0].hidden)))
                for index, frame in enumerate(features):
                    if frame.sample_start != index*1920 or audio[index].sample_start != frame.sample_start:
                        raise ValueError("PCM/features timestamp mismatch")
                    stream.write(struct.pack("<q", frame.sample_start)+frame.codes.astype("<i4").tobytes()+frame.hidden.astype("<f4").tobytes())
            aux_path = output / (stem+".align.json")
            command = [str(aligner), "--model", str(align_model), "--wav", str(wav_path), "--cuda", "--out", str(aux_path)]
            # Greedy phone recognition is inspectable; supplied readings can be
            # used for forced timing, but neither is treated as ground truth.
            if item.get("kana"): command += ["--kana", item["kana"]]
            with (output / (stem+".align.log")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=log, check=True, timeout=60)
            aux = json.loads(aux_path.read_text())
            if not aux.get("phones"): raise ValueError("teacher recognized no phonemes")
            frames = build_frames(aux, secondary_strength=0)
            activity = max(f["v"].get("jawOpen", 0) for f in frames)
            if activity < .1: raise ValueError("teacher motion is nearly neutral")
            performance = output / (stem+".performance.json")
            performance.write_text(json.dumps(dict(format="vhuman.performance.v1", controls=names, frames=frames)))
            target = output / (stem+".npz")
            take = capture(feature_path, performance, target, revision, names, ranges)
            take.update(split=item["split"], text=item["text"], rms=rms, jaw_max=activity,
                        audio_sha256=sha256(wav_path), alignment_sha256=sha256(aux_path))
            takes.append(take)
            receipts.append(dict(path=target.name, sha256=sha256(target), source="original complete Qwen utterance and Apache Japanese CTC/original viseme teacher",
                revision=revision, license="Apache-2.0", roles=["motion-training"]))
            print(json.dumps(dict(captured=stem, frames=len(audio), rms=rms, jaw_max=activity)), flush=True)
    finally: worker.close()
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(dict(format="vhuman.motion_corpus.v1", purpose="diagnostic", tts_revision=revision, text_feed=text_feed,
        names=names, ranges=ranges, provenance=receipts, takes=takes,
        aligner_weights_sha256=sha256(align_model), sentences_sha256=sha256(sentences),
        teacher_source_sha256=sha256(Path(__file__).parents[3] / "rig/speech.py"),
        limitations="Offline CTC/viseme supervision; phonetic alignment and avatar quality require independent validation"), indent=2))
    return dict(manifest=str(manifest), takes=len(takes))
