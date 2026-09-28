"""Optional SenseVoiceSmall GGUF audio-to-emotion extraction for speech takes.

The model emits one categorical tag per utterance. Short overlapping windows
give the rig a coarse timeline; these labels have no calibrated confidence.
"""
from __future__ import annotations

import hashlib
import math
import re
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np

from .. import gpu
from ..service import ROOT

DEFAULT_RUNNER = ROOT / "tmp/vhuman-emotion/runtime/llama-funasr-sensevoice"
DEFAULT_MODEL = ROOT / "tmp/vhuman-emotion/sensevoice-small-q8.gguf"
MODEL_NAME = "FunAudioLLM/SenseVoiceSmall-GGUF"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 4.0
HOP_SECONDS = 2.0
TAG_LABELS = {
    "HAPPY": "joy", "SAD": "sadness", "ANGRY": "anger",
    "DISGUSTED": "disgust", "SURPRISED": "surprise", "FEARFUL": "fear",
    "NEUTRAL": "neutral", "EMO_UNKNOWN": "neutral",
}


def availability(runner=DEFAULT_RUNNER, model=DEFAULT_MODEL) -> dict:
    runner, model = Path(runner), Path(model)
    missing = [str(path) for path, ok in ((runner, runner.is_file() and path_executable(runner)),
                                         (model, model.is_file())) if not ok]
    return {"available": not missing, "model": MODEL_NAME, "missing": missing}


def path_executable(path: Path) -> bool:
    return bool(path.stat().st_mode & 0o111)


def _read_mono_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        count = wav.getnframes()
        if wav.getcomptype() != "NONE" or width != 2 or not 1 <= channels <= 2 or rate <= 0:
            raise ValueError("emotion extraction needs mono/stereo PCM16 WAV")
        if count <= 0 or count / rate > 600:
            raise ValueError("emotion extraction needs 0–600 seconds of audio")
        values = np.frombuffer(wav.readframes(count), dtype="<i2").astype(np.float64)
    return values.reshape(-1, channels).mean(axis=1) / 32768.0, rate


def _resample_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE:
        return samples
    if rate > SAMPLE_RATE:
        # Windowed-sinc antialiasing before linear interpolation. The few dozen
        # taps are enough for the 24 kHz TTS WAVs and avoid a new DSP dependency.
        taps = np.arange(-48, 49)
        cutoff = .46 * SAMPLE_RATE / rate
        kernel = 2 * cutoff * np.sinc(2 * cutoff * taps) * np.hamming(len(taps))
        samples = np.convolve(samples, kernel / kernel.sum(), mode="same")
    count = max(1, round(len(samples) * SAMPLE_RATE / rate))
    return np.interp(np.arange(count) * rate / SAMPLE_RATE, np.arange(len(samples)), samples)


def _windows(duration: float) -> list[tuple[float, float]]:
    if not math.isfinite(duration) or duration <= 0 or duration > 600:
        raise ValueError("invalid emotion audio duration")
    starts = [0.0]
    while starts[-1] + WINDOW_SECONDS < duration:
        nxt = min(starts[-1] + HOP_SECONDS, duration - WINDOW_SECONDS)
        if nxt - starts[-1] < .05:
            break
        starts.append(nxt)
    return [(start, min(duration, start + WINDOW_SECONDS)) for start in starts]


def parse_tag(output: str) -> tuple[str, str]:
    tags = re.findall(r"<\|([A-Z_]+)\|>", output)
    for tag in tags:
        if tag in TAG_LABELS:
            return tag, TAG_LABELS[tag]
    raise ValueError("SenseVoice returned no emotion tag")


def _keys(windows: list[dict], duration: float) -> list[dict]:
    def weights(label):
        return {} if label == "neutral" else {label: 1.0}

    keys = [{"t": 0.0, "weights": weights(windows[0]["label"])}]
    for before, after in zip(windows, windows[1:]):
        if before["label"] == after["label"]:
            continue
        boundary = (before["center"] + after["center"]) / 2
        left = max(keys[-1]["t"] + .001, boundary - .2)
        right = min(duration, boundary + .2)
        if left < right:
            keys.append({"t": round(left, 3), "weights": weights(before["label"])})
            keys.append({"t": round(right, 3), "weights": weights(after["label"])})
    return keys


def _infer(runner: Path, model: Path, wav: Path, cancel: threading.Event) -> tuple[str, str]:
    proc = subprocess.Popen([str(runner), "-m", str(model), "-a", str(wav), "--keep-tags"],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stop = threading.Event()

    def watch():
        while not stop.wait(.2):
            if cancel.is_set() and proc.poll() is None:
                proc.terminate()

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        output, _ = proc.communicate()
    finally:
        stop.set()
        watcher.join(timeout=1)
    if cancel.is_set():
        raise gpu.Cancelled("cancelled")
    if proc.returncode:
        raise RuntimeError("SenseVoice failed: " + output[-800:])
    return parse_tag(output)


def extract(wav: Path, duration: float, work: Path, cancel: threading.Event, progress,
            *, runner=DEFAULT_RUNNER, model=DEFAULT_MODEL) -> dict:
    """Infer coarse categorical emotion keys from an existing aligned WAV."""
    wav, work = Path(wav).resolve(), Path(work).resolve()
    runner, model = Path(runner).resolve(), Path(model).resolve()
    status = availability(runner, model)
    if not status["available"]:
        raise ValueError("SenseVoice runtime/model missing: " + ", ".join(status["missing"]))
    samples, rate = _read_mono_pcm16(wav)
    samples = _resample_16k(samples, rate)
    actual_duration = len(samples) / SAMPLE_RATE
    if abs(actual_duration - duration) > .1:
        raise ValueError("emotion WAV and alignment durations disagree")
    spans = _windows(duration)
    windows = []
    for i, (start, end) in enumerate(spans):
        if cancel.is_set():
            raise gpu.Cancelled("cancelled")
        clip = samples[round(start * SAMPLE_RATE):round(end * SAMPLE_RATE)]
        if len(clip) < SAMPLE_RATE // 4:
            raise ValueError("emotion clip is too short")
        if float(np.sqrt(np.mean(clip * clip))) < .003:
            tag, label = "SILENCE", "neutral"
        else:
            path = work / f"emotion-window-{i}.wav"
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(SAMPLE_RATE)
                out.writeframes(np.clip(np.rint(clip * 32768), -32768, 32767).astype("<i2").tobytes())
            try:
                tag, label = _infer(runner, model, path, cancel)
            finally:
                path.unlink(missing_ok=True)
        windows.append({"start": round(start, 3), "end": round(end, 3),
                        "center": round((start + end) / 2, 3), "tag": tag, "label": label})
        progress(.72 + .07 * (i + 1) / len(spans), f"analyzed emotion {i + 1}/{len(spans)}")
    with model.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    return {"format": "vhuman.emotion.v1", "provider": "SenseVoiceSmall", "model": MODEL_NAME,
            "model_author": "Alibaba Group", "model_url": "https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF",
            "model_sha256": digest,
            "language": "ja", "window_seconds": WINDOW_SECONDS, "windows": windows,
            "emotion_keyframes": _keys(windows, duration)}
