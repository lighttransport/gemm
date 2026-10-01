"""Offline speech takes: Qwen3-TTS or WAV -> ja_align -> facial rig controls.

The optional emotion_provider argument accepts a callable returning named
emotion weights at a timestamp. No model is loaded by this module.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import random
import shutil
import subprocess
import threading
import uuid
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .. import gpu
from ..service import ROOT
from . import rigdef, usd

SPEECH = ROOT / "speech"
DEFAULT_MODEL = Path("/mnt/disk1/models/speech/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEFAULT_ALIGNER = Path("/mnt/disk1/models/speech/japanese-wav2vec2-large-hiragana-ctc/ja_align.safetensors")
FPS = 30
VISEMES = ("sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou")
TAKE_FILES = ("manifest.json", "audio.wav", "align.json", "animation.json", "animation.usda", "lightrig.txt",
              "emotion.json", "soft_tissue.usda", "soft_tissue_report.json", "fit_report.json")


def availability(model=DEFAULT_MODEL, aligner=DEFAULT_ALIGNER, backend="auto") -> dict:
    selected = gpu.backend() if backend == "auto" else backend
    runner = SPEECH / "build" / ("tts_ja" if selected == "cpu" else f"tts_ja_{selected}")
    missing = [str(path) for path, ok in ((runner, runner.is_file()),
                                         (Path(model), Path(model).is_dir()),
                                         (Path(aligner), Path(aligner).is_file())) if not ok]
    return {"available": not missing, "backend": selected, "missing": missing}

# The vowel/closed-lip poses match the rig viewer's A/E/I/O/U/M presets.
A = {"jawOpen": .55, "mouthStretchLeft": .15, "mouthStretchRight": .15}
E = {"jawOpen": .25, "mouthStretchLeft": .5, "mouthStretchRight": .5,
     "mouthSmileLeft": .2, "mouthSmileRight": .2}
I = {"jawOpen": .12, "mouthSmileLeft": .45, "mouthSmileRight": .45,
     "mouthStretchLeft": .3, "mouthStretchRight": .3}
O = {"jawOpen": .45, "mouthFunnel": .7, "mouthPucker": .3}
U = {"jawOpen": .12, "mouthPucker": .9, "mouthFunnel": .35}
M = {"mouthPressLeft": .6, "mouthPressRight": .6, "mouthRollLower": .3, "mouthRollUpper": .2}


def _mix(*parts: tuple[dict, float]) -> dict:
    out = {}
    for pose, weight in parts:
        for name, value in pose.items():
            out[name] = out.get(name, 0.0) + weight * value
    return out


VISEME_POSES = {
    "sil": {}, "PP": M, "FF": _mix((M, .4), (E, .2)),
    "TH": dict(_mix((E, .3)), tongueOut=.2),
    "DD": dict(_mix((E, .3)), tongueUp=.2),
    "kk": _mix((A, .4)), "CH": _mix((E, .25), (O, .2)),
    "SS": _mix((I, .25)), "nn": {"tongueUp": .3},
    "RR": dict(_mix((E, .3)), tongueCurlUp=.2),
    "aa": A, "E": E, "ih": I, "oh": O, "ou": U,
}
EMOTION_POSES = {
    "neutral": {},
    "joy": {"mouthSmileLeft": 1, "mouthSmileRight": 1, "cheekSquintLeft": .5,
            "cheekSquintRight": .5, "eyeSquintLeft": .25, "eyeSquintRight": .25},
    "sadness": {"browInnerUp": .8, "mouthFrownLeft": .9, "mouthFrownRight": .9,
                "eyeLookDownLeft": .2, "eyeLookDownRight": .2},
    "anger": {"browDownLeft": 1, "browDownRight": 1, "eyeSquintLeft": .5,
              "eyeSquintRight": .5, "mouthPressLeft": .6, "mouthPressRight": .6},
    "disgust": {"noseSneerLeft": 1, "noseSneerRight": 1, "mouthUpperUpLeft": .7,
                "mouthUpperUpRight": .7, "browDownLeft": .4, "browDownRight": .4},
    "surprise": {"browInnerUp": 1, "browOuterUpLeft": .8, "browOuterUpRight": .8,
                 "eyeWideLeft": .9, "eyeWideRight": .9, "jawOpen": .4},
    "fear": {"browInnerUp": .8, "browOuterUpLeft": .5, "browOuterUpRight": .5,
             "eyeWideLeft": .7, "eyeWideRight": .7, "mouthFrownLeft": .35,
             "mouthFrownRight": .35},
}
EMOTION_MOUTH = {"mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
                 "mouthDimpleLeft", "mouthDimpleRight"}


def _unit(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number in [0, 1]") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be a number in [0, 1]")
    return number


def validate_emotions(keys, duration: float) -> list[dict]:
    if keys is None:
        return []
    if not isinstance(keys, list) or len(keys) > 256:
        raise ValueError("emotion_keyframes must be a list of at most 256 keyframes")
    result = []
    prev = -1.0
    for key in keys:
        if not isinstance(key, dict) or not isinstance(key.get("weights"), dict):
            raise ValueError("emotion keyframe needs t and weights")
        try:
            t = float(key["t"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("emotion keyframe needs a finite time") from exc
        if not math.isfinite(t) or not prev < t <= duration or t < 0:
            raise ValueError("emotion keyframes must have increasing times within the audio")
        weights = {}
        for name, weight in key["weights"].items():
            if name not in EMOTION_POSES:
                raise ValueError(f"unknown emotion: {name}")
            weights[name] = _unit(weight, f"emotion {name}")
        result.append({"t": t, "weights": weights})
        prev = t
    return result


def _emotion_at(keys: list[dict], t: float) -> dict:
    if not keys:
        return {}
    if t <= keys[0]["t"]:
        return keys[0]["weights"]
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        if t <= b["t"]:
            u = (t - a["t"]) / (b["t"] - a["t"])
            return {n: (1 - u) * a["weights"].get(n, 0) + u * b["weights"].get(n, 0)
                    for n in set(a["weights"]) | set(b["weights"])}
    return keys[-1]["weights"]


def _prosody(aux: dict, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample the aligner's 100 Hz energy and voiced pitch on the rig timeline."""
    data = aux.get("prosody") or {}
    hop = float(data.get("hop", 0))
    rms = np.asarray(data.get("rms_db", []), dtype=np.float64)
    f0 = np.asarray(data.get("f0_hz", []), dtype=np.float64)
    if hop <= 0 or not math.isfinite(hop) or len(rms) < 2 or len(f0) != len(rms):
        return np.ones(len(times)), np.zeros(len(times))
    if not np.isfinite(rms).all() or not np.isfinite(f0).all():
        raise ValueError("invalid prosody values")
    peak = float(np.percentile(rms, 95))
    floor = peak - 35.0
    energy = np.clip((np.interp(times, np.arange(len(rms)) * hop, rms) - floor) / 35.0, 0, 1)
    voiced = f0[(f0 > 60) & (f0 < 600) & (rms > floor)]
    pitch = np.zeros(len(times))
    if len(voiced):
        sampled = np.interp(times, np.arange(len(f0)) * hop, f0)
        mask = sampled > 60
        pitch[mask] = np.clip(np.log2(sampled[mask] / np.percentile(voiced, 25)) / .5, -1, 1)
    return energy, pitch


def _phone_at(aux: dict, t: float, frame_index: int, fps: float) -> tuple[str, float]:
    intervals = aux.get("intervals") or ()
    best_phone, best_score = "", (0.0, -1)
    bilabial = {"p", "py", "b", "by", "m", "my"}
    lingual = {"n", "ny", "N", "t", "ty", "d", "dy"}
    for i, interval in enumerate(intervals):
        start, end = float(interval["start"]), float(interval["end"])
        if end <= start or t < start - .025 or t > end + .03:
            continue
        phone = str(interval["s"])
        following = str(intervals[i + 1]["s"]) if i + 1 < len(intervals) else ""
        if phone == "cl":
            phone = following  # Japanese geminate anticipates the next closure.
        elif phone == "N" and following in bilabial:
            phone = "m"  # Moraic nasal assimilates before a bilabial.
        if phone not in bilabial | lingual:
            continue
        # A 20 ms phone can otherwise fall between 30 fps samples. Hold its
        # center at full closure and allow a short anticipatory/release ramp.
        strength = ((t - start + .025) / .025 if t < start else
                    (end + .03 - t) / .03 if t > end else 1.0)
        if frame_index == int(math.floor((start + end) * .5 * fps + .5)):
            strength = 1.0
        score = (max(0.0, min(1.0, strength)), int(phone in bilabial))
        if score > best_score:
            best_phone, best_score = phone, score
    return best_phone, best_score[0]


def _secondary(times: np.ndarray, duration: float, seed: int) -> list[dict]:
    """Repeatable blink and gaze tracks, with a small head follow-through."""
    rng = random.Random(seed)
    blinks = []
    t = .8 + rng.uniform(0, .45)
    while t < duration - .22:
        blinks.append(t)
        t += rng.uniform(2.8, 4.5)
    targets = [(0.0, 0.0, 0.0)]
    t = .55 + rng.uniform(0, .3)
    while t < duration - .4:
        targets.append((t, rng.uniform(-.13, .13), rng.uniform(-.07, .07)))
        t += rng.uniform(2.0, 3.2)
    result = []
    for now in times:
        blink = 0.0
        for start in blinks:
            phase = now - start
            if 0 <= phase < .075:
                blink = max(blink, phase / .075)
            elif .075 <= phase < .205:
                blink = max(blink, 1 - (phase - .075) / .13)
        previous, target = targets[0], targets[0]
        for point in targets[1:]:
            if point[0] <= now:
                previous, target = target, point
            else:
                break
        blend = np.clip((now - target[0]) / .2, 0, 1)
        blend = blend * blend * (3 - 2 * blend)
        fade = min(1.0, now / .35, max(0.0, (duration - now) / .35))
        gx = float((previous[1] * (1 - blend) + target[1] * blend) * fade)
        gy = float((previous[2] * (1 - blend) + target[2] * blend) * fade)
        result.append({"blink": blink, "gx": gx, "gy": gy})
    return result


def build_frames(aux: dict, emotion_keyframes=None, speech_strength: float = 1.0,
                 emotion_strength: float = .6, emotion_provider=None,
                 secondary_seed: int = 7, secondary_strength: float = 1.0) -> list[dict]:
    """Produce finite, clamped 60-control frames at the aligner's sample times."""
    if aux.get("format") != "ja_align.v1":
        raise ValueError("expected ja_align.v1")
    vis = aux.get("visemes") or {}
    fps = float(vis.get("fps", 0))
    duration = float(aux.get("duration", 0))
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid alignment timing")
    if tuple(vis.get("names", ())) != VISEMES:
        raise ValueError("unknown viseme order")
    values = np.asarray(vis.get("frames"), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(VISEMES) or len(values) != math.ceil(duration * fps) + 1:
        raise ValueError("invalid viseme frame shape")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1.001).any():
        raise ValueError("invalid viseme weights")
    speech_strength = _unit(speech_strength, "speech_strength")
    emotion_strength = _unit(emotion_strength, "emotion_strength")
    secondary_strength = _unit(secondary_strength, "secondary_strength")
    if type(secondary_seed) is not int or not 0 <= secondary_seed < 2**64:
        raise ValueError("secondary_seed must be a nonnegative integer")
    keys = validate_emotions(emotion_keyframes, duration)
    names = rigdef.CONTROLS
    matrix = np.array([[VISEME_POSES[v].get(n, 0) for n in names] for v in VISEMES], np.float64)
    speech = values @ matrix
    # Symmetric three-frame filter adds no offset to the offline timeline.
    speech = np.pad(speech, ((1, 1), (0, 0)), mode="edge")
    speech = .25 * speech[:-2] + .5 * speech[1:-1] + .25 * speech[2:]
    times = np.minimum(np.arange(len(speech)) / fps, duration)
    energy, pitch = _prosody(aux, times)
    secondary = _secondary(times, duration, secondary_seed)
    index = {name: i for i, name in enumerate(names)}
    frames = []
    for i, row in enumerate(speech):
        t = float(times[i])
        out = np.clip(row * speech_strength, 0, 1)
        # RMS controls vowel aperture; pitch accent adds a subtle brow lift.
        out[index["jawOpen"]] *= .78 + .22 * energy[i]
        out[index["browInnerUp"]] += .045 * max(0.0, pitch[i]) * energy[i] * speech_strength
        phone, closure = _phone_at(aux, t, i, fps)
        if phone in ("p", "py", "b", "by", "m", "my"):
            # On this rig even a tiny residual jaw/press opens a visible teeth
            # slit. The rest lips already meet, so seal these at the peak.
            seal = min(1.0, closure / .7)
            out[index["jawOpen"]] *= 1 - seal
            out[index["mouthClose"]] += .8 * closure * speech_strength
            for side in ("mouthPressLeft", "mouthPressRight"):
                out[index[side]] *= 1 - seal
            if phone in ("m", "my"):
                out[index["tongueUp"]] *= 1 - closure
        elif phone in ("n", "ny", "N", "t", "ty", "d", "dy"):
            out[index["tongueUp"]] = max(out[index["tongueUp"]], .4 * closure * speech_strength)
        weights = _emotion_at(keys, t)
        if emotion_provider is not None:
            provided = emotion_provider(t)
            if not isinstance(provided, dict):
                raise ValueError("emotion provider must return a mapping")
            weights = {**weights, **provided}
        for emotion, weight in weights.items():
            if emotion not in EMOTION_POSES:
                raise ValueError(f"unknown emotion: {emotion}")
            weight = _unit(weight, f"emotion {emotion}") * emotion_strength
            for name, value in EMOTION_POSES[emotion].items():
                if name in EMOTION_MOUTH:
                    out[index[name]] += .35 * weight * value
                elif not name.startswith("mouth") and not name.startswith("jaw") and not name.startswith("tongue"):
                    out[index[name]] += weight * value
        motion = secondary[i]
        blink = motion["blink"] * secondary_strength
        for side in ("eyeBlinkLeft", "eyeBlinkRight"):
            out[index[side]] = max(out[index[side]], blink)
        gx, gy = motion["gx"] * secondary_strength, motion["gy"] * secondary_strength
        for name, value in (("eyeLookOutLeft", max(0, gx)), ("eyeLookInRight", max(0, gx)),
                            ("eyeLookInLeft", max(0, -gx)), ("eyeLookOutRight", max(0, -gx)),
                            ("eyeLookUpLeft", max(0, gy)), ("eyeLookUpRight", max(0, gy)),
                            ("eyeLookDownLeft", max(0, -gy)), ("eyeLookDownRight", max(0, -gy))):
            out[index[name]] += value
        out[index["headYaw"]] = .23 * gx
        out[index["headPitch"]] = .16 * gy + .025 * pitch[i] * energy[i] * speech_strength
        for name in rigdef.SIGNED:
            out[index[name]] = np.clip(out[index[name]], -1, 1)
        for name in names:
            if name not in rigdef.SIGNED:
                out[index[name]] = np.clip(out[index[name]], 0, 1)
        if i == len(speech) - 1:
            out[:] = 0
        frames.append({"t": round(t, 6), "v": {n: round(float(v), 5) for n, v in zip(names, out) if abs(v) >= 1e-5}})
    return frames


def write_lightrig(frames: list[dict], path: Path) -> None:
    with path.open("w") as fh:
        for frame in frames:
            fields = [frame["t"], 0.0] + [frame["v"].get(name, 0.0) for name in rigdef.LR_FACE_V1]
            fh.write(" ".join(f"{v:.6f}" for v in fields) + "\n")


def _run(cmd: list[str], cancel, progress) -> None:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", str(min(16, os.cpu_count() or 16)))
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    stop = threading.Event()

    def watch():
        while not stop.wait(.2):
            if cancel.is_set() and proc.poll() is None:
                proc.terminate()

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    tail = []
    try:
        for line in proc.stdout:
            tail = (tail + [line.strip()])[-8:]
        rc = proc.wait()
    finally:
        stop.set()
        watcher.join(timeout=1)
        proc.stdout.close()
    if cancel.is_set():
        raise gpu.Cancelled("cancelled")
    if rc:
        raise RuntimeError("speech command failed: " + " | ".join(tail[-4:]))
    progress(.7, "speech aligned")


def speech_job(service, request: dict, progress, cancel, *, model=DEFAULT_MODEL, aligner=DEFAULT_ALIGNER,
               backend="auto", allow_wav=False, emotion_runner=None, emotion_model=None) -> dict:
    """Generate a take; model paths come from server options, never HTTP JSON."""
    head = request.get("head_id")
    rig_path = service.rig_file(head, "rig.json")
    text = request.get("text")
    wav = request.get("wav")
    source_take = request.get("source_take")
    if sum(bool(x) for x in (text, wav, source_take)) != 1:
        raise ValueError("provide exactly one of text, wav or source_take")
    if wav and not allow_wav:
        raise ValueError("WAV paths are available only through the CLI")
    if text and (not isinstance(text, str) or len(text) > 1000 or "\0" in text):
        raise ValueError("text must be at most 1000 characters")
    for key in ("kana", "transcript"):
        value = request.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > 1000 or "\0" in value):
            raise ValueError(f"{key} must be at most 1000 characters")
    if request.get("ref_wav") and not allow_wav:
        raise ValueError("reference WAV paths are available only through the CLI")
    seed = request.get("seed") if request.get("seed") is not None else 7
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be a nonnegative integer")
    auto_emotion = request.get("auto_emotion", False)
    if type(auto_emotion) is not bool:
        raise ValueError("auto_emotion must be a boolean")
    if auto_emotion:
        from . import emotion
        emotion_runner = emotion_runner or emotion.DEFAULT_RUNNER
        emotion_model = emotion_model or emotion.DEFAULT_MODEL
        if not emotion.availability(emotion_runner, emotion_model)["available"]:
            raise ValueError("SenseVoice runtime or model missing; see server/vhuman/rig/README.md")
    selected = gpu.backend() if backend == "auto" else backend
    if selected not in ("cpu", "cuda", "rocm"):
        raise ValueError("backend must be auto, cpu, cuda or rocm")
    out_root = rig_path.parent / "takes"
    out_root.mkdir(exist_ok=True)
    take_id = uuid.uuid4().hex[:12]
    stage = out_root / (".partial-" + take_id)
    stage.mkdir()
    final = out_root / take_id
    try:
        progress(.03, "preparing speech")
        source_meta = {}
        if source_take:
            src = service.take_file(head, source_take, "audio.wav").parent
            source_meta = json.loads((src / "manifest.json").read_text())
            shutil.copyfile(src / "audio.wav", stage / "audio.wav")
            shutil.copyfile(src / "align.json", stage / "align.json")
        elif wav:
            src = Path(wav)
            if not src.is_file():
                raise ValueError(f"no such WAV: {src}")
            shutil.copyfile(src, stage / "audio.wav")
            runner = SPEECH / "build" / ("ja_align" if selected == "cpu" else f"ja_align_{selected}")
            if not runner.is_file() or not Path(aligner).is_file():
                raise ValueError("ja_align runner or aligner weights missing; see speech/README.md")
            cmd = [str(runner), "--model", str(aligner), "--wav", str(stage / "audio.wav"),
                   "--fps", str(FPS), "--out", str(stage / "align.json")]
            if request.get("kana"):
                cmd += ["--kana", request["kana"]]
            if selected in ("cuda", "rocm"):
                cmd += ["--rocm" if selected == "rocm" else "--cuda", "--device", str(gpu.device_index())]
            with gpu.device_session(1536, cancel) if selected in ("cuda", "rocm") else nullcontext():
                _run(cmd, cancel, progress)
        else:
            runner = SPEECH / "build" / ("tts_ja" if selected == "cpu" else f"tts_ja_{selected}")
            if not runner.is_file() or not Path(model).is_dir() or not Path(aligner).is_file():
                raise ValueError("tts_ja runner or model weights missing; see speech/README.md")
            cmd = [str(runner), "--model", str(model), "--aligner", str(aligner), "--text", text,
                   "--backend", selected, "--device", str(gpu.device_index()), "--fps", str(FPS), "--out", str(stage / "audio.wav"),
                   "--aux", str(stage / "align.json")]
            for arg, key in (("--speaker", "speaker"), ("--instruct", "instruct"), ("--kana", "kana"),
                             ("--ref-wav", "ref_wav"), ("--ref-text", "ref_text")):
                if request.get(key) is not None:
                    cmd += [arg, str(request[key])]
            cmd += ["--seed", str(seed)]
            if request.get("xvec_only"):
                cmd += ["--xvec-only"]
            with gpu.device_session(1536, cancel) if selected in ("cuda", "rocm") else nullcontext():
                _run(cmd, cancel, progress)
        if cancel.is_set():
            raise gpu.Cancelled("cancelled")
        aux = json.loads((stage / "align.json").read_text())
        if text and not aux.get("phones"):
            raise ValueError("no speech detected in TTS output; try another seed or a longer sentence")
        emotion_keys = request.get("emotion_keyframes")
        analysis = None
        if auto_emotion:
            analysis = emotion.extract(stage / "audio.wav", aux["duration"], stage, cancel, progress,
                                       runner=emotion_runner, model=emotion_model, backend=selected)
            emotion_keys = analysis["emotion_keyframes"]
            (stage / "emotion.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
        motion_seed = (seed if request.get("seed") is not None or text else
                       source_meta.get("secondary_seed", source_meta.get("seed")))
        if type(motion_seed) is not int or not 0 <= motion_seed < 2**64:
            motion_seed = 7
        secondary_strength = _unit(request.get("secondary_strength", 1), "secondary_strength")
        frames = build_frames(aux, emotion_keys, request.get("speech_strength", 1),
                              request.get("emotion_strength", .6), secondary_seed=motion_seed,
                              secondary_strength=secondary_strength)
        progress(.8, "writing animation")
        animation = {"format": "vhuman.performance.v1", "fps": aux["visemes"]["fps"],
                     "duration": aux["duration"], "controls": list(rigdef.CONTROLS), "frames": frames,
                     "emotion_keyframes": emotion_keys or [], "emotion_source": "SenseVoiceSmall" if auto_emotion else "manual",
                     "speech_strength": _unit(request.get("speech_strength", 1), "speech_strength"),
                     "emotion_strength": _unit(request.get("emotion_strength", .6), "emotion_strength"),
                     "secondary_seed": motion_seed, "secondary_strength": secondary_strength}
        (stage / "animation.json").write_text(json.dumps(animation, ensure_ascii=False, separators=(",", ":")))
        write_lightrig(frames, stage / "lightrig.txt")
        rig_bytes = rig_path.read_bytes()
        rig = rigdef.Rig(json.loads(rig_bytes), folder=rig_path.parent)
        usd.write_track(rig, [f["t"] for f in frames], [f["v"] for f in frames], stage / "animation.usda",
                        fps=float(animation["fps"]), rig_layer=str(rig_path.parent / "rig.usda"))
        manifest = {"id": take_id, "head_id": head, "format": animation["format"],
                    "source": "text" if text else "wav" if wav else "source_take",
                    "source_take": source_take,
                    "text": text or request.get("transcript") or source_meta.get("text", ""),
                    "duration": aux["duration"], "fps": animation["fps"],
                    "frames": len(frames), "backend": "reused" if source_take else selected,
                    "speaker": request.get("speaker") if text else source_meta.get("speaker"),
                    "seed": seed if text else source_meta.get("seed"),
                    "secondary_seed": motion_seed, "secondary_strength": secondary_strength,
                    "emotion_source": animation["emotion_source"],
                    "emotion_model": analysis["model"] if auto_emotion else None,
                    "emotion_model_author": analysis["model_author"] if auto_emotion else None,
                    "rig_sha256": hashlib.sha256(rig_bytes).hexdigest()[:16]}
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        progress(.99, "take ready")
        stage.rename(final)
        return service.take_summary(head, take_id)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
