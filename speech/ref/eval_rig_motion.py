"""Audio/rig take diagnostics for a Japanese speech-motion review set.

These are consistency checks, not a perceptual or ground-truth face score.
Run from the repository root with one or more rig/takes/<id> directories.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from server.vhuman.rig import rigdef

BILABIAL = {"p", "py", "b", "by", "m", "my"}


def diagnostics(take: Path) -> dict:
    align = json.loads((take / "align.json").read_text())
    animation = json.loads((take / "animation.json").read_text())
    manifest = json.loads((take / "manifest.json").read_text())
    frames = animation["frames"]
    times, face_track = rigdef.read_track(take / "lightrig.txt")
    if len(times) != len(frames):
        raise ValueError(f"{take}: LightRig frame count differs")
    face_error = max(abs(frame["v"].get(name, 0) - row[name])
                     for frame, row in zip(frames, face_track) for name in rigdef.LR_FACE_V1)
    face_error = max(face_error, max(abs(float(t) - frame["t"])
                                      for t, frame in zip(times, frames)))
    intervals = align.get("intervals", [])
    closures, nasals, silence_jaw = [], [], []
    for i, interval in enumerate(intervals):
        phone = interval["s"]
        following = intervals[i + 1]["s"] if i + 1 < len(intervals) else ""
        effective = following if phone == "cl" else "m" if phone == "N" and following in BILABIAL else phone
        start, end = float(interval["start"]), float(interval["end"])
        if effective in BILABIAL:
            window = [frame for frame in frames
                      if start - .025 <= frame["t"] <= end + .03]
            if window:
                peak_frame = max(window, key=lambda frame: frame["v"].get("mouthClose", 0))
                closures.append({"phone": phone, "start": start, "duration": round(end - start, 3),
                                 "peak": round(peak_frame["v"].get("mouthClose", 0), 5),
                                 "jaw_at_peak": round(peak_frame["v"].get("jawOpen", 0), 5),
                                 "press_at_peak": round(max(peak_frame["v"].get("mouthPressLeft", 0),
                                                            peak_frame["v"].get("mouthPressRight", 0)), 5)})
        elif phone in ("n", "ny", "N"):
            midpoint = (start + end) * .5
            frame = min(frames, key=lambda f: abs(f["t"] - midpoint))
            nasals.append(round(frame["v"].get("mouthClose", 0), 5))
        if phone == "sil":
            silence_jaw.extend(frame["v"].get("jawOpen", 0) for frame in frames
                               if start + .08 <= frame["t"] <= end - .08)
    blink = [frame["v"].get("eyeBlinkLeft", 0) for frame in frames]
    blink_count = sum(blink[i] > .5 and blink[i] >= blink[i - 1] and blink[i] > blink[i + 1]
                      for i in range(1, len(blink) - 1))
    confidence = [float(p["conf"]) for p in align.get("phones", [])]
    prosody = align.get("prosody") or {}
    rms = np.asarray(prosody.get("rms_db", []), dtype=np.float64)
    hop = float(prosody.get("hop", 0))
    active_unaligned = None
    if len(rms) and hop > 0 and np.isfinite(rms).all():
        active = np.flatnonzero(rms > np.percentile(rms, 95) - 30)
        if len(active):
            missing = sum(not any(interval["s"] != "sil" and
                                  interval["start"] <= index * hop < interval["end"]
                                  for interval in intervals) for index in active)
            active_unaligned = round(missing / len(active), 4)
    text_chars = sum(char.isalnum() for char in manifest.get("text", ""))
    result = {"take_id": manifest["id"], "text": manifest.get("text", ""),
              "duration": align["duration"], "align_mode": align.get("mode"),
              "phones": len(confidence),
              "median_phone_confidence": round(statistics.median(confidence), 4) if confidence else None,
              "phones_per_text_char": round(len(confidence) / text_chars, 4) if text_chars else None,
              "active_audio_unaligned_fraction": active_unaligned,
              "bilabial_closures": closures,
              "bilabial_coverage_0_7": round(sum(c["peak"] >= .7 for c in closures) / len(closures), 4)
              if closures else None,
              "nonlabial_nasal_peak": max(nasals, default=None),
              "silence_jaw_mean": round(statistics.mean(silence_jaw), 5) if silence_jaw else None,
              "blink_count": blink_count,
              "gaze_peak": round(max((frame["v"].get(name, 0) for frame in frames
                                       for name in ("eyeLookInLeft", "eyeLookOutLeft",
                                                    "eyeLookUpLeft", "eyeLookDownLeft")), default=0), 5),
              "head_peak": round(max((abs(frame["v"].get(name, 0)) for frame in frames
                                       for name in rigdef.SIGNED), default=0), 5),
              "face_track_max_error": round(face_error, 7),
              "neutral_final_frame": frames[-1]["v"] == {}}
    warnings = []
    if not confidence:
        warnings.append("no aligned phones")
    elif result["median_phone_confidence"] < .7:
        warnings.append("low median phone confidence; review alignment")
    if result["phones_per_text_char"] is not None and result["phones_per_text_char"] < .5:
        warnings.append("few phones for reference text; review transcript and alignment")
    if active_unaligned is not None and active_unaligned > .45:
        warnings.append("high energy outside aligned speech; review background or missed phones")
    if result["bilabial_coverage_0_7"] is not None and result["bilabial_coverage_0_7"] < 1:
        warnings.append("a bilabial misses 0.7 closure")
    if any(c["peak"] >= .7 and (c["jaw_at_peak"] > .01 or c["press_at_peak"] > .05)
           for c in closures):
        warnings.append("residual jaw or lip press at bilabial peak may expose teeth")
    if result["silence_jaw_mean"] is not None and result["silence_jaw_mean"] > .1:
        warnings.append("jaw moves during interior silence")
    if face_error > 1e-5:
        warnings.append("LightRig face track differs from JSON")
    if not result["neutral_final_frame"]:
        warnings.append("final frame is not neutral")
    result["warnings"] = warnings
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("takes", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="write full JSON report here")
    args = parser.parse_args()
    reports = [diagnostics(take) for take in args.takes]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    for report in reports:
        print(f"{report['take_id']} {report['duration']:.2f}s phones={report['phones']} "
              f"conf={report['median_phone_confidence']} bilabial={report['bilabial_coverage_0_7']} "
              f"blink={report['blink_count']} warnings={len(report['warnings'])}")


if __name__ == "__main__":
    main()
