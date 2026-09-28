"""Select a small, repeatable local ReazonSpeech set for facial-motion review.

Requires access to the gated ReazonSpeech corpus and datasets 2.x/3.x. The
source clips stay in the output directory; this script never publishes them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np


def select_clips(rows, *, seed: int, per_bucket: int) -> list[dict]:
    """Hash-rank 2–4 s and 4–8 s clips independently, with content checks."""
    candidates = {"short": [], "long": []}
    for row in rows:
        audio = row["audio"]
        samples = np.asarray(audio["array"], dtype=np.float32)
        rate = int(audio["sampling_rate"])
        if samples.ndim != 1 or rate <= 0 or not np.isfinite(samples).all():
            continue
        duration = len(samples) / rate
        bucket = "short" if 2 <= duration < 4 else "long" if 4 <= duration <= 8 else None
        text = str(row.get("transcription") or "").strip()
        if bucket is None or len(text) < 4:
            continue
        rms = math.sqrt(float(np.mean(samples * samples)))
        if rms < .01 or float(np.max(np.abs(samples))) >= .999:
            continue
        name = str(row["name"])
        rank = hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()
        candidates[bucket].append((rank, name, text, rate, samples.copy(), duration))
        # Keep a few alternatives for repeated text or unsuitable audio on review.
        candidates[bucket].sort(key=lambda item: item[0])
        del candidates[bucket][per_bucket * 4:]
    selected, seen_text = [], set()
    for bucket in ("short", "long"):
        count = 0
        for rank, name, text, rate, samples, duration in candidates[bucket]:
            if text in seen_text:
                continue
            selected.append({"bucket": bucket, "rank": rank, "name": name,
                             "transcription": text, "sample_rate": rate,
                             "samples": samples, "duration": duration})
            seen_text.add(text)
            count += 1
            if count == per_bucket:
                break
    return selected


def write_set(clips: list[dict], out: Path, fingerprint: str, seed: int) -> None:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "clips").mkdir()
    manifest = {"source": "reazon-research/reazonspeech", "config": "tiny", "split": "train",
                "fingerprint": fingerprint, "seed": seed, "clips": []}
    for i, clip in enumerate(clips):
        clip_id = f"rs_{i:02d}"
        path = out / "clips" / f"{clip_id}.wav"
        pcm = (np.clip(clip["samples"], -1, 1) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(clip["sample_rate"])
            wav.writeframes(pcm.tobytes())
        manifest["clips"].append({"id": clip_id, "source_name": clip["name"],
                                  "transcription": clip["transcription"],
                                  "bucket": clip["bucket"], "duration": round(clip["duration"], 3),
                                  "wav": str(path.relative_to(out))})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with (out / "review.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("clip_id", "take_id", "rater", "lip_sync_1_5", "closures_1_5",
                         "prosody_1_5", "naturalness_1_5", "blink_gaze_1_5", "notes"))
        for clip in manifest["clips"]:
            writer.writerow((clip["id"], "", "", "", "", "", "", "", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output directory (use repo tmp/)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-bucket", type=int, default=6, help="clips each from 2–4 s and 4–8 s")
    args = parser.parse_args()
    if args.per_bucket < 1 or args.seed < 0:
        parser.error("seed and per-bucket must be nonnegative, with at least one clip per bucket")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("install datasets>=2,<4 and its audio dependencies in a local environment") from exc
    ds = load_dataset("reazon-research/reazonspeech", "tiny", split="train", trust_remote_code=True)
    clips = select_clips(ds, seed=args.seed, per_bucket=args.per_bucket)
    if len(clips) != 2 * args.per_bucket:
        raise SystemExit(f"only {len(clips)} suitable clips found; requested {2 * args.per_bucket}")
    write_set(clips, args.out, getattr(ds, "_fingerprint", "unknown"), args.seed)
    print(f"Wrote {len(clips)} clips and review.csv to {args.out}")


if __name__ == "__main__":
    main()
