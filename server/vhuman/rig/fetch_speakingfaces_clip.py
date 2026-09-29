"""Fetch one bounded, public SpeakingFaces RGB/audio clip for fit validation.

Dataset: ISSAI SpeakingFaces, CC BY 4.0, https://issai.nu.edu.kz/download-speaking-faces/
The large subject archive is read by HTTP ranges; only one 72-frame clip is
stored locally. Run with TMPDIR pointing inside the repository.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from remotezip import RemoteZip

URL = "https://huggingface.co/datasets/issai/Speaking_Faces/resolve/main/image_audio/sub_1_ia.zip"
PREFIX = "1_1_2_7_611"


def fetch(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    with RemoteZip(URL) as archive:
        prefix = "sub_1_ia/trial_1/rgb_image_cmd/" + PREFIX + "_"
        images = [name for name in archive.namelist() if name.startswith(prefix) and name.endswith("_2.png")]
        images.sort(key=lambda name: int(name.removesuffix("_2.png").split("_")[-1]))
        if len(images) != 72:
            raise ValueError(f"expected 72 RGB frames, found {len(images)}")
        audio_name = "sub_1_ia/trial_1/mic1_audio_cmd_trim/" + PREFIX + "_1.wav"
        audio = out / "audio.wav"
        audio.write_bytes(archive.read(audio_name))
        for i, name in enumerate(images):
            (frames / f"{i:05d}.png").write_bytes(archive.read(name))
    video = out / "speakingfaces_subject1_trial1.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-framerate", "28", "-i", str(frames / "%05d.png"), "-i", str(audio),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    str(video)], check=True, timeout=180)
    (out / "source.json").write_text(json.dumps({"dataset": "ISSAI SpeakingFaces",
        "license": "CC BY 4.0", "url": URL, "prefix": PREFIX, "fps": 28,
        "frames": len(images), "video": video.name}, indent=2))
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tmp/vhuman-rig/speakingfaces"))
    args = parser.parse_args()
    print(fetch(args.out))
