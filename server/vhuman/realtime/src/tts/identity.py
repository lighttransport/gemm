"""Verify direct-TTS checkpoint identity against the loaded talker weights."""
from pathlib import Path
from ..avatar.provenance import sha256


def verify_model(model, revision):
    if len(revision) != 64 or sha256(Path(model) / "model.safetensors") != revision:
        raise ValueError("TTS revision must be the actual model.safetensors SHA256")
