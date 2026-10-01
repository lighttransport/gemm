"""Explicit artifact receipts; research fixtures cannot enter training silently."""
import hashlib
from pathlib import Path

PERMISSIVE = frozenset(("Apache-2.0", "MIT", "BSD-3-Clause", "CC0-1.0", "CC-BY-4.0"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_receipts(records, role="appearance-training"):
    if not isinstance(records, list) or not records:
        raise ValueError("artifact provenance is required")
    for record in records:
        if not isinstance(record, dict) or record.get("license") not in PERMISSIVE:
            raise ValueError("uncleared/research artifact cannot enter this pipeline")
        source = str(record.get("source", "")).lower()
        if "qwen-image-2.1" in source or "qwen-image-2.1" in str(record.get("generator", "")).lower():
            raise ValueError("restricted identity source cannot be relabeled as permissive")
        if role not in record.get("roles", []):
            raise ValueError(f"artifact not cleared for {role}")
        digest = record.get("sha256", "")
        if not record.get("source") or not record.get("revision") or len(digest) != 64:
            raise ValueError("incomplete artifact receipt")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError("invalid artifact digest") from exc


def verify_files(records, root, role="appearance-training"):
    validate_receipts(records, role)
    root = Path(root).resolve()
    for record in records:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or sha256(path) != record["sha256"]:
            raise ValueError("reference path escapes root or checksum differs")
