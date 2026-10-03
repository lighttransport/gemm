"""Durable, bounded tile checkpoints; reconstruct candidates without retraining."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import zlib

import numpy as np

from .common import GIB, require_disk, atomic_json


class TileStore:
    def __init__(self, path, identity, cap=8*GIB, reserve=4*GIB):
        self.path, self.cap, self.reserve = Path(path), cap, reserve
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (identity TEXT)")
        old = self.db.execute("SELECT identity FROM metadata").fetchone()
        if old and old[0] != identity:
            self.db.close()
            raise ValueError("GSQ checkpoint source/config/activations/codec changed")
        if not old:
            self.db.execute("INSERT INTO metadata VALUES (?)", (identity,))
        self.db.execute("CREATE TABLE IF NOT EXISTS tiles (key TEXT PRIMARY KEY, baseline TEXT, checksum TEXT, delta BLOB, metrics TEXT)")
        self.db.commit()

    @staticmethod
    def digest(raw):
        return hashlib.sha256(memoryview(raw)).hexdigest()

    def restore(self, key, baseline):
        row = self.db.execute("SELECT baseline,checksum,delta,metrics FROM tiles WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        if row[0] != self.digest(baseline):
            raise ValueError(f"Native baseline changed: {key}")
        raw = baseline.copy()
        if row[2]:
            delta = np.frombuffer(zlib.decompress(row[2]), dtype=np.uint8)
            if delta.shape != raw.shape:
                raise ValueError(f"Invalid checkpoint delta: {key}")
            raw ^= delta
        if self.digest(raw) != row[1]:
            raise ValueError(f"Corrupt GSQ tile checkpoint: {key}")
        return raw, json.loads(row[3])

    def save(self, key, baseline, raw, metrics):
        delta = b"" if np.array_equal(raw, baseline) else zlib.compress(np.bitwise_xor(raw, baseline).tobytes())
        # Leave space for SQLite pages/journals, activation state and logs.
        if self.path.stat().st_size + len(delta) + 65536 > self.cap:
            raise RuntimeError("GSQ resume checkpoint cap reached; expand available storage and set GSQ_CHECKPOINT_GIB before continuing")
        require_disk(self.path.parent, len(delta)+65536, self.reserve)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO tiles VALUES (?,?,?,?,?)", (key, self.digest(baseline), self.digest(raw), delta, json.dumps(metrics)))

    def close(self):
        self.db.close()


def write_payload(writer, tensors, temporary, journal, identity, reserve):
    """Resume a single GGUF at fsynced tensor boundaries, verifying old bytes."""
    temporary, journal = Path(temporary), Path(journal)
    header = journal.with_suffix(".header.tmp")
    writer.write_header_to_file(path=header)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.close()
    header_bytes = header.read_bytes()
    header.unlink()
    signature = hashlib.sha256(header_bytes).hexdigest()
    state = {"identity": identity, "header": signature, "offset": len(header_bytes), "tensors": {}}
    if temporary.exists():
        if not journal.exists():
            raise ValueError(f"Interrupted GGUF has no resume journal: {temporary}")
        state = json.loads(journal.read_text())
        if state["identity"] != identity or state["header"] != signature:
            raise ValueError("Export checkpoint source/config/selection/layout changed")
        with temporary.open("rb") as stream:
            if stream.read(len(header_bytes)) != header_bytes:
                raise ValueError("Interrupted GGUF header is corrupt")
            for name, record in state["tensors"].items():
                stream.seek(record["start"])
                checksum, left = hashlib.sha256(), record["bytes"]
                while left:
                    chunk = stream.read(min(left, 8*2**20))
                    if not chunk:
                        raise ValueError(f"Truncated saved tensor: {name}")
                    checksum.update(chunk)
                    left -= len(chunk)
                if checksum.hexdigest() != record["hash"]:
                    raise ValueError(f"Corrupt saved tensor: {name}")
    else:
        with temporary.open("xb") as stream:
            stream.write(header_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(journal, state)
    saved_names = list(state["tensors"])
    if saved_names != [row[0] for row in tensors[:len(saved_names)]]:
        raise ValueError("Export checkpoint is not a tensor prefix")
    remaining = sum(c.nbytes for _, _, _, chunks in tensors[len(saved_names):] for c in chunks)
    require_disk(temporary.parent, remaining+32*2**20, reserve)
    with temporary.open("r+b") as stream:
        stream.truncate(state["offset"])
        stream.seek(state["offset"])
        for name, _, _, chunks in tensors:
            ti = writer.tensors[0].pop(name)
            expected = hashlib.sha256()
            for raw in chunks:
                expected.update(memoryview(raw))
            if name in state["tensors"]:
                if expected.hexdigest() != state["tensors"][name]["hash"]:
                    raise ValueError(f"Candidate changed since export: {name}")
                continue
            writer.write_padding(stream, stream.tell())
            start = stream.tell()
            written = 0
            for raw in chunks:
                raw.tofile(stream)
                written += raw.nbytes
            if written != ti.nbytes:
                raise ValueError(f"Payload mismatch: {name}")
            writer.write_padding(stream, written)
            stream.flush()
            os.fsync(stream.fileno())
            state["tensors"][name] = {"start": start, "bytes": written, "hash": expected.hexdigest()}
            state["offset"] = stream.tell()
            atomic_json(journal, state)
    return {name: record["hash"] for name, record in state["tensors"].items()}
