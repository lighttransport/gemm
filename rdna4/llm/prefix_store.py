"""Remember agents' stable system prefixes across server restarts.

The runner's shared prefix snapshots (tools block, system turn) live in host
memory, so after a restart every agent's first request re-prefills its whole
system prompt (Claude Code: ~15.7K tokens, ~31 s).  The shim records the
prefix boundary texts of successful requests here, and a new server replays
the most recent ones as zero-output requests to rebuild those snapshots
before the first agent request arrives.

The file holds system prompts (tool schemas, AGENTS.md/CLAUDE.md contents),
so it is written with mode 0600.
"""
import hashlib
import json
import os
import tempfile
import threading
import time


class PrefixStore:
    def __init__(self, path, capacity=6, min_chars=2000):
        self.path = path
        self.capacity = capacity
        self.min_chars = min_chars
        self.lock = threading.Lock()
        self.entries = {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for key, entry in data.get("prefixes", {}).items():
                if (isinstance(entry, dict) and isinstance(entry.get("boundaries"), list) and
                        entry["boundaries"] and
                        all(isinstance(b, str) for b in entry["boundaries"]) and
                        isinstance(entry.get("last"), (int, float)) and
                        isinstance(entry.get("hits", 0), int)):
                    self.entries[key] = entry
        except (OSError, ValueError, AttributeError):
            self.entries = {}

    def record(self, boundaries):
        """Remember one request's prefix boundaries (longest last)."""
        if not boundaries or len(boundaries[-1]) < self.min_chars:
            return
        key = hashlib.sha256(boundaries[-1].encode("utf-8")).hexdigest()
        with self.lock:
            entry = self.entries.get(key)
            changed = entry is None or entry["boundaries"] != boundaries
            self.entries[key] = {"boundaries": list(boundaries), "last": time.time(),
                                 "hits": (entry or {}).get("hits", 0) + 1}
            if len(self.entries) > self.capacity:
                oldest = min(self.entries, key=lambda k: self.entries[k]["last"])
                del self.entries[oldest]
                changed = True
            # Rewrite on content changes, and at most once a minute for recency.
            if changed or time.time() - getattr(self, "_written", 0) > 60:
                self._write()

    def _write(self):
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".prefixes-")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "prefixes": self.entries}, f, ensure_ascii=False)
            os.replace(tmp, self.path)
            self._written = time.time()
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def warmup_list(self):
        """Stored boundary lists, most recently used first."""
        with self.lock:
            ordered = sorted(self.entries.values(), key=lambda e: e["last"], reverse=True)
            return [e["boundaries"] for e in ordered]
