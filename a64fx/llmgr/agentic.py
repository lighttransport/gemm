#!/usr/bin/env python3
"""Runner-neutral context and managed-cache primitives for llmgr.

This module deliberately has no HTTP or runner dependencies.  It gives the
OpenAI adapter and the native runner adapter the same small set of invariants:
one active operation per context, explicit tool-call continuation, and cache
manifests that are published atomically.
"""

from __future__ import absolute_import

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid


class ContextError(ValueError):
    pass


_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def cache_identity(model, tokenizer, runner_abi, layout, dtype, cache_key,
                   prefix_ids):
    """Return a non-secret immutable identity for a managed prefix entry."""
    payload = {
        "model": model,
        "tokenizer": tokenizer,
        "runner_abi": runner_abi,
        "layout": layout,
        "dtype": dtype,
        "cache_key": cache_key,
        "prefix_ids": list(prefix_ids),
    }
    digest = hashlib.sha256(_json_bytes(payload)).hexdigest()
    return "pfx-" + digest


class ManagedCacheStore(object):
    """Atomic, manifest-first storage for complete distributed cache sets."""

    schema_version = 1

    def __init__(self, root, ttl_seconds=7 * 24 * 3600):
        self.root = os.path.abspath(root)
        self.ttl_seconds = int(ttl_seconds)
        self._lock = threading.RLock()
        if not os.path.isdir(self.root):
            os.makedirs(self.root)

    def path(self, identity):
        if not _SAFE_NAME.match(identity):
            raise ContextError("invalid managed cache identity")
        return os.path.join(self.root, identity)

    def publish(self, identity, source_dir, metadata, shard_names=None):
        """Publish an existing shard directory without exposing partial state."""
        target = self.path(identity)
        source_dir = os.path.abspath(source_dir)
        if not os.path.isdir(source_dir):
            raise ContextError("cache source directory does not exist")
        names = sorted(shard_names or [n for n in os.listdir(source_dir)
                                       if n.endswith(".bin")])
        if not names:
            raise ContextError("cache has no shard files")
        for name in names:
            if not _SAFE_NAME.match(name):
                raise ContextError("invalid cache shard name")
            if not os.path.isfile(os.path.join(source_dir, name)):
                raise ContextError("cache shard is missing: " + name)
        manifest = dict(metadata)
        manifest.update({
            "schema_version": self.schema_version,
            "identity": identity,
            "shards": names,
            "shard_count": len(names),
            "created": time.time(),
        })
        with self._lock:
            tmp = tempfile.mkdtemp(prefix="." + identity + ".", dir=self.root)
            try:
                for name in names:
                    src = os.path.join(source_dir, name)
                    dst = os.path.join(tmp, name)
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copyfile(src, dst)
                with open(os.path.join(tmp, "manifest.json"), "wb") as f:
                    f.write(_json_bytes(manifest) + b"\n")
                    f.flush()
                    os.fsync(f.fileno())
                if os.path.exists(target):
                    shutil.rmtree(target)
                os.replace(tmp, target)
            finally:
                if os.path.isdir(tmp):
                    shutil.rmtree(tmp)
        return manifest

    def validate(self, identity, expected):
        """Return a complete manifest, or ``None`` for a cache miss."""
        try:
            path = self.path(identity)
            with open(os.path.join(path, "manifest.json"), "rb") as f:
                manifest = json.loads(f.read().decode("utf-8"))
            if manifest.get("schema_version") != self.schema_version:
                return None
            for key, value in expected.items():
                if manifest.get(key) != value:
                    return None
            shards = manifest.get("shards")
            if not isinstance(shards, list) or not shards:
                return None
            if manifest.get("shard_count") != len(shards):
                return None
            if any(not _SAFE_NAME.match(name) or
                   not os.path.isfile(os.path.join(path, name))
                   for name in shards):
                return None
            if self.ttl_seconds and time.time() - os.path.getmtime(path) > self.ttl_seconds:
                return None
            os.utime(path, None)
            return manifest
        except (IOError, OSError, ValueError, TypeError):
            return None

    def remove(self, identity):
        with self._lock:
            path = self.path(identity)
            if os.path.isdir(path):
                shutil.rmtree(path)
                return True
        return False


class ContextState(object):
    def __init__(self, context_id, model):
        self.context_id = context_id
        self.model = model
        self.lock = threading.Lock()
        self.created = time.time()
        self.updated = self.created
        self.last_response_id = None
        self.pending_tools = {}
        self.checkpoint = None
        self.prefix_identity = None

    def reserve(self):
        return self.lock

    def record_response(self, response_id, tool_calls=None):
        self.last_response_id = response_id
        self.pending_tools = {}
        for call in tool_calls or []:
            call_id = call.get("id")
            if call_id:
                self.pending_tools[call_id] = dict(call)
        self.updated = time.time()

    def accept_tool_results(self, previous_response_id, results):
        if previous_response_id != self.last_response_id:
            raise ContextError("tool results reference a stale response")
        seen = set()
        for result in results:
            call_id = result.get("tool_call_id")
            if call_id in seen or call_id not in self.pending_tools:
                raise ContextError("unknown or duplicate tool call: %s" % call_id)
            seen.add(call_id)
        if seen != set(self.pending_tools):
            raise ContextError("tool results are incomplete")
        self.pending_tools = {}
        self.updated = time.time()


class ContextRegistry(object):
    def __init__(self, max_contexts=1024):
        self.max_contexts = int(max_contexts)
        self._lock = threading.RLock()
        self._contexts = {}

    def get_or_create(self, context_id=None, model=""):
        context_id = context_id or "ctx_" + uuid.uuid4().hex
        if not isinstance(context_id, str) or not _SAFE_NAME.match(context_id):
            raise ContextError("invalid context_id")
        with self._lock:
            context = self._contexts.get(context_id)
            if context is not None and context.model != model:
                raise ContextError("context model cannot change")
            if context is None:
                if len(self._contexts) >= self.max_contexts:
                    raise ContextError("context registry is full")
                context = ContextState(context_id, model)
                self._contexts[context_id] = context
            context.updated = time.time()
            return context

    def get(self, context_id):
        with self._lock:
            return self._contexts.get(context_id)

    def find_response(self, response_id):
        with self._lock:
            for context in self._contexts.values():
                if context.last_response_id == response_id:
                    return context
        return None

    def delete(self, context_id):
        with self._lock:
            return self._contexts.pop(context_id, None) is not None

    def info(self):
        with self._lock:
            return [{"context_id": c.context_id, "model": c.model,
                     "created": c.created, "updated": c.updated,
                     "last_response_id": c.last_response_id,
                     "pending_tools": len(c.pending_tools),
                     "checkpoint": c.checkpoint}
                    for c in self._contexts.values()]


def batch_result(context_id, response=None, error=None):
    result = {"context_id": context_id}
    if error is not None:
        result["error"] = {"message": str(error), "type": "context_error"}
    else:
        result["response"] = response
    return result
