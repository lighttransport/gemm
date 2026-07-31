#!/usr/bin/env python3
import json
import os
import tempfile
import threading
import unittest
import sys

sys.path.insert(0, os.path.dirname(__file__))

import agentic


class ContextRegistryTest(unittest.TestCase):
    def test_context_is_stable_and_model_bound(self):
        registry = agentic.ContextRegistry()
        first = registry.get_or_create("codex", "k3")
        self.assertIs(first, registry.get_or_create("codex", "k3"))
        with self.assertRaises(agentic.ContextError):
            registry.get_or_create("codex", "other")

    def test_tool_results_require_complete_current_call_set(self):
        context = agentic.ContextState("ctx", "k3")
        context.record_response("resp-1", [{"id": "call-a"}, {"id": "call-b"}])
        with self.assertRaises(agentic.ContextError):
            context.accept_tool_results("resp-0", [])
        with self.assertRaises(agentic.ContextError):
            context.accept_tool_results("resp-1", [{"tool_call_id": "call-a"}])
        context.accept_tool_results("resp-1", [
            {"tool_call_id": "call-a"}, {"tool_call_id": "call-b"}])
        self.assertEqual(context.pending_tools, {})

    def test_context_lock_serializes_same_context(self):
        context = agentic.ContextState("ctx", "k3")
        order = []
        entered = threading.Event()
        release = threading.Event()

        def first():
            with context.reserve():
                order.append("first")
                entered.set()
                release.wait(2)

        def second():
            entered.wait(2)
            with context.reserve():
                order.append("second")

        one = threading.Thread(target=first)
        two = threading.Thread(target=second)
        one.start(); two.start()
        entered.wait(2)
        self.assertEqual(order, ["first"])
        release.set()
        one.join(2); two.join(2)
        self.assertEqual(order, ["first", "second"])


class ManagedCacheTest(unittest.TestCase):
    def test_publish_and_validate_complete_shard_set(self):
        with tempfile.TemporaryDirectory() as root, \
             tempfile.TemporaryDirectory() as source:
            with open(os.path.join(source, "rank-000.bin"), "wb") as f:
                f.write(b"cache-a")
            with open(os.path.join(source, "rank-001.bin"), "wb") as f:
                f.write(b"cache-b")
            store = agentic.ManagedCacheStore(root, ttl_seconds=3600)
            manifest = store.publish("pfx-test", source,
                                     {"model": "k3", "tokens": 32})
            self.assertEqual(manifest["shard_count"], 2)
            got = store.validate("pfx-test", {"model": "k3", "tokens": 32})
            self.assertEqual(got["identity"], "pfx-test")
            self.assertIsNone(store.validate("pfx-test", {"model": "other"}))

    def test_partial_or_unknown_shards_are_cache_misses(self):
        with tempfile.TemporaryDirectory() as root:
            store = agentic.ManagedCacheStore(root)
            path = store.path("pfx-test")
            os.makedirs(path)
            with open(os.path.join(path, "manifest.json"), "w") as f:
                json.dump({"schema_version": 1, "identity": "pfx-test",
                           "shards": ["rank-000.bin", "rank-001.bin"],
                           "shard_count": 2}, f)
            with open(os.path.join(path, "rank-000.bin"), "wb") as f:
                f.write(b"only-one")
            self.assertIsNone(store.validate("pfx-test", {}))


if __name__ == "__main__":
    unittest.main()
