#!/usr/bin/env python3
import os
import io
import queue
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import agentic
import llmgr_server as server


class AgenticServerTest(unittest.TestCase):
    def _handler(self):
        h = object.__new__(server.Handler)
        out = {}
        h._send_json = lambda body, status=200: out.update(
            body=body, status=status)
        h._err = lambda msg, status=400: out.update(
            body={"error": msg}, status=status)
        return h, out

    def test_responses_batch_isolates_bad_context(self):
        h, out = self._handler()
        done = threading.Event(); done.set()
        jobs = []

        def fake_submit(body, **kwargs):
            job = SimpleNamespace(id="resp-%d" % (len(jobs) + 1),
                                  done=done, error=None,
                                  result={"choices": [{"message": {
                                      "content": "ok"}}]},
                                  context_id=kwargs.get("context_id"))
            jobs.append(job)
            return job

        def fake_translate(body):
            if body.get("input") == "bad":
                raise ValueError("bad input")
            return {"ids": [1], "max_new": 1, "stream": True}

        with mock.patch.object(server, "_ready_serve", return_value=object()), \
             mock.patch.object(server.models, "get_by_openai_model",
                               return_value=server.models.get("k3")), \
             mock.patch.object(server.laguna_openai, "responses_request",
                               side_effect=fake_translate), \
             mock.patch.object(server.laguna_openai, "native_request",
                               return_value=({"ids": [1]}, None)), \
             mock.patch.object(server.laguna_openai, "responses_response",
                               side_effect=lambda body, result, request_id=None:
                               {"id": request_id, "output": result}), \
             mock.patch.object(server, "_apply_prompt_cache",
                               side_effect=lambda body, adapter: body), \
             mock.patch.object(server._inference, "submit", side_effect=fake_submit):
            h._post_openai_batch({
                "model": "k3",
                "contexts": [
                    {"context_id": "a", "input": "good"},
                    {"context_id": "b", "input": "bad"},
                ]})
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["object"], "response.batch")
        self.assertEqual(out["body"]["status"], "partial")
        self.assertEqual(out["body"]["data"][0]["context_id"], "a")
        self.assertIn("error", out["body"]["data"][1])

    def test_checkpoint_restore_directive_is_recorded(self):
        context = server._contexts.get_or_create("checkpoint-test", "k3")
        h, out = self._handler()
        h._post_context_checkpoint("/contexts/checkpoint-test/checkpoints", {
            "name": "coding"})
        self.assertEqual(out["status"], 202)
        self.assertEqual(context.checkpoint["state"], "requested")
        h._post_context_restore("/contexts/checkpoint-test/restore", {})
        self.assertEqual(out["status"], 409)
        with tempfile.TemporaryDirectory() as root, \
             mock.patch.object(server, "_managed_cache",
                               agentic.ManagedCacheStore(root)):
            staging = context.checkpoint["staging_path"]
            os.makedirs(staging)
            with open(os.path.join(staging, "rank-000.bin"), "wb") as f:
                f.write(b"checkpoint")
            h._finalize_context_checkpoint(context)
            self.assertEqual(context.checkpoint["state"], "complete")
        h._post_context_restore("/contexts/checkpoint-test/restore", {})
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["cache_load"], context.checkpoint["path"])
        server._contexts.delete("checkpoint-test")

    def test_stream_batch_tags_interleaved_context_events(self):
        h, _out = self._handler()
        h.send_response = lambda status: None
        h.send_header = lambda name, value: None
        h.end_headers = lambda: None
        h.wfile = io.BytesIO()
        events = []
        h._sse = lambda event: events.append(event)
        jobs = []

        def make_job(name):
            event_queue = queue.Queue()
            event_queue.put({"event": "token", "text": name})
            event_queue.put(None)
            job = SimpleNamespace(id="job-" + name, events=event_queue,
                                  error=None, result={"choices": [{
                                      "message": {"content": name}}]})
            jobs.append(job)
            return job

        contexts = [server._contexts.get_or_create("stream-a", "k3"),
                    server._contexts.get_or_create("stream-b", "k3")]
        submitted = [({"context_id": "stream-a", "enable_thinking": False},
                      contexts[0], make_job("A")),
                     ({"context_id": "stream-b", "enable_thinking": False},
                      contexts[1], make_job("B"))]
        with mock.patch.object(h, "_submit_batch_item", side_effect=submitted), \
             mock.patch.object(server.laguna_openai, "responses_response",
                               side_effect=lambda body, result, request_id=None:
                               {"id": request_id}), \
             mock.patch.object(server._inference, "cancel"):
            h._stream_openai_batch({"stream": True}, [{"context_id": "stream-a"},
                                                        {"context_id": "stream-b"}])
        created = [e for e in events if e["type"] == "response.created"]
        completed = [e for e in events if e["type"] == "response.completed"]
        deltas = [e for e in events if e["type"] == "response.output_text.delta"]
        self.assertEqual({e["context_id"] for e in created}, {"stream-a", "stream-b"})
        self.assertEqual({e["context_id"] for e in completed}, {"stream-a", "stream-b"})
        self.assertEqual({e["context_id"] for e in deltas}, {"stream-a", "stream-b"})

    def test_tool_continuation_validates_pending_calls(self):
        context = server._contexts.get_or_create("tool-context", "k3")
        context.record_response("resp-tool", [{"id": "call-tool"}])
        h, _out = self._handler()
        with self.assertRaises(agentic.ContextError):
            h._accept_tool_continuation(context, {
                "previous_response_id": "wrong",
                "input": [{"type": "function_call_output",
                           "call_id": "call-tool"}]})
        h._accept_tool_continuation(context, {
            "previous_response_id": "resp-tool",
            "input": [{"type": "function_call_output",
                       "call_id": "call-tool"}]})
        self.assertEqual(context.pending_tools, {})
        server._contexts.delete("tool-context")

    def test_anthropic_messages_route_uses_llmgr_context_metadata(self):
        h, out = self._handler()
        done = threading.Event(); done.set()
        job = SimpleNamespace(id="msg-1", done=done, error=None,
                              context_id="claude-context", result={
                                  "choices": [{"message": {
                                      "content": "hello"},
                                      "finish_reason": "stop"}],
                                  "usage": {}})
        request = {"model": "claude-sonnet-4", "max_tokens": 8,
                   "metadata": {"context_id": "claude-context"},
                   "messages": [{"role": "user", "content": "hello"}]}
        with mock.patch.object(server, "_ready_serve", return_value=object()), \
             mock.patch.object(server.laguna_openai, "native_request",
                               return_value=({"ids": [1], "max_new": 1}, None)), \
             mock.patch.object(server._inference, "submit", return_value=job):
            h._post_anthropic(request)
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["type"], "message")
        self.assertEqual(out["body"]["content"][0]["text"], "hello")
        server._contexts.delete("claude-context")

    def test_anthropic_count_tokens_uses_native_tokenizer_path(self):
        h, out = self._handler()
        with mock.patch.object(server.laguna_openai, "native_request",
                               return_value=({"ids": [1, 2, 3]}, None)):
            h._post_anthropic_count_tokens({
                "messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"], {"input_tokens": 3})


if __name__ == "__main__":
    unittest.main()
