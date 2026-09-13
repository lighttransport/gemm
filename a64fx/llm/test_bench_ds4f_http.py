#!/usr/bin/env python3
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import bench_ds4f_http as bench


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        events = [
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": {"usage": {
                "input_tokens": 100, "output_tokens": 8,
                "input_tokens_details": {"cached_tokens": 80}}}},
        ]
        for event in events:
            yield ("data: " + json.dumps(event) + "\n\n").encode()


class BenchmarkTest(unittest.TestCase):
    def test_stream_usage_reports_uncached_prompt_and_cache(self):
        with mock.patch.object(bench.urllib.request, "urlopen",
                               return_value=FakeResponse()):
            result = bench.request("http://unused/v1/responses", {
                "input": "hello", "stream": True})
        self.assertEqual(result["prompt_tokens"], 100)
        self.assertEqual(result["cached_tokens"], 80)
        self.assertEqual(result["uncached_prompt_tokens"], 20)
        self.assertEqual(result["completion_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
