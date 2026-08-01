import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import llmgr_server as server


class FakeResponse:
    def __init__(self):
        self.lines = iter([
            b'{"event":"start"}\n',
            b'{"event":"token","id":1,"index":0}\n',
            b'{"event":"token","id":2,"index":1}\n',
            b'{"event":"done","n":2,"stop":"eos"}\n'])

    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def __iter__(self): return self
    def __next__(self): return next(self.lines)
    def close(self): pass


class FakeTokenizer:
    def decode(self, ids, raw=False):
        self.assert_raw = raw
        return {1: "why", 2: "why</think>answer"}[len(ids)]


class InferenceQueueTest(unittest.TestCase):
    def test_stream_is_collected_and_structured(self):
        queue = server.InferenceQueue(2)
        body = {"model": "laguna-s21", "messages": [{"role": "user",
                                                        "content": "q"}]}
        native = {"ids": [2, 3], "max_new": 2}
        with mock.patch.object(server, "_ready_laguna",
                               return_value=SimpleNamespace(port=8080, id="run-1")), \
             mock.patch.object(server.urllib.request, "urlopen",
                               return_value=FakeResponse()):
            job = queue.submit(body, native=native, tokenizer=FakeTokenizer(),
                               chat=True)
            self.assertTrue(job.done.wait(2.0))
        self.assertEqual(job.state, "done")
        message = job.result["choices"][0]["message"]
        self.assertEqual(message["reasoning"], "why")
        self.assertEqual(message["content"], "answer")
        self.assertEqual(job.result["usage"]["prompt_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
