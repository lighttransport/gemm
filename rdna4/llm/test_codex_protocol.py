"""GPU-independent regression for runner stdout transaction alignment."""
import base64
import io
import queue
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from codex_server import Backend


class ProtocolTest(unittest.TestCase):
    def test_startup_waits_for_runner_ready(self):
        backend = Backend.__new__(Backend)
        backend.ready = False
        backend.proc = SimpleNamespace(stdout=io.StringIO("loading\nREADY\n"),
                                       poll=lambda: None)
        with patch("codex_server.sys.stderr"):
            backend._wait_ready()
        self.assertTrue(backend.ready)

    def test_startup_reports_runner_failure(self):
        backend = Backend.__new__(Backend)
        backend.ready = False
        backend.proc = SimpleNamespace(stdout=io.StringIO("loading\n"),
                                       poll=lambda: 1)
        with self.assertRaisesRegex(RuntimeError, "runner exited before READY"):
            backend._wait_ready()

    def test_queued_disconnect_cannot_cancel_active_request(self):
        replies = queue.Queue()
        submitted = threading.Event()
        stdin = io.StringIO()
        backend = Backend.__new__(Backend)
        backend.lock = threading.Lock()
        backend.cancel_lock = threading.Lock()
        backend.active_cancel = None
        backend.ready = True
        backend.proc = SimpleNamespace(
            pid=12345, poll=lambda: None,
            stdin=SimpleNamespace(write=stdin.write, flush=submitted.set),
            stdout=SimpleNamespace(readline=lambda: replies.get(timeout=5)))
        first, second = threading.Event(), threading.Event()

        def generate(event):
            return backend.generate("test", 4, 1, .95, 40, 0, .01,
                                    cancellation=event)

        with patch("codex_server.os.kill") as kill, ThreadPoolExecutor(2) as pool:
            active = pool.submit(generate, first)
            self.assertTrue(submitted.wait(2))
            queued = pool.submit(generate, second)
            second.set()
            backend.cancel(second)
            kill.assert_not_called()
            self.assertFalse(first.is_set())
            backend.cancel(first)
            kill.assert_called_once()
            replies.put("OK 0 0 0 cancelled \n")
            self.assertEqual(active.result(timeout=2)[4], "cancelled")
            self.assertEqual(queued.result(timeout=2)[4], "cancelled")
            self.assertEqual(stdin.getvalue().count("REQ "), 1)
            backend.cancel(first)
            self.assertEqual(kill.call_count, 1)

    def test_cancelled_during_startup_is_not_dispatched(self):
        backend = Backend.__new__(Backend)
        backend.lock = threading.Lock()
        backend.cancel_lock = threading.Lock()
        backend.active_cancel = None
        backend.ready = False
        backend.proc = SimpleNamespace(poll=lambda: None, stdin=io.StringIO(),
                                       stdout=io.StringIO("READY\n"))
        cancelled = threading.Event()
        cancelled.set()
        with patch("codex_server.os.kill") as kill:
            backend.cancel(cancelled)
            result = backend.generate("test", 4, 1, .95, 40, 0, .01,
                                      cancellation=cancelled)
            self.assertEqual(result[4], "cancelled")
            self.assertEqual(backend.proc.stdin.getvalue(), "")
            kill.assert_not_called()

    def test_diagnostics_do_not_shift_answers(self):
        def reply(text):
            return "OK 0 10 1 stop " + base64.b64encode(text.encode()).decode() + "\n"

        backend = Backend.__new__(Backend)
        backend.lock = threading.Lock()
        backend.cancel_lock = threading.Lock()
        backend.active_cancel = None
        backend.ready = False
        backend.proc = SimpleNamespace(
            poll=lambda: None, stdin=io.StringIO(),
            stdout=io.StringIO("startup diagnostic\nREADY\n"
                               "Clearing modules and retrying hipModuleLoad\n"
                               + reply("first") + reply("second")
                               + "OK 0 10 0 stop \n"))
        self.assertEqual(backend.generate("one", 4, 1, .95, 40, 0, .01)[0], "first")
        self.assertEqual(backend.generate("two", 4, 1, .95, 40, 0, .01)[0], "second")
        self.assertEqual(backend.generate("empty", 4, 1, .95, 40, 0, .01)[0], "")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.generate("three", 4, 1, .95, 40, 0, .01)


if __name__ == "__main__":
    unittest.main()
