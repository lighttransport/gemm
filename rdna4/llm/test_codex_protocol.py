"""GPU-independent regression for runner stdout transaction alignment."""
import base64
import hashlib
import io
import os
import queue
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from codex_server import Backend, responses_input_messages, runner_command


class ProtocolTest(unittest.TestCase):
    def test_qwen35_runner_command_uses_exact_server_profile(self):
        args = SimpleNamespace(
            runner="./test_hip_llm", model="target.gguf", context=65536,
            moe_cache_mb=0, coding=False, qwen4_coding_profile=False,
            qwen4_exact=False, qwen4_mtp=None, qwen4_mtp_draft=1,
            qwen4_mtp_cache_mb=128, qwen4_mtp_verify="scalar",
            qwen35_server_profile=True,
        )
        command = runner_command(args)
        self.assertIn("--stdio-server", command)
        self.assertIn("--kv-cache", command)
        self.assertIn("q8q8", command)
        self.assertIn("--context-cache-entries", command)
        self.assertIn("--context-cache-max-mib", command)
        self.assertEqual(command[-2:], ["--sampling-profile", "llama"])

        args.qwen35_server_profile = False
        args.qwen35_dflash2 = "draft.gguf"
        args.qwen35_dflash2_draft = 7
        dflash_command = runner_command(args)
        self.assertIn("--qwen35-dflash2", dflash_command)
        self.assertEqual(dflash_command[-4:], ["--qwen35-dflash2", "draft.gguf",
                                               "--qwen35-dflash2-draft", "7"])

    def test_qwen35_snapshot_budget_is_forwarded(self):
        args = SimpleNamespace(
            runner="./test_hip_llm", model="target.gguf", context=65536,
            moe_cache_mb=0, coding=False, qwen4_coding_profile=False,
            qwen4_exact=False, qwen4_mtp=None, qwen4_mtp_draft=1,
            qwen4_mtp_cache_mb=128, qwen4_mtp_verify="scalar",
            qwen35_server_profile=False, qwen35_dflash2="draft.gguf",
            qwen35_dflash2_draft=7, qwen35_snapshot_max_tokens=65536,
        )
        command = runner_command(args)
        self.assertIn("--qwen35-snapshot-max-tokens", command)
        i = command.index("--qwen35-snapshot-max-tokens")
        self.assertEqual(command[i + 1], "65536")

    def test_qwen35_profiles_are_mutually_exclusive(self):
        args = SimpleNamespace(
            runner="./test_hip_llm", model="target.gguf", context=4096,
            moe_cache_mb=0, coding=False, qwen4_coding_profile=False,
            qwen4_exact=False, qwen4_mtp=None, qwen35_server_profile=True,
            qwen35_dflash2="draft.gguf",
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            runner_command(args)

    def test_dflash_rejects_qwen4_mtp(self):
        args = SimpleNamespace(
            runner="./test_hip_llm", model="target.gguf", context=4096,
            moe_cache_mb=0, coding=False, qwen4_coding_profile=False,
            qwen4_exact=False, qwen4_mtp="draft.gguf",
            qwen35_server_profile=False, qwen35_dflash2="dflash.gguf",
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            runner_command(args)

    def test_qwen35_mtp_runner_command_is_exact_and_mutually_exclusive(self):
        args = SimpleNamespace(
            runner="./test_hip_llm", model="target.gguf", context=4096,
            moe_cache_mb=0, coding=False, qwen4_coding_profile=False,
            qwen4_exact=False, qwen4_mtp=None, qwen35_server_profile=False,
            qwen35_dflash2=None, qwen35_mtp="nextn.gguf",
            qwen35_mtp_draft=3, qwen35_mtp_window=True,
        )
        command = runner_command(args)
        self.assertEqual(command[-5:], ["--qwen35-mtp", "nextn.gguf",
                                        "--qwen35-mtp-draft", "3",
                                        "--qwen35-mtp-window"])
        args.qwen35_dflash2 = "dflash.gguf"
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            runner_command(args)

    def test_seed_uses_versioned_reference_sampler_request(self):
        backend = Backend.__new__(Backend)
        backend.lock = threading.Lock()
        backend.cancel_lock = threading.Lock()
        backend.active_cancel = None
        backend.ready = True
        backend.proc = SimpleNamespace(
            poll=lambda: None, stdin=io.StringIO(),
            stdout=io.StringIO("OK 0 1 0 stop \n"))
        backend.generate("test", 4, .6, .95, 0, 1.5, 1.1, .05,
                         seed=42, frequency=.2, penalty_last_n=32)
        identity = hashlib.sha256(b"shared").hexdigest()
        self.assertEqual(backend.proc.stdin.getvalue(),
                         f"REQ3 {identity} 42 4 0.6 0.95 0 1.5 1.1 0.05 0.2 32 - dGVzdA==\n")

    def test_cache_identity_is_stable_and_namespaced(self):
        backend = Backend.__new__(Backend)
        backend.lock = threading.Lock()
        backend.cancel_lock = threading.Lock()
        backend.active_cancel = None
        backend.ready = True
        backend.proc = SimpleNamespace(
            poll=lambda: None, stdin=io.StringIO(),
            stdout=io.StringIO("OK 0 1 0 stop \nOK 0 1 0 stop \n"))
        backend.generate("test", 0, 0, .95, 20, 0, 1, 0,
                         cache_key="conversation-a")
        backend.generate("test", 0, 0, .95, 20, 0, 1, 0,
                         cache_key="conversation-b")
        lines = backend.proc.stdin.getvalue().splitlines()
        self.assertEqual(lines[0].split()[1],
                         hashlib.sha256(b"conversation-a").hexdigest())
        self.assertEqual(lines[1].split()[1],
                         hashlib.sha256(b"conversation-b").hexdigest())
        self.assertNotEqual(lines[0].split()[1], lines[1].split()[1])

    def test_oversized_runner_request_is_rejected_before_dispatch(self):
        backend = Backend.__new__(Backend)
        with self.assertRaisesRegex(ValueError, "runner protocol limit"):
            backend.generate("x" * (4 * 1024 * 1024), 1, 0, .95, 20,
                             0, 1, 0)

    def test_responses_tool_output_is_preserved(self):
        messages = responses_input_messages([
            {"role": "user", "content": "Call echo."},
            {"type": "function_call_output", "call_id": "call_1",
             "output": {"value": 21}},
        ])
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertIn("call_id=call_1", messages[-1]["content"])
        self.assertIn('"value": 21', messages[-1]["content"])

    def test_responses_function_call_is_preserved(self):
        messages = responses_input_messages([
            {"role": "user", "content": "Call echo."},
            {"type": "function_call", "call_id": "call_1", "name": "echo",
             "arguments": '{"text":"hello"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "hello"},
        ])
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertIn("<function=echo>", messages[1]["content"])
        self.assertIn("<parameter=text>\nhello", messages[1]["content"])
        self.assertEqual(messages[2]["role"], "tool")

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

    def test_startup_readiness_has_a_deadline(self):
        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, "r")
        backend = Backend.__new__(Backend)
        backend.ready = False
        backend.proc = SimpleNamespace(stdout=stream, poll=lambda: None)
        try:
            with self.assertRaisesRegex(RuntimeError, "ready before timeout"):
                backend._wait_ready(0.01)
        finally:
            stream.close()
            os.close(write_fd)

    def test_health_reports_dead_runner(self):
        backend = Backend.__new__(Backend)
        backend.ready = True
        backend.proc = SimpleNamespace(poll=lambda: 1)
        self.assertEqual(backend.health(), {
            "status": "unavailable",
            "runner_alive": False,
            "runner_exit_status": 1,
            "active_request": None,
            "queued_requests": 0,
        })

    def test_close_reaps_running_runner(self):
        class FakeProc:
            def __init__(self):
                self.exit_status = None
                self.terminated = False
                self.wait_calls = 0

            def poll(self):
                return self.exit_status

            def terminate(self):
                self.terminated = True
                self.exit_status = 0

            def wait(self, timeout=None):
                self.wait_calls += 1
                return self.exit_status

        backend = Backend.__new__(Backend)
        backend.proc = FakeProc()
        backend.close()
        self.assertTrue(backend.proc.terminated)
        self.assertEqual(backend.proc.wait_calls, 1)

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

        def generate(event, request_id):
            return backend.generate("test", 4, 1, .95, 40, 0, 1, .01,
                                    cancellation=event, request_id=request_id)

        with patch("codex_server.os.kill") as kill, ThreadPoolExecutor(2) as pool:
            active = pool.submit(generate, first, "active")
            self.assertTrue(submitted.wait(2))
            queued = pool.submit(generate, second, "queued")
            for _ in range(100):
                with backend.cancel_lock:
                    if "queued" in backend.request_cancellations:
                        break
                time.sleep(0.005)
            self.assertTrue(backend.cancel_request("queued"))
            kill.assert_not_called()
            self.assertFalse(first.is_set())
            self.assertTrue(backend.cancel_request("active"))
            kill.assert_called_once()
            replies.put("OK 0 0 0 cancelled \n")
            self.assertEqual(active.result(timeout=2)[4], "cancelled")
            self.assertEqual(queued.result(timeout=2)[4], "cancelled")
            self.assertEqual(stdin.getvalue().count("REQ3 "), 1)
            self.assertFalse(backend.cancel_request("active"))
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
            result = backend.generate("test", 4, 1, .95, 40, 0, 1, .01,
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
        self.assertEqual(backend.generate("one", 4, 1, .95, 40, 0, 1, .01)[0], "first")
        self.assertEqual(backend.generate("two", 4, 1, .95, 40, 0, 1, .01)[0], "second")
        self.assertEqual(backend.generate("empty", 4, 1, .95, 40, 0, 1, .01)[0], "")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.generate("three", 4, 1, .95, 40, 0, 1, .01)


if __name__ == "__main__":
    unittest.main()
