#!/usr/bin/env python3
"""Test suite for the bash-over-HTTP server + client.

Runs the server in-process on an ephemeral port and exercises the client
library plus the raw HTTP edge cases. Standard-library `unittest`; the client
side needs `requests`.

    python3 -m unittest test_bash_http -v
        (or)
    python3 test_bash_http.py
"""

import threading
import time
import unittest

import requests

import bash_http_server as srv
import bash_http_client as bh

_httpd = None


def setUpModule():
    """Start the server in a background thread on an ephemeral port."""
    global _httpd
    _httpd = srv.ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    port = _httpd.server_address[1]
    threading.Thread(target=_httpd.serve_forever, daemon=True).start()
    bh.BASE_URL = "http://127.0.0.1:%d" % port


def tearDownModule():
    bh._http.close()
    if _httpd is not None:
        _httpd.shutdown()
        _httpd.server_close()


class CoreBehavior(unittest.TestCase):
    def setUp(self):
        self.sid = bh.new_session()
        self.addCleanup(self._safe_close)

    def _safe_close(self):
        try:
            bh.close(self.sid)
        except Exception:
            pass

    def test_state_persists_across_runs(self):
        bh.run(self.sid, "cd /tmp")
        bh.run(self.sid, "X=42")
        r = bh.run(self.sid, "echo $X; pwd")
        self.assertEqual(r["code"], 0)
        self.assertIn("42", r["stdout"])
        self.assertIn("/tmp", r["stdout"])

    def test_exit_code_propagates(self):
        self.assertEqual(bh.run(self.sid, "true")["code"], 0)
        self.assertEqual(bh.run(self.sid, "false")["code"], 1)
        # subshell exit, so the session's own shell is not terminated
        self.assertEqual(bh.run(self.sid, "(exit 7)")["code"], 7)

    def test_streaming_is_incremental(self):
        chunks = []
        ts = []
        r = bh.run(self.sid,
                   "for i in 1 2 3; do echo $i; sleep 0.3; done",
                   on_stdout=lambda c: (chunks.append(c), ts.append(time.time())))
        self.assertEqual(r["code"], 0)
        self.assertIn("1", r["stdout"])
        self.assertIn("3", r["stdout"])
        # More than one chunk arriving over time => genuinely incremental.
        self.assertGreaterEqual(len(chunks), 2)
        self.assertGreater(ts[-1] - ts[0], 0.2)

    def test_truncation(self):
        r = bh.run(self.sid, "yes | head -100000", max_bytes=200)
        self.assertTrue(r["truncated"])
        self.assertFalse(r["timed_out"])
        self.assertLessEqual(r["bytes"], 200)
        # session remains usable and clean after truncation
        self.assertEqual(bh.run(self.sid, "echo ok")["stdout"].strip(), "ok")

    def test_timeout_is_recoverable(self):
        t0 = time.time()
        r = bh.run(self.sid, "sleep 30", timeout=2)
        self.assertTrue(r["timed_out"])
        self.assertIsNone(r["code"])
        self.assertLess(time.time() - t0, 6)
        # still reusable on the same session
        self.assertEqual(bh.run(self.sid, "echo alive")["stdout"].strip(),
                         "alive")

    def test_interrupt_completes_promptly(self):
        out = []
        th = threading.Thread(target=lambda: out.append(
            bh.run(self.sid, "echo go; sleep 30; echo done", timeout=40)))
        th.start()
        time.sleep(1.2)
        self.assertEqual(bh.interrupt(self.sid), {"interrupted": True})
        th.join(timeout=10)
        self.assertTrue(out, "interrupted run did not return promptly")
        r = out[0]
        self.assertEqual(r["code"], 130)        # 128 + SIGINT
        self.assertIn("go", r["stdout"])
        self.assertNotIn("done", r["stdout"])   # aborted before echo done
        # session clean afterwards
        self.assertEqual(bh.run(self.sid, "echo recovered")["stdout"].strip(),
                         "recovered")

    def test_dumb_terminal_and_width(self):
        self.assertEqual(bh.run(self.sid, "echo $TERM")["stdout"].strip(),
                         "dumb")
        self.assertEqual(
            bh.run(self.sid, "tput cols 2>/dev/null || echo 0")["stdout"]
            .strip(), str(srv.PTY_COLS))


class EdgeCases(unittest.TestCase):
    def test_unknown_session_404(self):
        resp = requests.post(bh.BASE_URL + "/run",
                             json={"session": "nope", "command": "echo hi"})
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json(), {"error": "no such session"})

    def test_interrupt_unknown_session_404(self):
        resp = requests.post(bh.BASE_URL + "/interrupt",
                             json={"session": "nope"})
        self.assertEqual(resp.status_code, 404)

    def test_concurrent_run_is_rejected(self):
        sid = bh.new_session()
        self.addCleanup(lambda: bh.close(sid))
        th = threading.Thread(
            target=lambda: bh.run(sid, "sleep 3; echo done", timeout=20))
        th.start()
        time.sleep(0.6)
        resp = requests.post(bh.BASE_URL + "/run",
                             json={"session": sid, "command": "echo second"})
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json(), {"error": "session busy"})
        th.join()

    def test_client_disconnect_kills_command(self):
        sid = bh.new_session()
        self.addCleanup(lambda: bh.close(sid))
        resp = requests.post(
            bh.BASE_URL + "/run",
            json={"session": sid,
                  "command": "for i in $(seq 1 50); do echo line$i; "
                             "sleep 0.2; done; echo SHOULD_NOT_FINISH"},
            stream=True)
        got = 0
        for ln in resp.iter_lines():
            if ln:
                got += 1
                if got >= 2:
                    break
        resp.close()  # abruptly drop the connection mid-command
        self.assertGreaterEqual(got, 2)

        # Poll until cleanup releases the session, then assert it is clean.
        deadline = time.time() + 10
        r = None
        while time.time() < deadline:
            try:
                r = bh.run(sid, "echo clean_now")
                break
            except RuntimeError:
                time.sleep(0.2)  # 409 busy while cleanup runs
        self.assertIsNotNone(r, "session never became usable again")
        self.assertEqual(r["stdout"].strip(), "clean_now")
        self.assertNotIn("SHOULD_NOT_FINISH", r["stdout"])
        self.assertNotIn("line", r["stdout"])


class Introspection(unittest.TestCase):
    def test_health(self):
        h = bh.health()
        self.assertTrue(h["ok"])
        self.assertEqual(h["version"], srv.VERSION)
        self.assertIn("uptime_seconds", h)
        self.assertGreaterEqual(h["sessions"], 0)

    def test_sessions_lists_cwd_and_busy(self):
        sid = bh.new_session()
        self.addCleanup(lambda: bh.close(sid))
        bh.run(sid, "cd /tmp")
        infos = {s["session"]: s for s in bh.sessions()}
        self.assertIn(sid, infos)
        self.assertTrue(infos[sid]["cwd"].endswith("/tmp"))
        self.assertFalse(infos[sid]["busy"])  # idle right now


class NewFeatures(unittest.TestCase):
    def test_normalize_strips_cr_and_ansi(self):
        sid = bh.new_session()
        self.addCleanup(lambda: bh.close(sid))
        cmd = r"printf 'a\r\nb\rc\n'; printf '\033[31mred\033[0m\n'"
        raw = bh.run(sid, cmd)
        norm = bh.run(sid, cmd, normalize=True)
        self.assertIn("\x1b", raw["stdout"])          # raw keeps escapes
        self.assertNotIn("\x1b", norm["stdout"])       # normalized strips them
        self.assertNotIn("\r", norm["stdout"])         # and CRs
        self.assertIn("red", norm["stdout"])
        self.assertIn("bc", norm["stdout"])            # 'b\rc' -> 'bc'

    def test_result_attribute_access(self):
        sid = bh.new_session()
        self.addCleanup(lambda: bh.close(sid))
        r = bh.run(sid, "echo hi")
        self.assertEqual(r.code, r["code"])
        self.assertEqual(r.stdout.strip(), "hi")
        self.assertFalse(r.truncated)

    def test_shell_context_manager(self):
        with bh.Shell() as sh:
            sid = sh.id
            sh.run("cd /tmp")
            self.assertEqual(sh.run("pwd").stdout.strip(), "/tmp")
        # session auto-closed on exit
        self.assertNotIn(sid, {s["session"] for s in bh.sessions()})

    def test_token_auth(self):
        prev_srv, prev_cli = srv.AUTH_TOKEN, bh.AUTH_TOKEN
        srv.AUTH_TOKEN = "s3cret"
        try:
            # no token -> 401
            r = requests.post(bh.BASE_URL + "/session", json={})
            self.assertEqual(r.status_code, 401)
            # wrong token -> 401
            r = requests.post(bh.BASE_URL + "/session", json={},
                              headers={"Authorization": "Bearer nope"})
            self.assertEqual(r.status_code, 401)
            # correct token via client -> works
            bh.AUTH_TOKEN = "s3cret"
            sid = bh.new_session()
            self.assertEqual(bh.run(sid, "echo ok").stdout.strip(), "ok")
            bh.close(sid)
        finally:
            srv.AUTH_TOKEN, bh.AUTH_TOKEN = prev_srv, prev_cli


class Hardening(unittest.TestCase):
    def _proc_state(self, pid):
        try:
            with open("/proc/%d/stat" % pid) as f:
                return f.read().split(")")[1].split()[0]
        except FileNotFoundError:
            return None  # fully gone

    def test_close_leaves_no_zombie(self):
        s = srv._create_session()
        pid = s.pid
        srv._remove_session(s.id)
        s.close()
        time.sleep(0.2)
        state = self._proc_state(pid)
        self.assertNotEqual(state, "Z", "closed session left a zombie")

    def test_close_is_idempotent(self):
        sid = bh.new_session()
        s = srv._get_session(sid)
        srv._remove_session(sid)
        s.close()
        s.close()  # must not raise

    def test_bad_input_is_400_not_500(self):
        sid = bh.new_session()
        self.addCleanup(lambda: bh.close(sid))
        r = requests.post(bh.BASE_URL + "/run",
                          json={"session": sid, "command": "echo hi",
                                "timeout": -5})
        self.assertEqual(r.status_code, 400)

    def test_interrupt_idle_session_does_not_leak(self):
        sid = bh.new_session()
        self.addCleanup(lambda: bh.close(sid))
        # Interrupt while nothing is running...
        bh.interrupt(sid)
        # ...then the next run must be clean (no stray ^C / blank junk).
        self.assertEqual(bh.run(sid, "echo clean").stdout.strip(), "clean")

    def test_max_sessions_cap(self):
        prev = srv.MAX_SESSIONS
        with srv._sessions_lock:
            base = len(srv._sessions)
        srv.MAX_SESSIONS = base + 1
        made = None
        try:
            made = bh.new_session()           # fills the one free slot
            r = requests.post(bh.BASE_URL + "/session", json={})
            self.assertEqual(r.status_code, 503)
        finally:
            srv.MAX_SESSIONS = prev
            if made:
                bh.close(made)


class Sweeper(unittest.TestCase):
    def test_reaps_idle_skips_busy(self):
        idle = bh.new_session()
        busy = bh.new_session()
        self.addCleanup(lambda: bh.close(busy) if srv._get_session(busy)
                        else None)

        s_idle = srv._get_session(idle)
        s_busy = srv._get_session(busy)
        self.assertIsNotNone(s_idle)
        self.assertIsNotNone(s_busy)

        # Backdate both past the TTL; hold the busy one's run lock.
        past = time.monotonic() - (srv.SESSION_IDLE_TTL + 10)
        s_idle.last_activity = past
        s_busy.last_activity = past
        s_busy._run_lock.acquire()
        try:
            reaped = srv._sweep_idle_sessions()
        finally:
            s_busy._run_lock.release()

        self.assertGreaterEqual(reaped, 1)
        self.assertIsNone(srv._get_session(idle))      # idle reaped
        self.assertIsNotNone(srv._get_session(busy))   # busy spared


if __name__ == "__main__":
    unittest.main(verbosity=2)
