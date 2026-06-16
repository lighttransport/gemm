#!/usr/bin/env python3
"""Persistent stateful bash-over-HTTP execution server.

Drives a remote bash shell for an LLM CLI agent inside a trusted VPN
(no authentication). Each session owns ONE long-lived interactive bash
process (via pty.fork) so cwd, shell variables and background jobs persist
across multiple /run calls. Output is streamed back incrementally as NDJSON.

Standard library only. Binds 127.0.0.1 only (exposed externally via SSH
remote-forward + bastion relay; never bind 0.0.0.0).
"""

import argparse
import fcntl
import hmac
import json
import os
import pty
import queue
import re
import select
import signal
import struct
import sys
import termios
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

try:
    from http.server import ThreadingHTTPServer
except ImportError:
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

VERSION = "1.0"

HOST = "127.0.0.1"
PORT = 21264

DEFAULT_TIMEOUT = 300
DEFAULT_MAX_BYTES = 1_000_000
DRAIN_TIMEOUT = 2.0  # best-effort post-interrupt drain window (seconds)
READ_CHUNK = 65536
MAX_BODY = 16 * 1024 * 1024  # cap request body size (commands are tiny)
MAX_SESSIONS = 256           # soft cap on concurrent sessions (0 = unlimited)

# (d) Quiet the shell at the source: a dumb terminal makes programs avoid
# colour/cursor escape sequences, and a wide fixed window stops them wrapping
# output to an arbitrary default width.
CHILD_TERM = "dumb"
PTY_ROWS = 50
PTY_COLS = 200

# (c) Idle-session sweeper: a session untouched for this long is reaped.
SESSION_IDLE_TTL = 1800.0   # seconds (30 min)
SWEEP_INTERVAL = 60.0       # how often the sweeper thread checks

# (f/i/h) Runtime config, set by main() from CLI flags / env vars.
AUTH_TOKEN = None   # if set, all requests must send "Authorization: Bearer ..."
VERBOSE = False     # log requests/errors to stderr when True

# (e) Output normalization: collapse CRLF, drop bare CR, strip ANSI/OSC escape
# sequences. TERM=dumb already suppresses most escapes; this cleans the rest so
# the emitted text is plain. Applied per stdout chunk (best-effort across chunk
# boundaries — a sequence split across two reads may survive).
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def _normalize_text(text: str) -> str:
    text = _ANSI_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "")
    return text


class Session:
    """A single persistent interactive bash shell behind a PTY.

    A background reader thread continuously reads the PTY master and fans the
    bytes out to per-call subscriber queues. A session is used strictly
    serially (the caller waits for one command to finish before sending the
    next), so at most one subscriber is active at a time, but the design keeps
    the registry general and thread-safe.
    """

    def __init__(self):
        self.id = uuid.uuid4().hex
        # pty.fork() gives us a child running bash and a master fd we own.
        self.pid, self.master_fd = pty.fork()
        if self.pid == 0:
            # Child: become an interactive bash. --norc keeps the prompt/env
            # predictable. A dumb TERM suppresses most colour/cursor escapes
            # at the source. exec replaces this process image.
            os.environ["TERM"] = CHILD_TERM
            os.execvp("/bin/bash", ["/bin/bash", "--norc", "-i"])
            os._exit(127)  # only reached if exec fails

        # Give the PTY a fixed, wide window so programs don't wrap to 80 cols.
        self._set_winsize(PTY_ROWS, PTY_COLS)

        self._subscribers = []          # list[queue.Queue]
        self._sub_lock = threading.Lock()
        self._interrupt_evt = threading.Event()  # set by /interrupt
        # (a) Serialize /run on a session: the protocol is serial, and two
        # concurrent runs would interleave output and cross sentinels.
        self._run_lock = threading.Lock()
        # (c) Track activity so the sweeper can reap idle sessions.
        self.last_activity = time.monotonic()
        self._alive = True
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._init_shell()

    def _set_winsize(self, rows, cols):
        try:
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ,
                        struct.pack("HHHH", rows, cols, 0, 0))
        except OSError:
            pass

    def touch(self):
        """Mark the session as recently used (resets the idle clock)."""
        self.last_activity = time.monotonic()

    def info(self):
        """Lightweight, non-intrusive snapshot for /sessions introspection.

        cwd is read from the bash process's /proc entry rather than by running
        a command, so it works even while the session is busy.
        """
        try:
            cwd = os.readlink("/proc/%d/cwd" % self.pid)
        except OSError:
            cwd = None
        return {
            "session": self.id,
            "pid": self.pid,
            "cwd": cwd,
            "busy": self._run_lock.locked(),
            "idle_seconds": round(time.monotonic() - self.last_activity, 1),
            "alive": self._alive,
        }

    def _init_shell(self):
        """Quiet the interactive shell so streamed output is clean.

        An interactive bash echoes every command we type and prints a prompt
        plus bracketed-paste escape sequences. Left alone, the echoed command
        contains the literal __DONE__ marker and would trip false completion
        detection, and the output would be full of prompt noise. We disable
        echo, blank the prompts, and turn off bracketed paste, then drain the
        setup output so it never reaches a /run caller.
        """
        # stty -echo: stop the tty echoing what we write to it.
        # PS1/PS2='' : no prompts interleaved with output.
        # enable-bracketed-paste off: no \e[?2004h/l wrappers around input.
        setup = (
            "stty -echo 2>/dev/null; "
            "PS1=''; PS2=''; "
            "bind 'set enable-bracketed-paste off' 2>/dev/null\n"
        )
        self._write(setup.encode())
        self._drain()

    # --- reader thread / subscriber fan-out -------------------------------

    def _read_loop(self):
        """Continuously read the PTY master and broadcast to subscribers."""
        while self._alive:
            try:
                r, _, _ = select.select([self.master_fd], [], [], 0.2)
            except (OSError, ValueError):
                break
            if not r:
                continue
            try:
                data = os.read(self.master_fd, READ_CHUNK)
            except OSError:
                break
            if not data:
                break  # EOF: shell exited
            with self._sub_lock:
                subs = list(self._subscribers)
            for q in subs:
                q.put(data)
        # Signal EOF to any waiting subscribers.
        self._alive = False
        with self._sub_lock:
            for q in self._subscribers:
                q.put(None)

    def _subscribe(self):
        q = queue.Queue()
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def _unsubscribe(self, q):
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _write(self, data: bytes):
        try:
            os.write(self.master_fd, data)
        except OSError:
            pass

    # --- public control ----------------------------------------------------

    def interrupt(self):
        """Send Ctrl-C to the running foreground command.

        Also flags any in-flight run() so it can re-arm its sentinel: an
        interactive bash discards the pre-queued sentinel printf when SIGINT
        aborts the command, so without re-arming the run would hang until its
        own timeout.
        """
        self._interrupt_evt.set()
        self._write(b"\x03")

    def close(self):
        """Terminate the shell and release resources (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._alive = False
        # Ask bash to exit, then let the reader thread notice _alive=False and
        # stop before we close the fd (avoids closing a fd it is mid-read on).
        self._write(b"\nexit\n")
        self._reader.join(timeout=0.5)
        try:
            os.close(self.master_fd)
        except OSError:
            pass
        # Make sure the child is actually dead AND reaped — a bare
        # waitpid(WNOHANG) right after SIGTERM races the child's exit and
        # leaves a zombie. Escalate and wait until it is gone.
        self._reap()

    def _reap(self):
        try:
            pid, _ = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                return
        except OSError:
            return  # already reaped or never existed
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(self.pid, sig)
            except OSError:
                return  # gone
            for _ in range(20):  # up to ~0.2s before escalating
                try:
                    pid, _ = os.waitpid(self.pid, os.WNOHANG)
                except OSError:
                    return
                if pid:
                    return
                time.sleep(0.01)
        try:
            os.waitpid(self.pid, 0)  # SIGKILL guarantees this won't block long
        except OSError:
            pass

    # --- command execution -------------------------------------------------

    def run(self, command: str, timeout: int, max_bytes: int,
            normalize: bool = False):
        """Execute one command, yielding NDJSON event dicts.

        Completion is detected with a unique sentinel marker (see below). The
        output size limit is SINGLE-STAGE: when cumulative bytes reach
        max_bytes we emit the remaining allowance, SIGINT the command, report
        a truncated completion and stop reading. A timeout SIGINTs the command
        and reports a recoverable error. After either interrupt we drain
        leftover output up to a sync marker so it can't leak into the next run.
        """
        # Sentinel mechanism: after the user command we print a marker line
        # carrying the command's exit code. We detect completion by scanning
        # the stream for this marker. The marker is unique per call so stray
        # text in normal output cannot trigger a false completion.
        marker = "__DONE_%s__" % uuid.uuid4().hex
        marker_b = marker.encode()

        q = self._subscribe()
        # Fresh command: clear any leftover interrupt flag from a prior run.
        self._interrupt_evt.clear()

        def decode(b):
            t = b.decode("utf-8", "replace")
            return _normalize_text(t) if normalize else t

        try:
            # Write the command, then the marker printf. The leading \n in the
            # printf guarantees the marker starts on its own line; we strip
            # that newline from emitted output below.
            sentinel = "printf '\\n%s %%d\\n' \"$?\"\n" % marker
            payload = command + "\n" + sentinel
            self._write(payload.encode())

            # We emit output incrementally, holding back only the minimal tail
            # that could be the start of a marker split across two reads (see
            # _holdback). Holding back a fixed len(marker) would buffer small
            # streamed lines until completion and defeat live streaming.
            buf = b""
            emitted_total = 0          # bytes actually emitted to client
            deadline = time.monotonic() + timeout
            interrupted = False        # an external /interrupt hit this run
            rearmed = False            # sentinel re-sent after the interrupt

            while True:
                # External /interrupt: SIGINT was already sent. Interactive
                # bash discards the pre-queued sentinel on SIGINT, so re-send
                # it once to get a prompt completion carrying the real $?
                # (130). We drain any duplicate marker after completing.
                if self._interrupt_evt.is_set() and not rearmed:
                    self._interrupt_evt.clear()
                    interrupted = True
                    rearmed = True
                    self._write(sentinel.encode())

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Timeout: stop the command, report recoverable error.
                    self.interrupt()
                    self._drain()
                    yield {"type": "error", "reason": "timeout",
                           "recoverable": True, "timeout": timeout,
                           "bytes": emitted_total}
                    return
                try:
                    data = q.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    continue
                if data is None:
                    # Shell EOF — emit whatever is buffered and finish.
                    out = buf
                    if out:
                        yield {"type": "stdout",
                               "data": decode(out)}
                        emitted_total += len(out)
                    yield {"type": "exit", "code": None, "truncated": False,
                           "bytes": emitted_total}
                    return

                buf += data

                idx = self._find_complete_marker(buf, marker_b)
                if idx is not None:
                    # Found the marker followed by a complete "<code>\n" line.
                    # Output is everything before the marker; strip the single
                    # trailing newline that precedes the marker.
                    out = buf[:idx]
                    if out.endswith(b"\n"):
                        out = out[:-1]
                    if out:
                        # Respect max_bytes on the final flush too.
                        allow = max_bytes - emitted_total
                        if len(out) > allow:
                            out = out[:max(allow, 0)]
                        if out:
                            yield {"type": "stdout",
                                   "data": decode(out)}
                            emitted_total += len(out)
                    # Parse the exit code from the marker line:
                    # "<marker> <code>\n"
                    tail = buf[idx + len(marker_b):]
                    code = self._parse_code(tail)
                    # If we re-armed after an interrupt, the original sentinel
                    # may have survived and emitted a duplicate marker line;
                    # drain it so it can't leak into the next /run.
                    if interrupted:
                        self._drain()
                    yield {"type": "exit", "code": code, "truncated": False,
                           "bytes": emitted_total}
                    return

                # No complete marker yet. Emit what's safe, holding back only:
                # (a) the minimal tail that could be a marker prefix split
                # across two reads (so small lines still stream immediately),
                # and (b) everything from any already-present marker start, so
                # we never emit a partial marker line whose "<code>\n" hasn't
                # arrived yet.
                mpos = buf.find(marker_b)
                if mpos != -1:
                    limit = mpos
                else:
                    limit = len(buf) - self._holdback(buf, marker_b)
                if limit > 0:
                    emit = buf[:limit]
                    buf = buf[limit:]
                    if emit:
                        allow = max_bytes - emitted_total
                        if len(emit) >= allow:
                            # Single-stage size limit reached: emit the final
                            # allowance, interrupt, drain, report truncated.
                            emit = emit[:allow]
                            if emit:
                                yield {"type": "stdout",
                                       "data": decode(emit)}
                                emitted_total += len(emit)
                            self.interrupt()
                            self._drain()
                            yield {"type": "exit", "code": None,
                                   "truncated": True, "bytes": emitted_total,
                                   "reason": "max_bytes_exceeded"}
                            return
                        yield {"type": "stdout",
                               "data": decode(emit)}
                        emitted_total += len(emit)
        finally:
            self._unsubscribe(q)

    @staticmethod
    def _holdback(buf: bytes, marker_b: bytes):
        """Bytes to retain so a marker split across reads is still detectable.

        Returns the length of the longest suffix of buf that is also a prefix
        of the marker (0 if none). Only that tail might be the beginning of a
        marker; everything before it is safe to emit now. This keeps small
        streamed lines flowing instead of buffering a fixed len(marker) bytes.
        """
        maxk = min(len(buf), len(marker_b))
        for k in range(maxk, 0, -1):
            if buf[-k:] == marker_b[:k]:
                return k
        return 0

    @staticmethod
    def _find_complete_marker(buf: bytes, marker_b: bytes):
        """Return the index of a COMPLETE marker line, else None.

        A completion requires the marker followed by " <int>\\n" (the exit
        code line). We only accept the marker once that whole line has
        arrived, so a marker whose code line is still in flight does not
        complete early. (With echo disabled the marker also can't appear in a
        command echo, but requiring the integer makes detection robust either
        way.)
        """
        start = 0
        while True:
            idx = buf.find(marker_b, start)
            if idx == -1:
                return None
            after = buf[idx + len(marker_b):]
            nl = after.find(b"\n")
            if nl == -1:
                # Marker present but its code line hasn't finished arriving.
                return None
            code_field = after[:nl].strip()
            if code_field.lstrip(b"-").isdigit():
                return idx
            # Not a valid code line (e.g. a stray match); skip and keep looking.
            start = idx + len(marker_b)

    @staticmethod
    def _parse_code(tail: bytes):
        """Extract the integer exit code from the bytes after the marker."""
        line = tail.split(b"\n", 1)[0].strip()
        try:
            return int(line)
        except (ValueError, TypeError):
            return None

    def _drain(self):
        """Discard leftover output up to a fresh sync marker.

        After SIGINT the command may still flush bytes (partial output, the
        original __DONE__ marker, a job-killed message). We write a unique sync
        marker and read/discard until we see it, so the next /run on this
        session starts clean. Best-effort: if it doesn't arrive in
        DRAIN_TIMEOUT seconds we give up silently.
        """
        sync = ("__SYNC_%s__" % uuid.uuid4().hex).encode()
        # Match the marker preceded by a real newline byte. The printf OUTPUT
        # is "\n<sync>\n" (0x0a before the marker); if the tty echoes the
        # printf command itself (echo can still be on at input time), that echo
        # contains "...'\n<sync>\n'..." where the "\n" is a literal backslash-n
        # — so "0x0a + sync" matches only the real output, never the echo.
        needle = b"\n" + sync
        q = self._subscribe()
        try:
            self._write(b"printf '\\n%s\\n'\n" % sync)
            buf = b""
            deadline = time.monotonic() + DRAIN_TIMEOUT
            while time.monotonic() < deadline:
                try:
                    data = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if data is None:
                    return
                buf += data
                if needle in buf:
                    return
                # Keep buffer bounded; retain only a possible split-marker tail.
                if len(buf) > 2 * len(needle):
                    buf = buf[-len(needle):]
        finally:
            self._unsubscribe(q)


# --- session registry ------------------------------------------------------

_sessions = {}
_sessions_lock = threading.Lock()
_START_TIME = time.monotonic()


def _create_session() -> Session:
    s = Session()
    with _sessions_lock:
        _sessions[s.id] = s
    return s


def _get_session(sid):
    with _sessions_lock:
        return _sessions.get(sid)


def _remove_session(sid):
    with _sessions_lock:
        return _sessions.pop(sid, None)


def _sweep_idle_sessions():
    """(c) Reap sessions untouched past SESSION_IDLE_TTL.

    A client that crashes without /close would otherwise leak a live bash
    forever. Sessions currently running a command (run lock held) are skipped
    so a long-running job is never reaped mid-flight.
    """
    now = time.monotonic()
    stale = []
    with _sessions_lock:
        for sid, s in list(_sessions.items()):
            if s._run_lock.locked():
                continue
            if now - s.last_activity > SESSION_IDLE_TTL:
                stale.append(_sessions.pop(sid))
    for s in stale:
        s.close()
    return len(stale)


def _sweeper_loop(stop_evt):
    while not stop_evt.wait(SWEEP_INTERVAL):
        try:
            _sweep_idle_sessions()
        except Exception:  # noqa: BLE001 — never let the sweeper die
            pass


# --- HTTP handler ----------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # (h) Quiet by default; log to stderr only when --verbose/env enabled.
        if VERBOSE:
            sys.stderr.write("[%s] %s\n" % (self.address_string(),
                                            fmt % args))

    def _authorized(self) -> bool:
        # (i) Optional shared-token auth. No token configured => open (trusted
        # VPN default). Otherwise require "Authorization: Bearer <token>".
        # Constant-time compare to avoid leaking the token via timing.
        if AUTH_TOKEN is None:
            return True
        got = self.headers.get("Authorization") or ""
        return hmac.compare_digest(got, "Bearer " + AUTH_TOKEN)

    def handle(self):
        # Swallow the noisy traceback socketserver prints when a client resets
        # the connection (expected during /run disconnects and keep-alive).
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0:
            return {}
        if length > MAX_BODY:
            # Consume and discard so keep-alive framing stays intact, then bail.
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, READ_CHUNK))
                if not chunk:
                    break
                remaining -= len(chunk)
            raise ValueError("request body too large")
        body = self.rfile.read(length)
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_404_session(self):
        self._send_json({"error": "no such session"}, status=404)

    def do_GET(self):
        # (g) Read-only introspection endpoints.
        if not self._authorized():
            return self._send_json({"error": "unauthorized"}, status=401)
        try:
            if self.path == "/health":
                self._send_json({
                    "ok": True,
                    "version": VERSION,
                    "uptime_seconds": round(time.monotonic() - _START_TIME, 1),
                    "sessions": len(_sessions),
                })
            elif self.path == "/sessions":
                with _sessions_lock:
                    sessions = [s.info() for s in _sessions.values()]
                self._send_json({"sessions": sessions, "count": len(sessions)})
            else:
                self._send_json({"error": "not found"}, status=404)
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001 — report errors, keep server up
            try:
                self._send_json({"error": str(e)}, status=500)
            except Exception:
                pass

    def do_POST(self):
        # Always consume the request body first: with keep-alive (pooled
        # clients), unread body bytes would corrupt the next request on the
        # same socket.
        try:
            self._body = self._read_json()
        except Exception:
            self._body = {}
        if not self._authorized():
            return self._send_json({"error": "unauthorized"}, status=401)
        try:
            if self.path == "/session":
                self._handle_session()
            elif self.path == "/run":
                self._handle_run()
            elif self.path == "/interrupt":
                self._handle_interrupt()
            elif self.path == "/close":
                self._handle_close()
            else:
                self._send_json({"error": "not found"}, status=404)
        except BrokenPipeError:
            # Client disconnected mid-stream; nothing to do.
            pass
        except Exception as e:  # noqa: BLE001 — report errors, keep server up
            try:
                self._send_json({"error": str(e)}, status=500)
            except Exception:
                pass

    def _handle_session(self):
        # Soft cap to avoid an accidental fork-bomb of shells. The sweeper
        # reaps idle ones; a client hitting the cap should /close some first.
        if MAX_SESSIONS:
            with _sessions_lock:
                if len(_sessions) >= MAX_SESSIONS:
                    return self._send_json(
                        {"error": "too many sessions"}, status=503)
        s = _create_session()
        self._send_json({"session": s.id})

    def _handle_interrupt(self):
        body = self._body
        s = _get_session(body.get("session"))
        if s is None:
            return self._send_404_session()
        s.touch()
        # Only signal when a command is actually running; sending Ctrl-C to an
        # idle shell would leave a stray "^C" that leaks into the next /run.
        if s._run_lock.locked():
            s.interrupt()
        self._send_json({"interrupted": True})

    def _handle_close(self):
        body = self._body
        s = _remove_session(body.get("session"))
        if s is None:
            return self._send_404_session()
        s.close()
        self._send_json({"closed": True})

    def _handle_run(self):
        body = self._body
        s = _get_session(body.get("session"))
        if s is None:
            return self._send_404_session()
        try:
            command = str(body.get("command", ""))
            timeout = int(body.get("timeout") or DEFAULT_TIMEOUT)
            max_bytes = int(body.get("max_bytes") or DEFAULT_MAX_BYTES)
            normalize = bool(body.get("normalize", False))  # (e)
            if timeout <= 0 or max_bytes <= 0:
                raise ValueError("timeout and max_bytes must be positive")
        except (TypeError, ValueError) as e:
            return self._send_json({"error": "bad request: %s" % e},
                                   status=400)

        # (a) A session is used serially. Refuse a concurrent /run rather than
        # let two commands interleave on the same shell and cross sentinels.
        if not s._run_lock.acquire(blocking=False):
            return self._send_json({"error": "session busy"}, status=409)
        try:
            s.touch()
            # Stream NDJSON via chunked transfer encoding. No Content-Length;
            # flush after every line so output appears incrementally.
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            # Past this point the response is committed: never raise out (that
            # would make do_POST try to send a second HTTP response onto the
            # same socket). On any error, clean the shell and end the stream.
            gen = s.run(command, timeout=timeout, max_bytes=max_bytes,
                        normalize=normalize)
            try:
                for event in gen:
                    line = (json.dumps(event) + "\n").encode("utf-8")
                    self._write_chunk(line)
                self._write_chunk(b"")  # terminating zero-length chunk
            except (BrokenPipeError, ConnectionResetError):
                # (b) Client vanished mid-stream. The command is still running
                # on the shell; stop it and drain so its output can't leak into
                # the next /run on this session.
                self._cleanup_after_stream(s)
            except Exception:  # noqa: BLE001 — stream is committed; just stop
                self._cleanup_after_stream(s)
            finally:
                gen.close()
        finally:
            s.touch()
            s._run_lock.release()

    @staticmethod
    def _cleanup_after_stream(s):
        try:
            s.interrupt()
            s._drain()
        except Exception:  # noqa: BLE001
            pass

    def _write_chunk(self, data: bytes):
        """Write one HTTP/1.1 chunked-transfer chunk and flush."""
        self.wfile.write(b"%X\r\n" % len(data))
        if data:
            self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()


def _parse_args(argv):
    """(f) Config from CLI flags, falling back to env vars, then defaults."""
    env = os.environ.get
    p = argparse.ArgumentParser(
        description="Persistent stateful bash-over-HTTP execution server.")
    p.add_argument("--host", default=env("BASH_HTTP_HOST", HOST),
                   help="bind address (default %(default)s; keep loopback)")
    p.add_argument("--port", type=int,
                   default=int(env("BASH_HTTP_PORT", PORT)),
                   help="bind port (default %(default)s)")
    p.add_argument("--token", default=env("BASH_HTTP_TOKEN"),
                   help="require 'Authorization: Bearer <token>' on requests")
    p.add_argument("--idle-ttl", type=float,
                   default=float(env("BASH_HTTP_IDLE_TTL", SESSION_IDLE_TTL)),
                   help="seconds before an idle session is reaped "
                        "(default %(default)s)")
    p.add_argument("--max-sessions", type=int,
                   default=int(env("BASH_HTTP_MAX_SESSIONS", MAX_SESSIONS)),
                   help="max concurrent sessions, 0=unlimited "
                        "(default %(default)s)")
    p.add_argument("--verbose", action="store_true",
                   default=env("BASH_HTTP_VERBOSE", "") not in ("", "0"),
                   help="log requests/errors to stderr")
    return p.parse_args(argv)


def main(argv=None):
    global AUTH_TOKEN, VERBOSE, SESSION_IDLE_TTL, MAX_SESSIONS
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    AUTH_TOKEN = args.token or None
    VERBOSE = args.verbose
    SESSION_IDLE_TTL = args.idle_ttl
    MAX_SESSIONS = args.max_sessions

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("bash-over-http server listening on http://%s:%d%s%s"
          % (args.host, args.port,
             "  (auth: token)" if AUTH_TOKEN else "  (auth: none)",
             "  (verbose)" if VERBOSE else ""), flush=True)

    # (c) Background sweeper for idle/orphan sessions.
    stop_evt = threading.Event()
    sweeper = threading.Thread(target=_sweeper_loop, args=(stop_evt,),
                               daemon=True)
    sweeper.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        with _sessions_lock:
            sessions = list(_sessions.values())
            _sessions.clear()
        for s in sessions:
            s.close()
        server.server_close()


if __name__ == "__main__":
    main()
