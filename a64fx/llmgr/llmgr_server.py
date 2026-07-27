#!/usr/bin/env python3
"""llmgr -- control HTTP port for the A64FX LLM runners.

Runs on the head node of an interactive or batch allocation and turns that
allocation into an interactive development box: compile, stage weights,
start/stop `mpiexec` runners, tail their logs, run inference against them, and
capture fapp profiles -- all over one HTTP port, reachable from a Fugaku
frontend through an SSH reverse tunnel (see pjsub_llmgr_12n.sh).

Design notes
------------
* llmgr is a **supervisor**, not a rank.  It is a plain Python process outside
  MPI; runners are spawned as `setsid` child process groups.  Teardown needs
  more than killpg, because mpiexec detaches via plexec -- see
  Child._sweep_tree().  No C code was changed.
* Inference is **proxied** to the runner's own HTTP server (laguna's --serve,
  laguna_serve.inc).  That deliberately keeps the rank-0 collective invariant
  ("every accepted connection reaches lsrv_bcast exactly once, or the other
  ranks strand inside the allreduce") inside the C code where it already holds.
* The same invariant means a request is never free: llmgr must NOT poll a
  runner to discover whether it is up, because rank 0 accepts as soon as it has
  loaded while its peers may be a minute behind, and probing that window
  desynchronises every rank.  Readiness is therefore read from the runner's own
  per-rank files -- see Adapter.readiness() and Child._check_ready().
* Arbitrary shell -- the actual point of the exercise -- is delegated wholesale
  to tools/bash_http_server.py under /bash/*, which already implements
  persistent PTY sessions with NDJSON streaming.

Standard library only.  Binds loopback.  Bearer auth is optional (set
LLMGR_TOKEN); unset, every request is accepted, which is the intended mode on
Fugaku's private fabric where the port is reachable only through an ssh tunnel.
"""

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import HTTPServer
from socketserver import ThreadingMixIn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)

import bash_http_server as bhs   # noqa: E402  (path set above)
import models                    # noqa: E402

try:
    from http.server import ThreadingHTTPServer
except ImportError:                                   # pragma: no cover
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

VERSION = "1.0"
DEFAULT_PORT = 21274          # bash-over-http owns 21264; stay clear of it
LOG_DIR = os.path.join(HERE, "logs")
STATE_DIR = os.path.join(HERE, "state")
PROF_DIR = os.path.join(HERE, "prof")

# After every rank reports "loaded", wait this long before admitting traffic.
# Cheap insurance: the readiness signal is a log line, not a handshake.
READY_SETTLE = 10.0
REAP_INTERVAL = 1.0
STOP_GRACE = 20.0             # seconds between SIGTERM and SIGKILL of a group
TAIL_DEFAULT = 200
FOLLOW_POLL = 0.4
UTOFU_DIR = os.path.join(REPO, "a64fx", "utofu-tests")

_START_TIME = time.monotonic()
_children = {}
_children_lock = threading.Lock()
_next_id = [0]
_stop_evt = threading.Event()


def _log(msg):
    if bhs.VERBOSE:
        sys.stderr.write("[llmgr] %s\n" % msg)
        sys.stderr.flush()


# --------------------------------------------------------------------------
# children


class Child:
    """One supervised process group: a build, a stage, a runner, a profile.

    `kind` is "serve" for long-lived runners with an HTTP port (llmgr waits
    passively for them to become ready, then proxies /generate to them) and
    "oneshot" for everything that runs to completion.
    """

    def __init__(self, cid, kind, label, argv, env, cwd, port=None, meta=None):
        self.id = cid
        self.kind = kind
        self.label = label
        self.argv = argv
        self.env = env
        self.cwd = cwd
        self.port = port
        self.meta = meta or {}
        self.dir = os.path.join(LOG_DIR, cid)
        self.log_path = os.path.join(self.dir, "log.txt")
        self.state = "starting"
        self.exit_code = None
        self.error = None
        self.started = time.time()
        self.ended = None
        self.proc = None
        self._logf = None
        self._loglock = threading.Lock()
        self._rank_prefix = os.path.join(self.dir, "rank")
        self._rank_pos = {}
        self._ready_since = None
        self.ready_detail = None

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        # Child ids restart at 1 with each llmgr process, so a directory of this
        # name may belong to an earlier instance. Its rank.* files would be read
        # from offset 0 and folded into this child's log -- a previous run's
        # output appearing in a fresh child's log. Set the old one aside rather
        # than deleting it; those logs are usually why someone restarted llmgr.
        if os.path.exists(self.dir):
            try:
                os.rename(self.dir, "%s.prev-%d"
                          % (self.dir, int(os.path.getmtime(self.dir))))
            except OSError:
                shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir, exist_ok=True)
        env = dict(os.environ)
        env.update(self.env or {})
        # Fugaku's mpiexec discards rank stdout as far as the launching process
        # is concerned, so the launchers are asked to add -of-proc with this
        # prefix (they honour MPIEXEC_OF_PROC; unset, they behave as before).
        # _collect_ranks then folds those files into this child's single log.
        env.setdefault("MPIEXEC_OF_PROC", self._rank_prefix)
        with open(os.path.join(self.dir, "cmd.txt"), "w") as f:
            f.write("cwd: %s\n" % self.cwd)
            f.write("env: %s\n" % json.dumps(self.env or {}, sort_keys=True))
            f.write("argv: %s\n" % " ".join(shlex.quote(a) for a in self.argv))
        self._logf = open(self.log_path, "ab", buffering=0)
        try:
            self.proc = subprocess.Popen(
                self.argv, cwd=self.cwd, env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                # Own process group. Necessary but not sufficient: mpiexec
                # re-execs into plexec, which leaves this session, so teardown
                # also needs _sweep_tree(). See its comment.
                preexec_fn=os.setsid,
            )
        except OSError as e:
            self.state = "failed"
            self.error = str(e)
            self.ended = time.time()
            self._close_log()
            raise
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._collect_ranks, daemon=True).start()
        # A oneshot has nothing to become ready for; a serve child is watched.
        self.state = "starting" if self.kind == "serve" else "running"
        _log("started %s pid=%d %s" % (self.id, self.proc.pid, self.label))

    # -- log plumbing ------------------------------------------------------

    def _write_log(self, data):
        with self._loglock:
            if self._logf is not None:
                try:
                    self._logf.write(data)
                except (ValueError, OSError):
                    pass

    def _pump(self):
        """Drain the child's own stdout (the launcher's echoes) into the log.

        A pipe rather than a file handle, so this thread and _collect_ranks can
        interleave into one log without fighting over the file offset.
        """
        try:
            for line in iter(self.proc.stdout.readline, b""):
                self._write_log(line)
        except (ValueError, OSError):
            pass
        finally:
            try:
                self.proc.stdout.close()
            except Exception:
                pass

    def _collect_ranks(self):
        """Fold mpiexec's -of-proc per-rank files into the child's log.

        Files appear as <dir>/rank.<step>.<rank>; each is followed from its last
        offset. Rank 0 is usually the interesting one, so lines are tagged.
        """
        while True:
            done = self.state in ("exited", "stopped", "failed")
            try:
                names = [n for n in os.listdir(self.dir)
                         if n.startswith("rank.")]
            except OSError:
                names = []
            for n in sorted(names):
                p = os.path.join(self.dir, n)
                pos = self._rank_pos.get(n, 0)
                try:
                    size = os.path.getsize(p)
                    if size <= pos:
                        continue
                    with open(p, "rb") as f:
                        f.seek(pos)
                        data = f.read(size - pos)
                    self._rank_pos[n] = size
                except OSError:
                    continue
                tag = ("[%s] " % n.split(".")[-1]).encode()
                self._write_log(b"".join(
                    tag + ln + b"\n" for ln in data.split(b"\n") if ln))
            if done:
                self._append_result_file()
                self._close_log()
                return
            time.sleep(0.5)

    def _append_result_file(self):
        """Fold a one-shot's result file into its log, once, on exit.

        Runners like gemma4_pp_runner answer into a file rather than stdout, so
        without this the log ends at "running PP pipeline" and the actual
        result is somewhere the caller has to know about.
        """
        p = self.meta.get("result_path")
        if not p or self.meta.get("result_captured"):
            return
        self.meta["result_captured"] = True
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError as e:
            self._write_log(("\n[llmgr] result file %s unreadable: %s\n"
                             % (p, e)).encode())
            return
        self._write_log(("\n[llmgr] --- result file %s ---\n" % p).encode())
        self._write_log(data if data.endswith(b"\n") else data + b"\n")

    def poll(self):
        """Update state from the OS. Called by the reaper thread."""
        if self.proc is None or self.state in ("exited", "stopped", "failed"):
            return
        rc = self.proc.poll()
        if rc is not None:
            self.exit_code = rc
            self.ended = time.time()
            self.state = "stopped" if self.state == "stopping" else "exited"
            # NB: the log is closed by _collect_ranks after its final sweep, so
            # the last rank-file bytes are not lost to a race with exit.
            _log("child %s exited rc=%s" % (self.id, rc))
            return
        if self.kind == "serve" and self.state == "starting":
            self._check_ready()

    def _check_ready(self):
        """Passive readiness: read the runner's own files, never probe it.

        An HTTP request to rank 0 is not free -- every accepted connection
        drives a collective across all ranks -- so probing a runner whose
        slowest rank is still loading desynchronises the job (see
        Adapter.readiness).  Instead the adapter reports when every rank has
        logged that it finished loading; then we still wait SETTLE seconds
        before admitting traffic.
        """
        adapter = self.meta.get("adapter")
        if adapter is None:
            return
        try:
            ok, detail = adapter.readiness(self.meta.get("config") or {},
                                           self.started)
        except Exception as e:                    # noqa: BLE001
            ok, detail = False, "readiness check failed: %s" % e
        self.ready_detail = detail
        if not ok:
            self._ready_since = None
            return
        if self._ready_since is None:
            self._ready_since = time.time()
            _log("child %s: all ranks loaded (%s); settling" % (self.id, detail))
            return
        if time.time() - self._ready_since >= READY_SETTLE:
            self.state = "ready"
            _log("child %s ready on port %s (%s)" % (self.id, self.port, detail))

    def stop(self, grace=STOP_GRACE):
        """Ask nicely, then insist.

        For a laguna serve child, POST /shutdown first: that stops every rank
        cleanly through the runner's own collective, which SIGTERM to the
        mpiexec group cannot do without risking half-exited ranks.
        """
        if self.proc is None or self.state in ("exited", "stopped", "failed"):
            return False
        self.state = "stopping"
        if self.kind == "serve" and self.port:
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:%d/shutdown" % self.port,
                    data=b"{}", method="POST")
                urllib.request.urlopen(req, timeout=5.0).read()
                _log("child %s: sent /shutdown" % self.id)
            except Exception as e:
                _log("child %s: /shutdown failed (%s), falling back to signals"
                     % (self.id, e))
        deadline = time.time() + grace
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            time.sleep(0.3)
        if self.proc.poll() is None:
            self._signal_group(signal.SIGTERM)
            deadline = time.time() + grace
            while time.time() < deadline and self.proc.poll() is None:
                time.sleep(0.3)
        if self.proc.poll() is None:
            _log("child %s: SIGKILL" % self.id)
            self._signal_group(signal.SIGKILL)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        # The direct child being gone does NOT mean the ranks are: mpiexec
        # detaches via plexec, so sweep by command line as well.
        self._sweep_tree()
        self.exit_code = self.proc.poll()
        self.ended = time.time()
        self.state = "stopped"
        return True

    def _signal_group(self, sig):
        try:
            os.killpg(os.getpgid(self.proc.pid), sig)
        except (ProcessLookupError, PermissionError) as e:
            _log("child %s: killpg %s failed: %s" % (self.id, sig, e))

    # -- reaping the mpiexec tree -----------------------------------------
    #
    # killpg alone is NOT enough on Fugaku.  mpiexec re-execs into plexec, which
    # puts itself in its own session, so the launcher's process group no longer
    # contains the ranks: the direct child dies, reports exit -15, and the whole
    # mpiexec/plexec/fapp/runner tree keeps running and keeps holding the nodes.
    # Worse, plexec and mpiexec ignore SIGTERM; only SIGKILL moves them, and a
    # surviving plexec makes the next mpiexec fail with
    # "PLE 0008 plexec must be started sequentially".
    #
    # The survivors are found by their command line rather than by process tree,
    # since the tree is exactly what got detached.  Every mpiexec llmgr launches
    # carries "-of-proc <this child's log dir>/rank", which is unique per child,
    # so it is a precise handle on this child's launchers.  The leaf ranks do not
    # carry it, so they are matched by the runner binary path instead -- but only
    # after the launchers are gone, so a concurrently running child's ranks are
    # never touched.

    def _pids_matching(self, needle):
        out = []
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            if pid == os.getpid():
                continue
            try:
                with open("/proc/%d/cmdline" % pid, "rb") as f:
                    cmd = f.read().replace(b"\0", b" ").decode(
                        "utf-8", "replace")
            except (OSError, IOError):
                continue
            if needle in cmd:
                out.append(pid)
        return out

    def _sweep_tree(self):
        """Kill anything left over from this child's mpiexec, group or not."""
        needles = [self._rank_prefix]
        leaf = self.meta.get("runner_bin")
        for needle in needles:
            pids = self._pids_matching(needle)
            if not pids:
                continue
            _log("child %s: %d detached launcher(s) survived killpg; SIGKILL %s"
                 % (self.id, len(pids), pids))
            # No SIGTERM pass: mpiexec and plexec ignore it, and every second
            # spent waiting is a second the nodes stay locked.
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        if leaf:
            time.sleep(1.0)
            for pid in self._pids_matching(leaf):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

    def _close_log(self):
        if self._logf is not None:
            try:
                self._logf.close()
            except Exception:
                pass
            self._logf = None

    # -- reporting ---------------------------------------------------------

    def info(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "state": self.state,
            "pid": self.proc.pid if self.proc else None,
            "port": self.port,
            "exit_code": self.exit_code,
            "error": self.error,
            "started": self.started,
            "uptime_seconds": round((self.ended or time.time()) - self.started, 1),
            "ready_detail": self.ready_detail,
            "log": self.log_path,
            "cwd": self.cwd,
            "argv": self.argv,
            "env": self.env,
            "meta": {k: v for k, v in self.meta.items() if k != "adapter"},
        }


def _new_child(kind, label, argv, env, cwd, port=None, meta=None):
    with _children_lock:
        _next_id[0] += 1
        cid = "%s-%d" % (kind if kind != "serve" else "run", _next_id[0])
        c = Child(cid, kind, label, argv, env, cwd, port=port, meta=meta)
        _children[cid] = c
    c.start()
    return c


def _get_child(cid):
    with _children_lock:
        return _children.get(cid)


def _reaper_loop():
    while not _stop_evt.wait(REAP_INTERVAL):
        with _children_lock:
            children = list(_children.values())
        for c in children:
            try:
                c.poll()
            except Exception as e:            # noqa: BLE001 -- never die
                _log("reaper: %s: %s" % (c.id, e))


def _runner_bin(adapter, cfg):
    """Leaf binary path, for the detached-rank sweep. Best effort."""
    try:
        return adapter.runner_bin(cfg)
    except (NotImplementedError, models.ConfigError):
        return None


def _port_in_use(port):
    """A serve child already holding this port would make readiness ambiguous."""
    with _children_lock:
        for c in _children.values():
            if c.port == port and c.state in ("starting", "ready", "stopping"):
                return c
    return None


# --------------------------------------------------------------------------
# helpers


def _run_sync(argv, cwd=None, env=None, timeout=60):
    """Short blocking command (node probes, topology). Never for runners."""
    e = dict(os.environ)
    e.update(env or {})
    try:
        p = subprocess.run(argv, cwd=cwd, env=e, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return -1, "timeout after %ss" % timeout
    except OSError as e2:
        return -1, str(e2)


def _fanout(script, np_=None, timeout=120):
    """Run a shell snippet on every node of the allocation via mpiexec.

    Fugaku's mpiexec does NOT deliver rank stdout to the launching process's
    pipe -- capturing it with subprocess yields nothing at all (which is why
    every runner in this repo writes rank files instead of printing).  The
    working mechanism is `-of-proc PREFIX`, which writes PREFIX.<step>.<rank>.
    That prefix must live on the SHARED filesystem: /tmp and /local are
    node-local, so only rank 0's file would ever be visible here.

    Needs the nodes to be free: while a serve runner owns them, mpiexec will
    block or fail.  That is reported, not hidden.
    """
    np_ = np_ or int(os.environ.get("PJM_MPI_PROC", "12"))
    d = tempfile.mkdtemp(prefix="fanout-", dir=STATE_DIR)
    try:
        prefix = os.path.join(d, "o")
        rc, txt = _run_sync(["mpiexec", "-np", str(np_), "-of-proc", prefix,
                             "sh", "-c", script], cwd=HERE, timeout=timeout)
        lines = []
        for name in sorted(os.listdir(d)):
            try:
                with open(os.path.join(d, name)) as f:
                    lines.extend(f.read().splitlines())
            except OSError:
                pass
        if rc != 0 and not lines:
            return rc, txt
        return rc, "\n".join(lines)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _tail_file(path, nlines):
    """Last nlines of a file, read from the end so a 1GB log stays cheap."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        block = 8192
        data = b""
        while size > 0 and data.count(b"\n") <= nlines:
            step = min(block, size)
            size -= step
            f.seek(size)
            data = f.read(step) + data
    text = data.decode("utf-8", "replace")
    lines = text.splitlines()
    return "\n".join(lines[-nlines:])


def _job_info():
    return {k: v for k, v in os.environ.items() if k.startswith("PJM_")}


# --------------------------------------------------------------------------
# HTTP


class Handler(bhs.Handler):
    """llmgr routes, plus /bash/* delegated to bash_http_server.

    bash_http_server's dispatch is purely `self.path`-based, so delegation is
    just prefix-stripping and a super() call -- its session table, PTY handling
    and NDJSON streaming come along untouched. Auth is shared: _authorized() is
    inherited and reads bhs.AUTH_TOKEN, which main() sets.
    """

    server_version = "llmgr/" + VERSION

    # -- plumbing ----------------------------------------------------------

    def _split(self):
        u = urllib.parse.urlsplit(self.path)
        return u.path, urllib.parse.parse_qs(u.query)

    def _bash_delegate(self, method):
        """Rewrite /bash/foo -> /foo and hand off to bash_http_server."""
        u = urllib.parse.urlsplit(self.path)
        sub = u.path[len("/bash"):] or "/"
        self.path = urllib.parse.urlunsplit(("", "", sub, u.query, ""))
        return method()

    def _err(self, msg, status=400):
        self._send_json({"error": msg}, status=status)

    def _dispatch(self, table, path, *args):
        for prefix, fn in table:
            if callable(prefix):
                m = prefix(path)
                if m is not None:
                    return fn(m, *args)
            elif path == prefix:
                return fn(*args)
        return self._err("not found: %s" % path, status=404)

    # -- GET ---------------------------------------------------------------

    def do_GET(self):
        if not self._authorized():
            return self._send_json({"error": "unauthorized"}, status=401)
        if self.path == "/bash" or self.path.startswith("/bash/"):
            return self._bash_delegate(super().do_GET)
        try:
            path, q = self._split()
            if path in ("/", "/health"):
                return self._get_health()
            if path == "/models":
                return self._send_json(models.describe())
            if path == "/nodes":
                return self._get_nodes(q)
            if path == "/runner":
                return self._get_runners()
            if path == "/stage/status":
                return self._get_stage_status(q)
            if path.startswith("/runner/") and path.endswith("/log"):
                return self._get_log(path.split("/")[2], q)
            if path.startswith("/profile/") and path.endswith("/artifacts"):
                return self._get_artifacts(path.split("/")[2])
            return self._err("not found: %s" % path, status=404)
        except BrokenPipeError:
            pass
        except models.ConfigError as e:
            self._err(str(e))
        except Exception as e:                    # noqa: BLE001 -- stay up
            try:
                self._err(str(e), status=500)
            except Exception:
                pass

    def _get_health(self):
        with _children_lock:
            children = [c.info() for c in _children.values()]
        self._send_json({
            "ok": True,
            "service": "llmgr",
            "version": VERSION,
            "uptime_seconds": round(time.monotonic() - _START_TIME, 1),
            "host": os.uname().nodename,
            "job_id": os.environ.get("PJM_JOBID"),
            "nodes": os.environ.get("PJM_NODE"),
            "mpi_proc": os.environ.get("PJM_MPI_PROC"),
            "repo": REPO,
            "children": children,
            "bash_sessions": len(bhs._sessions),
        })

    def _get_nodes(self, q):
        out = {
            "head": os.uname().nodename,
            "job": _job_info(),
            "local": {},
        }
        for label, argv in (("uptime", ["uptime"]),
                            ("meminfo", ["sh", "-c",
                                         "grep -E 'MemTotal|MemAvailable' /proc/meminfo"]),
                            ("local_fs", ["sh", "-c", "df -h /local 2>&1 | tail -1"])):
            rc, txt = _run_sync(argv, timeout=10)
            out["local"][label] = txt.strip()
        if q.get("fanout", ["1"])[0] not in ("0", "false", ""):
            rc, txt = _fanout(
                'echo "rank=${PMIX_RANK:-?} host=$(hostname) '
                'mem_avail_kb=$(awk \'/MemAvailable/{print $2}\' /proc/meminfo) '
                'local=$(df -Pk /local 2>/dev/null | tail -1 | awk \'{print $3"/"$2}\')"',
                timeout=int(q.get("timeout", ["120"])[0]))
            out["fanout_rc"] = rc
            out["fanout"] = txt.strip().splitlines()
            if rc != 0:
                out["fanout_note"] = (
                    "mpiexec failed -- the allocation may be busy with a "
                    "running runner; stop it or pass ?fanout=0")
        topo = os.path.join(HERE, "tofu_topo.txt")
        if os.path.exists(topo):
            with open(topo) as f:
                out["topo"] = f.read().splitlines()
        self._send_json(out)

    def _get_runners(self):
        with _children_lock:
            children = [c.info() for c in _children.values()]
        self._send_json({"children": children, "count": len(children)})

    def _get_log(self, cid, q):
        c = _get_child(cid)
        if c is None:
            return self._err("no such child: %s" % cid, status=404)
        nlines = int(q.get("tail", [str(TAIL_DEFAULT)])[0])
        follow = q.get("follow", ["0"])[0] not in ("0", "false", "")
        if not follow:
            return self._send_json({"id": cid, "state": c.state,
                                    "exit_code": c.exit_code,
                                    "log": _tail_file(c.log_path, nlines)})
        # Streaming tail: chunked text, ends when the child does.
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self._write_chunk(_tail_file(c.log_path, nlines).encode() + b"\n")
        pos = os.path.getsize(c.log_path) if os.path.exists(c.log_path) else 0
        deadline = time.time() + float(q.get("timeout", ["3600"])[0])
        while time.time() < deadline:
            try:
                size = os.path.getsize(c.log_path)
            except OSError:
                size = pos
            if size > pos:
                with open(c.log_path, "rb") as f:
                    f.seek(pos)
                    data = f.read(size - pos)
                pos = size
                self._write_chunk(data)
            elif c.state in ("exited", "stopped", "failed"):
                break
            else:
                time.sleep(FOLLOW_POLL)
        self._write_chunk(("\n[llmgr] child %s state=%s exit_code=%s\n"
                           % (cid, c.state, c.exit_code)).encode())
        self._write_chunk(b"")

    def _get_stage_status(self, q):
        model = q.get("model", ["laguna"])[0]
        adapter = models.get(model)
        cfg = {k: v[0] for k, v in q.items()}
        stage_dir = adapter.stage_dir(cfg)
        out = {"model": model, "stage_dir": stage_dir,
               "model_dir": adapter.model_dir(cfg)}
        if q.get("fanout", ["1"])[0] not in ("0", "false", ""):
            rc, txt = _fanout(
                'd=%s; echo "rank=${PMIX_RANK:-?} host=$(hostname) '
                'files=$(ls -1 $d 2>/dev/null | wc -l) '
                'bytes=$(du -sb $d 2>/dev/null | cut -f1)"'
                % shlex.quote(stage_dir), timeout=180)
            out["fanout_rc"] = rc
            out["nodes"] = txt.strip().splitlines()
            if rc != 0:
                out["note"] = ("mpiexec failed -- allocation busy? "
                               "pass ?fanout=0 for the head node only")
        else:
            out["head"] = _run_sync(
                ["sh", "-c", "ls -1 %s 2>/dev/null | wc -l"
                 % shlex.quote(stage_dir)])[1].strip()
        self._send_json(out)

    def _get_artifacts(self, cid):
        c = _get_child(cid)
        if c is None:
            return self._err("no such child: %s" % cid, status=404)
        d = c.meta.get("prof_dir")
        if not d or not os.path.isdir(d):
            return self._err("child %s has no profile directory" % cid, status=404)
        files = []
        for root, _dirs, names in os.walk(d):
            for n in sorted(names):
                p = os.path.join(root, n)
                try:
                    files.append({"path": p, "size": os.path.getsize(p)})
                except OSError:
                    pass
        self._send_json({"id": cid, "state": c.state, "exit_code": c.exit_code,
                         "prof_dir": d, "artifacts": files,
                         "count": len(files)})

    # -- POST --------------------------------------------------------------

    def do_POST(self):
        if self.path == "/bash" or self.path.startswith("/bash/"):
            return self._bash_delegate(super().do_POST)
        try:
            self._body = self._read_json()
        except Exception:
            self._body = {}
        if not self._authorized():
            return self._send_json({"error": "unauthorized"}, status=401)
        try:
            path, _q = self._split()
            body = self._body if isinstance(self._body, dict) else {}
            if path == "/build":
                return self._post_simple(body, "build")
            if path == "/stage":
                return self._post_simple(body, "stage")
            if path == "/runner/start":
                return self._post_runner_start(body)
            if path == "/runner/stop":
                return self._post_runner_stop(body)
            if path == "/generate":
                return self._post_generate(body)
            if path == "/profile":
                return self._post_profile(body)
            if path == "/kv":
                return self._post_kv(body)
            if path == "/shutdown":
                return self._post_shutdown(body)
            return self._err("not found: %s" % path, status=404)
        except BrokenPipeError:
            pass
        except models.ConfigError as e:
            self._err(str(e))
        except NotImplementedError as e:
            self._err(str(e))
        except Exception as e:                    # noqa: BLE001 -- stay up
            try:
                self._err(str(e), status=500)
            except Exception:
                pass

    def _post_simple(self, body, mode):
        """/build and /stage: one-shot, identical shape."""
        adapter = models.get(body.get("model", "laguna"))
        argv, env, cwd = adapter.launch(mode, body)
        c = _new_child("oneshot", "%s:%s" % (mode, adapter.name),
                       argv, env, cwd,
                       meta={"model": adapter.name, "mode": mode})
        self._send_json({"id": c.id, "state": c.state, "log": c.log_path,
                         "argv": argv}, status=202)

    def _post_runner_start(self, body):
        adapter = models.get(body.get("model", "laguna"))
        mode = body.get("mode") or ("serve" if adapter.supports_serve else "generate")
        if mode not in ("serve", "generate"):
            return self._err("mode must be serve or generate")
        if mode == "serve" and not adapter.supports_serve:
            return self._err("%s has no serve mode -- use mode=generate "
                             "(one-shot) and read the log" % adapter.name)
        if mode == "serve":
            port = models._int(body, "port", required=True)
            busy = _port_in_use(port)
            if busy:
                return self._err("port %d already held by child %s"
                                 % (port, busy.id), status=409)
            argv, env, cwd = adapter.launch("serve", body)
            c = _new_child("serve", "serve:%s" % adapter.name, argv, env, cwd,
                           port=port, meta={"model": adapter.name,
                                            "adapter": adapter,
                                            "runner_bin": _runner_bin(adapter, body),
                                            "config": body})
        else:
            argv, env, cwd = adapter.launch("generate", body)
            c = _new_child("oneshot", "generate:%s" % adapter.name,
                           argv, env, cwd,
                           meta={"model": adapter.name, "config": body,
                                 "runner_bin": _runner_bin(adapter, body),
                                 "result_path": adapter.result_path(body)})
        self._send_json({"id": c.id, "state": c.state, "port": c.port,
                         "log": c.log_path, "argv": argv}, status=202)

    def _post_runner_stop(self, body):
        cid = body.get("id")
        if not cid:
            return self._err("missing 'id'")
        c = _get_child(cid)
        if c is None:
            return self._err("no such child: %s" % cid, status=404)
        stopped = c.stop(grace=float(body.get("grace", STOP_GRACE)))
        self._send_json({"id": cid, "stopped": stopped, "state": c.state,
                         "exit_code": c.exit_code})

    def _post_generate(self, body):
        """Proxy to the runner's own HTTP server (ids-in / ids-out)."""
        cid = body.pop("id", None)
        if not cid:
            return self._err("missing 'id' (the serve child to talk to)")
        c = _get_child(cid)
        if c is None:
            return self._err("no such child: %s" % cid, status=404)
        if c.kind != "serve":
            return self._err("child %s is not a serve runner" % cid)
        if c.state != "ready":
            return self._err("child %s is %s, not ready" % (cid, c.state),
                             status=503)
        timeout = float(body.pop("timeout", 900.0))
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:%d/generate" % c.port, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = r.read()
                status = r.getcode()
        except urllib.error.HTTPError as e:
            payload, status = e.read(), e.code
        except Exception as e:
            return self._err("runner %s unreachable: %s" % (cid, e), status=502)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _post_profile(self, body):
        """fapp-wrapped run of the per-rank binary.

        fapp collects one PMU dataset per process, so it must wrap the runner
        binary inside mpiexec -- not the launcher script. Consequence: weights
        must already be staged and a tofu_topo.txt is generated first.
        """
        adapter = models.get(body.get("model", "laguna"))
        runner = adapter.runner_bin(body)
        if not os.path.exists(runner):
            return self._err("runner binary missing: %s (POST /build first)"
                             % runner)
        argv = adapter.profile_argv(body)
        event = str(body.get("event", "statistics"))
        np_ = models._int(body, "np", adapter.default_np())
        tag = body.get("tag") or time.strftime("%Y%m%d-%H%M%S")
        work = os.path.join(PROF_DIR, "%s_%s" % (adapter.name, tag))
        os.makedirs(work, exist_ok=True)

        rc, txt = _run_sync(["sh", "-c",
                             "make -C %s tofu_topo_helper >/dev/null 2>&1; "
                             "mpiexec -np %d %s/tofu_topo_helper"
                             % (shlex.quote(UTOFU_DIR), np_,
                                shlex.quote(UTOFU_DIR))],
                            cwd=work, timeout=300)
        # tofu_topo_helper writes tofu_topo.txt into cwd; its stdout is lost to
        # the mpiexec behaviour described in _fanout, so check the file, not rc.
        if rc == 0 and not os.path.exists(os.path.join(work, "tofu_topo.txt")):
            rc, txt = 1, "tofu_topo.txt was not produced\n" + txt
        if rc != 0:
            return self._err("topology discovery failed: %s" % txt[-2000:],
                             status=500)

        quoted = " ".join(shlex.quote(a) for a in argv)
        script = (
            'r=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}}; '
            'd="prof_rank${r}_pa1"; rm -rf "$d"; '
            'fapp -C -d "$d" -Icpupa,nompi -Hevent=%s %s %s'
            % (shlex.quote(event), shlex.quote(runner), quoted))
        # The export pass turns each per-rank dataset into CSV + text.
        export = (
            'for d in prof_rank*_pa1; do [ -d "$d" ] || continue; b=${d%%_pa1}; '
            'fapppx -A -d "$d" -Icpupa,nompi -tcsv  -o "${b}.csv" '
            '>/dev/null 2>&1 || true; '
            'fapppx -A -d "$d" -Icpupa,nompi -ttext -o "${b}.txt" '
            '>/dev/null 2>&1 || true; done')
        # -of-proc so the ranks' own output reaches this child's log; the
        # prefix comes from the env Child.start() sets. Same reason as _fanout.
        full = ('set -x; mpiexec -np %d -of-proc "$MPIEXEC_OF_PROC" sh -c %s; '
                'set +x; %s' % (np_, shlex.quote(script), export))
        env = models._env_overrides(body)
        c = _new_child("oneshot", "profile:%s" % adapter.name,
                       ["sh", "-c", full], env, work,
                       meta={"model": adapter.name, "prof_dir": work,
                             "runner_bin": runner, "event": event})
        self._send_json({"id": c.id, "state": c.state, "prof_dir": work,
                         "log": c.log_path,
                         "artifacts": "/profile/%s/artifacts" % c.id},
                        status=202)

    def _post_kv(self, body):
        """KV cache control.

        Honest about the seam: the laguna runner has no KV endpoint on the
        wire, so save/load/clear are start-time flags. This records the intent
        against a child and reports the flags to re-start with rather than
        pretending a live API exists. `stats` reports what is observable.
        """
        action = body.get("action")
        if action not in ("save", "load", "clear", "stats"):
            return self._err("action must be save|load|clear|stats")
        cid = body.get("id")
        c = _get_child(cid) if cid else None
        if cid and c is None:
            return self._err("no such child: %s" % cid, status=404)
        if action == "stats":
            out = {"action": "stats"}
            if c is not None:
                out["child"] = c.info()
                if c.kind == "serve" and c.state == "ready":
                    try:
                        with urllib.request.urlopen(
                                "http://127.0.0.1:%d/health" % c.port,
                                timeout=5.0) as r:
                            out["runner_health"] = json.loads(
                                r.read().decode("utf-8", "replace"))
                    except Exception as e:
                        out["runner_health_error"] = str(e)
            return self._send_json(out)
        path = body.get("path")
        if action in ("save", "load") and not path:
            return self._err("%s needs 'path'" % action)
        if action == "clear":
            # Clearing means restarting the runner: the KV lives in each rank's
            # memory and there is no wire command to drop it.
            return self._send_json({
                "action": "clear",
                "applied": False,
                "note": "KV lives in rank-local memory with no wire command to "
                        "drop it; POST /runner/stop then /runner/start to clear.",
            })
        flags = ["--kv-%s" % action, str(path)]
        if c is not None:
            c.meta.setdefault("kv", []).append({"action": action, "path": path})
        self._send_json({
            "action": action,
            "applied": False,
            "restart_flags": flags,
            "note": "start-time flags: pass them in 'extra' on the next "
                    "/runner/start; the running child is unaffected.",
        })

    def _post_shutdown(self, body):
        with _children_lock:
            children = list(_children.values())
        stopped = []
        for c in children:
            if c.state in ("starting", "ready", "running"):
                try:
                    c.stop(grace=float(body.get("grace", 10)))
                    stopped.append(c.id)
                except Exception as e:            # noqa: BLE001
                    _log("shutdown: %s: %s" % (c.id, e))
        self._send_json({"ok": True, "stopped": stopped})
        threading.Thread(target=_deferred_exit, daemon=True).start()


def _deferred_exit():
    """Let the /shutdown response flush before tearing the server down."""
    time.sleep(0.5)
    _stop_evt.set()
    os.kill(os.getpid(), signal.SIGTERM)


# --------------------------------------------------------------------------
# main


def _parse_args(argv):
    env = os.environ.get
    p = argparse.ArgumentParser(description="llmgr control port for A64FX runners")
    p.add_argument("--host", default=env("LLMGR_HOST", "127.0.0.1"),
                   help="bind address (default %(default)s; keep loopback)")
    p.add_argument("--port", type=int, default=int(env("LLMGR_PORT", DEFAULT_PORT)),
                   help="bind port (default %(default)s)")
    p.add_argument("--token", default=env("LLMGR_TOKEN"),
                   help="bearer token; optional -- unset means open, which is "
                        "fine for a loopback port on a private fabric")
    p.add_argument("--allow-public-bind", action="store_true",
                   default=env("LLMGR_ALLOW_PUBLIC_BIND", "") not in ("", "0"),
                   help="permit a non-loopback bind (compute nodes sit on "
                        "shared 10.x subnets -- almost never what you want)")
    p.add_argument("--idle-ttl", type=float,
                   default=float(env("LLMGR_IDLE_TTL", "1800")),
                   help="bash session idle reap (default %(default)s s)")
    p.add_argument("--verbose", action="store_true",
                   default=env("LLMGR_VERBOSE", "") not in ("", "0"))
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    # A token is optional: the port is loopback-only and Fugaku's fabric is not
    # reachable from the internet, so the tunnel endpoint is the trust boundary.
    # Set LLMGR_TOKEN to turn bearer auth back on (e.g. on a shared frontend,
    # where any local user could reach a forwarded port).
    if not args.token:
        sys.stderr.write(
            "WARNING: no token set -- every request is accepted. This port runs "
            "arbitrary shell as %s; keep it on loopback.\n"
            % os.environ.get("USER", "?"))
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.allow_public_bind:
        sys.stderr.write(
            "FATAL: refusing to bind %s. Compute nodes sit on shared 10.x "
            "Tofu/admin subnets; use the reverse tunnel, or pass "
            "--allow-public-bind if you really mean it.\n" % args.host)
        return 2

    # One token, one verbosity setting for both surfaces: our Handler inherits
    # bash_http_server's _authorized(), which reads these module globals. None
    # means open -- that is bash_http_server's own "trusted network" default.
    bhs.AUTH_TOKEN = args.token or None
    bhs.VERBOSE = args.verbose
    bhs.SESSION_IDLE_TTL = args.idle_ttl

    for d in (LOG_DIR, STATE_DIR, PROF_DIR):
        os.makedirs(d, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)

    state_path = os.path.join(
        STATE_DIR, "llmgr.%s.env" % os.environ.get("PJM_JOBID", "nojob"))
    with open(state_path, "w") as f:
        f.write("LLMGR_HOST=%s\nLLMGR_PORT=%d\nLLMGR_NODE=%s\n"
                "LLMGR_JOBID=%s\nLLMGR_PID=%d\nLLMGR_REPO=%s\n"
                % (args.host, args.port, os.uname().nodename,
                   os.environ.get("PJM_JOBID", ""), os.getpid(), REPO))
    print("llmgr listening on http://%s:%d (node %s, job %s)  [auth: %s]"
          % (args.host, args.port, os.uname().nodename,
             os.environ.get("PJM_JOBID", "-"),
             "token" if bhs.AUTH_TOKEN else "none"), flush=True)
    print("state: %s" % state_path, flush=True)

    threading.Thread(target=_reaper_loop, daemon=True).start()
    threading.Thread(target=bhs._sweeper_loop, args=(_stop_evt,),
                     daemon=True).start()

    def _term(_sig, _frm):
        _stop_evt.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop_evt.set()
        with _children_lock:
            children = list(_children.values())
        for c in children:
            try:
                c.stop(grace=5)
            except Exception:                     # noqa: BLE001
                pass
        with bhs._sessions_lock:
            sessions = list(bhs._sessions.values())
            bhs._sessions.clear()
        for s in sessions:
            s.close()
        server.server_close()
        try:
            os.unlink(state_path)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
