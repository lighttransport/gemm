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
import hashlib
import json
import os
import queue
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
import uuid
from http.server import HTTPServer
from socketserver import ThreadingMixIn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)

import bash_http_server as bhs   # noqa: E402  (path set above)
import models                    # noqa: E402
import laguna_openai             # noqa: E402
import agentic                   # noqa: E402
import anthropic_api             # noqa: E402

try:
    from http.server import ThreadingHTTPServer
except ImportError:                                   # pragma: no cover
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

VERSION = "1.1"
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
_contexts = agentic.ContextRegistry(
    int(os.environ.get("LLMGR_MAX_CONTEXTS", "1024")))
_managed_cache = agentic.ManagedCacheStore(
    os.environ.get("LLMGR_CACHE_ROOT", os.path.join(STATE_DIR, "cache")),
    int(os.environ.get("LLMGR_CACHE_TTL", str(7 * 24 * 3600))))


def _ready_serve(model):
    with _children_lock:
        ready = [c for c in _children.values()
                 if c.kind == "serve" and c.state == "ready" and
                 c.meta.get("model") == model]
    return ready[0] if len(ready) == 1 else None


def _resolve_adapter(model=None):
    if model is None:
        model = models.default_model()
    try:
        return models.get_by_openai_model(model)
    except models.ConfigError:
        return models.get(model)


def _default_openai_model():
    adapter = _resolve_adapter()
    return adapter.openai_models[0] if adapter.openai_models else adapter.name


class InferenceJob:
    def __init__(self, jid, body, native, tokenizer, chat, context_id=None,
                 model=None):
        self.id = jid
        self.body = body
        self.native = native
        self.tokenizer = tokenizer
        self.chat = chat
        self.context_id = context_id
        self.model = model
        self.events = queue.Queue()
        self.done = threading.Event()
        self.cancelled = threading.Event()
        self.state = "queued"
        self.error = None
        self.result = None
        self.created = time.time()
        self.started = None

    def info(self):
        return {"id": self.id, "state": self.state, "created": self.created,
                "started": self.started, "cancelled": self.cancelled.is_set(),
                "error": self.error, "context_id": self.context_id}


class InferenceQueue:
    """One bounded FIFO feeding the one default Laguna runner."""
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.pending = queue.Queue(maxsize=capacity)
        self.jobs = {}
        self.lock = threading.Lock()
        self.seq = 0
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, body, chat=True, native=None, tokenizer=None,
               context_id=None, model=""):
        context = _contexts.get_or_create(context_id, model=model)
        with self.lock:
            self.seq += 1
            jid = "infer-%d" % self.seq
            job = InferenceJob(jid, body, native, tokenizer, chat,
                               context.context_id, model=model)
            self.jobs[jid] = job
        try:
            self.pending.put_nowait(job)
        except queue.Full:
            with self.lock:
                self.jobs.pop(jid, None)
            raise OverflowError("inference queue is full")
        with self.lock:
            finished = [key for key, old in self.jobs.items() if old.done.is_set()]
            for key in finished[:-128]:
                self.jobs.pop(key, None)
        return job

    def cancel(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
        if job is None:
            return None
        job.cancelled.set()
        if job.state == "queued":
            job.state = "cancelled"
            job.done.set()
        return job

    def info(self):
        with self.lock:
            jobs = [j.info() for j in self.jobs.values()
                    if not j.done.is_set()]
        runners = []
        with _children_lock:
            runners = [c for c in _children.values()
                       if c.kind == "serve" and c.state == "ready"]
        return {"capacity": self.capacity, "depth": self.pending.qsize(),
                "jobs": jobs, "runner": runners[0].id if len(runners) == 1 else None}

    def _work(self):
        while not _stop_evt.is_set():
            try:
                job = self.pending.get(timeout=0.5)
            except queue.Empty:
                continue
            if job.cancelled.is_set():
                job.done.set(); self.pending.task_done(); continue
            try:
                context = _contexts.get(job.context_id)
                if context is None:
                    raise RuntimeError("inference context was deleted")
                with context.reserve():
                    self._run(job)
            except Exception as e:                # noqa: BLE001
                job.error = str(e); job.state = "failed"
                job.events.put({"event": "error", "error": str(e)})
            finally:
                job.done.set(); job.events.put(None); self.pending.task_done()

    def _run(self, job):
        adapter = _resolve_adapter(job.model or models.default_model())
        runner = _ready_serve(adapter.name)
        if runner is None:
            raise RuntimeError("exactly one ready runner for model %s is required" %
                               adapter.name)
        job.state = "running"; job.started = time.time()
        native = dict(job.native)
        native["stream"] = True
        prompt_ids = list(native.get("ids", []))
        request = urllib.request.Request(
            "http://127.0.0.1:%d/generate" % runner.port,
            data=json.dumps(native).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        ids = []
        meta = {}
        previous = ""
        with urllib.request.urlopen(request, timeout=3600.0) as response:
            for raw in response:
                if job.cancelled.is_set():
                    response.close(); job.state = "cancelled"; return
                event = json.loads(raw.decode("utf-8", "replace"))
                if event.get("event") == "token":
                    ids.append(int(event["id"]))
                    text = job.tokenizer.decode(ids, raw=job.chat) if job.tokenizer else ""
                    delta = text[len(previous):] if text.startswith(previous) else text
                    previous = text
                    event["text"] = delta
                elif event.get("event") == "done":
                    meta = event
                job.events.put(event)
        meta["ids"] = ids
        meta.setdefault("n", len(ids))
        if job.tokenizer:
            meta["prompt_ids"] = prompt_ids
            job.result = adapter.completion_response(
                job.body, job.tokenizer.decode(ids, raw=job.chat), meta,
                chat=job.chat, request_id=job.id)
        else:
            meta.pop("event", None)
            job.result = meta
        job.state = "done"


_inference = InferenceQueue(int(os.environ.get("LLMGR_QUEUE_CAPACITY", "8")))


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
                # start_new_session is thread-safe; preexec_fn=os.setsid is not,
                # now that the inference FIFO has a permanent worker thread.
                start_new_session=True,
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


def _cache_set_stats(path, expected=None, shard_prefix=None):
    out = {"path": path, "exists": os.path.isdir(path), "shards": 0,
           "bytes": 0}
    if not out["exists"]:
        return out
    try:
        names = os.listdir(path)
    except OSError as e:
        out["error"] = str(e)
        return out
    for name in names:
        if not (name.endswith(".bin") and
                (name.startswith(shard_prefix) if shard_prefix is not None
                 else name.startswith("k3_ep_cache_"))):
            continue
        p = os.path.join(path, name)
        try:
            if os.path.isfile(p):
                out["shards"] += 1
                out["bytes"] += os.path.getsize(p)
        except OSError:
            continue
    if expected is not None:
        out["expected_shards"] = expected
        out["complete"] = out["shards"] == expected
    return out


def _apply_prompt_cache(body, adapter):
    """Map the OpenAI prompt-cache key to a private, shared K3 cache set.

    Explicit runner paths are deliberately left untouched.  The key is
    hashed and scoped by layout-affecting runner settings so arbitrary client
    strings never become filesystem paths and incompatible K3 layouts do not
    share a cache directory.
    """
    key = body.get("prompt_cache_key")
    if key is None or adapter.name != "k3":
        return body
    if not isinstance(key, str) or not key or len(key) > 512 or "\0" in key:
        raise ValueError("prompt_cache_key must be a non-empty string of at most 512 characters")
    if body.get("cache_load") is not None or body.get("cache_save") is not None:
        return body
    fields = [
        body.get("model", "k3"), body.get("variant", "default"),
        body.get("np", adapter.default_np()), body.get("tp_np", "auto"),
        body.get("layer", 1), body.get("layers", 1),
        body.get("threads", 48), body.get("kda_threads", 8),
        body.get("mla_cache", body.get("mla_cache_dtype", "bf16")),
    ]
    scope = "\x1f".join(str(x) for x in fields)
    digest = hashlib.sha256((scope + "\x1e" + key).encode("utf-8")).hexdigest()
    path = os.path.join(STATE_DIR, "openai-cache", "k3-" + digest)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = dict(body)
    out["cache_save"] = path
    try:
        expected = int(body.get("np", adapter.default_np()))
    except (TypeError, ValueError):
        expected = adapter.default_np()
    try:
        layer = int(body.get("layer", 1))
        layers = int(body.get("layers", 1))
        threads = int(body.get("threads", 48))
        shard_prefix = "k3_ep_cache_l%03d_%03d_n%03d_t%03d_" % (
            layer, layers, expected, threads)
    except (TypeError, ValueError):
        shard_prefix = None
    if _cache_set_stats(path, expected, shard_prefix).get("complete"):
        out["cache_load"] = path
    return out


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

    def _authorized(self):
        token = getattr(bhs, "AUTH_TOKEN", None)
        if token and self.headers.get("x-api-key") == token:
            return True
        return super()._authorized()

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
            path = path.rstrip("/") or "/"
            if path == "/v1/contexts" or path.startswith("/v1/contexts/"):
                path = path[3:]
            if path in ("/", "/health"):
                return self._get_health()
            if path == "/models":
                return self._send_json(models.describe())
            if path == "/v1/models":
                entries = []
                seen = set()
                for spec in models.describe().values():
                    for model_id in spec.get("openai_models", ()):
                        if model_id in seen:
                            continue
                        seen.add(model_id)
                        entries.append({"id": model_id, "object": "model",
                                        "created": 0, "owned_by": "poolside"})
                return self._send_json({"object": "list", "data": entries})
            if path == "/inference/queue":
                return self._send_json(_inference.info())
            if path == "/contexts":
                return self._send_json({"object": "list",
                                        "data": _contexts.info()})
            if path.startswith("/contexts/") and path.endswith("/checkpoints"):
                context_id = path.split("/")[2]
                context = _contexts.get(context_id)
                if context is None:
                    return self._err("no such context: %s" % context_id,
                                     status=404)
                return self._send_json({"object": "list",
                                        "data": ([context.checkpoint]
                                                  if context.checkpoint else [])})
            if path.startswith("/contexts/"):
                context_id = path.split("/")[2]
                context = _contexts.get(context_id)
                if context is None:
                    return self._err("no such context: %s" % context_id,
                                     status=404)
                return self._send_json({"object": "context",
                                        "context_id": context.context_id,
                                        "model": context.model,
                                        "last_response_id": context.last_response_id,
                                        "pending_tools": len(context.pending_tools),
                                        "checkpoint": context.checkpoint})
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
        model = q.get("model", [models.default_model()])[0]
        adapter = _resolve_adapter(model)
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
            path = path.rstrip("/") or "/"
            if path == "/v1/contexts" or path.startswith("/v1/contexts/"):
                path = path[3:]
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
            if path in ("/v1/chat/completions", "/chat/completions"):
                return self._post_openai(body, chat=True)
            if path == "/v1/responses":
                return self._post_openai_responses(body)
            if path == "/v1/messages":
                return self._post_anthropic(body)
            if path == "/v1/messages/count_tokens":
                return self._post_anthropic_count_tokens(body)
            if path in ("/v1/completions", "/completion"):
                return self._post_openai(body, chat=False)
            if path == "/inference/cancel":
                return self._post_inference_cancel(body)
            if path == "/profile":
                return self._post_profile(body)
            if path == "/kv":
                return self._post_kv(body)
            if path.startswith("/contexts/") and path.endswith("/checkpoints"):
                return self._post_context_checkpoint(path, body)
            if path.startswith("/contexts/") and path.endswith("/restore"):
                return self._post_context_restore(path, body)
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

    def do_DELETE(self):
        if not self._authorized():
            return self._send_json({"error": "unauthorized"}, status=401)
        path, _q = self._split()
        path = path.rstrip("/") or "/"
        if path == "/v1/contexts" or path.startswith("/v1/contexts/"):
            path = path[3:]
        parts = path.split("/")
        if len(parts) == 5 and parts[1] == "contexts" and parts[3] == "checkpoints":
            context = _contexts.get(parts[2])
            if context is None:
                return self._err("no such context: %s" % parts[2], status=404)
            if not context.checkpoint or context.checkpoint.get("name") != parts[4]:
                return self._err("checkpoint is not present", status=404)
            context.checkpoint = None
            return self._send_json({"deleted": True, "context_id": parts[2],
                                    "name": parts[4]})
        return self._err("not found: %s" % path, status=404)

    def _post_simple(self, body, mode):
        """/build and /stage: one-shot, identical shape."""
        adapter = _resolve_adapter(body.get("model"))
        argv, env, cwd = adapter.launch(mode, body)
        c = _new_child("oneshot", "%s:%s" % (mode, adapter.name),
                       argv, env, cwd,
                       meta={"model": adapter.name, "mode": mode})
        self._send_json({"id": c.id, "state": c.state, "log": c.log_path,
                         "argv": argv}, status=202)

    def _post_runner_start(self, body):
        adapter = _resolve_adapter(body.get("model"))
        mode = body.get("mode") or ("serve" if adapter.supports_serve else "generate")
        if mode not in ("serve", "generate"):
            return self._err("mode must be serve or generate")
        if mode == "serve" and not adapter.supports_serve:
            return self._err("%s has no serve mode -- use mode=generate "
                             "(one-shot) and read the log" % adapter.name)
        if mode == "serve":
            with _children_lock:
                active = [x for x in _children.values()
                          if x.kind == "serve" and
                          x.meta.get("model") == adapter.name and
                          x.state in ("starting", "ready", "stopping")]
            if active:
                return self._err(
                    "serve runner for model %s is already active; stop it before starting another"
                    % adapter.name, status=409)
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

    def _proxy_protocol(self, adapter, path, body):
        """Forward a wire-compatible request to an adapter-owned frontend."""
        runner = _ready_serve(adapter.name)
        if runner is None:
            return self._err("ready runner for model %s not found" % adapter.name,
                             status=503)
        request = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (runner.port, path),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            response = urllib.request.urlopen(request, timeout=3600.0)
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                payload = json.loads(raw.decode("utf-8", "replace"))
            except (ValueError, UnicodeDecodeError):
                payload = {"error": raw.decode("utf-8", "replace")}
            return self._send_json(payload, status=e.code)
        except (OSError, urllib.error.URLError) as e:
            return self._err("proxy to %s failed: %s" % (adapter.name, e),
                             status=502)
        if body.get("stream"):
            self.close_connection = True
            self.send_response(response.status)
            self.send_header("Content-Type", response.headers.get(
                "Content-Type", "text/event-stream"))
            self.send_header("Cache-Control", response.headers.get(
                "Cache-Control", "no-cache"))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                # SSE is line-delimited.  readline() preserves token timing;
                # a large read would buffer the whole generation until EOF.
                for chunk in iter(response.readline, b""):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                response.close()
            return None
        try:
            payload = json.loads(response.read().decode("utf-8", "replace"))
        except (ValueError, UnicodeDecodeError) as e:
            return self._err("invalid %s response: %s" % (adapter.name, e),
                             status=502)
        finally:
            response.close()
        return self._send_json(payload, status=response.status)

    def _post_generate(self, body):
        """Queue the native ids-in/ids-out request against the default runner."""
        req = dict(body)
        requested = req.pop("id", None)
        model = req.get("model", models.default_model())
        adapter = _resolve_adapter(model)
        runner = _ready_serve(adapter.name)
        if runner is None:
            return self._err("exactly one ready runner for model %s is required" %
                             adapter.name, status=503)
        if requested and requested != runner.id:
            return self._err("%s is not the active default runner" % requested,
                             status=409)
        req.pop("timeout", None)
        if not isinstance(req.get("ids"), list):
            return self._err("ids must be an array")
        try:
            job = _inference.submit(body, native=req, tokenizer=None, chat=False,
                                    context_id=body.get("context_id"),
                                    model=adapter.name)
        except OverflowError as e:
            return self._err(str(e), status=429)
        if body.get("stream"):
            return self._stream_native(job)
        if not job.done.wait(float(body.get("timeout", 900.0))):
            _inference.cancel(job.id)
            return self._err("inference timed out", status=504)
        if job.error:
            return self._err(job.error, status=502)
        self._send_json(job.result)

    def _stream_native(self, job):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                event = job.events.get()
                if event is None:
                    break
                self.wfile.write((json.dumps(event, separators=(",", ":")) +
                                  "\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            _inference.cancel(job.id)

    def _post_inference_cancel(self, body):
        job = _inference.cancel(body.get("id"))
        if job is None:
            return self._err("no such inference job", status=404)
        self._send_json(job.info())

    def _post_openai_responses(self, body):
        if "contexts" in body:
            try:
                if _resolve_adapter(body.get("model")).proxy_protocol:
                    return self._err("context batches are not supported by the "
                                     "DS4F wire proxy", status=400)
            except models.ConfigError as e:
                return self._err(str(e), status=400)
            return self._post_openai_batch(body)
        try:
            adapter = _resolve_adapter(body.get("model"))
            if not adapter.supports_serve:
                return self._err("%s does not support OpenAI-style serve operations" %
                                 adapter.name, status=400)
            if adapter.proxy_protocol:
                return self._proxy_protocol(adapter, "/v1/responses", body)
            translated = adapter.responses_request(body)
        except (models.ConfigError, ValueError) as e:
            return self._err(str(e), status=400)
        return self._post_openai(translated, chat=True, response_api=True,
                                 response_body=body)

    def _anthropic_context_id(self, body):
        metadata = body.get("metadata") or {}
        return (body.get("context_id") or metadata.get("context_id") or
                metadata.get("llmgr_context_id"))

    def _accept_anthropic_tool_continuation(self, context, body):
        results = []
        for message in body.get("messages") or []:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    results.append({"tool_call_id": block.get("tool_use_id")})
        if results:
            metadata = body.get("metadata") or {}
            previous = (body.get("previous_response_id") or
                        metadata.get("previous_response_id") or
                        context.last_response_id)
            context.accept_tool_results(previous, results)

    def _post_anthropic(self, body):
        if not isinstance(body, dict):
            return self._err("request body must be an object")
        model_name = body.get("llmgr_model", _default_openai_model())
        try:
            adapter = models.get_by_openai_model(model_name)
        except models.ConfigError as e:
            return self._err(str(e), status=400)
        if not adapter.supports_serve:
            return self._err("%s does not support Anthropic-style serve operations" % adapter.name,
                             status=400)
        if _ready_serve(adapter.name) is None:
            return self._err("ready runner for model %s not found" % adapter.name,
                             status=503)
        if adapter.proxy_protocol:
            return self._proxy_protocol(adapter, "/v1/messages", body)
        try:
            request = anthropic_api.request(
                body, model=(adapter.openai_models[0]
                             if adapter.openai_models else model_name))
            context = _contexts.get_or_create(self._anthropic_context_id(body),
                                               model=adapter.name)
            self._accept_anthropic_tool_continuation(context, body)
            request["context_id"] = context.context_id
            request = _apply_prompt_cache(request, adapter)
            native, tokenizer = adapter.native_request(request, chat=True)
            job = _inference.submit(request, native=native, tokenizer=tokenizer,
                                    chat=True, context_id=context.context_id,
                                    model=adapter.name)
        except OverflowError as e:
            return self._err(str(e), status=429)
        except (ValueError, OSError, agentic.ContextError) as e:
            return self._err(str(e))
        if body.get("stream"):
            return self._stream_anthropic(job, body, context)
        if not job.done.wait(float(body.get("timeout", 3600.0))):
            _inference.cancel(job.id)
            return self._err("inference timed out", status=504)
        if job.error:
            return self._err(job.error, status=502)
        self._record_context_response(context, job)
        self._send_json(anthropic_api.response(body, job.result,
                                               request_id=job.id))

    def _post_anthropic_count_tokens(self, body):
        try:
            adapter = _resolve_adapter(body.get("llmgr_model"))
            if adapter.proxy_protocol:
                return self._proxy_protocol(adapter, "/v1/messages/count_tokens", body)
            model_name = body.get("llmgr_model")
            request = anthropic_api.request(
                body, model=(adapter.openai_models[0]
                             if adapter.openai_models else model_name))
            native, _tokenizer = adapter.native_request(request, chat=True)
        except (ValueError, OSError) as e:
            return self._err(str(e))
        self._send_json({"input_tokens": len(native.get("ids", []))})

    def _stream_anthropic(self, job, body, context):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self._sse(anthropic_api.stream_start(body, job.id))
            self._sse({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "text", "text": ""}})
            for event in iter(job.events.get, None):
                if event.get("event") == "token" and event.get("text"):
                    self._sse(anthropic_api.stream_text(0, event["text"]))
            if job.error:
                self._sse({"type": "error", "error": {
                    "type": "runner_error", "message": job.error}})
            elif job.result:
                self._record_context_response(context, job)
                self._sse(anthropic_api.stream_stop(0))
                final = anthropic_api.response(body, job.result, request_id=job.id)
                next_index = 1
                for block in final.get("content", []):
                    if block.get("type") != "tool_use":
                        continue
                    self._sse({"type": "content_block_start", "index": next_index,
                               "content_block": {"type": "tool_use",
                                                 "id": block.get("id"),
                                                 "name": block.get("name"),
                                                 "input": {}}})
                    self._sse({"type": "content_block_delta", "index": next_index,
                               "delta": {"type": "input_json_delta",
                                         "partial_json": json.dumps(
                                             block.get("input", {}),
                                             ensure_ascii=False)}})
                    self._sse(anthropic_api.stream_stop(next_index))
                    next_index += 1
                self._sse(anthropic_api.stream_done(job.result))
                self._sse({"type": "message_stop"})
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            _inference.cancel(job.id)

    def _accept_tool_continuation(self, context, request):
        items = request.get("input")
        if not isinstance(items, list):
            return
        results = []
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "function_call_output":
                continue
            call_id = item.get("call_id") or item.get("tool_call_id")
            results.append({"tool_call_id": call_id})
        if results:
            previous = request.get("previous_response_id") or context.last_response_id
            context.accept_tool_results(previous, results)

    def _record_context_response(self, context, job):
        calls = []
        try:
            choice = (job.result.get("choices") or [{}])[0]
            calls = (choice.get("message") or {}).get("tool_calls") or []
        except (AttributeError, TypeError):
            pass
        context.record_response(job.id, calls)

    def _submit_batch_item(self, body, item):
        if not isinstance(item, dict):
            raise agentic.ContextError("context item must be an object")
        request = dict(body)
        request.pop("contexts", None)
        request.update(item)
        model = request.get("model", body.get("model", _default_openai_model()))
        adapter = models.get_by_openai_model(model)
        if not adapter.supports_serve:
            raise ValueError("%s does not support OpenAI-style serve operations" % adapter.name)
        if _ready_serve(adapter.name) is None:
            raise RuntimeError("ready runner for model %s not found" % adapter.name)
        translated = adapter.responses_request(request)
        context_id = request.get("context_id") or request.get("conversation_id")
        if not context_id and request.get("previous_response_id"):
            prior = _contexts.find_response(request["previous_response_id"])
            if prior is None:
                raise agentic.ContextError("unknown previous_response_id")
            context_id = prior.context_id
        context = _contexts.get_or_create(context_id, model=adapter.name)
        self._accept_tool_continuation(context, request)
        translated["context_id"] = context.context_id
        checkpoint = context.checkpoint or {}
        if not translated.get("cache_load") and checkpoint.get("state") == "restore_requested":
            translated["cache_load"] = checkpoint["path"]
        if not translated.get("cache_save") and checkpoint.get("state") == "requested":
            translated["cache_save"] = checkpoint["staging_path"]
        translated = _apply_prompt_cache(translated, adapter)
        native, tokenizer = adapter.native_request(translated, chat=True)
        job = _inference.submit(translated, native=native, tokenizer=tokenizer,
                                chat=True, context_id=context.context_id,
                                model=adapter.name)
        return request, context, job

    def _post_openai_batch(self, body):
        contexts = body.get("contexts")
        if not isinstance(contexts, list) or not contexts:
            return self._err("contexts must be a non-empty array")
        if body.get("stream"):
            return self._stream_openai_batch(body, contexts)
        results = []
        for item in contexts:
            try:
                request, context, job = self._submit_batch_item(body, item)
                timeout = float(request.get("timeout", 3600.0))
                if not job.done.wait(timeout):
                    _inference.cancel(job.id)
                    raise RuntimeError("inference timed out")
                if job.error:
                    raise RuntimeError(job.error)
                self._record_context_response(context, job)
                if context.checkpoint:
                    if context.checkpoint.get("state") == "requested":
                        self._finalize_context_checkpoint(context)
                    elif context.checkpoint.get("state") == "restore_requested":
                        context.checkpoint["state"] = "restored"
                response = adapter.responses_response(request, job.result,
                                                       request_id=job.id)
                results.append(agentic.batch_result(context.context_id,
                                                    response=response))
            except (ValueError, OSError, RuntimeError, OverflowError,
                    agentic.ContextError) as e:
                results.append(agentic.batch_result(
                    item.get("context_id", "") if isinstance(item, dict) else "",
                    error=e))
        failed = any("error" in result for result in results)
        return self._send_json({"id": "batch_" + uuid.uuid4().hex,
                                "object": "response.batch",
                                "status": "partial" if failed else "completed",
                                "data": results})

    def _stream_openai_batch(self, body, items):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        jobs = {}
        try:
            for item in items:
                try:
                    request, context, job = self._submit_batch_item(body, item)
                    jobs[job.id] = {"request": request, "context": context,
                                    "job": job, "reasoning": bool(
                                        request.get("enable_thinking", True) and
                                        request.get("reasoning_effort") != "none")}
                    self._sse({"type": "response.created",
                               "context_id": context.context_id,
                               "response": {"id": job.id, "object": "response",
                                             "model": request.get("model", _default_openai_model()),
                                             "status": "in_progress"}})
                except (ValueError, OSError, RuntimeError, OverflowError,
                        agentic.ContextError) as e:
                    self._sse({"type": "response.failed",
                               "context_id": item.get("context_id", "")
                               if isinstance(item, dict) else "",
                               "error": {"message": str(e), "type": "context_error"}})
            while jobs:
                progressed = False
                for jid, state in list(jobs.items()):
                    job = state["job"]
                    while True:
                        try:
                            event = job.events.get_nowait()
                        except queue.Empty:
                            break
                        progressed = True
                        if event is None:
                            context = state["context"]
                            if job.error:
                                self._sse({"type": "response.failed",
                                           "context_id": context.context_id,
                                           "error": {"message": job.error}})
                            elif job.result:
                                self._record_context_response(context, job)
                                if context.checkpoint:
                                    if context.checkpoint.get("state") == "requested":
                                        self._finalize_context_checkpoint(context)
                                    elif context.checkpoint.get("state") == "restore_requested":
                                        context.checkpoint["state"] = "restored"
                                self._sse({"type": "response.completed",
                                           "context_id": context.context_id,
                                           "response": _resolve_adapter(
                                               getattr(state["job"], "model", None) or
                                               state["request"].get("model")).responses_response(
                                                   state["request"], job.result,
                                                   request_id=job.id)})
                            jobs.pop(jid, None)
                            break
                        if event.get("event") != "token" or not event.get("text"):
                            continue
                        delta_text = event["text"]
                        if delta_text.startswith("<think>"):
                            delta_text = delta_text[len("<think>"):]
                        if state["reasoning"]:
                            if "</think>" in delta_text:
                                before, delta_text = delta_text.split("</think>", 1)
                                if before:
                                    self._sse({"type": "response.reasoning_summary_text.delta",
                                               "context_id": state["context"].context_id,
                                               "delta": before})
                                state["reasoning"] = False
                            else:
                                self._sse({"type": "response.reasoning_summary_text.delta",
                                           "context_id": state["context"].context_id,
                                           "delta": delta_text})
                                continue
                        if delta_text:
                            if "<tool_call>" in delta_text:
                                delta_text = delta_text.split("<tool_call>", 1)[0]
                            if delta_text:
                                self._sse({"type": "response.output_text.delta",
                                           "context_id": state["context"].context_id,
                                           "delta": delta_text, "output_index": 0,
                                           "content_index": 0})
                if jobs and not progressed:
                    time.sleep(0.01)
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            for state in jobs.values():
                _inference.cancel(state["job"].id)

    def _post_openai(self, body, chat, response_api=False, response_body=None):
        model = body.get("model", _default_openai_model())
        try:
            adapter = models.get_by_openai_model(model)
        except models.ConfigError as e:
            return self._err(str(e), status=400)
        if not adapter.supports_serve:
            return self._err("%s does not support OpenAI-style serve operations" % adapter.name,
                             status=400)
        if _ready_serve(adapter.name) is None:
            return self._err("ready runner for model %s not found" % adapter.name, status=503)
        if adapter.proxy_protocol:
            path = "/v1/chat/completions" if chat else "/v1/completions"
            return self._proxy_protocol(adapter, path, body)
        try:
            original = response_body if isinstance(response_body, dict) else body
            context_id = original.get("context_id") or original.get("conversation_id")
            if not context_id and original.get("previous_response_id"):
                prior = _contexts.find_response(original["previous_response_id"])
                if prior is None:
                    return self._err("unknown previous_response_id", status=409)
                context_id = prior.context_id
            body = dict(body)
            if context_id:
                body["context_id"] = context_id
            context = _contexts.get_or_create(context_id, model=adapter.name)
            self._accept_tool_continuation(context, original)
            checkpoint = context.checkpoint or {}
            if not body.get("cache_load") and checkpoint.get("state") == "restore_requested":
                body["cache_load"] = checkpoint["path"]
            if not body.get("cache_save") and checkpoint.get("state") == "requested":
                body["cache_save"] = checkpoint["staging_path"]
            body = _apply_prompt_cache(body, adapter)
            native, tokenizer = adapter.native_request(body, chat=chat)
            job = _inference.submit(body, native=native, tokenizer=tokenizer,
                                    chat=chat, context_id=context_id,
                                    model=adapter.name)
        except OverflowError as e:
            return self._err(str(e), status=429)
        except (ValueError, OSError) as e:
            return self._err(str(e))
        if body.get("stream"):
            if response_api:
                return self._stream_responses(job, response_body)
            return self._stream_openai(job, body, chat)
        if not job.done.wait(float(body.get("timeout", 3600.0))):
            _inference.cancel(job.id)
            return self._err("inference timed out", status=504)
        if job.error:
            return self._err(job.error, status=502)
        context = _contexts.get(getattr(job, "context_id", None))
        if context is not None:
            self._record_context_response(context, job)
            if context.checkpoint:
                if body.get("cache_save") and context.checkpoint.get("state") == "requested":
                    self._finalize_context_checkpoint(context)
                elif body.get("cache_load") and context.checkpoint.get("state") == "restore_requested":
                    context.checkpoint["state"] = "restored"
        result = (adapter.responses_response(response_body, job.result,
                                             request_id=job.id)
                  if response_api else job.result)
        self._send_json(result)

    def _stream_openai(self, job, body, chat):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        reasoning = chat and body.get("enable_thinking", True) and \
            body.get("reasoning_effort") != "none"
        tool_mode = False
        try:
            if chat:
                self._sse({"id": job.id, "object": "chat.completion.chunk",
                           "created": int(time.time()),
                           "model": body.get("model", _default_openai_model()),
                           "choices": [{"index": 0, "delta": {"role": "assistant"},
                                        "finish_reason": None}]})
            while True:
                event = job.events.get()
                if event is None:
                    break
                if event.get("event") != "token" or not event.get("text"):
                    continue
                delta_text = event["text"]
                delta = {}
                if delta_text.startswith("<think>"):
                    delta_text = delta_text[len("<think>"):]
                if reasoning:
                    if "</think>" in delta_text:
                        before, delta_text = delta_text.split("</think>", 1)
                        if before:
                            delta["reasoning_content"] = before
                            delta["reasoning"] = before
                        reasoning = False
                    else:
                        delta["reasoning_content"] = delta_text
                        delta["reasoning"] = delta_text
                if not reasoning and delta_text:
                    if tool_mode:
                        delta_text = ""
                    elif "<tool_call>" in delta_text:
                        delta_text = delta_text.split("<tool_call>", 1)[0]
                        tool_mode = True
                    if delta_text:
                        delta["content"] = delta_text
                if chat and not delta:
                    continue
                kind = "chat.completion.chunk" if chat else "text_completion"
                choice = {"index": 0, "finish_reason": None}
                choice["delta" if chat else "text"] = delta if chat else delta_text
                self._sse({"id": job.id, "object": kind,
                           "created": int(time.time()),
                           "model": body.get("model", _default_openai_model()),
                           "choices": [choice]})
            if job.error:
                self._sse({"error": {"message": job.error, "type": "runner_error"}})
            elif job.result:
                final = job.result["choices"][0]
                delta = {}
                if chat and final.get("message", {}).get("tool_calls"):
                    delta["tool_calls"] = final["message"]["tool_calls"]
                choice = {"index": 0, "finish_reason": final.get("finish_reason")}
                choice["delta" if chat else "text"] = delta if chat else ""
                self._sse({"id": job.id,
                           "object": "chat.completion.chunk" if chat else "text_completion",
                           "created": int(time.time()),
                           "model": body.get("model", _default_openai_model()),
                           "choices": [choice]})
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            _inference.cancel(job.id)

    def _stream_responses(self, job, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self._sse({"type": "response.created", "response": {
                "id": job.id, "object": "response", "model": body.get(
                    "model", _default_openai_model()), "status": "in_progress"}})
            reasoning = body.get("enable_thinking", True) and \
                body.get("reasoning_effort") != "none"
            for event in iter(job.events.get, None):
                if event.get("event") != "token" or not event.get("text"):
                    continue
                delta_text = event["text"]
                if delta_text.startswith("<think>"):
                    delta_text = delta_text[len("<think>"):]
                if reasoning:
                    if "</think>" in delta_text:
                        before, delta_text = delta_text.split("</think>", 1)
                        if before:
                            self._sse({"type": "response.reasoning_summary_text.delta",
                                      "delta": before})
                        reasoning = False
                    else:
                        self._sse({"type": "response.reasoning_summary_text.delta",
                                  "delta": delta_text})
                        continue
                if delta_text:
                    if "<tool_call>" in delta_text:
                        delta_text = delta_text.split("<tool_call>", 1)[0]
                    if delta_text:
                        self._sse({"type": "response.output_text.delta",
                                  "delta": delta_text, "output_index": 0,
                                  "content_index": 0})
            if job.error:
                self._sse({"type": "response.failed",
                           "error": {"message": job.error}})
            elif job.result:
                context = _contexts.get(getattr(job, "context_id", None))
                if context is not None:
                    self._record_context_response(context, job)
                    if context.checkpoint:
                        if context.checkpoint.get("state") == "requested":
                            self._finalize_context_checkpoint(context)
                        elif context.checkpoint.get("state") == "restore_requested":
                            context.checkpoint["state"] = "restored"
                adapter = _resolve_adapter(getattr(job, "model", None) or
                                           body.get("model"))
                self._sse({"type": "response.completed",
                           "response": adapter.responses_response(
                               body, job.result, request_id=job.id)})
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            _inference.cancel(job.id)

    def _sse(self, obj):
        data = ("data: " + json.dumps(obj, ensure_ascii=False,
                                      separators=(",", ":")) + "\n\n").encode("utf-8")
        self.wfile.write(data); self.wfile.flush()

    def _post_profile(self, body):
        """fapp-wrapped run of the per-rank binary.

        fapp collects one PMU dataset per process, so it must wrap the runner
        binary inside mpiexec -- not the launcher script. Consequence: weights
        must already be staged and a tofu_topo.txt is generated first.
        """
        adapter = _resolve_adapter(body.get("model"))
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

    def _post_context_checkpoint(self, path, body):
        context_id = path.split("/")[2]
        context = _contexts.get(context_id)
        if context is None:
            return self._err("no such context: %s" % context_id, status=404)
        if context.model != "k3":
            return self._err("model %s does not support managed checkpoints" % context.model,
                             status=400)
        name = body.get("name")
        if not isinstance(name, str) or not agentic._SAFE_NAME.match(name):
            return self._err("checkpoint name must match [A-Za-z0-9._-]{1,128}")
        identity = "checkpoint-%s-%s" % (context_id, name)
        checkpoint_path = _managed_cache.path(identity)
        staging_path = os.path.join(_managed_cache.root,
                                    ".staging-%s-%s" % (context_id, name))
        context.checkpoint = {"name": name, "identity": identity,
                              "path": checkpoint_path,
                              "staging_path": staging_path,
                              "state": "requested", "updated": time.time()}
        return self._send_json({"object": "checkpoint",
                                "context_id": context_id,
                                "checkpoint": context.checkpoint}, status=202)

    def _finalize_context_checkpoint(self, context):
        checkpoint = context.checkpoint
        if not checkpoint or checkpoint.get("state") != "requested":
            return
        staging = checkpoint.get("staging_path")
        try:
            manifest = _managed_cache.publish(
                checkpoint["identity"], staging,
                {"model": context.model, "context_id": context.context_id,
                 "checkpoint_name": checkpoint["name"]})
            checkpoint["manifest"] = manifest
            checkpoint["state"] = "complete"
            checkpoint["updated"] = time.time()
            shutil.rmtree(staging)
        except (OSError, ValueError, agentic.ContextError) as e:
            checkpoint["state"] = "save_failed"
            checkpoint["error"] = str(e)
            checkpoint["updated"] = time.time()

    def _post_context_restore(self, path, body):
        context_id = path.split("/")[2]
        context = _contexts.get(context_id)
        if context is None:
            return self._err("no such context: %s" % context_id, status=404)
        checkpoint = context.checkpoint
        if checkpoint is None:
            return self._err("context has no checkpoint", status=404)
        if checkpoint.get("state") != "complete":
            return self._err("checkpoint is not complete", status=409)
        if body.get("name") and body["name"] != checkpoint["name"]:
            return self._err("checkpoint is not present", status=404)
        checkpoint = dict(checkpoint)
        checkpoint["state"] = "restore_requested"
        checkpoint["updated"] = time.time()
        context.checkpoint = checkpoint
        return self._send_json({"object": "checkpoint",
                                "context_id": context_id,
                                "checkpoint": checkpoint,
                                "cache_load": checkpoint["path"]})

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
        model = body.get("model")
        if model is None and c is not None:
            model = c.meta.get("model")
        if model is None:
            model = models.default_model()
        adapter = _resolve_adapter(model)
        if cid and c is None:
            return self._err("no such child: %s" % cid, status=404)
        if action == "stats":
            out = {"action": "stats"}
            stats_path = body.get("path")
            if stats_path is not None:
                if not isinstance(stats_path, str):
                    return self._err("stats path must be a string")
                if "\0" in stats_path:
                    return self._err("stats path must not contain NUL")
                expected = body.get("np")
                try:
                    expected = int(expected) if expected is not None else None
                except (TypeError, ValueError):
                    return self._err("stats np must be an integer")
                out["cache"] = _cache_set_stats(stats_path, expected)
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
        if action in ("save", "load") and not isinstance(path, str):
            return self._err("%s path must be a string" % action)
        if action in ("save", "load") and "\0" in path:
            return self._err("%s path must not contain NUL" % action)
        if action == "clear":
            # Clearing means restarting the runner: the KV lives in each rank's
            # memory and there is no wire command to drop it.
            return self._send_json({
                "action": "clear",
                "applied": False,
                "note": "KV lives in rank-local memory with no wire command to "
                        "drop it; POST /runner/stop then /runner/start to clear.",
            })
        flags = adapter.cache_flags(action, path) if action in ("save", "load") else []
        if c is not None:
            c.meta.setdefault("kv", []).append({"action": action, "path": path})
        note = "pass these as the next /runner/start request body fields; the running child is unaffected."
        if not flags and action in ("save", "load"):
            if not adapter.supports_cache:
                note = "model does not support cache restart flags"
            else:
                note = ("runner-specific restart flags unavailable; pass "
                        "cache_load/cache_save in /runner/start.")
        self._send_json({
            "action": action,
            "applied": False,
            "restart_flags": flags,
            "note": note,
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
