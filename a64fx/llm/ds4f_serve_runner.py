#!/usr/bin/env python3
"""ds4f_serve_runner.py -- single-node persistent DS4F serve loop.

Implements the shared-FS protocol that a64fx/llm/ds4f_serve.py drives:

  <BASE>.req       prompt + sampling header (written by the frontend)
  <BASE>.reqseq    monotonically increasing request sequence
  <BASE>.resp      generated token ids (space-separated) written by the runner
  <BASE>.respseq   sequence of the last completed response

Request body (frontend, ds4f_serve.py infer()):
  hdr = "max_new temp top_p top_k presence_penalty repeat_penalty seed slot ctl"
  [cache_path, if ctl != 0]
  prompt ids (space-separated)
  ctl bit0 = load KV prefix from the next request line before prefill
  ctl bit1 = save KV prefix to the following request line after generation

The model runs through libds4f_serve.so (ctypes); the runner owns the chat
loop, sampling penalties, prefix caching, and context/slot management.

Env: DS4F_SERVE_BASE, DS4F_STAGE_DIR (required), DS4F_SERVE_USE_HIP (1),
     DS4F_HIP_DEVICE, DS4F_MAXPOS, LLM_THREADS, DS4F_CMGS,
     DS4F_SERVE_PREFIX_CACHE (1), DS4F_SERVE_SLOTS (>=1).
"""
import argparse, ctypes, hashlib, json, os, selectors, signal, socket, sys, time

def _term(sig, frame):
    raise KeyboardInterrupt

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.environ.get("DS4F_SERVE_LIB", os.path.join(HERE, "libds4f_serve.so"))


class Sampling(ctypes.Structure):
    _fields_ = [("temperature", ctypes.c_double),
                ("top_p", ctypes.c_double),
                ("top_k", ctypes.c_int),
                ("presence_penalty", ctypes.c_double),
                ("repeat_penalty", ctypes.c_double),
                ("seed", ctypes.c_longlong)]


def load_lib(path):
    if not os.path.exists(path):
        sys.exit("libds4f_serve.so not found at %s (run a64fx/llm/build_ds4f_serve.sh)" % path)
    lib = ctypes.CDLL(path)
    lib.ds4f_serve_open.restype = ctypes.c_void_p
    lib.ds4f_serve_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
                                    ctypes.c_char_p, ctypes.c_size_t]
    if hasattr(lib, "ds4f_serve_configure_hip_prefill"):
        lib.ds4f_serve_configure_hip_prefill.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 18
        lib.ds4f_serve_configure_hip_prefill.restype = ctypes.c_int
    lib.ds4f_serve_close.argtypes = [ctypes.c_void_p]
    lib.ds4f_serve_prefill.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                       ctypes.c_int, ctypes.c_int]
    lib.ds4f_serve_prefill.restype = ctypes.c_int
    lib.ds4f_serve_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.ds4f_serve_decode.restype = ctypes.c_int
    lib.ds4f_serve_sample.argtypes = [ctypes.c_void_p, ctypes.POINTER(Sampling)]
    lib.ds4f_serve_sample.restype = ctypes.c_int
    lib.ds4f_serve_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.ds4f_serve_logits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.ds4f_serve_kv_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.ds4f_serve_kv_restore.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.ds4f_serve_reset.argtypes = [ctypes.c_void_p]
    lib.ds4f_serve_pos.argtypes = [ctypes.c_void_p]
    lib.ds4f_serve_pos.restype = ctypes.c_int
    lib.ds4f_serve_eos.argtypes = [ctypes.c_void_p]
    lib.ds4f_serve_eos.restype = ctypes.c_int
    lib.ds4f_serve_vocab.argtypes = [ctypes.c_void_p]
    lib.ds4f_serve_vocab.restype = ctypes.c_int
    lib.ds4f_serve_maxpos.argtypes = [ctypes.c_void_p]
    lib.ds4f_serve_maxpos.restype = ctypes.c_int
    lib.ds4f_serve_context_bytes.argtypes = [ctypes.c_void_p]
    lib.ds4f_serve_context_bytes.restype = ctypes.c_size_t
    lib.ds4f_serve_context_export.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.ds4f_serve_context_export.restype = ctypes.c_int
    lib.ds4f_serve_context_import.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.ds4f_serve_context_import.restype = ctypes.c_int
    lib.ds4f_serve_decode_batch.argtypes = [ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t)]
    lib.ds4f_serve_decode_batch.restype = ctypes.c_int
    lib.ds4f_serve_decode_batch_reserve.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ds4f_serve_decode_batch_reserve.restype = ctypes.c_int
    lib.ds4f_serve_slot_import.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_size_t]
    lib.ds4f_serve_slot_import.restype = ctypes.c_int
    lib.ds4f_serve_slot_context_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ds4f_serve_slot_context_bytes.restype = ctypes.c_size_t
    lib.ds4f_serve_slot_export.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_size_t]
    lib.ds4f_serve_slot_export.restype = ctypes.c_int
    lib.ds4f_serve_slot_sample.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                           ctypes.POINTER(Sampling)]
    lib.ds4f_serve_slot_sample.restype = ctypes.c_int
    lib.ds4f_serve_slot_release.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ds4f_serve_slot_release.restype = ctypes.c_int
    lib.ds4f_serve_decode_slots.argtypes = [ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.ds4f_serve_decode_slots.restype = ctypes.c_int
    if hasattr(lib, "ds4f_serve_cache_hot_experts"):
        lib.ds4f_serve_cache_hot_experts.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 3
        lib.ds4f_serve_cache_hot_experts.restype = ctypes.c_int
    if hasattr(lib, "ds4f_serve_enable_route_telemetry"):
        lib.ds4f_serve_enable_route_telemetry.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.ds4f_serve_enable_route_telemetry.restype = ctypes.c_int
    return lib


class Serve(object):
    def __init__(self, lib, stage_dir, use_hip, hip_device, threads, cmgs, max_pos,
                 hip_prefill):
        err = ctypes.create_string_buffer(512)
        self._s = lib.ds4f_serve_open(stage_dir.encode(), int(use_hip), int(hip_device),
                                      int(threads), int(cmgs), int(max_pos), err, len(err))
        if not self._s:
            sys.exit("ds4f_serve_open failed: %s" % err.value.decode())
        self.lib = lib
        if use_hip and hip_prefill[0]:
            if not hasattr(lib, "ds4f_serve_configure_hip_prefill"):
                sys.exit("serving library lacks tuned HIP prefill support; rebuild it")
            rc = lib.ds4f_serve_configure_hip_prefill(self._s,
                                                       *map(int, hip_prefill))
            if rc != 0:
                sys.exit("failed to configure tuned HIP prefill (rc=%d)" % rc)
        self.vocab = lib.ds4f_serve_vocab(self._s)
        self.eos = lib.ds4f_serve_eos(self._s)
        self.logits = lib.ds4f_serve_logits(self._s, ctypes.byref(ctypes.c_int(0)))

    def close(self):
        if self._s:
            self.lib.ds4f_serve_close(self._s)
            self._s = None

    def prefill(self, ids, pos0):
        arr = (ctypes.c_int * len(ids))(*ids)
        return self.lib.ds4f_serve_prefill(self._s, arr, len(ids), pos0)

    def decode(self, token, pos):
        return self.lib.ds4f_serve_decode(self._s, int(token), int(pos))

    def sample(self, sp):
        return self.lib.ds4f_serve_sample(self._s, ctypes.byref(sp))

    def kv_save(self, path):
        return self.lib.ds4f_serve_kv_save(self._s, path.encode())

    def kv_restore(self, path):
        return self.lib.ds4f_serve_kv_restore(self._s, path.encode())

    def reset(self):
        return self.lib.ds4f_serve_reset(self._s)

    def pos(self):
        return self.lib.ds4f_serve_pos(self._s)

    def maxpos(self):
        return self.lib.ds4f_serve_maxpos(self._s)

    def context_export(self):
        n = self.lib.ds4f_serve_context_bytes(self._s)
        blob = bytearray(n)
        buf = (ctypes.c_char * n).from_buffer(blob)
        if not n or self.lib.ds4f_serve_context_export(self._s, buf, n) != 0:
            raise RuntimeError("context export failed")
        return blob

    def context_import(self, blob):
        if isinstance(blob, bytearray):
            buf = (ctypes.c_char * len(blob)).from_buffer(blob)
        else:
            buf = ctypes.create_string_buffer(blob, len(blob))
        return self.lib.ds4f_serve_context_import(self._s, buf, len(blob))

    def decode_batch(self, blobs, tokens):
        n = len(blobs)
        # Context images are hundreds of MiB at an 8K window. Keep one mutable
        # image per job and let the native batch call update it in place. The
        # previous input-buffer + output-buffer scheme temporarily tripled the
        # live context memory and copied every image twice per generated token.
        work = [b if isinstance(b, bytearray) else bytearray(b) for b in blobs]
        ilen = [len(b) for b in work]
        # A compact v2 image grows by one KV row per layer plus one history
        # token. One MiB covers the current model with ample format headroom.
        for b in work: b.extend(b"\0" * (1 << 20))
        views = [(ctypes.c_char * len(b)).from_buffer(b) for b in work]
        ia = (ctypes.c_void_p * n)(*[ctypes.addressof(v) for v in views])
        oa = (ctypes.c_void_p * n)(*[ctypes.addressof(v) for v in views])
        il = (ctypes.c_size_t * n)(*ilen)
        oc = (ctypes.c_size_t * n)(*[len(b) for b in work])
        ol = (ctypes.c_size_t * n)()
        ta = (ctypes.c_int * n)(*tokens)
        rc = self.lib.ds4f_serve_decode_batch(self._s, ta, n, ia, il, oa, oc, ol)
        if rc != 0: raise RuntimeError("batched decode failed rc=%d" % rc)
        del views
        for k, b in enumerate(work):
            if ol[k] > len(b): raise RuntimeError("batched decode returned oversized context")
            del b[ol[k]:]
        return work

    def reserve_decode_batch(self, capacity):
        return self.lib.ds4f_serve_decode_batch_reserve(self._s, int(capacity))

    def slot_import(self, slot, blob):
        if isinstance(blob, bytearray):
            buf = (ctypes.c_char * len(blob)).from_buffer(blob)
        else:
            buf = ctypes.create_string_buffer(blob, len(blob))
        return self.lib.ds4f_serve_slot_import(self._s, slot, buf, len(blob))

    def slot_export(self, slot):
        n = self.lib.ds4f_serve_slot_context_bytes(self._s, slot)
        blob = bytearray(n); buf = (ctypes.c_char * n).from_buffer(blob)
        if not n or self.lib.ds4f_serve_slot_export(self._s, slot, buf, n) != 0:
            raise RuntimeError("slot export failed")
        return blob

    def slot_sample(self, slot, sp):
        return self.lib.ds4f_serve_slot_sample(self._s, slot, ctypes.byref(sp))

    def slot_release(self, slot):
        return self.lib.ds4f_serve_slot_release(self._s, slot)

    def decode_slots(self, slots, tokens):
        n = len(slots)
        sa = (ctypes.c_int * n)(*slots); ta = (ctypes.c_int * n)(*tokens)
        return self.lib.ds4f_serve_decode_slots(self._s, sa, ta, n)

    def cache_hot_experts(self, cache_mb, reserve_mb, stats):
        if not hasattr(self.lib, "ds4f_serve_cache_hot_experts"):
            return -1
        return self.lib.ds4f_serve_cache_hot_experts(
            self._s, int(cache_mb), int(reserve_mb), int(stats))

    def enable_route_telemetry(self, enabled):
        if not hasattr(self.lib, "ds4f_serve_enable_route_telemetry"):
            return -1
        return self.lib.ds4f_serve_enable_route_telemetry(self._s, int(enabled))


def env_i(k, d):
    return int(os.environ.get(k, d))


# The current HIP prefill path is unstable for very long single requests.
# Keep a head/tail window until the long-context kernel is hardened; this
# prevents a native crash from wedging the HTTP frontend indefinitely.
PREFILL_MAX_TOKENS = env_i("DS4F_SERVE_PREFILL_MAX_TOKENS", 2048)
PREFILL_CHUNK_TOKENS = max(1, env_i("DS4F_SERVE_PREFILL_CHUNK_TOKENS", 32))


def prefill_chunked(sess, ids, pos0):
    """Prefill in small HIP-safe chunks instead of one large dispatch."""
    for off in range(0, len(ids), PREFILL_CHUNK_TOKENS):
        chunk = ids[off:off + PREFILL_CHUNK_TOKENS]
        if chunk and sess.prefill(chunk, pos0 + off) != 0:
            return -1
    return 0


class Context(object):
    def __init__(self, cid, internal=False):
        self.id = cid
        self.internal = internal
        self.blob = None
        self.tokens = []
        self.disk = None
        self.last_access = time.time()
        self.active_job = None


class Job(object):
    def __init__(self, sock, request):
        self.sock, self.request = sock, request
        self.id = str(request.get("job_id") or ("job-%x" % id(self)))
        self.context_id = str(request.get("context_id") or self.id)
        self.ephemeral = bool(request.get("ephemeral", False))
        self.prompt = [int(x) for x in request.get("prompt", [])]
        self.max_new = max(0, int(request.get("max_new", 512)))
        self.sp = Sampling(float(request.get("temperature", 0.0)),
                           float(request.get("top_p", 1.0)),
                           int(request.get("top_k", 0)),
                           float(request.get("presence_penalty", 0.0)),
                           float(request.get("repeat_penalty", 1.0)),
                           int(request.get("seed", 1)))
        self.phase, self.off, self.out = "prepare", 0, []
        self.working = None
        self.started = time.time()
        self.prefill_started = self.started
        self.decode_started = None
        self.cancelled = False
        self.slot = -1
        self.cache_path = request.get("cache_path")
        self.cache_load = bool(request.get("cache_load", False))
        self.cache_save_path = request.get("cache_save_path")
        self.cached_tokens = max(0, int(request.get("cached_tokens", 0)))
        self.cache_hit = False
        self.cache_saved = False
        self.prompt_cached = False


class CooperativeServer(object):
    def __init__(self, sess, path, context_dir, memory_ttl, disk_ttl,
                 memory_mb, disk_mb, prefill_quantum, decode_quantum, quantum_ms,
                 decode_batch_size, single_prefill_quantum,
                 expert_cache_mb=0, expert_cache_reserve_mb=1536,
                 expert_cache_stats=0):
        self.sess, self.path = sess, path
        self.context_dir = context_dir
        self.memory_ttl, self.disk_ttl = memory_ttl, disk_ttl
        self.memory_cap, self.disk_cap = memory_mb << 20, disk_mb << 20
        self.prefill_q = max(1, prefill_quantum)
        self.single_prefill_q = max(self.prefill_q, single_prefill_quantum)
        self.decode_q = max(1, decode_quantum)
        self.decode_batch_size = max(1, decode_batch_size)
        self.batch_steps = self.batch_sequences = 0
        self.expert_cache_mb = expert_cache_mb
        self.expert_cache_reserve_mb = expert_cache_reserve_mb
        self.expert_cache_stats = expert_cache_stats
        self.expert_cache_initialized = False
        if self.decode_batch_size > 1 and self.sess.reserve_decode_batch(self.decode_batch_size) != 0:
            raise RuntimeError("native decode batch reservation failed")
        self.free_slots = list(range(self.decode_batch_size)) if self.decode_batch_size > 1 else []
        self.quantum_s = max(0.001, quantum_ms / 1000.0)
        self.contexts, self.jobs, self.current = {}, [], None
        os.makedirs(context_dir, mode=0o700, exist_ok=True)
        for name in os.listdir(context_dir):
            if not name.endswith(".json"): continue
            try:
                mp = os.path.join(context_dir, name)
                with open(mp) as f: meta = json.load(f)
                cid = str(meta["context_id"]); bp = mp[:-5] + ".ctx"
                if not os.path.isfile(bp): continue
                c = Context(cid, bool(meta.get("internal", False)))
                c.tokens = [int(x) for x in meta.get("tokens", [])]
                c.last_access = float(meta.get("last_access", os.path.getmtime(mp)))
                c.disk = bp; self.contexts[cid] = c
            except (OSError, ValueError, TypeError, KeyError):
                continue
        try: os.unlink(path)
        except FileNotFoundError: pass
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(path); os.chmod(path, 0o600); self.listener.listen(64)
        self.listener.setblocking(False)
        print("[runner] cooperative socket=%s context_dir=%s" % (path, context_dir),
              file=sys.stderr, flush=True)

    def send(self, job, event):
        try:
            job.sock.sendall((json.dumps(event, separators=(",", ":")) + "\n").encode())
            return True
        except OSError:
            job.cancelled = True
            return False

    def maybe_cache_hot_experts(self, job, decode_ready=False):
        if not self.expert_cache_mb or self.expert_cache_initialized:
            return
        cold_prompt = max(0, len(job.prompt) - int(job.cached_tokens))
        if cold_prompt < 256 and not (decode_ready and len(job.out) >= 8):
            return
        admitted = self.sess.cache_hot_experts(
            self.expert_cache_mb, self.expert_cache_reserve_mb,
            self.expert_cache_stats)
        if admitted < 0:
            raise RuntimeError("HIP prompt-hot expert cache setup failed")
        self.expert_cache_initialized = True
        print("[runner] prompt-hot expert cache admitted=%d training_prompt=%d decode=%d" %
              (admitted, cold_prompt, len(job.out)), file=sys.stderr, flush=True)

    def disk_paths(self, cid):
        key = hashlib.sha256(cid.encode()).hexdigest()
        return os.path.join(self.context_dir, key + ".ctx"), os.path.join(self.context_dir, key + ".json")

    def load_context(self, ctx):
        if ctx.blob is not None: return True
        bp, mp = self.disk_paths(ctx.id)
        try:
            with open(bp, "rb") as f: ctx.blob = f.read()
            with open(mp) as f: meta = json.load(f)
            if meta.get("context_id") != ctx.id: raise ValueError("identity mismatch")
            ctx.tokens = [int(x) for x in meta.get("tokens", [])]
            ctx.disk = bp
            return True
        except (OSError, ValueError, TypeError):
            ctx.blob = None; ctx.tokens = []
            return False

    def spill(self, ctx):
        if ctx.blob is None or ctx.active_job is not None: return
        bp, mp = self.disk_paths(ctx.id)
        tb, tm = bp + ".tmp.%d" % os.getpid(), mp + ".tmp.%d" % os.getpid()
        with open(tb, "wb") as f: f.write(ctx.blob); f.flush(); os.fsync(f.fileno())
        with open(tm, "w") as f:
            json.dump({"schema": 1, "context_id": ctx.id, "tokens": ctx.tokens,
                       "last_access": ctx.last_access, "internal": ctx.internal},
                      f, separators=(",", ":"))
            f.flush(); os.fsync(f.fileno())
        os.chmod(tb, 0o600); os.chmod(tm, 0o600)
        os.replace(tb, bp); os.replace(tm, mp)
        ctx.disk, ctx.blob = bp, None

    def maintain(self):
        now = time.time()
        warm = [c for c in self.contexts.values() if c.blob is not None and c.active_job is None]
        for c in warm:
            if now - c.last_access >= self.memory_ttl: self.spill(c)
        warm = sorted((c for c in self.contexts.values() if c.blob is not None and c.active_job is None),
                      key=lambda c: c.last_access)
        used = sum(len(c.blob) for c in warm)
        for c in warm:
            if used <= self.memory_cap: break
            n = len(c.blob); self.spill(c); used -= n
        for c in list(self.contexts.values()):
            if c.active_job is None and c.disk and now - c.last_access >= self.disk_ttl:
                bp, mp = self.disk_paths(c.id)
                for p in (bp, mp):
                    try: os.unlink(p)
                    except OSError: pass
                self.contexts.pop(c.id, None)
        disk = sorted((c for c in self.contexts.values() if c.disk), key=lambda c: c.last_access)
        used = sum(os.path.getsize(c.disk) for c in disk if os.path.exists(c.disk))
        for c in disk:
            if used <= self.disk_cap: break
            bp, mp = self.disk_paths(c.id); n = os.path.getsize(bp) if os.path.exists(bp) else 0
            for p in (bp, mp):
                try: os.unlink(p)
                except OSError: pass
            c.disk = None; used -= n
            if c.blob is None: self.contexts.pop(c.id, None)

    def status(self):
        now = time.time()
        return [{"id": c.id, "state": (c.active_job.phase if c.active_job else "warm" if c.blob is not None else "disk"),
                 "tokens": len(c.tokens), "last_access": c.last_access,
                 "idle_seconds": max(0, now - c.last_access),
                 "bytes": len(c.blob) if c.blob is not None else
                          (os.path.getsize(c.disk) if c.disk and os.path.exists(c.disk) else 0),
                 "active_job": c.active_job.id if c.active_job else None,
                 "prompt_total": len(c.active_job.prompt) if c.active_job else 0,
                 "prompt_processed": c.active_job.off if c.active_job else 0,
                 "completion_tokens": len(c.active_job.out) if c.active_job else 0,
                 "cache_hit": bool(c.active_job.cache_hit) if c.active_job else False,
                 "cached_tokens": c.active_job.cached_tokens if c.active_job and c.active_job.cache_hit else 0,
                 "started": c.active_job.started if c.active_job else None,
                 "decode_started": c.active_job.decode_started if c.active_job else None}
                for c in self.contexts.values() if not c.internal]

    def shared_prefix(self, prompt, exclude):
        candidates = [c for c in self.contexts.values()
                      if c is not exclude and c.active_job is None and c.tokens and
                      len(prompt) >= len(c.tokens) and prompt[:len(c.tokens)] == c.tokens]
        candidates.sort(key=lambda c: len(c.tokens), reverse=True)
        for c in candidates:
            if self.load_context(c):
                c.last_access = time.time()
                return c
        return None

    def cache_prompt(self, job):
        if job.prompt_cached or not job.prompt: return
        raw = json.dumps(job.prompt, separators=(",", ":")).encode()
        cid = "@prompt:" + hashlib.sha256(raw).hexdigest()
        c = self.contexts.get(cid)
        if c is None:
            c = Context(cid, internal=True); self.contexts[cid] = c
        c.blob = self.sess.context_export()
        c.tokens = list(job.prompt); c.last_access = time.time(); c.internal = True
        if c.disk:
            bp, mp = self.disk_paths(c.id)
            for p in (bp, mp):
                try: os.unlink(p)
                except OSError: pass
        c.disk = None; job.prompt_cached = True

    def accept(self):
        while True:
            try: conn, _ = self.listener.accept()
            except BlockingIOError: return
            conn.settimeout(0.05)
            try:
                data = b""
                while b"\n" not in data:
                    part = conn.recv(65536)
                    if not part: raise ValueError("closed request")
                    data += part
                req = json.loads(data.split(b"\n", 1)[0])
                op = req.get("op", "generate")
                if op == "contexts":
                    conn.sendall((json.dumps({"event": "contexts", "data": self.status(),
                        "decode_batch": {"enabled": self.decode_batch_size > 1,
                                         "capacity": self.decode_batch_size,
                                         "steps": self.batch_steps,
                                         "sequences": self.batch_sequences}}) + "\n").encode()); conn.close(); continue
                if op == "delete":
                    cid = str(req.get("context_id", "")); c = self.contexts.get(cid)
                    if c and c.active_job: code = 409
                    else:
                        if c:
                            bp, mp = self.disk_paths(cid)
                            for p in (bp, mp):
                                try: os.unlink(p)
                                except OSError: pass
                            self.contexts.pop(cid, None)
                        code = 204
                    conn.sendall((json.dumps({"event": "deleted", "status": code}) + "\n").encode()); conn.close(); continue
                job = Job(conn, req); ctx = self.contexts.setdefault(job.context_id, Context(job.context_id))
                if ctx.active_job is not None:
                    conn.sendall(b'{"event":"error","status":409,"error":"context busy"}\n'); conn.close(); continue
                ctx.active_job = job; self.jobs.append(job)
                self.send(job, {"event": "accepted", "job_id": job.id, "context_id": job.context_id})
            except Exception as exc:
                try: conn.sendall((json.dumps({"event": "error", "error": str(exc)}) + "\n").encode()); conn.close()
                except OSError: pass

    def activate(self, job):
        if self.current is job: return
        if self.current and not self.current.cancelled and self.current.phase not in ("done", "error"):
            self.current.working = self.sess.context_export()
        if job.working is not None:
            if self.sess.context_import(job.working) != 0: raise RuntimeError("working context restore failed")
        else:
            ctx = self.contexts[job.context_id]
            limit = max(1, self.sess.maxpos() - min(job.max_new, self.sess.maxpos() - 1))
            if len(job.prompt) > limit:
                job.prompt = truncate_prompt(job.prompt, limit)
            self.load_context(ctx)
            reuse = bool(ctx.blob is not None and len(job.prompt) >= len(ctx.tokens) and
                         job.prompt[:len(ctx.tokens)] == ctx.tokens)
            if reuse:
                if self.sess.context_import(ctx.blob) != 0: raise RuntimeError("context restore failed")
                job.off = len(ctx.tokens)
                job.cache_hit = True; job.cached_tokens = job.off
            else:
                shared = self.shared_prefix(job.prompt, ctx)
                if shared is not None:
                    if self.sess.context_import(shared.blob) != 0:
                        raise RuntimeError("shared prefix restore failed")
                    job.off = len(shared.tokens)
                    job.cache_hit = True; job.cached_tokens = job.off
            if (not job.cache_hit and job.cache_load and job.cache_path and
                  os.path.isfile(job.cache_path)):
                self.sess.reset()
                if self.sess.kv_restore(job.cache_path) != 0:
                    raise RuntimeError("prefix cache restore failed")
                restored = self.sess.pos()
                if restored != job.cached_tokens or restored > len(job.prompt):
                    self.sess.reset()
                    raise RuntimeError("prefix cache length mismatch")
                job.off = restored; job.cache_hit = True
            elif not job.cache_hit:
                self.sess.reset(); job.off = 0
            job.phase = "prefill" if job.off < len(job.prompt) else "decode"
        self.current = job

    def finish(self, job, error=None):
        ctx = self.contexts[job.context_id]
        if job.slot >= 0:
            try:
                job.working = self.sess.slot_export(job.slot)
            except Exception as exc:
                if error is None: error = exc
            self.sess.slot_release(job.slot); self.free_slots.append(job.slot)
            job.slot = -1; self.current = None
        if not job.cancelled and error is None:
            ctx.blob = job.working if job.working is not None and self.current is not job \
                       else self.sess.context_export()
            ctx.tokens = job.prompt + job.out
            ctx.last_access = time.time()
            if ctx.disk:
                bp, mp = self.disk_paths(ctx.id)
                for p in (bp, mp):
                    try: os.unlink(p)
                    except OSError: pass
            ctx.disk = None
            self.send(job, {"event": "final", "tokens": job.out,
                            "context_id": job.context_id,
                            "elapsed_ms": int((time.time() - job.started) * 1000)})
        elif error is not None:
            self.send(job, {"event": "error", "error": str(error)})
        ctx.active_job = None; job.phase = "done"
        try: job.sock.close()
        except OSError: pass
        if job in self.jobs: self.jobs.remove(job)
        if self.current is job: self.current = None
        if job.ephemeral: self.contexts.pop(job.context_id, None)

    def step(self, job):
        try:
            self.activate(job)
            if job.cancelled: return self.finish(job)
            deadline = time.time() + self.quantum_s
            if job.phase == "prefill":
                quantum = self.single_prefill_q if len(self.jobs) == 1 else self.prefill_q
                end = min(len(job.prompt), job.off + quantum)
                if end > job.off and self.sess.prefill(job.prompt[job.off:end], job.off) != 0:
                    raise RuntimeError("prefill failed at %d" % job.off)
                job.off = end
                self.send(job, {"event": "progress", "phase": "prefill",
                                "processed": job.off, "total": len(job.prompt)})
                if job.off >= len(job.prompt):
                    self.maybe_cache_hot_experts(job)
                    self.cache_prompt(job)
                    if job.cache_save_path and not job.cache_saved:
                        if self.sess.kv_save(job.cache_save_path) != 0:
                            raise RuntimeError("prefix cache save failed")
                        job.cache_saved = True
                    job.phase = "decode"; job.decode_started = time.time()
            if job.phase == "decode" and time.time() < deadline:
                for _ in range(self.decode_q):
                    if len(job.out) >= job.max_new or self.sess.pos() >= self.sess.maxpos():
                        return self.finish(job)
                    tok = self.sess.sample(job.sp)
                    if tok < 0: raise RuntimeError("sampling failed")
                    if tok == self.sess.eos: return self.finish(job)
                    pos = self.sess.pos()
                    if self.sess.decode(tok, pos) < 0: raise RuntimeError("decode failed at %d" % pos)
                    job.out.append(tok); self.send(job, {"event": "token", "token": tok})
                    self.maybe_cache_hot_experts(job, decode_ready=True)
                    if time.time() >= deadline: break
        except Exception as exc:
            self.finish(job, exc)

    def step_decode_batch(self, batch):
        ready, tokens, slots = [], [], []
        try:
            # Sampling remains per-context (temperature/history/RNG), then the
            # expensive transformer forward is shared across the group.
            for job in batch:
                if job.slot < 0:
                    self.activate(job)
                    blob = self.sess.context_export()
                    if not self.free_slots: raise RuntimeError("no native decode slot available")
                    job.slot = self.free_slots.pop(0)
                    if self.sess.slot_import(job.slot, blob) != 0:
                        self.free_slots.append(job.slot); job.slot = -1
                        raise RuntimeError("native decode slot import failed")
                    job.working = None; self.current = None
                if job.cancelled or len(job.out) >= job.max_new or \
                        len(job.prompt) + len(job.out) >= self.sess.maxpos():
                    self.finish(job); continue
                tok = self.sess.slot_sample(job.slot, job.sp)
                if tok < 0: raise RuntimeError("sampling failed")
                if tok == self.sess.eos: self.finish(job); continue
                ready.append(job); tokens.append(tok); slots.append(job.slot)
            if not ready: return
            if self.sess.decode_slots(slots, tokens) != 0:
                raise RuntimeError("native slot decode failed")
            self.batch_steps += 1; self.batch_sequences += len(ready)
            if self.batch_steps == 1 or (self.batch_steps % 32) == 0:
                print("[runner] decode_batch steps=%d last_n=%d sequences=%d" %
                      (self.batch_steps, len(ready), self.batch_sequences),
                      file=sys.stderr, flush=True)
            self.current = None
            for job, tok in zip(ready, tokens):
                job.out.append(tok)
                self.send(job, {"event": "token", "token": tok, "batch": len(ready)})
                self.maybe_cache_hot_experts(job, decode_ready=True)
                if job.cancelled or len(job.out) >= job.max_new:
                    self.finish(job)
        except Exception as exc:
            self.current = None
            for job in batch:
                if job in self.jobs: self.finish(job, exc)

    def run(self):
        try:
            while True:
                self.accept()
                decode = [j for j in self.jobs if j.phase == "decode" and not j.cancelled]
                admitting = [j for j in self.jobs if j.phase != "decode" and not j.cancelled]
                # Fill one batch cohort before decoding it. Without this gate,
                # the first two completed prefills form a permanent B=2 group
                # and starve the remaining members that would have made B=4.
                if (self.decode_batch_size > 1 and admitting and
                        len(self.jobs) <= self.decode_batch_size):
                    job = admitting[0]; self.jobs.remove(job); self.jobs.append(job); self.step(job)
                elif (decode and self.decode_batch_size > 1 and
                      (len(decode) >= 2 or decode[0].slot >= 0)):
                    self.step_decode_batch(decode[:self.decode_batch_size])
                elif self.jobs:
                    job = self.jobs.pop(0); self.jobs.append(job); self.step(job)
                else: time.sleep(0.005)
                self.maintain()
        finally:
            self.listener.close()
            try: os.unlink(self.path)
            except OSError: pass


def run_serve(sess, base, prefix_cache, slots):
    req = base + ".req"; resp = base + ".resp"
    reqseq = base + ".reqseq"; respseq = base + ".respseq"
    error = base + ".error"
    # per-slot cache paths (slot 0 is the live context; others are switched in)
    slot_path = [base + ".slot.%d" % i for i in range(slots)]
    syscache = os.environ.get("DS4F_SERVE_SYSCACHE")
    if syscache and os.path.exists(syscache):
        restored = sess.kv_restore(syscache)
        print("[runner] system-prompt cache %s from %s (pos=%d)" %
              ("loaded" if restored == 0 else "restore failed", syscache, sess.pos()),
              file=sys.stderr, flush=True)
    done = 0
    print("[runner] serving on %s.* slots=%d prefix_cache=%d" % (base, slots, prefix_cache),
          file=sys.stderr, flush=True)
    while True:
        # wait for a request
        try:
            with open(reqseq) as f:
                rs = int(f.read().strip() or 0)
        except (OSError, ValueError):
            rs = 0
        if rs <= done:
            time.sleep(0.005)
            continue
        prompt = []
        t0 = time.time()
        try:
            # Read and validate the complete request before touching the session.
            with open(req) as f:
                body = f.read().splitlines()
            if not body:
                raise ValueError("empty request")
            hdr = body[0].split()
            if len(hdr) < 9:
                raise ValueError("bad header: %r" % hdr)
            max_new, temp, top_p, top_k = int(hdr[0]), float(hdr[1]), float(hdr[2]), int(hdr[3])
            pres, rep, seed, slot, ctl = (float(hdr[4]), float(hdr[5]), int(hdr[6]),
                                          int(hdr[7]), int(hdr[8]))
            li = 1
            cache_path = None
            save_path = None
            if ctl & 1:
                if len(body) <= li: raise ValueError("missing cache path")
                cache_path = body[li]; li += 1
            if ctl & 2:
                if len(body) <= li: raise ValueError("missing save path")
                save_path = body[li]; li += 1
            prompt = [int(x) for x in body[li].split()] if len(body) > li else []
            stream_path = (base + ".tok") if (ctl & 4) else None
            gen = generate(sess, prompt, max_new, temp, top_p, top_k, pres, rep, seed,
                           slot, ctl, cache_path, prefix_cache, slots, slot_path,
                           stream_path, save_path)
            payload = " ".join(map(str, gen))
            try: os.unlink(error)
            except FileNotFoundError: pass
        except Exception as exc:
            gen = []
            payload = ""
            message = "%s: %s" % (type(exc).__name__, exc)
            print("[runner] seq %d FAILED: %s" % (rs, message), file=sys.stderr, flush=True)
            tmp = "%s.tmp.%d" % (error, os.getpid())
            with open(tmp, "w") as f:
                f.write(message + "\n")
            os.replace(tmp, error)
        tmp = "%s.tmp.%d" % (resp, os.getpid())
        with open(tmp, "w") as f:
            f.write(payload)
        os.replace(tmp, resp)
        done = rs
        with open(respseq, "w") as f:
            f.write(str(rs))
        elapsed = max(time.time() - t0, 1e-6)
        print("[runner] seq %d ids=%d gen=%d %.2fs (%.2f tok/s)" %
              (rs, len(prompt), len(gen), elapsed,
               (len(gen) / elapsed) if gen else 0.0),
              file=sys.stderr, flush=True)


def truncate_prompt(prompt, limit):
    """Keep the system prompt / tool definitions (the head) plus the recent
    tail when the conversation would exceed the context ceiling.  The head and
    the tail are both kept whole; the middle turns are dropped."""
    if len(prompt) <= limit:
        return prompt
    head = max(1, limit * 2 // 3)
    tail = limit - head
    if tail < 1:
        return prompt[-limit:]
    return prompt[:head] + prompt[-tail:]


def generate(sess, prompt, max_new, temp, top_p, top_k, pres, rep, seed,
             slot, ctl, cache_path, prefix_cache, slots, slot_path,
             stream_path=None, save_path=None):
    sp = Sampling(temp, top_p, top_k, pres, rep, seed)
    maxpos = sess.maxpos()
    limit = max(1, min(maxpos - max_new, PREFILL_MAX_TOKENS))
    if len(prompt) > limit:
        print("[runner] truncating prompt %d -> %d tokens" % (len(prompt), limit),
              file=sys.stderr, flush=True)
        prompt = truncate_prompt(prompt, limit)
    start = sess.pos()
    _dbg = os.environ.get("DS4F_SERVE_DEBUG")
    _t0 = time.time()

    if ctl & 1 and cache_path and os.path.exists(cache_path):
        # load a cached prefix; prefill only the tokens after the cached length.
        # The cache only applies when its length is a prefix of the prompt; a
        # longer cache (e.g. the previous turn rendered the tool call in fewer
        # tokens than the model generated) must fall back to a fresh prefill.
        restored = sess.kv_restore(cache_path)
        cached = sess.pos()
        if restored != 0 or cached > len(prompt):
            if restored != 0:
                print("[runner] KV restore failed; falling back to full prefill: %s" % cache_path,
                      file=sys.stderr, flush=True)
            sess.reset()
            if prompt:
                if prefill_chunked(sess, prompt, 0) != 0:
                    raise RuntimeError("full prefill failed")
        else:
            tail = prompt[cached:]
            if tail:
                if prefill_chunked(sess, tail, cached) != 0:
                    raise RuntimeError("cached-tail prefill failed")
    elif prefix_cache and slots > 1 and os.path.exists(slot_path[slot % slots]):
        if sess.kv_restore(slot_path[slot % slots]) != 0:
            raise RuntimeError("slot KV restore failed")
        cached = sess.pos()
        tail = prompt[cached:]
        if tail:
            if prefill_chunked(sess, tail, cached) != 0:
                raise RuntimeError("slot-tail prefill failed")
    else:
        # a fresh (no-cache, no-slot) request carries the full conversation;
        # the session must start at position 0 or the KV is written at the
        # previous turn's position (context corruption).
        if sess.pos() != 0:
            sess.reset()
        if prompt:
            if prefill_chunked(sess, prompt, 0) != 0:
                raise RuntimeError("prefill failed")

    prefill_s = max(time.time() - _t0, 1e-9)

    # the generation loop: sample + decode until max_new or EOS
    out = []
    pos = sess.pos()
    decode_t0 = time.time()
    tf = None
    if stream_path:
        tf = open(stream_path, "w")
    for _ in range(max_new):
        tok = sess.sample(sp)
        if tok < 0:
            raise RuntimeError("sampling failed")
        if tok == sess.eos:
            break
        ar = sess.decode(tok, pos)
        if ar < 0:
            raise RuntimeError("decode failed at position %d" % pos)
        out.append(tok)
        pos += 1
        if tf is not None:
            tf.write(str(tok) + "\n")
            tf.flush()
        if ar == sess.eos:
            break
        if pos >= maxpos:
            break
    if tf is not None:
        tf.close()

    decode_s = max(time.time() - decode_t0, 1e-9)
    if os.environ.get("DS4F_SERVE_BENCH"):
        print("[bench] prefill tokens=%d seconds=%.6f tok/s=%.3f; "
              "decode tokens=%d seconds=%.6f tok/s=%.3f" %
              (len(prompt), prefill_s, len(prompt) / prefill_s,
               len(out), decode_s, len(out) / decode_s if out else 0.0),
              file=sys.stderr, flush=True)

    if _dbg:
        print("[runner] gen prefill+decode %.1fs gen=%d pos=%d" %
              (time.time() - _t0, len(out), sess.pos()), file=sys.stderr, flush=True)
    # cache save: the KV up to the current position (prefix for the next turn)
    spath = save_path or cache_path
    if ctl & 2 and spath:
        _ts = time.time()
        saved = sess.kv_save(spath)
        if saved != 0:
            raise RuntimeError("KV save failed: %s" % spath)
        if _dbg:
            print("[runner] kv_save %.2fs rc=%s" % (time.time() - _ts, saved),
                  file=sys.stderr, flush=True)
    elif prefix_cache and slots > 1:
        if sess.kv_save(slot_path[slot % slots]) != 0:
            raise RuntimeError("slot KV save failed")
    return out


def daemonize():
    if os.fork() > 0: os._exit(0)          # first fork: parent exits
    os.setsid()                            # new session
    if os.fork() > 0: os._exit(0)          # second fork: detach from tty
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        try: os.dup2(devnull, fd)
        except OSError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--unix-socket")
    ap.add_argument("--context-dir")
    ap.add_argument("--context-memory-ttl-sec", type=int, default=600)
    ap.add_argument("--context-disk-ttl-sec", type=int, default=86400)
    ap.add_argument("--context-memory-mb", type=int, default=512)
    ap.add_argument("--context-disk-mb", type=int, default=8192)
    ap.add_argument("--prefill-quantum-tokens", type=int, default=32)
    ap.add_argument("--single-prefill-quantum-tokens", type=int, default=2048)
    ap.add_argument("--hip-prefill-tuned", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-fused-shared-ffn", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-prefill-attn", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-tb2-batch", type=int, choices=(0, 1), default=0,
                    help="issue one exact Tier-B2 GPU window partial per layer/tile")
    ap.add_argument("--hip-qkv-fuse", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-qkv-device-chain", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-attn-device-chain", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-attn-no-d2h", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-routed-ffn", type=int, choices=(0, 1), default=0,
                    help="stream routed experts to GPU during prefill (CPU is faster on Gen3 PCIe)")
    ap.add_argument("--hip-decode-routed-ffn", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-fp8-wmma", type=int, choices=(0, 1, 2), default=2)
    ap.add_argument("--hip-bf16-wmma", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-attn-wmma", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-oproj-group-wmma", type=int, choices=(0, 1, 2), default=2)
    ap.add_argument("--hip-mxfp4-wmma", type=int, choices=(0, 1, 2), default=1)
    ap.add_argument("--hip-expert-stream", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-expert-pinned-staging", type=int, choices=(0, 1), default=0,
                    help="coalesce streamed expert weights through two pinned host buffers")
    ap.add_argument("--hip-expert-cache-mb", default="0",
                    help="0 disables, auto uses free VRAM, or an explicit MiB budget")
    ap.add_argument("--hip-expert-cache-reserve-mb", type=int, default=1536)
    ap.add_argument("--hip-expert-cache-stats", type=int, choices=(0, 1), default=0)
    ap.add_argument("--hip-block-threads", type=int, choices=(64, 128, 256), default=128)
    # Accepted here so the single-node wrapper can expose one unified program
    # argument list; the value itself configures the HTTP frontend.
    ap.add_argument("--agent-cache-max-tokens", type=int, default=14336)
    ap.add_argument("--default-temperature", type=float, default=0.0)
    ap.add_argument("--default-top-p", type=float, default=1.0)
    ap.add_argument("--runner-timeout-sec", type=float, default=3600.0)
    ap.add_argument("--decode-quantum-tokens", type=int, default=4)
    ap.add_argument("--decode-batch-size", type=int, default=1)
    ap.add_argument("--scheduler-quantum-ms", type=int, default=250)
    args = ap.parse_args()
    try:
        expert_cache_mb = (-1 if str(args.hip_expert_cache_mb).lower() == "auto"
                           else int(args.hip_expert_cache_mb))
    except ValueError:
        ap.error("--hip-expert-cache-mb must be 0, auto, or an integer MiB value")
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    if args.daemon:
        daemonize()
    base = os.environ.get("DS4F_SERVE_BASE", "/tmp/ds4f_serve")
    stage = os.environ.get("DS4F_STAGE_DIR")
    if not stage:
        sys.exit("DS4F_STAGE_DIR is required (the single-node staged manifest dir)")
    lib = load_lib(os.environ.get("DS4F_SERVE_LIB", LIB))
    sess = Serve(lib, stage,
                 use_hip=env_i("DS4F_SERVE_USE_HIP", 0),
                 hip_device=env_i("DS4F_HIP_DEVICE", 0),
                 threads=env_i("LLM_THREADS", 16),
                 cmgs=env_i("DS4F_CMGS", 1),
                 max_pos=env_i("DS4F_MAXPOS", 16384),
                 hip_prefill=(args.hip_prefill_tuned, args.hip_fused_shared_ffn,
                              args.hip_prefill_attn, args.hip_qkv_fuse,
                              args.hip_qkv_device_chain, args.hip_attn_device_chain,
                              args.hip_attn_no_d2h, args.hip_routed_ffn,
                              args.hip_fp8_wmma, args.hip_bf16_wmma,
                              args.hip_attn_wmma, args.hip_oproj_group_wmma,
                              args.hip_mxfp4_wmma, args.hip_expert_stream,
                              args.hip_block_threads, args.hip_expert_pinned_staging,
                              args.hip_tb2_batch, args.hip_decode_routed_ffn))
    slots = max(1, env_i("DS4F_SERVE_SLOTS", 1))
    prefix_cache = env_i("DS4F_SERVE_PREFIX_CACHE", 1)
    if expert_cache_mb and sess.enable_route_telemetry(True) != 0:
        sys.exit("serving library lacks routed-expert telemetry support; rebuild it")
    try:
        if args.unix_socket:
            CooperativeServer(sess, args.unix_socket,
                              args.context_dir or (base + ".contexts"),
                              args.context_memory_ttl_sec, args.context_disk_ttl_sec,
                              args.context_memory_mb, args.context_disk_mb,
                              args.prefill_quantum_tokens, args.decode_quantum_tokens,
                              args.scheduler_quantum_ms, args.decode_batch_size,
                              args.single_prefill_quantum_tokens,
                              expert_cache_mb, args.hip_expert_cache_reserve_mb,
                              args.hip_expert_cache_stats).run()
        else:
            run_serve(sess, base, prefix_cache, slots)
    except KeyboardInterrupt:
        pass
    finally:
        sess.close()


if __name__ == "__main__":
    main()
