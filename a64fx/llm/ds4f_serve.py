#!/usr/bin/env python3
"""OpenAI-compatible HTTP API for the DeepSeek-V4-Flash 11-node EP runner (DS4F_SERVE mode).

Runs on the controller node; drives the persistent `ds4f_ep_runner` (loaded once, looping on
requests) over shared-FS files:  <BASE>.req / .reqseq  (prompt in) and  <BASE>.resp / .respseq
(generated ids out). The runner's 11 ranks all read the same request -> lockstep, no broadcast.

Endpoints:
  POST /v1/chat/completions  {"messages":[...], "tools":[...], "stream":bool, ...}  (OpenAI chat)
  POST /v1/responses         Responses API requests for Codex
  POST /v1/messages          Anthropic Messages requests for Claude Code
  POST /v1/completions       {"prompt": str, "max_tokens": int, ...sampling}        (OpenAI text)
  POST /completion           {"prompt": str, "n_predict": int, ...sampling}         (llama.cpp)
  GET  /v1/models            {"object":"list","data":[{"id":"ds4f",...}]}
  GET  /health

DS4F-Flash ships NO chat_template and no role/tool special tokens (base-model tokenizer), so chat
uses a plain-text role convention and tool-calling is PROMPT-INJECTED: the server appends the tool
schemas to the system prompt with an instruction to emit `<tool_call>{...}</tool_call>`, then parses
that text back into OpenAI `tool_calls`. Streaming is PSEUDO-STREAM: the runner generates the full
turn (blocking), then this frontend emits it as SSE deltas (pi's openai-completions client forces
stream:true). Sampling params (all optional): temperature (<=0 => greedy, default 0), top_p (1.0),
top_k (0=off), presence_penalty (0), repeat_penalty / repetition_penalty (1.0), seed.

Env: PORT (8080), TOK (~/models/ds4f/tokenizer.json), DS4F_SERVE_BASE,
DS4F_SERVE_AGENT_CACHE_DIR, DS4F_SERVE_TIMEOUT (1200s).
Start via run_ds4f_serve_11n.sh (which launches the runner first, then this)."""
import argparse, hashlib, http.server, json, os, re, socket, socketserver, subprocess, sys, tempfile, threading, time, uuid


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.environ.get("TOK", os.path.expanduser("~/models/ds4f/tokenizer.json"))
TOKCLI = os.path.join(HERE, "tools", "ds4f_tokenizer.py")
BASE = os.environ.get("DS4F_SERVE_BASE", "/tmp/ds4f_serve")
REQ, RESP, REQSEQ, RESPSEQ = BASE + ".req", BASE + ".resp", BASE + ".reqseq", BASE + ".respseq"
ERROR = BASE + ".error"
PORT = int(os.environ.get("PORT", "8080"))
TIMEOUT = float(os.environ.get("DS4F_SERVE_TIMEOUT", "1200"))
MODEL_ID = "ds4f"
RUNNER_SOCKET = None
RESPONSE_STATE_DIR = BASE + ".contexts/responses"
_runner_decode_batch = {"enabled": False, "capacity": 1, "steps": 0, "sequences": 0}
_last_runner_contexts = []
_cooperative_progress = {"active": False}
_cooperative_progress_lock = threading.Lock()
_lock = threading.Lock()        # the runner is single-stream: serialize requests
_seq = 0
_started_at = time.time()


def progress_snapshot():
    """Return a llama.cpp-style lightweight progress snapshot.

    The file-backed runner has no token callback while idle, so report the
    request queue state and stable zero counters rather than returning 404.
    """
    req = os.path.exists(REQ)
    try:
        seq = int(open(REQSEQ).read().strip() or 0)
    except (OSError, ValueError):
        seq = 0
    try:
        resp_seq = int(open(RESPSEQ).read().strip() or 0)
    except (OSError, ValueError):
        resp_seq = 0
    contexts = runner_contexts() if RUNNER_SOCKET else []
    live = [c for c in contexts if c.get("state") == "active"]
    if not live:
        live = [c for c in contexts if c.get("state") in ("prepare", "prefill", "decode")]
    active = bool(live) if RUNNER_SOCKET else bool(req and seq > resp_seq)
    current = live[0] if live else {}
    with _cooperative_progress_lock:
        direct = dict(_cooperative_progress)
    if RUNNER_SOCKET and direct.get("active"):
        active = True
        current = direct
    now = time.time()
    started = current.get("started") or now
    decode_started = current.get("decode_started") or now
    processed = int(current.get("prompt_processed", 0))
    completed = int(current.get("completion_tokens", 0))
    return {
        "active": active,
        "job_id": current.get("active_job") if RUNNER_SOCKET else
                  (str(seq) if active else None),
        "model": MODEL_ID,
        "precision": "ds4f",
        "phase": current.get("state", "decode") if active else "idle",
        "prompt_tokens": int(current.get("prompt_total", 0)),
        "prompt_processed": processed,
        "prompt_tps": processed / max(now - started, 1e-6),
        "completion_tokens": completed,
        "completion_total": completed,
        "decode_tps": completed / max(now - decode_started, 1e-6),
        "cache_hit": bool(current.get("cache_hit", False)),
        "gpu": "hip" if os.environ.get("DS4F_SERVE_USE_HIP", "0") == "1" else "cpu",
        "elapsed_ms": int(((now - started) if active else (now - _started_at)) * 1000),
        "queue_depth": max(0, len(live) - 1),
        "contexts": {"active": max(len(live), 1 if active and RUNNER_SOCKET else 0),
                     "warm": sum(c.get("state") == "warm" for c in contexts),
                     "disk": sum(c.get("state") == "disk" for c in contexts)},
        "decode_batch": dict(_runner_decode_batch),
    }

# ---- concurrent batched decode (DS4F_SERVE_BATCH>1): a dispatcher thread collects queued requests
# within a short window and submits them as ONE "BATCH" request the runner decodes together (dense
# GEMM + EP reduce amortize -> ~2x aggregate throughput). Greedy only (the batch runner ignores
# sampling params); cache ops fall back to the single-request path. ----
import queue as _queue
BATCH = int(os.environ.get("DS4F_SERVE_BATCH", "1"))
BATCH_WINDOW = float(os.environ.get("DS4F_SERVE_BATCH_WINDOW", "0.03"))   # 30 ms coalesce window
_batch_q = _queue.Queue()

class _BReq:
    __slots__ = ("ids", "max_new", "ev", "gen", "samp")
    def __init__(self, ids, max_new, samp=None):
        self.ids, self.max_new, self.ev, self.gen = ids, max_new, threading.Event(), []
        self.samp = samp

def _batch_dispatcher():
    global _seq
    while True:
        first = _batch_q.get()
        batch = [first]
        t_end = time.time() + BATCH_WINDOW
        while len(batch) < BATCH:
            to = t_end - time.time()
            if to <= 0: break
            try: batch.append(_batch_q.get(timeout=to))
            except _queue.Empty: break
        _seq += 1
        with open(REQ, "w") as f:
            f.write("BATCH %d\n" % len(batch))
            for r in batch:
                sp = r.samp or {"temperature": 0.0, "top_p": 1.0, "top_k": 0,
                                "repeat_penalty": 1.0, "presence_penalty": 0.0, "seed": None}
                seed = sp.get("seed") if sp.get("seed") is not None else 0
                # per-seq line: "max_new temp top_p top_k seed rep_pen pres_pen" (runner parses; greedy if temp<=0)
                f.write("%d %g %g %d %d %g %g\n" % (r.max_new, sp["temperature"], sp["top_p"],
                        sp["top_k"], seed, sp["repeat_penalty"], sp["presence_penalty"]))
                f.write(" ".join(map(str, r.ids)) + "\n")
        with open(REQSEQ, "w") as f:
            f.write(str(_seq) + "\n")
        t0 = time.time()
        while True:
            try:
                with open(RESPSEQ) as f: rs = int(f.read().strip() or 0)
            except (OSError, ValueError): rs = 0
            if rs >= _seq: break
            if time.time() - t0 > TIMEOUT:
                for r in batch: r.ev.set()
                break
            time.sleep(0.005)
        try:
            lines = open(RESP).read().splitlines()
            n = int(lines[0]) if lines else 0
        except (OSError, ValueError, IndexError):
            n = 0; lines = []
        for i, r in enumerate(batch):
            if i < n and 1 + i < len(lines) and lines[1 + i].strip():
                r.gen = [int(x) for x in lines[1 + i].split()]
            r.ev.set()

def infer_batched(prompt, max_tokens, samp=None):
    ids = encode(prompt)
    if not ids: return [], [], ""
    r = _BReq(ids, max_tokens, samp)
    _batch_q.put(r)
    if not r.ev.wait(TIMEOUT): raise TimeoutError("runner timeout")
    return ids, r.gen, decode(r.gen)

# ---- DYNAMIC continuous batching (DS4F_SERVE_DYNAMIC): each request enqueues itself (q.<id> file +
# atomic qhead bump) and polls for its own r.<id> response. The runner admits mid-flight, so a fast
# request is NOT blocked behind slow batch-mates. No coalescing thread needed. ----
DYNAMIC = int(os.environ.get("DS4F_SERVE_DYNAMIC", "0"))
_qlock = threading.Lock()
_qnext = 0

def infer_dynamic(prompt, max_tokens, samp=None):
    global _qnext
    ids = encode(prompt)
    if not ids: return [], [], ""
    if samp is None:
        samp = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "presence_penalty": 0.0,
                "repeat_penalty": 1.0, "seed": None}
    seed = samp.get("seed") if samp.get("seed") is not None else 0   # 0 -> runner derives from request id
    with _qlock:
        myid = _qnext; _qnext += 1
        tmp = "%s.q.%d.t" % (BASE, myid)        # write atomically: the runner probes q.<id> existence,
        with open(tmp, "w") as f:               # so it must never observe a half-written file
            # first line: "max_new temp top_p top_k seed rep_pen pres_pen" (per-sequence sampling)
            f.write("%d %g %g %d %d %g %g\n" % (max_tokens, samp["temperature"], samp["top_p"],
                    samp["top_k"], seed, samp["repeat_penalty"], samp["presence_penalty"]))
            f.write(" ".join(map(str, ids)) + "\n")
        os.rename(tmp, "%s.q.%d" % (BASE, myid))  # atomic publish
        with open(BASE + ".qhead", "w") as f:   # client-side id bookkeeping (runner probes q.<id>, not qhead)
            f.write(str(_qnext) + "\n")
    rf = "%s.r.%d" % (BASE, myid)
    t0 = time.time()
    while not os.path.exists(rf):               # runner renames the finished response into place (atomic)
        if time.time() - t0 > TIMEOUT: raise TimeoutError("runner timeout")
        time.sleep(0.004)
    gen = [int(x) for x in open(rf).read().split()]
    try: os.unlink(rf)
    except OSError: pass
    return ids, gen, decode(gen)

# ---- SOCKET transport (DS4F_SERVE_SOCK): the runner's rank 0 hosts a TCP listener over the Tofu IP and
# publishes "<ip> <port>" to <base>.sock. Each request opens a connection, sends [u32 BE len][payload],
# reads [u32 BE len][gen ids]. Eliminates the file protocol's ~tens-of-seconds cross-node FS-cache
# admission/response latency (control->compute TCP is ~1 ms; a request is then compute-bound). ----
SOCK = int(os.environ.get("DS4F_SERVE_SOCK", "0"))
_sock_addr = None
def _sock_get_addr():
    global _sock_addr
    if _sock_addr: return _sock_addr
    for _ in range(int(TIMEOUT * 10)):
        try:
            ip, port = open(BASE + ".sock").read().split(); _sock_addr = (ip, int(port)); return _sock_addr
        except Exception: time.sleep(0.1)
    raise RuntimeError("no %s.sock (runner not in DS4F_SERVE_SOCK mode?)" % BASE)

def infer_socket(prompt, max_tokens, samp=None):
    import socket, struct
    ids = encode(prompt)
    if not ids: return [], [], ""
    if samp is None:
        samp = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "presence_penalty": 0.0,
                "repeat_penalty": 1.0, "seed": None}
    seed = samp.get("seed") if samp.get("seed") is not None else 0
    payload = ("%d %g %g %d %d %g %g\n%s\n" % (max_tokens, samp["temperature"], samp["top_p"], samp["top_k"],
               seed, samp["repeat_penalty"], samp["presence_penalty"], " ".join(map(str, ids)))).encode()
    ip, port = _sock_get_addr()
    s = socket.socket(); s.settimeout(TIMEOUT); s.connect((ip, port))
    try:
        s.sendall(struct.pack("!I", len(payload)) + payload)
        hdr = b""
        while len(hdr) < 4: hdr += s.recv(4 - len(hdr))
        n = struct.unpack("!I", hdr)[0]; body = b""
        while len(body) < n: body += s.recv(n - len(body))
    finally:
        s.close()
    gen = [int(x) for x in body.split()]
    return ids, gen, decode(gen)

# Plain-text chat template (this checkpoint has no chat_template / role tokens).
BOS = "<｜begin▁of▁sentence｜>"
ROLE_TAG = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}
# Stop markers: the base model can run past its turn -> truncate the completion at the next role tag.
STOP_MARKERS = ["\nUser:", "\nSystem:", "\nTool:", "\n\nUser:", BOS, "<｜end▁of▁sentence｜>"]
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOL_INSTRUCTIONS = (
    "You can call tools. To call a tool, output EXACTLY one line per call of the form\n"
    "<tool_call>{\"name\": \"<tool_name>\", \"arguments\": {<json args>}}</tool_call>\n"
    "and nothing else on that line. Emit a tool call only when you need it; otherwise answer "
    "normally. Available tools (JSON schema):\n"
)


def _tok(args):
    # Python 3.6 (this node) lacks subprocess.run(capture_output=); use PIPE + universal_newlines.
    return subprocess.run([sys.executable, TOKCLI] + args,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def encode(text):
    pf = of = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(text); pf = f.name
        of = pf + ".ids"
        _tok(["encode", "--tokenizer", TOK, "--prompt-file", pf, "--out", of])
        with open(of) as g:
            return [int(x) for x in g.read().split()]
    finally:
        for p in (pf, of):
            if p:
                try: os.unlink(p)
                except OSError: pass


def decode(ids):
    idf = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ids", delete=False) as f:
            f.write(" ".join(map(str, ids))); idf = f.name
        return _tok(["decode", "--tokenizer", TOK, "--ids-file", idf]).stdout
    finally:
        if idf:
            try: os.unlink(idf)
            except OSError: pass


# conversation prefix cache: the KV snapshot (BASE.conv) + the token ids of the
# last prefilled conversation.  The chat handler reuses it when the next request
# extends it, skipping the re-prefill of everything before the new turn.
_conv = {"agent": None, "path": None, "ids": None}
_conv_lock = threading.Lock()
# Durable system-prefix caches are separate from DS4F_SERVE_SYSCACHE, which is
# retained as the runner's legacy single-file checkpoint/preload knob.
AGENT_CACHE_ROOT = os.environ.get("DS4F_SERVE_AGENT_CACHE_DIR", BASE + ".agent-cache")
_cache_lock = threading.Lock()
_cache_stats = {"hits": 0, "misses": 0, "writes": 0}
# Native KV snapshots are reliable for short prefixes but can terminate the
# HIP runner on long prompt saves. Keep long requests uncached and let them
# use the normal bounded-context prefill path instead.
CACHE_MAX_TOKENS = 8192

# Responses API state is keyed by previous_response_id.  It is deliberately
# bounded: Chat/Anthropic clients normally resend their complete history.
_response_contexts = {}
_response_context_ids = {}
_response_context_order = []
_response_context_lock = threading.Lock()
_MAX_RESPONSE_CONTEXTS = int(os.environ.get("DS4F_SERVE_MAX_RESPONSE_CONTEXTS", "256"))


def _cache_identity(agent, prefix_ids):
    token_identity = os.path.abspath(TOK)
    try:
        st = os.stat(TOK)
        token_identity += ":%d:%d" % (st.st_size, int(getattr(st, "st_mtime_ns", st.st_mtime * 1e9)))
    except OSError:
        pass
    raw = ("ds4f-agent-cache-v1\0%s\0%s\0%s\0%s" %
           (agent, MODEL_ID, token_identity, " ".join(map(str, prefix_ids)))).encode()
    return hashlib.sha256(raw).hexdigest()


def _cache_paths(agent, prefix_ids):
    key = _cache_identity(agent, prefix_ids)
    directory = os.path.join(AGENT_CACHE_ROOT, agent)
    return (key, directory, os.path.join(directory, key + ".kv"),
            os.path.join(directory, key + ".json"))


def _read_agent_cache(agent, prefix_ids):
    key, directory, kv_path, meta_path = _cache_paths(agent, prefix_ids)
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if (meta.get("schema_version") != 1 or meta.get("agent") != agent or
                meta.get("model") != MODEL_ID or meta.get("key") != key or
                meta.get("token_ids") != prefix_ids or not os.path.isfile(kv_path) or
                os.path.getsize(kv_path) < 16):
            return None
    except (OSError, ValueError, TypeError):
        return None
    return {"agent": agent, "key": key, "ids": prefix_ids, "path": kv_path,
            "directory": directory, "hit": True}


def _write_agent_cache(agent, prefix_text, prefix_ids):
    key, directory, kv_path, meta_path = _cache_paths(agent, prefix_ids)
    os.makedirs(directory, exist_ok=True)
    tmp_kv = "%s.tmp.%d" % (kv_path, os.getpid())
    tmp_meta = "%s.tmp.%d" % (meta_path, os.getpid())
    try:
        infer(prefix_text, 0, {"temperature": 0.0, "top_p": 1.0, "top_k": 0,
                              "presence_penalty": 0.0, "repeat_penalty": 1.0,
                              "seed": None}, cache_save=True, save_path=tmp_kv)
        if not os.path.isfile(tmp_kv) or os.path.getsize(tmp_kv) < 16:
            raise RuntimeError("runner did not produce a complete system cache")
        meta = {"schema_version": 1, "agent": agent, "model": MODEL_ID,
                "key": key, "tokenizer": os.path.abspath(TOK),
                "token_count": len(prefix_ids), "token_ids": prefix_ids}
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, separators=(",", ":"))
        os.replace(tmp_kv, kv_path)
        os.replace(tmp_meta, meta_path)
        _cache_stats["writes"] += 1
        return {"agent": agent, "key": key, "ids": prefix_ids, "path": kv_path,
                "directory": directory, "hit": False}
    finally:
        for path in (tmp_kv, tmp_meta):
            try:
                os.unlink(path)
            except OSError:
                pass


def prepare_agent_cache(agent, messages, tools, prompt_ids):
    """Return (prompt_ids, cache_path, cache_load, cache_tokens).

    The metadata sidecar makes a frontend restart safe: the raw KV file is
    never restored unless its exact tokenized prefix matches this request.
    """
    prefix_text = sys_prefix_text(messages, tools)
    prefix_ids = encode(prefix_text)
    if len(prefix_ids) > CACHE_MAX_TOKENS:
        print("[cache] agent=%s SKIP prefix_tokens=%d exceeds limit=%d" %
              (agent, len(prefix_ids), CACHE_MAX_TOKENS),
              file=sys.stderr, flush=True)
        return prompt_ids, None, False, 0
    cache = None
    with _cache_lock:
        cache = _read_agent_cache(agent, prefix_ids)
        if cache is None:
            _cache_stats["misses"] += 1
            try:
                cache = _write_agent_cache(agent, prefix_text, prefix_ids)
            except (OSError, RuntimeError) as e:
                print("[cache] agent=%s WRITE_FAILED: %s" % (agent, e),
                      file=sys.stderr, flush=True)
                cache = {"path": None, "hit": False, "ids": []}
        else:
            _cache_stats["hits"] += 1
    cached = cache["ids"] if cache and cache.get("hit") else []
    print("[cache] agent=%s %s prefix_tokens=%d key=%s" %
          (agent, "HIT" if cache.get("hit") else "WRITE", len(cached),
           cache.get("key", "none")), file=sys.stderr, flush=True)
    return prompt_ids, cache["path"], bool(cache.get("hit")), len(cached)


def _runner_rpc(payload, streaming=False, timeout=None):
    if not RUNNER_SOCKET:
        raise RuntimeError("cooperative runner socket is not configured")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(TIMEOUT if timeout is None else timeout)
    s.connect(RUNNER_SOCKET)
    s.sendall((json.dumps(payload, separators=(",", ":")) + "\n").encode())
    f = s.makefile("r")
    if streaming:
        return s, f
    try:
        line = f.readline()
        return json.loads(line) if line else {"event": "error", "error": "runner closed"}
    finally:
        f.close(); s.close()


def runner_contexts():
    global _runner_decode_batch, _last_runner_contexts
    try:
        reply = _runner_rpc({"op": "contexts"}, timeout=0.1)
        _runner_decode_batch = reply.get("decode_batch", _runner_decode_batch)
        if reply.get("event") == "contexts":
            _last_runner_contexts = reply.get("data", [])
        return _last_runner_contexts
    except (OSError, ValueError, RuntimeError):
        return _last_runner_contexts


def runner_delete_context(context_id):
    return _runner_rpc({"op": "delete", "context_id": context_id})


def infer_cooperative(prompt, max_tokens, samp, context_id=None, stream_path=None,
                      cache_path=None, cache_load=False, cache_save=False,
                      save_path=None, cached_tokens=0):
    ids = encode(prompt)
    ephemeral = context_id is None
    req = {"op": "generate", "job_id": uuid.uuid4().hex,
           "context_id": context_id or ("stateless-" + uuid.uuid4().hex),
           "ephemeral": ephemeral,
           "prompt": ids, "max_new": max_tokens,
           "temperature": samp["temperature"], "top_p": samp["top_p"],
           "top_k": samp["top_k"], "presence_penalty": samp["presence_penalty"],
           "repeat_penalty": samp["repeat_penalty"],
           "seed": samp["seed"] if samp["seed"] is not None else 1,
           "cache_path": cache_path, "cache_load": bool(cache_load),
           "cache_save_path": save_path if cache_save else None,
           "cached_tokens": int(cached_tokens)}
    with _cooperative_progress_lock:
        _cooperative_progress.update({
            "active": True, "active_job": req["job_id"], "state": "prefill",
            "prompt_total": len(ids),
            "prompt_processed": int(cached_tokens) if cache_load else 0,
            "completion_tokens": 0, "started": time.time(),
            "decode_started": None, "cache_hit": bool(cache_load),
            "cached_tokens": int(cached_tokens) if cache_load else 0})
    s, f = _runner_rpc(req, streaming=True)
    gen = []
    tf = open(stream_path, "w") if stream_path else None
    try:
        for line in f:
            event = json.loads(line)
            if event.get("event") == "progress":
                with _cooperative_progress_lock:
                    _cooperative_progress.update({
                        "state": event.get("phase", "prefill"),
                        "prompt_processed": int(event.get("processed", 0)),
                        "prompt_total": int(event.get("total", len(ids)))})
            elif event.get("event") == "token":
                tok = int(event["token"]); gen.append(tok)
                with _cooperative_progress_lock:
                    if _cooperative_progress.get("decode_started") is None:
                        _cooperative_progress["decode_started"] = time.time()
                    _cooperative_progress.update({"state": "decode",
                                                  "completion_tokens": len(gen)})
                if tf: tf.write(str(tok) + "\n"); tf.flush()
            elif event.get("event") == "final":
                gen = [int(x) for x in event.get("tokens", gen)]
                break
            elif event.get("event") == "error":
                raise RuntimeError(event.get("error", "runner error"))
    finally:
        with _cooperative_progress_lock:
            _cooperative_progress["active"] = False
        if tf: tf.close()
        f.close(); s.close()
    return ids, gen, decode(gen)


def infer(prompt, max_tokens, samp, slot=0, cache_path=None, cache_load=False, cache_save=False,
          stream=False, save_path=None, context_id=None, stream_path=None,
          cached_tokens=0):
    global _seq
    if RUNNER_SOCKET:
        return infer_cooperative(prompt, max_tokens, samp, context_id, stream_path,
                                 cache_path, cache_load, cache_save, save_path,
                                 cached_tokens)
    # concurrent batched decode: route greedy, non-cache requests through the dispatcher (the runner
    # is in DS4F_SERVE_BATCH mode -> the single-request protocol is not served there).
    if BATCH > 1 and not (cache_load or cache_save):
        # all batched paths support per-sequence sampling; SOCK is the low-latency dynamic transport
        if DYNAMIC and SOCK: return infer_socket(prompt, max_tokens, samp)
        return infer_dynamic(prompt, max_tokens, samp) if DYNAMIC else infer_batched(prompt, max_tokens, samp)
    with _lock:
        ids = encode(prompt)
        if not ids and not cache_save:
            return [], [], ""
        seed = samp["seed"] if samp["seed"] is not None else (_seq + 1)
        # bit0=load-before, bit1=save-after, bit2=stream-tokens-to BASE.tok
        ctl = (1 if cache_load else 0) | (2 if cache_save else 0) | (4 if stream else 0)
        # header: "max_new temp top_p top_k presence_penalty repeat_penalty seed slot ctl"; if ctl!=0 the
        # load path line (bit0) and/or save path line (bit1) follow; then the prompt ids.  (runner parses)
        hdr = "%d %g %g %d %g %g %d %d %d" % (max_tokens, samp["temperature"], samp["top_p"], samp["top_k"],
                                              samp["presence_penalty"], samp["repeat_penalty"], seed, slot, ctl)
        body = hdr + "\n"
        if ctl:
            if ctl & 1: body += (cache_path or "") + "\n"
            if ctl & 2: body += ((save_path or cache_path) or "") + "\n"
        body += " ".join(map(str, ids)) + "\n"
        with open(REQ, "w") as f:
            f.write(body)
        try:
            os.unlink(ERROR)
        except FileNotFoundError:
            pass
        _seq += 1
        with open(REQSEQ, "w") as f:
            f.write(str(_seq) + "\n")                # write req then bump seq -> runner reads a complete file
        t0 = time.time()
        _dbg = os.environ.get("DS4F_SERVE_DEBUG")
        while True:
            try:
                with open(RESPSEQ) as f:
                    rs = int(f.read().strip() or 0)
            except (OSError, ValueError):
                rs = 0
            if _dbg and rs > 0:
                pass
            if rs >= _seq:
                break
            if time.time() - t0 > TIMEOUT:
                raise TimeoutError("runner timeout")
            time.sleep(0.01)
        if _dbg: print("[infer] wait %.2f rs=%d seq=%d" % (time.time() - t0, rs, _seq), flush=True)
        if os.path.exists(ERROR):
            try:
                with open(ERROR) as f:
                    message = f.read().strip()
            finally:
                try: os.unlink(ERROR)
                except FileNotFoundError: pass
            raise RuntimeError("DS4F runner failed: %s" % (message or "unknown error"))
        with open(RESP) as f:
            gen = [int(x) for x in f.read().split()]
        return ids, gen, decode(gen)


# ------------------------------- chat helpers -------------------------------
def parse_sampling(body):
    seed = body.get("seed", None)
    return {
        "temperature": float(body.get("temperature", 0.0)),   # <=0 -> greedy
        "top_p":       float(body.get("top_p", 1.0)),
        "top_k":       int(body.get("top_k", 0)),             # 0 -> disabled
        "presence_penalty": float(body.get("presence_penalty", 0.0)),
        "repeat_penalty":   float(body.get("repeat_penalty", body.get("repetition_penalty", 1.0))),
        "seed":        None if seed is None else int(seed),
    }


def _content_to_text(content):
    """OpenAI content may be a string or a list of typed blocks; flatten to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text", "") if b.get("type") == "text" else "")
            else:
                parts.append(str(b))
        return "".join(parts)
    return str(content)


def render_tools(tools):
    """Prompt-injected tool schemas + calling instructions (appended to the system block)."""
    if not tools:
        return ""
    schemas = []
    for t in tools:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        schemas.append(json.dumps({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        }, ensure_ascii=False))
    return TOOL_INSTRUCTIONS + "\n".join(schemas) + "\n"


def _render_assistant(msg):
    """Assistant turn -> text, including any prior tool_calls rendered back in our <tool_call> form."""
    out = _content_to_text(msg.get("content"))
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        out += ("\n" if out else "") + '<tool_call>{"name": %s, "arguments": %s}</tool_call>' % (
            json.dumps(fn.get("name", ""), ensure_ascii=False), args)
    return out


def build_chat_prompt(messages, tools):
    """messages[] -> a single plain-text prompt ending in 'Assistant:' for the model to continue."""
    sys_txt = ""
    turns = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            sys_txt += (("\n\n" if sys_txt else "") + _content_to_text(m.get("content")))
        elif role == "assistant":
            turns.append(("assistant", _render_assistant(m)))
        elif role == "tool":
            # tool result fed back to the model
            turns.append(("tool", _content_to_text(m.get("content"))))
        else:  # user (and any unknown role)
            turns.append(("user", _content_to_text(m.get("content"))))
    tool_txt = render_tools(tools)
    if tool_txt:
        sys_txt = (sys_txt + "\n\n" + tool_txt) if sys_txt else tool_txt
    prompt = BOS
    if sys_txt.strip():
        prompt += ROLE_TAG["system"] + ": " + sys_txt.strip() + "\n"
    for role, text in turns:
        prompt += "%s: %s\n" % (ROLE_TAG[role], text)
    prompt += ROLE_TAG["assistant"] + ":"
    return prompt


def anthropic_tools_to_openai(tools):
    """Anthropic tool defs -> OpenAI function tools (build_chat_prompt's format)."""
    out = []
    for t in (tools or []):
        if not isinstance(t, dict):
            continue
        name = t.get("name", "")
        if not name:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return out


def anthropic_messages_to_openai(messages):
    """Anthropic messages[] -> OpenAI messages[] the chat prompt builder expects."""
    out = []
    for m in (messages or []):
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        text_parts, tool_uses, tool_results = [], [], []
        for b in content:
            t = b.get("type")
            if t == "text":
                text_parts.append(b.get("text", ""))
            elif t == "tool_use":
                tool_uses.append(b)
            elif t == "tool_result":
                tool_results.append(b)
        if role == "assistant":
            msg = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_uses:
                tcs = []
                for tu in tool_uses:
                    tcs.append({"id": tu.get("id", "call_0"), "type": "function",
                                "function": {"name": tu.get("name", ""),
                                             "arguments": json.dumps(tu.get("input", {}),
                                                                      ensure_ascii=False)}})
                msg["tool_calls"] = tcs
            out.append(msg)
        else:
            # Keep tool results in request order; Claude can send text and
            # tool_result blocks together in one user message.
            pending_text = []
            for block in content:
                if block.get("type") == "text":
                    pending_text.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    if pending_text:
                        out.append({"role": "user", "content": "".join(pending_text)})
                        pending_text = []
                    out.append({"role": "tool",
                                "tool_call_id": block.get("tool_use_id", "call_0"),
                                "content": _content_to_text(block.get("content"))})
            if pending_text:
                out.append({"role": "user", "content": "".join(pending_text)})
    return out


def sys_prefix_text(messages, tools):
    """The fixed prompt prefix every conversation shares (BOS + system + tools)."""
    sys_txt = ""
    for m in messages:
        if m.get("role") == "system":
            sys_txt += (("\n\n" if sys_txt else "") + _content_to_text(m.get("content")))
    tool_txt = render_tools(tools)
    if tool_txt:
        sys_txt = (sys_txt + "\n\n" + tool_txt) if sys_txt else tool_txt
    p = BOS
    if sys_txt.strip():
        p += ROLE_TAG["system"] + ": " + sys_txt.strip() + "\n"
    return p


def parse_completion(text, hit_eos):
    """Truncate the base-model output at the next role marker, then split off tool calls.
    Returns (content_text, tool_calls_list, finish_reason)."""
    orig_len = len(text)
    cut = orig_len
    for mk in STOP_MARKERS:
        i = text.find(mk)
        if i != -1:
            cut = min(cut, i)
    truncated = cut < orig_len
    text = text[:cut]
    tool_calls = []
    for i, mobj in enumerate(TOOL_CALL_RE.finditer(text)):
        try:
            call = json.loads(mobj.group(1))
        except (ValueError, TypeError):
            continue
        name = call.get("name", "")
        args = call.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        tool_calls.append({
            "id": "call_%d" % i, "type": "function",
            "function": {"name": name, "arguments": args},
        })
    if tool_calls:
        # strip the tool-call syntax out of the visible content
        content = TOOL_CALL_RE.sub("", text).strip()
        return content, tool_calls, "tool_calls"
    return text.strip(), [], ("stop" if (hit_eos or truncated) else "length")


def responses_tools_to_openai(tools):
    """Codex/Responses-API tool defs -> OpenAI function tools."""
    out = []
    for t in (tools or []):
        if not isinstance(t, dict) or not t.get("name"):
            continue
        out.append({"type": "function", "function": {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {"type": "object", "properties": {}})}})
    return out


def responses_input_to_openai(body):
    """Responses-API request -> OpenAI messages[] for the chat prompt builder."""
    messages = []
    instructions = body.get("instructions") or ""
    if isinstance(instructions, list):
        instructions = "".join(b.get("text", "") for b in instructions if isinstance(b, dict))
    if instructions:
        messages.append({"role": "system", "content": instructions})
    incoming = body.get("input", [])
    if isinstance(incoming, str):
        incoming = [{"type": "message", "role": "user", "content": incoming}]
    elif isinstance(incoming, dict):
        incoming = [incoming]
    for item in incoming:
        t = item.get("type") if isinstance(item, dict) else None
        if t == "message":
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join((b.get("text", "") if isinstance(b, dict) else str(b))
                               for b in content)
            else:
                text = str(content)
            messages.append({"role": role, "content": text})
        elif t in ("input_text", "output_text"):
            messages.append({"role": "user", "content": str(item.get("text", ""))})
        elif t == "function_call":
            messages.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": item.get("call_id", "call_0"), "type": "function",
                 "function": {"name": item.get("name", ""),
                              "arguments": item.get("arguments", "{}")}}]})
        elif t == "function_call_output":
            messages.append({"role": "tool",
                             "tool_call_id": item.get("call_id", "call_0"),
                             "content": str(item.get("output", ""))})
    return messages


def _response_context_messages(body):
    previous = body.get("previous_response_id")
    if not previous:
        return responses_input_to_openai(body)
    with _response_context_lock:
        saved = _response_contexts.get(previous)
    if saved is None:
        saved = _load_response_state(previous)
    if saved is None:
        raise KeyError(previous)
    messages = json.loads(json.dumps(saved))
    # Continuation requests carry only new function_call_output items.
    continuation = dict(body)
    continuation["instructions"] = ""
    messages.extend(responses_input_to_openai(continuation))
    return messages


def _remember_response(response_id, messages, context_id=None):
    with _response_context_lock:
        _response_contexts[response_id] = json.loads(json.dumps(messages))
        if context_id: _response_context_ids[response_id] = context_id
        if response_id in _response_context_order:
            _response_context_order.remove(response_id)
        _response_context_order.append(response_id)
        while len(_response_context_order) > _MAX_RESPONSE_CONTEXTS:
            old = _response_context_order.pop(0)
            _response_contexts.pop(old, None)
            _response_context_ids.pop(old, None)
    if context_id:
        os.makedirs(RESPONSE_STATE_DIR, mode=0o700, exist_ok=True)
        key = hashlib.sha256(response_id.encode()).hexdigest()
        path = os.path.join(RESPONSE_STATE_DIR, key + ".json")
        tmp = path + ".tmp.%d" % os.getpid()
        with open(tmp, "w") as f:
            json.dump({"schema": 1, "response_id": response_id,
                       "context_id": context_id, "messages": messages,
                       "saved_at": time.time()}, f, separators=(",", ":"))
            f.flush(); os.fsync(f.fileno())
        os.chmod(tmp, 0o600); os.replace(tmp, path)


def _load_response_state(response_id):
    key = hashlib.sha256(str(response_id).encode()).hexdigest()
    try:
        with open(os.path.join(RESPONSE_STATE_DIR, key + ".json")) as f: state = json.load(f)
        if state.get("schema") != 1 or state.get("response_id") != response_id: return None
        with _response_context_lock:
            _response_contexts[response_id] = state["messages"]
            _response_context_ids[response_id] = state.get("context_id")
        return state["messages"]
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _body_context_id(body, headers=None):
    cid = body.get("context_id") if isinstance(body, dict) else None
    if not cid and headers is not None: cid = headers.get("X-DS4F-Context-ID")
    if cid is None: return None
    cid = str(cid)
    if not cid or len(cid) > 256: raise ValueError("invalid context_id")
    return cid


def _select_cache(agent, messages, tools, prompt):
    """Decide the KV prefix to restore for this prompt (conversation reuse,
    system-prompt reuse, or fresh) and where to save the new KV snapshot."""
    if RUNNER_SOCKET:
        ids_all = encode(prompt)
        _, cache_path, reuse, cached_tokens = prepare_agent_cache(
            agent, messages, tools, ids_all)
        return ids_all, reuse, cache_path, None, cached_tokens
    with _conv_lock:
        ids_all = encode(prompt)
        prev = _conv["ids"]
        conv_reuse = _conv["agent"] == agent and _conv["path"] is not None and prev is not None and \
            len(ids_all) >= len(prev) and ids_all[:len(prev)] == prev
        # Conversation snapshots are agent-scoped too.  Reusing the previous
        # agent's path would let a Codex turn overwrite Claude's context (and
        # vice versa) when requests are interleaved.
        cpath = BASE + ".conv." + agent
        if conv_reuse:
            reuse = True
            cache_path = cpath
            cached_tokens = len(prev)
        else:
            _, cache_path, reuse, cached_tokens = prepare_agent_cache(agent, messages, tools, ids_all)
        save_path = cpath if len(ids_all) <= CACHE_MAX_TOKENS else None
        if save_path is None:
            print("[cache] agent=%s SKIP conversation_tokens=%d exceeds limit=%d" %
                  (agent, len(ids_all), CACHE_MAX_TOKENS),
                  file=sys.stderr, flush=True)
        _conv["agent"] = agent
        _conv["ids"] = ids_all
        _conv["path"] = cpath
    return ids_all, reuse, cache_path, save_path, cached_tokens


def anthropic_blocks(content, tool_calls, finish, hit_eos):
    """(content, OpenAI tool_calls, finish_reason) -> Anthropic content blocks +
    stop_reason."""
    blocks = []
    if content:
        blocks.append({"type": "text", "text": content})
    for tc in (tool_calls or []):
        fn = tc.get("function", {})
        try:
            inp = json.loads(fn.get("arguments", "{}"))
        except (ValueError, TypeError):
            inp = {}
        blocks.append({"type": "tool_use", "id": tc.get("id", "call_0"),
                       "name": fn.get("name", ""), "input": inp})
    if tool_calls:
        stop = "tool_use"
    elif finish == "length":
        stop = "max_tokens"
    else:
        stop = "end_turn"
    return blocks, stop


def stream_visible_delta(state, piece):
    """Return only safe visible text; hold tool-call marker/JSON until EOF."""
    state["raw"] += piece
    raw = state["raw"]
    marker = "<tool_call>"
    cut = raw.find(marker)
    if cut >= 0:
        visible = raw[:cut]
    else:
        visible = raw
        for n in range(1, len(marker)):
            if raw.endswith(marker[:n]):
                visible = raw[:-n]
                break
    delta = visible[state["emitted"]:]
    state["emitted"] = len(visible)
    return delta


class H(http.server.BaseHTTPRequestHandler):
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # The stream has no Content-Length; close it after [DONE] so urllib,
        # curl, and llmgr's wire proxy can detect end-of-response.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

    def _sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path in ("/health", "/"):
            self._json(200, {"status": "ok", "model": MODEL_ID})
        elif self.path.rstrip("/") in ("/progress", "/v1/progress", "/status",
                                        "/v1/status", "/v1/metrics"):
            self._json(200, progress_snapshot())
        elif self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{
                "id": MODEL_ID, "object": "model", "created": 0, "owned_by": "deepseek-ai",
                "context_window": 16384, "max_tokens": 4096,
            }]})
        elif self.path.rstrip("/") == "/v1/contexts":
            self._json(200, {"object": "list", "data": runner_contexts()})
        elif self.path.startswith("/v1/contexts/"):
            cid = self.path.split("/", 3)[-1]
            match = next((c for c in runner_contexts() if c.get("id") == cid), None)
            self._json(200 if match else 404, match or {"error": "context not found"})
        else:
            self._json(404, {"error": "not found"})

    def do_DELETE(self):
        if not self.path.startswith("/v1/contexts/"):
            return self._json(404, {"error": "not found"})
        cid = self.path.split("/", 3)[-1]
        try: reply = runner_delete_context(cid)
        except Exception as exc: return self._json(503, {"error": str(exc)})
        status = int(reply.get("status", 500))
        self._json(status, {} if status == 204 else reply)

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return None

    def do_POST(self):
        path = self.path.rstrip("/")
        if path in ("/v1/chat/completions", "/chat/completions"):
            return self.chat()
        if path == "/v1/messages/count_tokens":
            return self.messages_count_tokens()
        if path in ("/v1/messages", "/messages"):
            return self.messages_anthropic()
        if path in ("/v1/responses", "/responses"):
            return self.responses_api()
        return self.completion()

    # ---- OpenAI chat completions (pi's path) ----
    def chat(self):
        body = self._read_body()
        if body is None:
            return self._json(400, {"error": "bad json"})
        messages = body.get("messages", [])
        tools = body.get("tools", [])
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 512)))
        samp = parse_sampling(body)
        try: context_id = _body_context_id(body, self.headers)
        except ValueError as exc: return self._json(400, {"error": str(exc)})
        prompt = build_chat_prompt(messages, tools)
        ids_all, reuse, cache_path, save_path, cached_tokens = _select_cache(
            "opencode", messages, tools, prompt)
        t0 = time.time()
        _dbg = os.environ.get("DS4F_SERVE_DEBUG")
        if _dbg: print("[chat] t0 %.2f reuse=%s cache=%s len=%d" %
                       (time.time(), reuse, cache_path, len(ids_all)), flush=True)
        if not stream:
            try:
                ids, gen, raw = infer(prompt, max_tokens, samp,
                                      cache_path=cache_path, cache_load=reuse, cache_save=True,
                                      save_path=save_path, context_id=context_id,
                                      cached_tokens=cached_tokens)
                if _dbg: print("[chat] infer %.2f gen=%d" % (time.time() - t0, len(gen)), flush=True)
            except TimeoutError:
                return self._json(504, {"error": "runner timeout"})
            except Exception as e:
                return self._json(500, {"error": str(e)})
            hit_eos = bool(gen and gen[-1] == 1)   # DS4F_EOS_ID == 1
            content, tool_calls, finish = parse_completion(raw, hit_eos)
            msg = {"role": "assistant", "content": content or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            return self._json(200, {
                "id": "chatcmpl-ds4f", "object": "chat.completion", "created": int(t0),
                "model": MODEL_ID,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(gen),
                          "total_tokens": len(ids) + len(gen),
                          "prompt_tokens_details": {"cached_tokens": cached_tokens}},
            })
        # real streaming: the runner appends each generated token id to BASE.tok
        # (ctl bit2); infer() runs in a thread and this handler tails the file,
        # decoding each token and emitting an SSE delta as it lands.
        self._sse_headers()
        head = {"id": "chatcmpl-ds4f", "object": "chat.completion.chunk", "created": int(t0),
                "model": MODEL_ID}
        self._sse({**head, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                        "finish_reason": None}]})
        tok_path = (BASE + ".tok." + uuid.uuid4().hex) if RUNNER_SOCKET else BASE + ".tok"
        err = {}
        done = {"gen": []}
        stream_state = {"raw": "", "emitted": 0}
        def _run():
            try:
                p_ids, p_gen, _ = infer(prompt, max_tokens, samp, stream=True,
                                        cache_path=cache_path, cache_load=reuse, cache_save=True,
                                        save_path=save_path, context_id=context_id,
                                        stream_path=tok_path, cached_tokens=cached_tokens)
                done["ids"] = p_ids
                done["gen"] = p_gen
            except Exception as e:
                err["e"] = e
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        seen = 0
        while True:
            try:
                lines = open(tok_path).read().splitlines()
            except OSError:
                lines = []
            for tok in lines[seen:]:
                try:
                    text = decode([int(tok)])
                except Exception:
                    text = ""
                text = stream_visible_delta(stream_state, text)
                if text:
                    self._sse({**head, "choices": [{"index": 0, "delta": {"content": text},
                                                    "finish_reason": None}]})
            seen = len(lines)
            if err:
                return self._json(500, {"error": str(err["e"])})
            if not t.is_alive() and seen >= len(lines):
                break
            time.sleep(0.02)
        gen = done["gen"]
        hit_eos = bool(gen and gen[-1] == 1)
        content, tool_calls, finish = parse_completion(decode(gen), hit_eos)
        usage = {"prompt_tokens": len(done.get("ids", ids)),
                 "completion_tokens": len(gen), "total_tokens": len(done.get("ids", ids)) + len(gen)}
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            self._sse({**head, "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": i, "id": tc.get("id", "call_%d" % i),
                                 "type": "function", "function": {
                                     "name": fn.get("name", ""),
                                     "arguments": fn.get("arguments", "{}")}}]},
                "finish_reason": None}]})
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
        self._sse({**head, "choices": [{"index": 0, "delta": {},
                                          "finish_reason": "tool_calls" if tool_calls else finish}],
                   "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # ---- Anthropic messages API (claude-code's wire protocol) ----
    def messages_count_tokens(self):
        body = self._read_body()
        if body is None:
            return self._json(400, {"error": "bad json"})
        tools = anthropic_tools_to_openai(body.get("tools", []))
        messages = anthropic_messages_to_openai(body.get("messages", []))
        system = body.get("system")
        if system and not any(m.get("role") == "system" for m in messages):
            if isinstance(system, list):
                system = "".join(b.get("text", "") for b in system
                                 if isinstance(b, dict) and b.get("type") == "text")
            messages.insert(0, {"role": "system", "content": system})
        return self._json(200, {"input_tokens": len(encode(
            build_chat_prompt(messages, tools)))})

    def messages_anthropic(self):
        body = self._read_body()
        if body is None:
            return self._json(400, {"error": "bad json"})
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 512)))
        samp = parse_sampling(body)
        try: context_id = _body_context_id(body, self.headers)
        except ValueError as exc: return self._json(400, {"error": str(exc)})
        tools = anthropic_tools_to_openai(body.get("tools", []))
        messages = anthropic_messages_to_openai(body.get("messages", []))
        system = body.get("system")
        if system and not any(m.get("role") == "system" for m in messages):
            if isinstance(system, list):
                system = "".join(b.get("text", "") for b in system if b.get("type") == "text")
            messages.insert(0, {"role": "system", "content": system})
        prompt = build_chat_prompt(messages, tools)
        ids_all, reuse, cache_path, save_path, cached_tokens = _select_cache(
            "claude-code", messages, tools, prompt)
        t0 = time.time()
        mid = "msg_ds4f_" + uuid.uuid4().hex
        if not stream:
            try:
                ids, gen, raw = infer(prompt, max_tokens, samp,
                                      cache_path=cache_path, cache_load=reuse,
                                      cache_save=True, save_path=save_path,
                                      context_id=context_id, cached_tokens=cached_tokens)
            except TimeoutError:
                return self._json(504, {"error": "runner timeout"})
            except Exception as e:
                return self._json(500, {"error": str(e)})
            hit_eos = bool(gen and gen[-1] == 1)
            content, tool_calls, finish = parse_completion(raw, hit_eos)
            blocks, stop = anthropic_blocks(content, tool_calls, finish, hit_eos)
            return self._json(200, {
                "id": mid, "type": "message", "role": "assistant", "model": MODEL_ID,
                "content": blocks, "stop_reason": stop,
                "usage": {"input_tokens": len(ids), "output_tokens": len(gen),
                          "cache_read_input_tokens": cached_tokens},
            })
        # streaming: content_block deltas over the .tok stream
        self._sse_headers()
        head = {"type": "message_start", "message": {
            "id": mid, "type": "message", "role": "assistant", "model": MODEL_ID,
            "content": [], "stop_reason": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}}}
        self._sse(head)
        self._sse({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}})
        tok_path = (BASE + ".tok." + uuid.uuid4().hex) if RUNNER_SOCKET else BASE + ".tok"
        err = {}
        done = {"gen": []}
        stream_state = {"raw": "", "emitted": 0}
        def _run():
            try:
                p_ids, p_gen, _ = infer(prompt, max_tokens, samp, stream=True,
                                        cache_path=cache_path, cache_load=reuse,
                                        cache_save=True, save_path=save_path,
                                        context_id=context_id, stream_path=tok_path,
                                        cached_tokens=cached_tokens)
                done["ids"] = p_ids
                done["gen"] = p_gen
            except Exception as e:
                err["e"] = e
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        seen = 0
        while True:
            try:
                lines = open(tok_path).read().splitlines()
            except OSError:
                lines = []
            for tok in lines[seen:]:
                try:
                    text = decode([int(tok)])
                except Exception:
                    text = ""
                text = stream_visible_delta(stream_state, text)
                if text:
                    self._sse({"type": "content_block_delta", "index": 0,
                               "delta": {"type": "text_delta", "text": text}})
            seen = len(lines)
            if err:
                return self._json(500, {"error": str(err["e"])})
            if not t.is_alive() and seen >= len(lines):
                break
            time.sleep(0.02)
        gen = done["gen"]
        hit_eos = bool(gen and gen[-1] == 1)
        content, tool_calls, finish = parse_completion(decode(gen), hit_eos)
        blocks, stop = anthropic_blocks(content, tool_calls, finish, hit_eos)
        self._sse({"type": "content_block_stop", "index": 0})
        if tool_calls:
            for i, blk in enumerate(blocks):
                if blk.get("type") != "tool_use":
                    continue
                self._sse({"type": "content_block_start", "index": 1 + i,
                           "content_block": {"type": "tool_use", "id": blk["id"],
                                             "name": blk["name"], "input": {}}})
                self._sse({"type": "content_block_delta", "index": 1 + i,
                           "delta": {"type": "input_json_delta",
                                      "partial_json": json.dumps(blk["input"], ensure_ascii=False)}})
                self._sse({"type": "content_block_stop", "index": 1 + i})
        self._sse({"type": "message_delta",
                   "delta": {"stop_reason": stop},
                   "usage": {"output_tokens": len(gen)}})
        self._sse({"type": "message_stop"})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # ---- OpenAI Responses API (codex's wire protocol) ----
    def responses_api(self):
        body = self._read_body()
        if body is None:
            return self._json(400, {"error": "bad json"})
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_output_tokens") or body.get("max_tokens") or 512)
        samp = parse_sampling(body)
        tools = responses_tools_to_openai(body.get("tools", []))
        try:
            messages = _response_context_messages(body)
        except KeyError as e:
            return self._json(409, {"error": "unknown previous_response_id: %s" % e.args[0]})
        prompt = build_chat_prompt(messages, tools)
        ids_all, reuse, cache_path, save_path, cached_tokens = _select_cache(
            "codex", messages, tools, prompt)
        t0 = time.time()
        rid = "resp_ds4f_" + uuid.uuid4().hex
        try: explicit_context = _body_context_id(body, self.headers)
        except ValueError as exc: return self._json(400, {"error": str(exc)})
        previous = body.get("previous_response_id")
        with _response_context_lock:
            inherited_context = _response_context_ids.get(previous) if previous else None
        context_id = explicit_context or inherited_context or ("ctx_" + uuid.uuid4().hex)
        use = {"input_tokens": 0, "output_tokens": 0}
        if not stream:
            try:
                ids, gen, raw = infer(prompt, max_tokens, samp,
                                      cache_path=cache_path, cache_load=reuse,
                                      cache_save=True, save_path=save_path,
                                      context_id=context_id, cached_tokens=cached_tokens)
            except TimeoutError:
                return self._json(504, {"error": "runner timeout"})
            except Exception as e:
                return self._json(500, {"error": str(e)})
            hit_eos = bool(gen and gen[-1] == 1)
            content, tool_calls, finish = parse_completion(raw, hit_eos)
            output = []
            if content:
                output.append({"type": "message", "role": "assistant", "status": "completed",
                               "content": [{"type": "output_text", "text": content, "annotations": []}]})
            for tc in (tool_calls or []):
                fn = tc.get("function", {})
                output.append({"type": "function_call", "id": tc.get("id", "fc_0"),
                               "call_id": tc.get("id", "fc_0"), "name": fn.get("name", ""),
                               "arguments": fn.get("arguments", "{}"), "status": "completed"})
            assistant = {"role": "assistant", "content": content or None}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            _remember_response(rid, messages + [assistant], context_id)
            return self._json(200, {
                "id": rid, "object": "response", "created_at": int(t0), "model": MODEL_ID,
                "status": "completed", "output": output,
                "usage": {"input_tokens": len(ids), "output_tokens": len(gen),
                          "total_tokens": len(ids) + len(gen),
                          "input_tokens_details": {"cached_tokens": cached_tokens}}})
        # streaming: codex requires the SSE event stream through response.completed
        self._sse_headers()
        mid = rid + "_msg"
        out_msg = {"id": mid, "type": "message", "role": "assistant", "status": "in_progress",
                   "content": []}
        self._sse({"type": "response.created", "response": {
            "id": rid, "object": "response", "created_at": int(t0), "model": MODEL_ID,
            "status": "in_progress", "output": [], "usage": use}})
        self._sse({"type": "response.in_progress", "response": {
            "id": rid, "object": "response", "status": "in_progress", "output": [],
            "usage": use}})
        self._sse({"type": "response.output_item.added", "output_index": 0,
                   "item": out_msg})
        self._sse({"type": "response.content_part.added", "item_id": mid,
                   "output_index": 0, "content_index": 0,
                   "part": {"type": "output_text", "text": "", "annotations": []}})
        tok_path = (BASE + ".tok." + rid) if RUNNER_SOCKET else BASE + ".tok"
        err = {}
        done = {"gen": []}
        stream_state = {"raw": "", "emitted": 0}
        def _run():
            try:
                p_ids, p_gen, _ = infer(prompt, max_tokens, samp, stream=True,
                                        cache_path=cache_path, cache_load=reuse,
                                        cache_save=True, save_path=save_path,
                                        context_id=context_id, stream_path=tok_path,
                                        cached_tokens=cached_tokens)
                done["ids"] = p_ids
                done["gen"] = p_gen
            except Exception as e:
                err["e"] = e
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        seen = 0
        while True:
            try:
                lines = open(tok_path).read().splitlines()
            except OSError:
                lines = []
            for tok in lines[seen:]:
                try:
                    text = decode([int(tok)])
                except Exception:
                    text = ""
                text = stream_visible_delta(stream_state, text)
                if text:
                    self._sse({"type": "response.output_text.delta", "item_id": mid,
                               "output_index": 0, "content_index": 0, "delta": text})
            seen = len(lines)
            if err:
                return self._json(500, {"error": str(err["e"])})
            if not t.is_alive() and seen >= len(lines):
                break
            time.sleep(0.02)
        gen = done["gen"]
        hit_eos = bool(gen and gen[-1] == 1)
        content, tool_calls, finish = parse_completion(decode(gen), hit_eos)
        self._sse({"type": "response.output_text.done", "item_id": mid,
                   "output_index": 0, "content_index": 0, "text": content})
        self._sse({"type": "response.content_part.done", "item_id": mid,
                   "output_index": 0, "content_index": 0,
                   "part": {"type": "output_text", "text": content, "annotations": []}})
        self._sse({"type": "response.output_item.done", "output_index": 0, "item": {
            "id": mid, "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": content, "annotations": []}]}})
        output = []
        if content:
            output.append({"type": "message", "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": content, "annotations": []}]})
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                fn = tc.get("function", {})
                fid = tc.get("id", "fc_%d" % i)
                self._sse({"type": "response.output_item.added", "output_index": 1 + i,
                           "item": {"id": fid, "type": "function_call", "call_id": fid,
                                    "name": fn.get("name", ""), "arguments": "",
                                    "status": "in_progress"}})
                self._sse({"type": "response.function_call_arguments.delta",
                           "item_id": fid, "output_index": 1 + i, "delta": fn.get("arguments", "")})
                self._sse({"type": "response.function_call_arguments.done",
                           "item_id": fid, "output_index": 1 + i,
                           "arguments": fn.get("arguments", "")})
                self._sse({"type": "response.output_item.done", "output_index": 1 + i,
                           "item": {"id": fid, "type": "function_call", "call_id": fid,
                                    "name": fn.get("name", ""),
                                    "arguments": fn.get("arguments", ""), "status": "completed"}})
                output.append({"type": "function_call", "id": fid, "call_id": fid,
                               "name": fn.get("name", ""),
                               "arguments": fn.get("arguments", ""), "status": "completed"})
        use = {"input_tokens": len(done.get("ids", [])), "output_tokens": len(gen),
               "total_tokens": len(done.get("ids", [])) + len(gen),
               "input_tokens_details": {"cached_tokens": cached_tokens}}
        assistant = {"role": "assistant", "content": content or None}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        _remember_response(rid, messages + [assistant], context_id)
        if RUNNER_SOCKET:
            try: os.unlink(tok_path)
            except OSError: pass
        self._sse({"type": "response.completed", "response": {
            "id": rid, "object": "response", "created_at": int(t0), "model": MODEL_ID,
            "status": "completed", "output": output, "usage": use}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # ---- legacy text completion (/v1/completions, /completion) ----
    def completion(self):
        body = self._read_body()
        if body is None:
            return self._json(400, {"error": "bad json"})
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "".join(map(str, prompt))
        max_tokens = int(body.get("max_tokens", body.get("n_predict", 128)))
        samp = parse_sampling(body)
        slot = int(body.get("slot", 0))
        cache_path = body.get("cache_path", None)
        cache_load = bool(body.get("cache_load", False))
        cache_save = bool(body.get("cache_save", False))
        t0 = time.time()
        try:
            ids, gen, text = infer(prompt, max_tokens, samp, slot, cache_path, cache_load, cache_save)
        except TimeoutError:
            return self._json(504, {"error": "runner timeout"})
        except Exception as e:
            return self._json(500, {"error": str(e)})
        stop = bool(gen and gen[-1] == 1)   # DS4F_EOS_ID == 1
        self._json(200, {
            "id": "cmpl-ds4f", "object": "text_completion", "model": MODEL_ID,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop" if stop else "length"}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(gen),
                      "total_tokens": len(ids) + len(gen)},
            "timings": {"wall_s": round(time.time() - t0, 3)},
        })

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner-socket")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--tokenizer", default=TOK)
    ap.add_argument("--response-state-dir", default=RESPONSE_STATE_DIR)
    ap.add_argument("--agent-cache-max-tokens", type=int, default=8192)
    ap.add_argument("--runner-timeout-sec", type=float, default=3600.0)
    args = ap.parse_args()
    RUNNER_SOCKET, PORT, TOK = args.runner_socket, args.port, args.tokenizer
    CACHE_MAX_TOKENS = max(0, args.agent_cache_max_tokens)
    TIMEOUT = max(1.0, args.runner_timeout_sec)
    RESPONSE_STATE_DIR = args.response_state_dir
    try:
        with open(RESPSEQ, "w") as f:
            f.write("0\n")                    # clear any stale response marker
    except OSError:
        pass
    if BATCH > 1 and DYNAMIC:
        try:                                  # sync _qnext to the runner's current queue head
            with open(BASE + ".qhead") as f: _qnext = int(f.read().strip() or 0)
        except (OSError, ValueError):
            pass
        print(f"[ds4f-serve] DYNAMIC continuous batching: B={BATCH}, mid-flight admission (greedy)", flush=True)
    elif BATCH > 1:
        try:                                  # sync _seq to the runner's current request counter
            with open(REQSEQ) as f: _seq = int(f.read().strip() or 0)
        except (OSError, ValueError):
            pass
        threading.Thread(target=_batch_dispatcher, daemon=True).start()
        print(f"[ds4f-serve] concurrent batched decode: B={BATCH}, coalesce window={BATCH_WINDOW*1e3:.0f}ms (greedy)", flush=True)
    print(f"[ds4f-serve] listening on http://0.0.0.0:{PORT}", flush=True)
    print(f"[ds4f-serve]   POST /v1/chat/completions  {{\"messages\":[...], \"tools\":[...]}}  (pi)", flush=True)
    print(f"[ds4f-serve]   POST /v1/completions       {{\"prompt\": \"...\", \"max_tokens\": 128}}", flush=True)
    print(f"[ds4f-serve]   GET  /v1/models  /health", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
