#!/usr/bin/env python3
"""OpenAI-compatible HTTP API for the DeepSeek-V4-Flash 11-node EP runner (DS4F_SERVE mode).

Runs on the controller node; drives the persistent `ds4f_ep_runner` (loaded once, looping on
requests) over shared-FS files:  <BASE>.req / .reqseq  (prompt in) and  <BASE>.resp / .respseq
(generated ids out). The runner's 11 ranks all read the same request -> lockstep, no broadcast.

Endpoints:
  POST /v1/chat/completions  {"messages":[...], "tools":[...], "stream":bool, ...}  (OpenAI chat)
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

Env: PORT (8080), TOK (~/models/ds4f/tokenizer.json), DS4F_SERVE_BASE, DS4F_SERVE_TIMEOUT (1200s).
Start via run_ds4f_serve_11n.sh (which launches the runner first, then this)."""
import http.server, json, os, re, socketserver, subprocess, sys, tempfile, threading, time


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.environ.get("TOK", os.path.expanduser("~/models/ds4f/tokenizer.json"))
TOKCLI = os.path.join(HERE, "tools", "ds4f_tokenizer.py")
BASE = os.environ.get("DS4F_SERVE_BASE", "/tmp/ds4f_serve")
REQ, RESP, REQSEQ, RESPSEQ = BASE + ".req", BASE + ".resp", BASE + ".reqseq", BASE + ".respseq"
PORT = int(os.environ.get("PORT", "8080"))
TIMEOUT = float(os.environ.get("DS4F_SERVE_TIMEOUT", "1200"))
MODEL_ID = "ds4f"
_lock = threading.Lock()        # the runner is single-stream: serialize requests
_seq = 0

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


def infer(prompt, max_tokens, samp, slot=0, cache_path=None, cache_load=False, cache_save=False):
    global _seq
    # concurrent batched decode: route greedy, non-cache requests through the dispatcher (the runner
    # is in DS4F_SERVE_BATCH mode -> the single-request protocol is not served there).
    if BATCH > 1 and not (cache_load or cache_save):
        # both batched paths now support per-sequence sampling
        return infer_dynamic(prompt, max_tokens, samp) if DYNAMIC else infer_batched(prompt, max_tokens, samp)
    with _lock:
        ids = encode(prompt)
        if not ids and not cache_save:
            return [], [], ""
        seed = samp["seed"] if samp["seed"] is not None else (_seq + 1)
        ctl = (1 if cache_load else 0) | (2 if cache_save else 0)   # bit0=load-before, bit1=save-after
        # header: "max_new temp top_p top_k presence_penalty repeat_penalty seed slot ctl"; if ctl!=0 the
        # NEXT line is the cache path; then the prompt ids.  (runner parses this)
        hdr = "%d %g %g %d %g %g %d %d %d" % (max_tokens, samp["temperature"], samp["top_p"], samp["top_k"],
                                              samp["presence_penalty"], samp["repeat_penalty"], seed, slot, ctl)
        body = hdr + "\n" + ((cache_path or "") + "\n" if ctl else "") + " ".join(map(str, ids)) + "\n"
        with open(REQ, "w") as f:
            f.write(body)
        _seq += 1
        with open(REQSEQ, "w") as f:
            f.write(str(_seq) + "\n")                # write req then bump seq -> runner reads a complete file
        t0 = time.time()
        while True:
            try:
                with open(RESPSEQ) as f:
                    rs = int(f.read().strip() or 0)
            except (OSError, ValueError):
                rs = 0
            if rs >= _seq:
                break
            if time.time() - t0 > TIMEOUT:
                raise TimeoutError("runner timeout")
            time.sleep(0.01)
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


class H(http.server.BaseHTTPRequestHandler):
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def _sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path in ("/health", "/"):
            self._json(200, {"status": "ok", "model": MODEL_ID})
        elif self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{
                "id": MODEL_ID, "object": "model", "created": 0, "owned_by": "deepseek-ai",
                "context_window": 16384, "max_tokens": 4096,
            }]})
        else:
            self._json(404, {"error": "not found"})

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
        prompt = build_chat_prompt(messages, tools)
        t0 = time.time()
        try:
            ids, gen, raw = infer(prompt, max_tokens, samp)
        except TimeoutError:
            return self._json(504, {"error": "runner timeout"})
        except Exception as e:
            return self._json(500, {"error": str(e)})
        hit_eos = bool(gen and gen[-1] == 1)   # DS4F_EOS_ID == 1
        content, tool_calls, finish = parse_completion(raw, hit_eos)
        created = int(t0)
        usage = {"prompt_tokens": len(ids), "completion_tokens": len(gen),
                 "total_tokens": len(ids) + len(gen)}
        if not stream:
            msg = {"role": "assistant", "content": content or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            return self._json(200, {
                "id": "chatcmpl-ds4f", "object": "chat.completion", "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": usage,
            })
        # pseudo-stream: emit the already-computed turn as SSE deltas
        self._sse_headers()
        head = {"id": "chatcmpl-ds4f", "object": "chat.completion.chunk", "created": created,
                "model": MODEL_ID}
        self._sse({**head, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                        "finish_reason": None}]})
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                self._sse({**head, "choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": i, "id": tc["id"], "type": "function",
                    "function": {"name": tc["function"]["name"],
                                 "arguments": tc["function"]["arguments"]},
                }]}, "finish_reason": None}]})
        elif content:
            self._sse({**head, "choices": [{"index": 0, "delta": {"content": content},
                                            "finish_reason": None}]})
        self._sse({**head, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                   "usage": usage})
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
