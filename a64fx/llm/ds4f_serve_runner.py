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
  ctl bit0 = load KV prefix from cache_path before prefill
  ctl bit1 = save KV prefix to cache_path after generation

The model runs through libds4f_serve.so (ctypes); the runner owns the chat
loop, sampling penalties, prefix caching, and context/slot management.

Env: DS4F_SERVE_BASE, DS4F_STAGE_DIR (required), DS4F_SERVE_USE_HIP (1),
     DS4F_HIP_DEVICE, DS4F_MAXPOS, LLM_THREADS, DS4F_CMGS,
     DS4F_SERVE_PREFIX_CACHE (1), DS4F_SERVE_SLOTS (>=1).
"""
import ctypes, os, signal, sys, time

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
    return lib


class Serve(object):
    def __init__(self, lib, stage_dir, use_hip, hip_device, threads, cmgs, max_pos):
        err = ctypes.create_string_buffer(512)
        self._s = lib.ds4f_serve_open(stage_dir.encode(), int(use_hip), int(hip_device),
                                      int(threads), int(cmgs), int(max_pos), err, len(err))
        if not self._s:
            sys.exit("ds4f_serve_open failed: %s" % err.value.decode())
        self.lib = lib
        self.vocab = lib.ds4f_serve_vocab(self._s)
        self.eos = lib.ds4f_serve_eos(self._s)
        self.logits = lib.ds4f_serve_logits(self._s, ctypes.byref(ctypes.c_int(0)))

    def close(self):
        self.lib.ds4f_serve_close(self._s)

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


def env_i(k, d):
    return int(os.environ.get(k, d))


def run_serve(sess, base, prefix_cache, slots):
    req = base + ".req"; resp = base + ".resp"
    reqseq = base + ".reqseq"; respseq = base + ".respseq"
    # per-slot cache paths (slot 0 is the live context; others are switched in)
    slot_path = [base + ".slot.%d" % i for i in range(slots)]
    syscache = os.environ.get("DS4F_SERVE_SYSCACHE")
    if syscache and os.path.exists(syscache):
        sess.kv_restore(syscache)
        print("[runner] system-prompt cache loaded from %s (pos=%d)" %
              (syscache, sess.pos()), file=sys.stderr, flush=True)
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
        # read the request body
        body = open(req).read().splitlines()
        hdr = body[0].split()
        if len(hdr) < 9:
            print("[runner] bad header: %r" % hdr, file=sys.stderr, flush=True)
            done = rs
            continue
        max_new, temp, top_p, top_k = int(hdr[0]), float(hdr[1]), float(hdr[2]), int(hdr[3])
        pres, rep, seed, slot, ctl = (float(hdr[4]), float(hdr[5]), int(hdr[6]),
                                      int(hdr[7]), int(hdr[8]))
        li = 1
        cache_path = None
        save_path = None
        if ctl != 0 and len(body) > li:
            if ctl & 1:
                cache_path = body[li]; li += 1
            if ctl & 2:
                save_path = body[li]; li += 1
        prompt = [int(x) for x in body[li].split()] if len(body) > li else []

        t0 = time.time()
        stream_path = (base + ".tok") if (ctl & 4) else None
        gen = generate(sess, prompt, max_new, temp, top_p, top_k, pres, rep, seed,
                       slot, ctl, cache_path, prefix_cache, slots, slot_path,
                       stream_path, save_path)
        with open(resp, "w") as f:
            f.write(" ".join(map(str, gen)))
        done = rs
        with open(respseq, "w") as f:
            f.write(str(rs))
        print("[runner] seq %d ids=%d gen=%d %.2fs (%.2f tok/s)" %
              (rs, len(prompt), len(gen), time.time() - t0,
               (len(gen) / max(time.time() - t0, 1e-6)) if gen else 0.0),
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
    limit = max(1, maxpos - max_new)
    prompt = truncate_prompt(prompt, limit)
    start = sess.pos()
    _dbg = os.environ.get("DS4F_SERVE_DEBUG")
    _t0 = time.time()

    if ctl & 1 and cache_path and os.path.exists(cache_path):
        # load a cached prefix; prefill only the tokens after the cached length.
        # The cache only applies when its length is a prefix of the prompt; a
        # longer cache (e.g. the previous turn rendered the tool call in fewer
        # tokens than the model generated) must fall back to a fresh prefill.
        sess.kv_restore(cache_path)
        cached = sess.pos()
        if cached > len(prompt):
            sess.reset()
            if prompt:
                sess.prefill(prompt, 0)
        else:
            tail = prompt[cached:]
            if tail:
                sess.prefill(tail, cached)
    elif prefix_cache and slots > 1 and os.path.exists(slot_path[slot % slots]):
        sess.kv_restore(slot_path[slot % slots])
        cached = sess.pos()
        tail = prompt[cached:]
        if tail:
            sess.prefill(tail, cached)
    else:
        # a fresh (no-cache, no-slot) request carries the full conversation;
        # the session must start at position 0 or the KV is written at the
        # previous turn's position (context corruption).
        if sess.pos() != 0:
            sess.reset()
        if prompt:
            sess.prefill(prompt, 0)

    # the generation loop: sample + decode until max_new or EOS
    out = []
    pos = sess.pos()
    tf = None
    if stream_path:
        tf = open(stream_path, "w")
    for _ in range(max_new):
        tok = sess.sample(sp)
        if tok == sess.eos:
            break
        ar = sess.decode(tok, pos)
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

    if _dbg:
        print("[runner] gen prefill+decode %.1fs gen=%d pos=%d" %
              (time.time() - _t0, len(out), sess.pos()), file=sys.stderr, flush=True)
    # cache save: the KV up to the current position (prefix for the next turn)
    spath = save_path or cache_path
    if ctl & 2 and spath:
        _ts = time.time()
        sess.kv_save(spath)
        if _dbg:
            print("[runner] kv_save %.2fs" % (time.time() - _ts), file=sys.stderr, flush=True)
    elif prefix_cache and slots > 1:
        sess.kv_save(slot_path[slot % slots])
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
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    if "--daemon" in sys.argv:
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
                 max_pos=env_i("DS4F_MAXPOS", 16384))
    slots = max(1, env_i("DS4F_SERVE_SLOTS", 1))
    prefix_cache = env_i("DS4F_SERVE_PREFIX_CACHE", 1)
    try:
        run_serve(sess, base, prefix_cache, slots)
    except KeyboardInterrupt:
        pass
    finally:
        sess.close()


if __name__ == "__main__":
    main()
