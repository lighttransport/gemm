#!/usr/bin/env python3
"""Client for the llmgr control port.

Run it from a Fugaku frontend against the reverse-tunnelled port, or on the
head node itself against loopback.

  export LLMGR_TOKEN=secret LLMGR_URL=http://127.0.0.1:21374
  llmgr_cli.py health
  llmgr_cli.py sh 'cd a64fx/laguna-s21 && make all'      # streams
  llmgr_cli.py build --model laguna --variant int4
  llmgr_cli.py stage --model laguna
  llmgr_cli.py start --model laguna --port 8080 --maxpos 8192
  llmgr_cli.py ps
  llmgr_cli.py log run-3 --follow
  llmgr_cli.py gen --ids 2,1841,563 --max-new 32
  llmgr_cli.py chat 'Explain A64FX SVE briefly' --stream
  llmgr_cli.py stop run-3

Standard library only.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_URL = os.environ.get("LLMGR_URL", "http://127.0.0.1:21374")


def call(args, method, path, body=None, stream=False, timeout=None):
    url = args.url.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if args.token:
        req.add_header("Authorization", "Bearer " + args.token)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        r = urllib.request.urlopen(req, timeout=timeout or args.timeout)
    except urllib.error.HTTPError as e:
        sys.stderr.write("HTTP %d: %s\n" % (e.code, e.read().decode("utf-8", "replace")))
        sys.exit(1)
    except Exception as e:                        # noqa: BLE001
        sys.stderr.write("request failed: %s\n" % e)
        sys.exit(1)
    if stream:
        for chunk in iter(lambda: r.read(4096), b""):
            sys.stdout.write(chunk.decode("utf-8", "replace"))
            sys.stdout.flush()
        return None
    return json.loads(r.read().decode("utf-8", "replace"))


def emit(obj):
    print(json.dumps(obj, indent=2, sort_keys=True))


# --- bash session helpers ---------------------------------------------------

def cmd_sh(args):
    """One command in a fresh (or named) bash session, streamed as it runs."""
    sid = args.session
    if not sid:
        sid = call(args, "POST", "/bash/session", {})["session"]
    url = args.url.rstrip("/") + "/bash/run"
    body = {"session": sid, "command": args.command,
            "timeout": args.command_timeout, "normalize": True}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 method="POST")
    if args.token:
        req.add_header("Authorization", "Bearer " + args.token)
    req.add_header("Content-Type", "application/json")
    code = 0
    try:
        r = urllib.request.urlopen(req, timeout=args.command_timeout + 60)
    except urllib.error.HTTPError as e:
        sys.stderr.write("HTTP %d: %s\n" % (e.code, e.read().decode("utf-8", "replace")))
        return 1
    buf = b""
    for chunk in iter(lambda: r.read(4096), b""):
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                ev = json.loads(line.decode("utf-8", "replace"))
            except ValueError:
                continue
            if ev.get("type") == "stdout":
                sys.stdout.write(ev.get("data", ""))
                sys.stdout.flush()
            elif ev.get("type") == "exit":
                code = ev.get("code", 0)
                if ev.get("truncated"):
                    sys.stderr.write("\n[llmgr] output truncated\n")
            elif ev.get("type") == "error":
                sys.stderr.write("\n[llmgr] %s\n" % ev.get("message", ev))
                code = code or 1
    if not args.session:
        call(args, "POST", "/bash/close", {"session": sid})
    else:
        sys.stderr.write("[llmgr] session %s kept\n" % sid)
    return code


# --- subcommands ------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=DEFAULT_URL,
                   help="llmgr base URL (default %(default)s, $LLMGR_URL)")
    p.add_argument("--token", default=os.environ.get("LLMGR_TOKEN"),
                   help="bearer token ($LLMGR_TOKEN)")
    p.add_argument("--timeout", type=float, default=120.0)
    # NB: no required=True -- Fugaku's system python3 is 3.6, where
    # add_subparsers() does not accept it. Checked by hand below instead.
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("health")
    sub.add_parser("models")
    sub.add_parser("ps", help="list children")
    n = sub.add_parser("nodes")
    n.add_argument("--no-fanout", action="store_true",
                   help="head node only (use while a runner owns the nodes)")

    s = sub.add_parser("sh", help="run one shell command (streamed)")
    s.add_argument("command")
    s.add_argument("--session", help="reuse/keep a named bash session id")
    s.add_argument("--command-timeout", type=float, default=1800.0)

    b = sub.add_parser("build")
    b.add_argument("--model", default="laguna")
    b.add_argument("--variant")
    b.add_argument("--kv-fp16", action="store_true",
                   help="experimental Laguna FP16 KV (FP8 variant only)")
    b.add_argument("--clean", action="store_true")

    st = sub.add_parser("stage")
    st.add_argument("--model", default="laguna")
    st.add_argument("--variant")
    st.add_argument("--stage-dir")
    st.add_argument("--model-dir")
    st.add_argument("--np", type=int)
    st.add_argument("--layer", type=int)
    st.add_argument("--experts")

    ss = sub.add_parser("stage-status")
    ss.add_argument("--model", default="laguna")
    ss.add_argument("--variant")
    ss.add_argument("--stage-dir")
    ss.add_argument("--model-dir")
    ss.add_argument("--np", type=int)
    ss.add_argument("--no-fanout", action="store_true")

    r = sub.add_parser("start", help="start a runner")
    r.add_argument("--model", default="laguna")
    r.add_argument("--variant")
    r.add_argument("--kv-fp16", action="store_true",
                   help="experimental Laguna FP16 KV (FP8 variant only)")
    r.add_argument("--quality-cpp", action="store_true",
                   help="Laguna one-shot: validate and repair generated C++")
    r.add_argument("--mode", choices=("serve", "generate"))
    r.add_argument("--port", type=int)
    r.add_argument("--maxpos", type=int)
    r.add_argument("--max-batch", type=int)
    r.add_argument("--pchunk", type=int)
    r.add_argument("--ar-groups", type=int)
    r.add_argument("--comm-robust", type=int, choices=(0, 1, 2))
    r.add_argument("--comm-poll-spins", type=int)
    r.add_argument("--layers", type=int)
    r.add_argument("--np", type=int)
    r.add_argument("--max-new", type=int)
    r.add_argument("--tokens", type=int, help="K3 partial-runner token steps")
    r.add_argument("--layer", type=int, help="K3 first decoder layer")
    r.add_argument("--experts", help="K3 expert range, e.g. 0-15")
    r.add_argument("--stage-dir")
    r.add_argument("--model-dir")
    r.add_argument("--result-dir")
    r.add_argument("--heartbeat-tokens", type=int)
    r.add_argument("--min-available-mib", type=int)
    r.add_argument("--prompt")
    r.add_argument("--ids", help="path to an ids file (generate mode)")
    r.add_argument("--prompt-ids", help="path to a prompt ids file (Gemma4)")
    r.add_argument("--gguf", help="model GGUF path (Gemma4)")
    r.add_argument("--mtp", help="Gemma4 TP MTP draft GGUF path")
    r.add_argument("--exclude", help="Gemma4 node coordinate to exclude, or none")
    r.add_argument("--threads", type=int, help="Gemma4 worker threads")
    r.add_argument("--spec-k", type=int, help="Gemma4 TP speculative draft length")
    r.add_argument("--batch", type=int, help="Gemma4 TP batch size")
    r.add_argument("--tp-skip-ar", action="store_true",
                   help="Gemma4 TP compute-only mode (output is invalid)")
    r.add_argument("--stage", action="store_true", help="stage before running")
    r.add_argument("--extra", nargs="*", default=[],
                   help="extra flags passed through to the runner")

    k = sub.add_parser("stop")
    k.add_argument("id")

    lg = sub.add_parser("log")
    lg.add_argument("id")
    lg.add_argument("--tail", type=int, default=200)
    lg.add_argument("--follow", action="store_true")

    g = sub.add_parser("gen", help="generate against a ready serve runner")
    g.add_argument("--id", help="optional assertion of the active runner id")
    g.add_argument("--ids", required=True, help="comma-separated token ids")
    g.add_argument("--max-new", type=int, default=32)
    g.add_argument("--sample", action="store_true")
    g.add_argument("--temp", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--top-p", type=float)
    g.add_argument("--seed", type=int)

    ch = sub.add_parser("chat", help="OpenAI-compatible Laguna chat")
    ch.add_argument("prompt")
    ch.add_argument("--system")
    ch.add_argument("--model", default="laguna-s21")
    ch.add_argument("--max-new", type=int, default=256)
    ch.add_argument("--temperature", type=float, default=0.0)
    ch.add_argument("--top-p", type=float)
    ch.add_argument("--seed", type=int)
    ch.add_argument("--no-think", action="store_true")
    ch.add_argument("--stream", action="store_true")

    sub.add_parser("queue", help="show the inference FIFO")
    ca = sub.add_parser("cancel", help="cancel a queued/running inference")
    ca.add_argument("id")

    pr = sub.add_parser("profile")
    pr.add_argument("--model", default="laguna")
    pr.add_argument("--variant")
    pr.add_argument("--kv-fp16", action="store_true",
                    help="experimental Laguna FP16 KV (FP8 variant only)")
    pr.add_argument("--ids", help="token ids file (required by Laguna, not K3)")
    pr.add_argument("--max-new", type=int, default=16)
    pr.add_argument("--tokens", type=int, help="K3 profile token steps")
    pr.add_argument("--layer", type=int, help="K3 profile decoder layer")
    pr.add_argument("--threads", type=int, help="K3 profile worker threads")
    pr.add_argument("--stage-dir", help="existing K3 rank-local stage")
    pr.add_argument("--event", default="statistics")
    pr.add_argument("--np", type=int)

    ar = sub.add_parser("artifacts")
    ar.add_argument("id")

    kv = sub.add_parser("kv")
    kv.add_argument("action", choices=("save", "load", "clear", "stats"))
    kv.add_argument("--id")
    kv.add_argument("--path")

    sub.add_parser("shutdown")

    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 2

    def opt(*keys):
        """Only send flags the user actually set: llmgr fills the defaults."""
        return {k: getattr(args, k) for k in keys
                if getattr(args, k, None) not in (None, False, [])}

    c = args.cmd
    if c == "health":
        emit(call(args, "GET", "/health"))
    elif c == "models":
        emit(call(args, "GET", "/models"))
    elif c == "ps":
        emit(call(args, "GET", "/runner"))
    elif c == "nodes":
        q = "?fanout=0" if args.no_fanout else ""
        emit(call(args, "GET", "/nodes" + q, timeout=300))
    elif c == "sh":
        return cmd_sh(args)
    elif c == "build":
        emit(call(args, "POST", "/build",
                  dict(model=args.model, **opt("variant", "kv_fp16", "clean"))))
    elif c == "stage":
        emit(call(args, "POST", "/stage",
                  dict(model=args.model,
                       **opt("variant", "stage_dir", "model_dir", "np",
                             "layer", "experts"))))
    elif c == "stage-status":
        q = {"model": args.model}
        q.update(opt("variant", "stage_dir", "model_dir", "np"))
        if args.no_fanout:
            q["fanout"] = "0"
        emit(call(args, "GET", "/stage/status?" + urllib.parse.urlencode(q),
                  timeout=300))
    elif c == "start":
        emit(call(args, "POST", "/runner/start",
                  dict(model=args.model,
                       **opt("variant", "kv_fp16", "quality_cpp", "mode", "port", "maxpos", "max_batch",
                             "pchunk", "ar_groups", "comm_robust",
                             "comm_poll_spins", "layers",
                             "np", "max_new", "tokens", "layer", "experts",
                             "stage_dir", "model_dir", "result_dir",
                             "heartbeat_tokens", "min_available_mib",
                             "prompt", "ids", "prompt_ids",
                             "gguf", "mtp", "exclude", "threads", "spec_k",
                             "batch", "tp_skip_ar", "stage",
                             "extra"))))
    elif c == "stop":
        emit(call(args, "POST", "/runner/stop", {"id": args.id}, timeout=180))
    elif c == "log":
        path = "/runner/%s/log?tail=%d" % (args.id, args.tail)
        if args.follow:
            call(args, "GET", path + "&follow=1", stream=True, timeout=None)
        else:
            print(call(args, "GET", path)["log"])
    elif c == "gen":
        ids = [int(x) for x in args.ids.replace(",", " ").split()]
        body = dict(id=args.id, ids=ids, max_new=args.max_new)
        body.update(opt("sample", "temp", "top_k", "top_p", "seed"))
        emit(call(args, "POST", "/generate", body, timeout=1800))
    elif c == "chat":
        messages = []
        if args.system is not None:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        body = {"model": args.model, "messages": messages,
                "max_completion_tokens": args.max_new,
                "temperature": args.temperature,
                "enable_thinking": not args.no_think,
                "stream": args.stream}
        body.update(opt("top_p", "seed"))
        if args.stream:
            call(args, "POST", "/v1/chat/completions", body,
                 stream=True, timeout=None)
        else:
            emit(call(args, "POST", "/v1/chat/completions", body,
                      timeout=3600))
    elif c == "queue":
        emit(call(args, "GET", "/inference/queue"))
    elif c == "cancel":
        emit(call(args, "POST", "/inference/cancel", {"id": args.id}))
    elif c == "profile":
        emit(call(args, "POST", "/profile",
                  dict(model=args.model, max_new=args.max_new,
                       event=args.event, **opt("variant", "kv_fp16", "np", "ids",
                                              "tokens", "layer", "threads",
                                              "stage_dir")), timeout=600))
    elif c == "artifacts":
        emit(call(args, "GET", "/profile/%s/artifacts" % args.id))
    elif c == "kv":
        emit(call(args, "POST", "/kv",
                  dict(action=args.action, **opt("id", "path"))))
    elif c == "shutdown":
        emit(call(args, "POST", "/shutdown", {}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
