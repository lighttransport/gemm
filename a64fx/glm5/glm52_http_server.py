#!/usr/bin/env python3
"""Small serialized HTTP gateway for the 12-node GLM-5.2 Q2 runner."""

import argparse
import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glm5_tokenizer import Tok, TOKJSON  # noqa: E402


class QueueFull(RuntimeError):
    pass


class WorkerUnavailable(RuntimeError):
    pass


class PersistentWorker:
    def __init__(self, args):
        stamp = "worker-%d-%d" % (int(time.time()), os.getpid())
        self.control = Path(args.work_dir) / stamp
        self.control.mkdir(mode=0o700)
        self.seq = 0
        self.stdout = open(str(self.control / "worker.stdout"), "w")
        self.stderr = open(str(self.control / "worker.stderr"), "w")
        cmd = [args.runner, "serve", "--no-stage", "--no-enforce", "--stable-outputs",
               "--kv-tier-bf16=0", "--ctx", str(args.max_context + args.max_tokens + 128),
               "--slots", str(args.max_slots), "--pchunk", str(args.worker_pchunk),
               "--serve-dir", str(self.control),
               "--set", "GLM5_SERVE_CAPACITY_PROBE=1"]
        self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=self.stdout, stderr=self.stderr,
                                     preexec_fn=os.setsid)
        deadline = time.time() + args.startup_timeout
        while not (self.control / "ready").is_file():
            if self.proc.poll() is not None:
                raise RuntimeError("persistent MPI worker exited during startup; see %s" % self.control)
            if time.time() >= deadline:
                self.stop()
                raise RuntimeError("persistent MPI worker startup timed out")
            time.sleep(0.25)

    def submit(self, batches, max_tokens, timeout):
        if self.proc.poll() is not None:
            raise WorkerUnavailable("persistent MPI worker exited; see %s" % self.control)
        self.seq += 1
        seq = self.seq
        prompts = self.control / ("prompts-%d.ids" % seq)
        prefix = self.control / ("generated-%d" % seq)
        prompts.write_text("".join(" ".join(map(str, ids)) + "\n" for ids in batches))
        done = self.control / "done"
        try:
            done.unlink()
        except FileNotFoundError:
            pass
        tmp = self.control / "request.tmp"
        tmp.write_text("%d %d %d %s %s\n" %
                       (seq, max_tokens, len(batches), prompts, prefix))
        os.replace(str(tmp), str(self.control / "request"))
        deadline = time.time() + timeout
        while True:
            if self.proc.poll() is not None:
                raise WorkerUnavailable("persistent MPI worker exited; see %s" % self.control)
            if done.is_file():
                fields = done.read_text().split()
                if len(fields) == 2 and int(fields[0]) == seq:
                    if int(fields[1]) != 0:
                        raise RuntimeError("persistent MPI request failed")
                    break
            if time.time() >= deadline:
                self.stop()
                raise subprocess.TimeoutExpired("persistent MPI request", timeout)
            time.sleep(0.05)
        outputs = []
        for i in range(len(batches)):
            path = Path("%s_%03d.txt" % (prefix, i))
            if not path.is_file():
                raise RuntimeError("persistent MPI output missing: %s" % path)
            outputs.append([int(x) for x in path.read_text().split()])
        # Completion is published only after every rank has finished with these paths.
        # Keep worker logs, but do not leak one prompt/output set per HTTP request.
        for path in [prompts, self.control / "request", done] + [
                Path("%s_%03d.txt" % (prefix, i)) for i in range(len(batches))]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return outputs

    def stop(self):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            (self.control / "stop").write_text("stop\n")
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, 15)
                    self.proc.wait(timeout=5)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.proc.pid, 9)
                    except ProcessLookupError:
                        pass
                    self.proc.wait()
        for stream in (getattr(self, "stdout", None), getattr(self, "stderr", None)):
            if stream:
                stream.close()


class Service:
    def __init__(self, args):
        self.args = args
        self.tokenizer = Tok(args.tokenizer)
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.admission = threading.BoundedSemaphore(getattr(args, "max_queue", 8) + 1)
        self.started = time.time()
        self.requests = 0
        self.completed = 0
        self.failed = 0
        self.rejected = 0
        self.waiting = 0
        self.inflight = 0
        self.contexts = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.service_seconds = 0.0
        self.queue_seconds = 0.0
        self.worker = PersistentWorker(args) if getattr(args, "persistent", False) else None
        if self.worker:
            atexit.register(self.worker.stop)

    def worker_alive(self):
        return not self.worker or self.worker.proc.poll() is None

    @contextmanager
    def admitted(self):
        if not self.worker_alive():
            raise WorkerUnavailable("persistent MPI worker is not running")
        if not self.admission.acquire(False):
            with self.state_lock:
                self.rejected += 1
            raise QueueFull("server queue is full")
        with self.state_lock:
            self.waiting += 1
        queued_at = time.time()
        running_at = None
        ok = False
        try:
            with self.lock:
                running_at = time.time()
                with self.state_lock:
                    self.waiting -= 1
                    self.inflight += 1
                    self.queue_seconds += running_at - queued_at
                yield
                ok = True
        finally:
            with self.state_lock:
                if running_at is not None:
                    self.inflight -= 1
                    self.service_seconds += time.time() - running_at
                else:
                    self.waiting -= 1
                if ok:
                    self.completed += 1
                else:
                    self.failed += 1
            self.admission.release()

    def record_usage(self, usage, contexts):
        with self.state_lock:
            self.contexts += contexts
            self.prompt_tokens += usage["prompt_tokens"]
            self.completion_tokens += usage["completion_tokens"]

    def snapshot(self):
        with self.state_lock:
            return {
                "status": "ok" if self.worker_alive() else "unavailable",
                "ready": self.worker_alive(), "busy": bool(self.inflight),
                "waiting": self.waiting, "inflight": self.inflight,
                "requests": self.requests, "completed": self.completed,
                "failed": self.failed, "rejected": self.rejected,
                "contexts": self.contexts,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "service_seconds": round(self.service_seconds, 3),
                "queue_seconds": round(self.queue_seconds, 3),
                "persistent": bool(self.worker),
                "worker_alive": bool(self.worker and self.worker.proc.poll() is None),
                "uptime_seconds": int(time.time() - self.started),
            }

    def complete(self, body):
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        max_tokens = body.get("max_tokens", self.args.max_tokens)
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= self.args.max_tokens:
            raise ValueError("max_tokens must be an integer in [1, %d]" % self.args.max_tokens)
        ids = self.tokenizer.chat(prompt, think=bool(body.get("think", False)))
        if len(ids) + max_tokens > self.args.max_context:
            raise ValueError("prompt plus max_tokens exceeds max_context")

        with self.admitted():
            self.requests += 1
            request_id = "glm52-%d-%06d" % (int(time.time()), self.requests)
            if self.worker:
                started = time.time()
                generated = self.worker.submit([ids], max_tokens, self.args.timeout)[0]
                text = self.tokenizer.decode(generated)
                result = {
                    "id": request_id, "object": "text_completion", "created": int(started),
                    "model": "glm-5.2-q2-a64fx-ep12",
                    "choices": [{"text": text, "index": 0,
                                 "finish_reason": "stop" if len(generated) < max_tokens else "length"}],
                    "usage": {"prompt_tokens": len(ids), "completion_tokens": len(generated),
                              "total_tokens": len(ids) + len(generated)},
                    "elapsed_seconds": round(time.time() - started, 3),
                }
                self.record_usage(result["usage"], 1)
                return result
            with tempfile.TemporaryDirectory(prefix=request_id + "-", dir=self.args.work_dir) as td:
                prompt_file = Path(td) / "prompt.ids"
                output_file = Path(td) / "generated.ids"
                prompt_file.write_text(" ".join(map(str, ids)) + "\n")
                ctx = len(ids) + max_tokens + 128
                cmd = [
                    self.args.runner, "generate", "--no-stage",
                    "--no-enforce", "--stable-outputs", "--ctx", str(ctx),
                    "--prompt-ids", str(prompt_file), "--gen-out", str(output_file),
                    "--max-new", str(max_tokens), "--min-new", "0",
                ]
                if len(ids) >= self.args.int4_threshold:
                    cmd += ["--kv-tier-bf16=0"]
                started = time.time()
                proc = subprocess.run(cmd, cwd=HERE, universal_newlines=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, timeout=self.args.timeout)
                if proc.returncode or not output_file.is_file():
                    tail = (proc.stderr or proc.stdout)[-2000:]
                    raise RuntimeError("runner failed (exit %d): %s" % (proc.returncode, tail))
                generated = [int(x) for x in output_file.read_text().split()]
                text = self.tokenizer.decode(generated)
                result = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(started),
                    "model": "glm-5.2-q2-a64fx-ep12",
                    "choices": [{"text": text, "index": 0,
                                 "finish_reason": "stop" if len(generated) < max_tokens else "length"}],
                    "usage": {"prompt_tokens": len(ids), "completion_tokens": len(generated),
                              "total_tokens": len(ids) + len(generated)},
                    "elapsed_seconds": round(time.time() - started, 3),
                }
                self.record_usage(result["usage"], 1)
                return result

    def complete_many(self, body):
        prompts = body.get("prompts")
        if not isinstance(prompts, list) or not prompts or not all(isinstance(x, str) and x for x in prompts):
            raise ValueError("prompts must be a non-empty array of strings")
        if len(prompts) > self.args.max_slots:
            raise ValueError("prompts exceeds max_slots (%d)" % self.args.max_slots)
        max_tokens = body.get("max_tokens", self.args.max_tokens)
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= self.args.max_tokens:
            raise ValueError("max_tokens must be an integer in [1, %d]" % self.args.max_tokens)
        batches = [self.tokenizer.chat(p, think=bool(body.get("think", False))) for p in prompts]
        lengths = [len(ids) + max_tokens for ids in batches]
        if max(lengths) > self.args.max_context:
            raise ValueError("a prompt plus max_tokens exceeds max_context")
        if sum(lengths) > self.args.max_total_context:
            raise ValueError("aggregate contexts exceed max_total_context")

        with self.admitted():
            self.requests += 1
            request_id = "glm52-batch-%d-%06d" % (int(time.time()), self.requests)
            if self.worker:
                started = time.time()
                generated_many = self.worker.submit(batches, max_tokens, self.args.timeout)
                choices = [{"text": self.tokenizer.decode(ids), "index": i,
                            "finish_reason": "stop" if len(ids) < max_tokens else "length"}
                           for i, ids in enumerate(generated_many)]
                prompt_tokens = sum(len(ids) for ids in batches)
                completion_tokens = sum(len(ids) for ids in generated_many)
                result = {
                    "id": request_id, "object": "text_completion.batch", "created": int(started),
                    "model": "glm-5.2-q2-a64fx-ep12", "choices": choices,
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                              "total_tokens": prompt_tokens + completion_tokens},
                    "contexts": len(batches), "elapsed_seconds": round(time.time() - started, 3),
                }
                self.record_usage(result["usage"], len(prompts))
                return result
            with tempfile.TemporaryDirectory(prefix=request_id + "-", dir=self.args.work_dir) as td:
                batch_file = Path(td) / "prompts.ids"
                out_prefix = str(Path(td) / "generated")
                batch_file.write_text("".join(" ".join(map(str, ids)) + "\n" for ids in batches))
                ctx = max(lengths) + 128
                cmd = [
                    self.args.runner, "generate", "--no-stage", "--no-enforce",
                    "--stable-outputs", "--ctx", str(ctx), "--prompts", str(batch_file),
                    "--slots", str(len(prompts)), "--out-prefix", out_prefix,
                    "--max-new", str(max_tokens),
                ]
                if max(len(ids) for ids in batches) >= self.args.int4_threshold:
                    cmd += ["--kv-tier-bf16=0"]
                started = time.time()
                proc = subprocess.run(cmd, cwd=HERE, universal_newlines=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      timeout=self.args.timeout)
                outputs = [Path("%s_%03d.txt" % (out_prefix, i)) for i in range(len(prompts))]
                if proc.returncode or not all(path.is_file() for path in outputs):
                    tail = (proc.stderr or proc.stdout)[-2000:]
                    raise RuntimeError("runner failed (exit %d): %s" % (proc.returncode, tail))
                choices = []
                completion_tokens = 0
                for i, path in enumerate(outputs):
                    generated = [int(x) for x in path.read_text().split()]
                    completion_tokens += len(generated)
                    choices.append({"text": self.tokenizer.decode(generated), "index": i,
                                    "finish_reason": "stop" if len(generated) < max_tokens else "length"})
                prompt_tokens = sum(len(ids) for ids in batches)
                result = {
                    "id": request_id, "object": "text_completion.batch", "created": int(started),
                    "model": "glm-5.2-q2-a64fx-ep12", "choices": choices,
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                              "total_tokens": prompt_tokens + completion_tokens},
                    "contexts": len(prompts), "elapsed_seconds": round(time.time() - started, 3),
                }
                self.record_usage(result["usage"], len(prompts))
                return result


class Handler(BaseHTTPRequestHandler):
    server_version = "glm52-http/1"

    def send_json(self, status, value):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path in ("/health", "/ready", "/metrics"):
            s = self.server.service
            snapshot = s.snapshot()
            status = 200 if self.path == "/metrics" or snapshot["ready"] else 503
            self.send_json(status, snapshot)
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/completions", "/v1/batch/completions", "/generate"):
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.server.service.args.max_body:
                raise ValueError("invalid request body size")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            if self.path == "/v1/batch/completions" or "prompts" in body:
                result = self.server.service.complete_many(body)
            else:
                result = self.server.service.complete(body)
            self.send_json(200, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except subprocess.TimeoutExpired:
            self.send_json(504, {"error": "runner timed out"})
        except QueueFull as exc:
            self.send_json(429, {"error": str(exc)})
        except WorkerUnavailable as exc:
            self.send_json(503, {"error": str(exc)})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--tokenizer", default=TOKJSON)
    p.add_argument("--runner", default=str(HERE / "run_glm52_q2_12n.sh"))
    p.add_argument("--work-dir", default=str(HERE / "logs"))
    p.add_argument("--max-context", type=int, default=262144)
    p.add_argument("--max-total-context", type=int, default=0)
    p.add_argument("--max-slots", type=int, default=3)
    p.add_argument("--worker-pchunk", type=int, default=64)
    p.add_argument("--max-queue", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--int4-threshold", type=int, default=23000)
    p.add_argument("--max-body", type=int, default=8 << 20)
    p.add_argument("--timeout", type=int, default=21600)
    p.add_argument("--startup-timeout", type=int, default=600)
    p.add_argument("--no-persistent", dest="persistent", action="store_false")
    p.set_defaults(persistent=True)
    args = p.parse_args()
    if args.max_queue < 0:
        p.error("--max-queue must be non-negative")
    if args.max_total_context <= 0:
        args.max_total_context = args.max_context * args.max_slots
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.service = Service(args)
    print("GLM52 HTTP listening on http://%s:%d" % (args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server.service.worker:
            server.service.worker.stop()


if __name__ == "__main__":
    main()
