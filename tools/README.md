# bash-over-HTTP

A persistent, stateful bash execution service for driving an LLM CLI agent
inside a **trusted VPN** (no authentication by default; optional shared-token
auth available — see Configuration). Each session is a long-lived
interactive bash shell (via `pty.fork`), so `cwd`, shell variables and
background jobs persist across calls. Output streams back incrementally as
NDJSON.

## Files

- `bash_http_server.py` — the server. **Python 3 standard library only.**
  Binds `127.0.0.1:21264` only (never `0.0.0.0`).
- `bash_http_client.py` — Python client library (requires `requests`).
- `bash_http_example.py` — runnable end-to-end demo.
- `test_bash_http.py` — `unittest` suite.

## Run

```sh
# 1. start the server (foreground)
python3 bash_http_server.py

# 2. in another shell, run the demo
python3 bash_http_example.py
```

The demo proves state persistence, incremental streaming, and output
truncation.

## Configuration

Flags (CLI) override environment variables, which override defaults:

| Flag          | Env var              | Default       | Purpose |
|---------------|----------------------|---------------|---------|
| `--host`      | `BASH_HTTP_HOST`     | `127.0.0.1`   | bind address (keep loopback) |
| `--port`      | `BASH_HTTP_PORT`     | `21264`       | bind port |
| `--token`     | `BASH_HTTP_TOKEN`    | _(none)_      | require `Authorization: Bearer <token>` |
| `--idle-ttl`  | `BASH_HTTP_IDLE_TTL` | `1800`        | seconds before an idle session is reaped |
| `--max-sessions` | `BASH_HTTP_MAX_SESSIONS` | `256`   | concurrent session cap (0 = unlimited) |
| `--verbose`   | `BASH_HTTP_VERBOSE`  | off           | log requests/errors to stderr |

```sh
BASH_HTTP_PORT=21299 python3 bash_http_server.py --token s3cret --verbose
```

When a token is configured, every request (POST and GET) must send
`Authorization: Bearer <token>` or gets `401 {"error":"unauthorized"}`.

## HTTP API (all POST, JSON bodies)

| Endpoint     | Body                                          | Response |
|--------------|-----------------------------------------------|----------|
| `/session`   | `{}`                                          | `{"session": "<id>"}` |
| `/run`       | `{"session","command","timeout"?,"max_bytes"?,"normalize"?}` | NDJSON stream (chunked) |
| `/interrupt` | `{"session"}`                                 | `{"interrupted": true}` |
| `/close`     | `{"session"}`                                 | `{"closed": true}` |

`normalize: true` makes the server strip ANSI/OSC escape sequences and collapse
CRLF/CR so the streamed text is plain (best-effort across chunk boundaries).

Read-only introspection (GET):

| Endpoint     | Response |
|--------------|----------|
| `GET /health`   | `{"ok":true,"version","uptime_seconds","sessions":<count>}` |
| `GET /sessions` | `{"sessions":[{"session","pid","cwd","busy","idle_seconds","alive"}...],"count"}` |

`cwd` in `/sessions` is read from the shell's `/proc/<pid>/cwd`, so it works
even while the session is busy.

Unknown session → HTTP 404 `{"error": "no such session"}`.
Concurrent `/run` on a session already running a command → HTTP 409
`{"error": "session busy"}` (sessions are used serially).

### `/run` NDJSON events

- `{"type":"stdout","data":"<text>"}` — incremental output chunk.
- `{"type":"exit","code":<int>,"truncated":false,"bytes":<int>}` — normal
  completion.
- `{"type":"exit","code":null,"truncated":true,"bytes":<int>,"reason":"max_bytes_exceeded"}`
  — output hit `max_bytes` (default 1,000,000); the command was interrupted.
  Treated as a completion ("output ends here").
- `{"type":"error","reason":"timeout","recoverable":true,"timeout":<int>,"bytes":<int>}`
  — command exceeded `timeout` (default 300s); the server sent SIGINT and the
  session is **still reusable**.

## Operational behavior

- **Serial sessions**: one command at a time per session; a concurrent `/run`
  gets `409 session busy`.
- **`exit` ends the session, not just the command**: each session is one
  persistent shell, so a command containing a bare `exit [N]` (or anything that
  kills the shell) terminates the *whole session* — that `/run` returns
  `code: null` and later calls get `404 no such session`. To surface a non-zero
  status without dropping the session, run it in a subshell — `( exit 7 )` →
  `code: 7` — or just let a failing tool set it (`ls /nope` → `code: 2`). The
  session shell stays alive either way.
- **Client disconnect**: if the client drops mid-`/run`, the server SIGINTs the
  running command and drains it so the session is left clean (briefly `409` busy
  while cleanup finishes).
- **Idle sweeper**: a session untouched for `SESSION_IDLE_TTL` (default 30 min)
  is reaped, so a crashed client can't leak a live shell. Sessions running a
  command are never reaped mid-flight. Any activity (`/run`, `/interrupt`)
  resets the idle clock.
- **Clean output**: the child shell runs with `TERM=dumb` and a fixed
  `200×50` PTY window, so programs emit far fewer colour/cursor escapes and
  don't wrap to 80 columns.
- **Clean teardown**: `/close` (and the sweeper) terminate and *reap* the
  child shell, so closed sessions leave no zombie processes.
- **Limits**: a soft session cap returns `503 {"error":"too many sessions"}`
  to guard against runaway shell creation; malformed `/run` input (e.g.
  non-positive `timeout`/`max_bytes`) returns `400`; request bodies are capped
  at 16 MiB.

## Client library

```python
import bash_http_client as bh

bh.BASE_URL   = "http://127.0.0.1:21264"   # server address
bh.AUTH_TOKEN = "s3cret"                    # only if the server requires a token

# functional style
sid = bh.new_session()
result = bh.run(sid, "echo $HOME; pwd", on_stdout=print, normalize=True)
print(result.code, result.stdout)   # attribute access; result["code"] also works
#  result fields: stdout, code, truncated, timed_out, bytes
bh.interrupt(sid)   # Ctrl-C the running command
bh.close(sid)       # terminate the shell

# context-manager style (auto-closes the session)
with bh.Shell() as sh:
    sh.run("cd /tmp")
    print(sh.run("pwd").stdout)

bh.health()         # {ok, version, uptime_seconds, sessions}
bh.sessions()       # [{session, pid, cwd, busy, idle_seconds, alive}, ...]
```

The client pools connections (keep-alive) via a shared `requests.Session`.

## Tests

```sh
python3 -m unittest test_bash_http -v
```

Runs the server in-process on an ephemeral port and covers state persistence,
incremental streaming, truncation, recoverable timeout, interrupt→130, the
concurrency guard, client-disconnect cleanup, the idle sweeper, and the
introspection endpoints.

## Remote exposure

The server binds loopback only. Expose it from a remote host with an SSH
remote-forward through a bastion, e.g.:

```sh
ssh -R 21264:127.0.0.1:21264 user@bastion
```

so the agent reaches `http://127.0.0.1:21264` on the bastion side. The server
itself never listens on a public interface.
