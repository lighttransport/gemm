#!/usr/bin/env python3
"""Client library for the bash-over-HTTP execution server.

Thin wrapper over the server's HTTP API. Uses `requests` (with a pooled
Session) for readable streaming of the NDJSON /run response.

    import bash_http_client as bh

    # functional style
    sid = bh.new_session()
    result = bh.run(sid, "echo hello", on_stdout=print)
    print(result.code, result["stdout"])   # attribute or item access
    bh.close(sid)

    # context-manager style (auto-closes the session)
    with bh.Shell() as sh:
        print(sh.run("pwd").stdout)

Configuration:
    bh.BASE_URL   = "http://127.0.0.1:21264"   # server address
    bh.AUTH_TOKEN = "secret"                    # if the server requires a token
"""

import json

import requests

BASE_URL = "http://127.0.0.1:21264"
AUTH_TOKEN = None   # set if the server was started with --token

# Pooled connection reuse across calls (keep-alive).
_http = requests.Session()


class Result(dict):
    """A /run summary that supports both item and attribute access.

    Backwards-compatible with the old dict return (r["code"]) while also
    allowing r.code / r.stdout / r.truncated / r.timed_out / r.bytes.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _url(base, path):
    return (base or BASE_URL).rstrip("/") + path


def _headers():
    return {"Authorization": "Bearer " + AUTH_TOKEN} if AUTH_TOKEN else {}


def _check(resp, session=None):
    if resp.status_code == 404:
        raise RuntimeError("no such session: %s" % session)
    if resp.status_code == 409:
        raise RuntimeError("session busy: %s" % session)
    if resp.status_code == 401:
        raise RuntimeError("unauthorized (check bh.AUTH_TOKEN)")
    resp.raise_for_status()


def new_session(base_url=None) -> str:
    """Create a session and return its id."""
    resp = _http.post(_url(base_url, "/session"), json={}, headers=_headers())
    _check(resp)
    return resp.json()["session"]


def health(base_url=None) -> dict:
    """Return server health: {ok, version, uptime_seconds, sessions}."""
    resp = _http.get(_url(base_url, "/health"), headers=_headers())
    _check(resp)
    return resp.json()


def sessions(base_url=None) -> list:
    """Return a list of active session info dicts (id, cwd, busy, idle, ...)."""
    resp = _http.get(_url(base_url, "/sessions"), headers=_headers())
    _check(resp)
    return resp.json()["sessions"]


def run(session, command, timeout=300, max_bytes=1_000_000, on_stdout=None,
        normalize=False, base_url=None) -> "Result":
    """Run a command on a session, streaming output.

    Iterates the NDJSON /run response line by line. Accumulates stdout text;
    if on_stdout is given it is called with each incremental chunk for live
    display. With normalize=True the server strips ANSI escapes and CRs so the
    text is plain.

    Returns a Result (dict + attribute access):
        {"stdout": <full text>, "code": <int|None>, "truncated": <bool>,
         "timed_out": <bool>, "bytes": <int>}
    """
    payload = {"session": session, "command": command, "timeout": timeout,
               "max_bytes": max_bytes, "normalize": normalize}
    chunks = []
    code = None
    truncated = False
    timed_out = False
    nbytes = 0

    resp = _http.post(_url(base_url, "/run"), json=payload, stream=True,
                      headers=_headers())
    _check(resp, session)

    for line in resp.iter_lines(decode_unicode=False):
        if not line:
            continue
        event = json.loads(line.decode("utf-8") if isinstance(line, bytes)
                           else line)
        etype = event.get("type")
        if etype == "stdout":
            text = event.get("data", "")
            chunks.append(text)
            if on_stdout is not None:
                on_stdout(text)
        elif etype == "exit":
            code = event.get("code")
            truncated = event.get("truncated", False)
            nbytes = event.get("bytes", nbytes)
        elif etype == "error":
            if event.get("reason") == "timeout":
                timed_out = True
            nbytes = event.get("bytes", nbytes)

    return Result(stdout="".join(chunks), code=code, truncated=truncated,
                  timed_out=timed_out, bytes=nbytes)


def interrupt(session, base_url=None):
    """Send Ctrl-C to the command currently running on the session."""
    resp = _http.post(_url(base_url, "/interrupt"),
                      json={"session": session}, headers=_headers())
    _check(resp, session)
    return resp.json()


def close(session, base_url=None):
    """Terminate the session's shell and remove it."""
    resp = _http.post(_url(base_url, "/close"),
                      json={"session": session}, headers=_headers())
    _check(resp, session)
    return resp.json()


class Shell:
    """Context manager wrapping one session; auto-closes on exit.

        with bh.Shell() as sh:
            sh.run("cd /tmp")
            print(sh.run("pwd").stdout)
    """

    def __init__(self, base_url=None):
        self.base_url = base_url
        self.id = None

    def __enter__(self):
        self.id = new_session(self.base_url)
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def run(self, command, **kw):
        kw.setdefault("base_url", self.base_url)
        return run(self.id, command, **kw)

    def interrupt(self):
        return interrupt(self.id, self.base_url)

    def close(self):
        if self.id is not None:
            try:
                close(self.id, self.base_url)
            finally:
                self.id = None
