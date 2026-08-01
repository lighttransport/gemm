#!/usr/bin/env python3
"""Usage example for the bash-over-HTTP client/server.

Start the server first:

    python3 bash_http_server.py

Then run this example:

    python3 bash_http_example.py

Demonstrates, against a single session:
  1. cd /tmp && pwd
  2. X=42; echo set
  3. echo $X; pwd          -> 42 and /tmp (state persists across calls)
  4. a streaming job        -> printed incrementally via on_stdout
  5. a truncation demo      -> truncated: true with a tiny max_bytes
"""

import sys

import bash_http_client as bh


def main():
    sid = bh.new_session()
    print("session: %s\n" % sid)

    print("== 1. cd /tmp && pwd ==")
    r = bh.run(sid, "cd /tmp && pwd")
    print("output: %r  code=%s\n" % (r["stdout"].strip(), r["code"]))

    print("== 2. X=42; echo set ==")
    r = bh.run(sid, "X=42; echo set")
    print("output: %r  code=%s\n" % (r["stdout"].strip(), r["code"]))

    print("== 3. echo $X; pwd  (proves state persistence) ==")
    r = bh.run(sid, "echo $X; pwd")
    print("output: %r  code=%s" % (r["stdout"].strip(), r["code"]))
    assert "42" in r["stdout"] and "/tmp" in r["stdout"], "state did not persist!"
    print("state persisted ✓\n")

    print("== 4. streaming: for i in 1 2 3; do echo $i; sleep 1; done ==")

    def live(chunk):
        sys.stdout.write("  [stream] " + chunk.replace("\n", "\n  [stream] "))
        sys.stdout.flush()

    r = bh.run(sid, "for i in 1 2 3; do echo $i; sleep 1; done",
               on_stdout=live)
    print("\n  code=%s\n" % r["code"])

    print("== 5. truncation: yes | head -100000 with max_bytes=200 ==")
    r = bh.run(sid, "yes | head -100000", max_bytes=200)
    print("truncated=%s  bytes=%s  code=%s" %
          (r["truncated"], r["bytes"], r["code"]))
    assert r["truncated"] is True, "expected truncation!"
    print("truncation works ✓\n")

    bh.close(sid)
    print("session closed.")


if __name__ == "__main__":
    main()
