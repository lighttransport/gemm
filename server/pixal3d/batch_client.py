#!/usr/bin/env python3
"""Submit and monitor a Pixal3D/Qwen Image batch using only the stdlib."""
import argparse, json, sys, time
from urllib.request import Request, urlopen

def call(url, method="GET", body=None, token=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    if token: headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, data=data, method=method, headers=headers), timeout=30) as r:
        return json.load(r)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs_file", help="JSON object containing a jobs array")
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--token", default=None)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--results", action="store_true")
    args = p.parse_args()
    with open(args.jobs_file, encoding="utf-8") as f: payload = json.load(f)
    if isinstance(payload, list): payload = {"jobs": payload}
    created = call(args.url.rstrip("/") + "/v1/batches", "POST", payload, args.token)
    print(json.dumps(created, indent=2)); batch_id = created["batch_id"]
    while True:
        suffix = "?results=1" if args.results else ""
        status = call(args.url.rstrip("/") + f"/v1/batches/{batch_id}{suffix}", token=args.token)
        print(json.dumps(status, indent=2), flush=True)
        if status["state"] in ("complete", "failed"): return 0 if status["state"] == "complete" else 1
        time.sleep(max(0.1, args.interval))

if __name__ == "__main__": sys.exit(main())
