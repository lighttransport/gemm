#!/usr/bin/env python3
"""Replay a reference dump (ref dir with in_*.npy, smp_noise.*, smp_out_*) through hip_praxis_server (binary protocol) and compare. numpy only.
  python3 test_server_replay.py --server rdna4/praxis/hip_praxis_server --policy DIR --ref tmp/praxis_ref/real_fp32 [--text in_text.npy --text-u in_text_u.npy]"""
import argparse, os, struct, subprocess, sys, time
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True); ap.add_argument("--policy", required=True); ap.add_argument("--ref", required=True)
ap.add_argument("--stream", default="2"); ap.add_argument("--calls", type=int, default=2); ap.add_argument("--skip-last-video", type=int, default=0)
a = ap.parse_args()
L = lambda n: np.load(os.path.join(a.ref, n + ".npy")).astype(np.float32)
anchor, noised, prop, nv, na = L("in_anchor"), L("in_noised"), L("in_proprio"), L("smp_noise.salt0.0"), L("smp_noise.salt1.0")
clean = L("in_clean") if os.path.exists(os.path.join(a.ref, "in_clean.npy")) else None
K = int(L("in_t_trunk").shape[1]); h, w = anchor.shape[3], anchor.shape[4]
srv = subprocess.Popen([a.server, "--policy", a.policy, "--stream", a.stream, "--text", os.path.join(a.ref, "in_text.npy"), "--text-u", os.path.join(a.ref, "in_text_u.npy")],
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def req(op, payload):
    srv.stdin.write(struct.pack("<IIQ", 0x51585250, op, len(payload)) + payload); srv.stdin.flush()
    m, st, ln = struct.unpack("<IiQ", srv.stdout.read(16))
    body = srv.stdout.read(ln) if ln else b""
    return st, body
def pad(b): return b + b"\0" * ((-len(b)) % 4)
for call in range(a.calls):
    p = struct.pack("<5i3d", K, h, w, 2, a.skip_last_video, 12.5, 4.0, 1.0)
    for s in (b"prompt", b" "): p += struct.pack("<i", len(s)) + pad(s)
    for x in (anchor, clean, prop, nv, na):
        if x is not None: p += np.ascontiguousarray(x).tobytes()
    t0 = time.time(); st, body = req(2, p); dt = time.time() - t0
    if st: print("status", st); sys.exit(1)
    hd = struct.unpack("<4d", body[:32]); C, F, hh, ww, A = struct.unpack("<5i", body[32:52])
    vid = np.frombuffer(body[52:52 + 4 * C * F * hh * ww], np.float32); act = np.frombuffer(body[52 + 4 * C * F * hh * ww:], np.float32)
    rv, ra = L("smp_out_video").ravel(), L("smp_out_action").ravel()
    rel = lambda x, r: float(np.linalg.norm(x - r) / (np.linalg.norm(r) + 1e-30))
    print(f"call {call}: {dt:.2f} s (server {hd[0]:.0f} ms, load {hd[1]:.0f} ms, peak {hd[2]:.2f} GB over ctx, after {hd[3]:.2f} GB)  video rel_l2 {rel(vid, rv):.3e}  action rel_l2 {rel(act, ra):.3e}")
st, body = req(5, b""); print("info:", body[32:].decode())
srv.stdin.write(struct.pack("<IIQ", 0x51585250, 6, 0)); srv.stdin.flush(); srv.wait()
