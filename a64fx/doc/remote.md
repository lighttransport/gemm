# Fugaku remote dev guide

Local edits are mirrored to Fugaku with **Mutagen**; you build and run **on Fugaku** (cross-
compile `fccpx` on a login node, native `fcc` only inside jobs). This doc is the end-to-end
setup: SSH → Mutagen sync → job submission → bash-over-HTTP pseudo-interactive shell.

- **SSH alias:** `fugaku` → `login.fugaku.r-ccs.riken.jp` (user `u14346`, key `~/.ssh/fugaku`).
  `ssh fugaku` load-balances across login nodes `fn01svNN` — a later connection may land on a
  *different* frontend, so anything pinned to one node (a tmux socket, an `ssh -L` forward) is
  invisible from the others. Pin one frontend (nested `ssh fn01svNN`) for stateful work.
- **Repo (remote):** `~/work/gemm/ds4p` (= `/vol0006/mdt0/data/hp250467/work/gemm/ds4p`).
- **Git lives on the remote** — branch/commit/push by running `git` over SSH on Fugaku, not
  locally (the local tree is just the Mutagen alpha).
- **Assets (remote only):** `~/models` (weights), `~/data`. Use `readlink -f` on the remote for
  absolute paths when a job needs them.
- **Compilers:** `fccpx`/`FCCpx` cross-compile on the (x86) login node; `fcc`/`FCC` are native
  aarch64 and only run inside a `pjsub` job (batch or `--interact`). Build flags:
  `-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast [-fopenmp]`.

---

## Dev setup procedure (new branch / workdir)

### 1. SSH access
Add to `~/.ssh/config`:
```
Host fugaku
    Hostname login.fugaku.r-ccs.riken.jp
    User u14346
    ForwardAgent yes
    IdentityFile ~/.ssh/fugaku
    ServerAliveInterval 60
    ServerAliveCountMax 4
```
Verify: `ssh fugaku hostname`.

### 2. Mutagen sync (local ↔ remote)
One session mirrors the working dir both ways. `mutagen-ds4p.sh` (local-only helper, not
committed) is the template — copy it per workdir, set the three vars, run it:
```sh
LOCAL_DIR=/home/syoyo/work/fugaku/work/gemm/ds4p \
REMOTE_DIR=fugaku:/home/u14346/work/gemm/ds4p \
SESSION=ds4p \
  ./mutagen-ds4p.sh        # mutagen sync terminate <SESSION> ; mutagen sync create ...
```
For a **new branch/workdir**: pick a fresh local dir + remote dir + unique `SESSION` name and
run the same script — e.g. `SESSION=ds4p-exp REMOTE_DIR=fugaku:~/work/gemm/ds4p-exp`. Create the
remote dir first (`ssh fugaku 'git -C ~/work/gemm clone <repo> ds4p-exp && cd ds4p-exp && git checkout -b <branch>'`),
then point the session's beta at it. Manage sessions with:
```sh
mutagen sync list                 # status, conflicts, file counts
mutagen sync flush  <SESSION>      # force a sync now
mutagen sync monitor <SESSION>     # live
mutagen sync terminate <SESSION>
```
**What syncs:** source only. The session ignores VCS, build dirs (`build`, `CMakeFiles`, …),
agent dirs (`.claude`/`.codex`/`.agents`), and crucially **`*.out`, `*.o`, `*.a`, `*.so`,
`output.*`, `*.log`** — so compiled binaries and **job stdout never come back to local**. Read
them over SSH (see below). Edits land on the remote within a second or so; if a job's `make`
seems to use stale source, `mutagen sync flush <SESSION>` before submitting.

### 3. Build + submit a job
Submit from the synced remote dir (Mutagen has already mirrored your edits):
```sh
ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory <path/to/job.sh>'
```
Jobs typically `make ... CC=fcc` (native build at job start, so they pick up synced source) then
`mpiexec`. Standard `#PJM` header (small-s2 multi-node example):
```
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=4x4x3:torus,elapse=02:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=48"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
```
- `rscgrp=small` (≤1 node-ish, fast queue, elapse ≤72h) vs `rscgrp=small-s2` (multi-node torus,
  **elapse must be > 3600s**). `freq=2000` normal / `2200` boost; `eco_state=0` uses both FPUs.
- **Read output over SSH** (it does not sync back). The combined log is `<job.sh>.<jobid>.out`
  in the *submit* directory (the dir you `cd`'d to, not the script's dir):
  ```sh
  ssh fugaku 'cd ~/work/gemm/ds4p && pjstat <jobid>'                 # QUE/RUN/EXT
  ssh fugaku 'cd ~/work/gemm/ds4p && pjstat --estimate <jobid>'      # est. start time
  ssh fugaku 'cd ~/work/gemm/ds4p && grep -E "SENTINEL|tok/s|NaN" job.sh.<jobid>.out'
  ```
  `logmsg()`-style runners often write to a per-rank file (e.g. `m3_ep_rank00.txt`) in the job's
  cwd, **not** stdout — cat that too. Note `.out` is LLIO-buffered: it may not flush until the
  job ends.
- **Per-job cwd isolation:** multi-node uTofu jobs write `tofu_topo.txt` + rank logs in cwd;
  concurrent jobs sharing a dir collide. Use `RUN="$M3/run_${PJM_JOBID}"; mkdir -p "$RUN"; cd "$RUN"`.
- **CODE=1907** at `tofu_topo_helper` is an intermittent uTofu-init flake; retry the helper
  several times with a backoff (episodes can last ~30 min, then clear) before giving up.

### 4. bash-over-HTTP pseudo-interactive shell
For an interactive feel (run many native commands without one `pjsub` per command), a one-node
batch job runs `tools/bash_http_server.py` on the compute node and reverse-tunnels it to a pinned
login node; a local `ssh -L` gives a stable `http://127.0.0.1:21364` endpoint. This survives for
the job's `elapse` (up to 72h on `rscgrp=small`) and both tunnel ends auto-reconnect.

```sh
export DS4P_BASH_HTTP_TOKEN=...        # required bearer token (never commit it)
export LOGIN_NODE=1                    # pin loginN.fugaku.r-ccs.riken.jp (1..8)

tools/open_bash_http_local_tunnel.sh    # local ssh -L ControlMaster -> loginN; writes state.env
tools/submit_bash_http_1n_over_ssh.sh   # pjsub the 1-node bridge (reverse tunnel to the SAME loginN)
python3 tools/test_bash_http_remote.py http://127.0.0.1:21364 "$DS4P_BASH_HTTP_TOKEN"
# optional: keep the local forward alive across drops
nohup tools/watch_bash_http_local_tunnel.sh > /tmp/ds4p-bash-http/watch.log 2>&1 &
# ... run commands via tools/bash_http_client.py (new_session / run / close) ...
tools/close_bash_http_local_tunnel.sh   # tear down when done
```
Pin **one** login node for both the local `ssh -L` owner and the compute `ssh -R` target — the
forwarded port exists on exactly that frontend. `LOGIN_NODE` (or explicit `REMOTE=`) is the single
knob honored by all the tunnel scripts; the reverse target must be a **hostname**, never an IP
(compute nodes can't reach the internal frontend IP) and never a comma list (`pjsub -x` treats
commas as separators → `ST=ERR`). Set `ELAPSE=24:00:00 MAX_RETRY=20` etc. at submit time.

Why it works (full path):
```text
local client -> 127.0.0.1:21364
             -> ssh -L -> loginN loopback:21364
             -> ssh -R -> compute loopback:21264
             -> bash_http_server.py
```

---

## Reference

### Interactive jobs (`pjsub --interact`)
TTY + `script(1)` to capture a session, or a pinned-frontend `tmux` driven by `send-keys` /
`capture-pane`. Because `ssh fugaku` round-robins frontends, always nest to the chosen node:
```sh
ssh fugaku 'ssh fn01sv03 "tmux capture-pane -pt <session> -S -120"'
```
`a64fx/interactive-test/fugaku_tmux_mcp.py` wraps tmux start/send/capture/tail/status/stop as a
stdio MCP server (`FUGAKU_FRONTEND`, `FUGAKU_TMUX_SESSION`, `FUGAKU_TRANSCRIPT`). The tmux path is
flaky under frontend round-robin — for reliable automated runs prefer batch jobs or the
bash-over-HTTP bridge.

### uTofu topology
Regenerate `tofu_topo.txt` for **every** allocation: `mpiexec -np <N> ./tofu_topo_helper` (writes
rank → Tofu 6D coordinate rows). Runners read it via `TOFU_TOPO_PATH` (defaults to cwd). Per-job
cwd isolation (above) keeps concurrent jobs from clobbering each other's topo file.

### Long bash-HTTP sessions & reconnection
`rscgrp=small` allows `elapse` up to `72:00:00`. Both tunnel ends are supervised: the job side
(`pjsub_bash_http_1n.sh`) re-selects a reachable frontend and re-opens `ssh -R` on drop; the
client side (`watch_bash_http_local_tunnel.sh`) rebuilds the local `ssh -L`. Retry budget
`MAX_RETRY` (default 10 consecutive; a healthy re-establish resets it), `MONITOR_INTERVAL` 15 s,
`KEEPALIVE_SECONDS` 0 = run to elapse. Validated 2026-06-13 (jobs 49224567 / 49224908 / 49226036 /
49226425): tunnel survived killing either end; native `fcc` build + run worked through the
forwarded endpoint. Notes: compute → login SSH needs `IdentitiesOnly=yes` (avoids "Too many
authentication failures"); `bash_http_server.py` has a Py3.6 `ThreadingHTTPServer` fallback.
